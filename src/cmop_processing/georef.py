from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from affine import Affine
from rasterio.transform import from_origin
from scipy.ndimage import map_coordinates


# ============================================================
# READ QGIS GCP POINTS
# ============================================================

def read_qgis_points(points_file):
    """Read enabled GCPs from a QGIS Georeferencer .points file.

    This workflow assumes sourceX/sourceY are map coordinates of the
    feature in the ORIGINAL raster and mapX/mapY are the surveyed/corrected
    coordinates, all in the raster CRS (normally EPSG:32610).
    """
    points_file = Path(points_file)
    if not points_file.exists():
        raise FileNotFoundError(f"Points file not found: {points_file}")

    df = pd.read_csv(points_file, comment="#")
    required = {"mapX", "mapY", "sourceX", "sourceY", "enable"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in .points file: {missing}")

    df = df[df["enable"] == 1].copy()
    if len(df) < 2:
        raise ValueError("At least two enabled GCPs are required for a Helmert transformation.")

    source_xy = df[["sourceX", "sourceY"]].to_numpy(dtype=float)
    destination_xy = df[["mapX", "mapY"]].to_numpy(dtype=float)
    return source_xy, destination_xy


# ============================================================
# FIT / APPLY HELMERT
# ============================================================

def fit_helmert(source_xy, destination_xy):
    """Fit source-map-coordinate -> corrected-map-coordinate 2-D Helmert."""
    src = np.asarray(source_xy, dtype=float)
    dst = np.asarray(destination_xy, dtype=float)

    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 2:
        raise ValueError("Source and destination coordinates must both have shape (n, 2).")
    if len(src) < 2:
        raise ValueError("At least two GCPs are required.")

    A, L = [], []
    for (x, y), (X, Y) in zip(src, dst):
        A.append([x, -y, 1, 0])
        A.append([y,  x, 0, 1])
        L.extend([X, Y])

    a, b, tx, ty = np.linalg.lstsq(
        np.asarray(A, dtype=float), np.asarray(L, dtype=float), rcond=None
    )[0]

    fitted = np.column_stack([
        a * src[:, 0] - b * src[:, 1] + tx,
        b * src[:, 0] + a * src[:, 1] + ty,
    ])
    residual_xy = dst - fitted
    residual_distance = np.hypot(residual_xy[:, 0], residual_xy[:, 1])

    return {
        "a": a,
        "b": b,
        "tx": tx,
        "ty": ty,
        "scale": np.hypot(a, b),
        "rotation_deg": np.degrees(np.arctan2(b, a)),
        "fitted": fitted,
        "residual_xy": residual_xy,
        "residual_distance": residual_distance,
        "rmse": np.sqrt(np.mean(residual_distance ** 2)),
    }


def helmert_to_affine(helmert):
    return Affine(
        helmert["a"], -helmert["b"], helmert["tx"],
        helmert["b"],  helmert["a"], helmert["ty"],
    )


def apply_helmert_xy(x, y, helmert):
    """Forward map: original map coordinate -> corrected map coordinate."""
    a, b = helmert["a"], helmert["b"]
    return (
        a * x - b * y + helmert["tx"],
        b * x + a * y + helmert["ty"],
    )


def inverse_helmert_xy(X, Y, helmert):
    """Inverse map: corrected map coordinate -> original map coordinate."""
    a, b = helmert["a"], helmert["b"]
    dx = X - helmert["tx"]
    dy = Y - helmert["ty"]
    denom = a * a + b * b
    return (
        (a * dx + b * dy) / denom,
        (-b * dx + a * dy) / denom,
    )


def affine_bounds(transform, width, height):
    corners = [
        transform * (0, 0),
        transform * (width, 0),
        transform * (0, height),
        transform * (width, height),
    ]
    xs = [p[0] for p in corners]
    ys = [p[1] for p in corners]
    return min(xs), min(ys), max(xs), max(ys)


# ============================================================
# COMMON NORTH-UP GRID
# ============================================================

def make_north_up_grid(reference_tif, helmert):
    """Build one north-up grid covering the forward-transformed reference raster.

    Resolution is the original reference-raster resolution. The Helmert scale
    changes where image content lands; it does NOT silently redefine the nominal
    output pixel size.
    """
    reference_tif = Path(reference_tif)
    with rasterio.open(reference_tif) as src:
        original_transform = src.transform
        left0, bottom0, right0, top0 = affine_bounds(
            original_transform, src.width, src.height
        )

        # Transform the four ORIGINAL raster corners through the Helmert.
        corners = [
            original_transform * (0, 0),
            original_transform * (src.width, 0),
            original_transform * (0, src.height),
            original_transform * (src.width, src.height),
        ]
        corrected_corners = [apply_helmert_xy(x, y, helmert) for x, y in corners]
        xs = [p[0] for p in corrected_corners]
        ys = [p[1] for p in corrected_corners]
        left, bottom, right, top = min(xs), min(ys), max(xs), max(ys)

        # Preserve original nominal ground sampling distance.
        pixel_width = np.hypot(original_transform.a, original_transform.d)
        pixel_height = np.hypot(original_transform.b, original_transform.e)

        width = int(np.ceil((right - left) / pixel_width))
        height = int(np.ceil((top - bottom) / pixel_height))
        dst_transform = from_origin(left, top, pixel_width, pixel_height)

    return {
        "transform": dst_transform,
        "width": width,
        "height": height,
        "pixel_width": pixel_width,
        "pixel_height": pixel_height,
        "original_bounds": (left0, bottom0, right0, top0),
        "corrected_bounds": (left, bottom, right, top),
    }


# ============================================================
# EXPLICIT IMAGE WARP
# ============================================================

def _warp_band_inverse_helmert(
    src,
    dst,
    band,
    helmert,
    dst_transform,
    dst_width,
    dst_height,
    block_rows=512,
):
    """Warp one band by explicit inverse mapping.

    For every destination pixel center:
      corrected map XY -> inverse Helmert -> original map XY -> source row/col.

    This guarantees that image FEATURES move from sourceX/sourceY toward
    mapX/mapY, rather than merely changing raster metadata.
    """
    inv_src = ~src.transform
    src_data = src.read(band)

    # map_coordinates works in array-center coordinates. Affine inverse returns
    # pixel-corner coordinates, so subtract 0.5 to obtain center-index coords.
    nodata = src.nodata
    if nodata is None:
        fill = np.nan if np.issubdtype(src_data.dtype, np.floating) else 0
    else:
        fill = nodata

    # Nearest-neighbor sampling does not interpolate values.
    # scipy map_coordinates accepts the native numeric array directly.
    work = src_data

    for row0 in range(0, dst_height, block_rows):
        h = min(block_rows, dst_height - row0)
        rows = np.arange(row0, row0 + h, dtype=np.float64) + 0.5
        cols = np.arange(dst_width, dtype=np.float64) + 0.5
        cc, rr = np.meshgrid(cols, rows)

        # Destination pixel centers -> corrected map coordinates.
        X = dst_transform.c + dst_transform.a * cc + dst_transform.b * rr
        Y = dst_transform.f + dst_transform.d * cc + dst_transform.e * rr

        # Corrected map coordinates -> ORIGINAL map coordinates.
        x, y = inverse_helmert_xy(X, Y, helmert)

        # Original map coordinates -> source pixel coordinates.
        src_col_corner = inv_src.a * x + inv_src.b * y + inv_src.c
        src_row_corner = inv_src.d * x + inv_src.e * y + inv_src.f
        src_cols = src_col_corner - 0.5
        src_rows = src_row_corner - 0.5

        warped = map_coordinates(
            work,
            [src_rows, src_cols],
            order=0,               # nearest-neighbor: preserve source pixel values
            mode="constant",
            cval=float(fill),
            prefilter=False,
        )

        if np.issubdtype(src_data.dtype, np.integer):
            info = np.iinfo(src_data.dtype)
            warped = np.clip(np.rint(warped), info.min, info.max).astype(src_data.dtype)
        else:
            warped = warped.astype(src_data.dtype, copy=False)

        dst.write(
            warped,
            band,
            window=rasterio.windows.Window(0, row0, dst_width, h),
        )


# ============================================================
# BATCH HELMERT GEOREFERENCE
# ============================================================

def batch_helmert_georeference(
    input_dir,
    output_dir,
    points_file,
    suffix="_GCP",
    expected_epsg=32610,
):
    """Apply one GCP-derived Helmert warp to every TIFF in a directory.

    All outputs are written to the SAME north-up grid. Raster values are sampled
    with nearest-neighbor interpolation so output values come directly from
    source pixels rather than being numerically averaged.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    points_file = Path(points_file)
    output_dir.mkdir(parents=True, exist_ok=True)

    source_xy, destination_xy = read_qgis_points(points_file)
    print(f"Loaded {len(source_xy)} enabled GCPs from {points_file.name}")

    helmert = fit_helmert(source_xy, destination_xy)
    print("\nHelmert transformation")
    print("----------------------")
    print(f"Scale:       {helmert['scale']:.9f}")
    print(f"Rotation:    {helmert['rotation_deg']:.6f} degrees")
    print(f"RMSE:        {helmert['rmse']:.4f} m")
    print("\nGCP residuals:")
    for i, residual in enumerate(helmert["residual_distance"], start=1):
        print(f"  GCP {i}: {residual:.4f} m")

    tif_files = sorted(
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".tif", ".tiff"}
    )
    if not tif_files:
        raise FileNotFoundError(f"No TIFF files found in:\n{input_dir}")
    print(f"\nFound {len(tif_files)} TIFF files.")

    # Validate the reference raster and CRS. Inputs may have different source
    # transforms/dimensions (for example a CHM), but they are all warped with
    # the same map-coordinate Helmert onto the shared output grid.
    reference_tif = tif_files[0]
    with rasterio.open(reference_tif) as ref:
        ref_crs = ref.crs
        if ref_crs is None or ref_crs.to_epsg() != expected_epsg:
            raise ValueError(
                f"{reference_tif.name} CRS is {ref_crs}; expected EPSG:{expected_epsg}."
            )

    for tif_path in tif_files:
        with rasterio.open(tif_path) as src:
            if src.crs is None or src.crs.to_epsg() != expected_epsg:
                raise ValueError(
                    f"{tif_path.name} has CRS {src.crs}; expected EPSG:{expected_epsg}."
                )

    grid = make_north_up_grid(reference_tif, helmert)
    dst_transform = grid["transform"]
    dst_width = grid["width"]
    dst_height = grid["height"]

    print("\nReference bounds")
    print("----------------")
    print(f"Original:  {grid['original_bounds']}")
    print(f"Corrected: {grid['corrected_bounds']}")

    print("\nCommon north-up output grid")
    print("---------------------------")
    print(f"Reference:    {reference_tif.name}")
    print(f"CRS:          EPSG:{expected_epsg}")
    print(f"Pixel width:  {grid['pixel_width']:.6f} m")
    print(f"Pixel height: {grid['pixel_height']:.6f} m")
    print(f"Columns:      {dst_width}")
    print(f"Rows:         {dst_height}\n")

    for src_path in tif_files:
        dst_path = output_dir / f"{src_path.stem}{suffix}{src_path.suffix}"
        print(f"Processing: {src_path.name}")

        with rasterio.open(src_path) as src:
            profile = src.profile.copy()

            # Do not inherit source TIFF block dimensions.
            # The output raster may have different dimensions, and GeoTIFF
            # tile dimensions must be multiples of 16.
            profile.pop("blockxsize", None)
            profile.pop("blockysize", None)

            profile.update(
                transform=dst_transform,
                width=dst_width,
                height=dst_height,
                crs=src.crs,
                compress="deflate",
                tiled=True,
                blockxsize=256,
                blockysize=256,
                BIGTIFF="IF_SAFER",
            )

            with rasterio.open(dst_path, "w", **profile) as dst:
                for band in range(1, src.count + 1):
                    _warp_band_inverse_helmert(
                        src=src,
                        dst=dst,
                        band=band,
                        helmert=helmert,
                        dst_transform=dst_transform,
                        dst_width=dst_width,
                        dst_height=dst_height,
                    )

        print(f"  -> {dst_path.name}")

    print("\n======================================")
    print("Finished")
    print("======================================")
    print("\nCorrected north-up TIFFs saved to:")
    print(output_dir)
    return helmert
