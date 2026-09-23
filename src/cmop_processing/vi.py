"""
vi.py

Vegetation index calculations for georeferenced raster data.

Raster calculations are performed window-by-window so that very large
rasters do not need to be loaded entirely into memory.

Included vegetation indices
----------------------------
NDVI   - Normalized Difference Vegetation Index
NDRE   - Normalized Difference Red Edge Index
GNDVI  - Green Normalized Difference Vegetation Index
EVI    - Enhanced Vegetation Index
SAVI   - Soil Adjusted Vegetation Index
MSAVI2 - Modified Soil Adjusted Vegetation Index 2

All output rasters are written as tiled, compressed float32 GeoTIFFs.

Notes
-----
For indices containing additive constants (EVI, SAVI, MSAVI2), input
reflectance should be expressed from 0 to 1.

If the source rasters store scaled reflectance, such as 0-10000, set
``scale_factor=10000``. The input arrays will then be divided by 10000
before calculation.
"""

from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import rasterio


# ---------------------------------------------------------------------
# Private raster-processing helper
# ---------------------------------------------------------------------

def _process_rasters(
    input_paths: Sequence[str | Path],
    output_path: str | Path,
    calculation: Callable[..., np.ndarray],
    *,
    scale_factor: float = 1.0,
    nodata: float = -9999.0,
) -> Path:
    """
    Apply a calculation to multiple aligned rasters window-by-window.

    Parameters
    ----------
    input_paths : sequence of str or Path
        Paths to input raster bands.

    output_path : str or Path
        Path of the output GeoTIFF.

    calculation : callable
        Function receiving one NumPy array per input raster and returning
        the calculated output array.

    scale_factor : float, optional
        Number by which input pixel values are divided before calculation.

        Examples
        --------
        Reflectance stored as 0-1:
            scale_factor=1.0

        Reflectance stored as 0-10000:
            scale_factor=10000

    nodata : float, optional
        NoData value written to the output raster.

    Returns
    -------
    Path
        Path to the created raster.

    Raises
    ------
    ValueError
        If fewer than one raster is provided, scale_factor is invalid,
        or the input rasters do not share the same grid.
    """

    if not input_paths:
        raise ValueError("At least one input raster must be provided.")

    if scale_factor <= 0:
        raise ValueError("scale_factor must be greater than zero.")

    input_paths = [Path(path) for path in input_paths]
    output_path = Path(output_path)

    # Create output directory if necessary
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Open all input rasters.
    datasets = [rasterio.open(path) for path in input_paths]

    try:
        reference = datasets[0]

        # -------------------------------------------------------------
        # Verify that all rasters are spatially aligned.
        # -------------------------------------------------------------
        for path, dataset in zip(input_paths[1:], datasets[1:]):

            if dataset.width != reference.width:
                raise ValueError(
                    f"Raster width does not match:\n"
                    f"Reference: {input_paths[0]}\n"
                    f"Raster:    {path}"
                )

            if dataset.height != reference.height:
                raise ValueError(
                    f"Raster height does not match:\n"
                    f"Reference: {input_paths[0]}\n"
                    f"Raster:    {path}"
                )

            if dataset.crs != reference.crs:
                raise ValueError(
                    f"Raster CRS does not match:\n"
                    f"Reference: {input_paths[0]} ({reference.crs})\n"
                    f"Raster:    {path} ({dataset.crs})"
                )

            if not dataset.transform.almost_equals(reference.transform):
                raise ValueError(
                    f"Raster transforms do not match:\n"
                    f"Reference: {input_paths[0]}\n"
                    f"Raster:    {path}"
                )

        # -------------------------------------------------------------
        # Configure output raster.
        # -------------------------------------------------------------
        profile = reference.profile.copy()

        profile.update(
            driver="GTiff",
            dtype="float32",
            count=1,
            nodata=nodata,
            compress="deflate",
            predictor=3,
            tiled=True,
            BIGTIFF="YES",
        )

        with rasterio.open(output_path, "w", **profile) as dst:

            # ---------------------------------------------------------
            # Process one raster window at a time.
            #
            # This is the important part for very large orthomosaics:
            # only the current window is held in RAM.
            # ---------------------------------------------------------
            for _, window in reference.block_windows(1):

                arrays = []
                valid_mask = np.ones(
                    (window.height, window.width),
                    dtype=bool,
                )

                for dataset in datasets:

                    # Read as a masked array so source NoData values
                    # are automatically identified.
                    data = dataset.read(
                        1,
                        window=window,
                        masked=True,
                    )

                    # Pixels masked in ANY input raster become invalid.
                    valid_mask &= ~np.ma.getmaskarray(data)

                    # Convert to float32 before doing calculations.
                    array = np.asarray(
                        data.filled(np.nan),
                        dtype=np.float32,
                    )

                    # Convert stored values into reflectance.
                    if scale_factor != 1.0:
                        array /= scale_factor

                    # NaN or infinite input pixels are also invalid.
                    valid_mask &= np.isfinite(array)

                    arrays.append(array)

                # -----------------------------------------------------
                # Perform VI calculation.
                # -----------------------------------------------------
                with np.errstate(
                    divide="ignore",
                    invalid="ignore",
                    over="ignore",
                ):
                    result = calculation(*arrays)

                result = np.asarray(result, dtype=np.float32)

                # Invalid calculation results should become NoData.
                valid_mask &= np.isfinite(result)

                output = np.full(
                    result.shape,
                    nodata,
                    dtype=np.float32,
                )

                output[valid_mask] = result[valid_mask]

                dst.write(
                    output,
                    1,
                    window=window,
                )

    finally:
        # Make sure files are closed even if calculation fails.
        for dataset in datasets:
            dataset.close()

    return output_path


# ---------------------------------------------------------------------
# NDVI
# ---------------------------------------------------------------------

def ndvi(
    nir_path: str | Path,
    red_path: str | Path,
    output_path: str | Path,
    *,
    scale_factor: float = 1.0,
) -> Path:
    """
    Calculate Normalized Difference Vegetation Index (NDVI).

    NDVI = (NIR - Red) / (NIR + Red)

    Tucker (1979)
    """

    def calculate(nir, red):
        denominator = nir + red

        return np.divide(
            nir - red,
            denominator,
            out=np.full_like(nir, np.nan),
            where=denominator != 0,
        )

    return _process_rasters(
        [nir_path, red_path],
        output_path,
        calculate,
        scale_factor=scale_factor,
    )


# ---------------------------------------------------------------------
# NDRE
# ---------------------------------------------------------------------

def ndre(
    nir_path: str | Path,
    red_edge_path: str | Path,
    output_path: str | Path,
    *,
    scale_factor: float = 1.0,
) -> Path:
    """
    Calculate Normalized Difference Red Edge Index (NDRE).

    NDRE = (NIR - RedEdge) / (NIR + RedEdge)
    """

    def calculate(nir, red_edge):
        denominator = nir + red_edge

        return np.divide(
            nir - red_edge,
            denominator,
            out=np.full_like(nir, np.nan),
            where=denominator != 0,
        )

    return _process_rasters(
        [nir_path, red_edge_path],
        output_path,
        calculate,
        scale_factor=scale_factor,
    )


# ---------------------------------------------------------------------
# GNDVI
# ---------------------------------------------------------------------

def gndvi(
    nir_path: str | Path,
    green_path: str | Path,
    output_path: str | Path,
    *,
    scale_factor: float = 1.0,
) -> Path:
    """
    Calculate Green Normalized Difference Vegetation Index (GNDVI).

    GNDVI = (NIR - Green) / (NIR + Green)

    Gitelson et al. (1996)
    """

    def calculate(nir, green):
        denominator = nir + green

        return np.divide(
            nir - green,
            denominator,
            out=np.full_like(nir, np.nan),
            where=denominator != 0,
        )

    return _process_rasters(
        [nir_path, green_path],
        output_path,
        calculate,
        scale_factor=scale_factor,
    )


# ---------------------------------------------------------------------
# EVI
# ---------------------------------------------------------------------

def evi(
    nir_path: str | Path,
    red_path: str | Path,
    blue_path: str | Path,
    output_path: str | Path,
    *,
    scale_factor: float = 1.0,
) -> Path:
    """
    Calculate Enhanced Vegetation Index (EVI).

                       2.5 * (NIR - Red)
    EVI = ------------------------------------------------
           NIR + 6*Red - 7.5*Blue + 1

    Huete et al. (2002)

    Important
    ---------
    Reflectance must be on a 0-1 scale.

    For rasters stored as 0-10000, use:

        scale_factor=10000
    """

    def calculate(nir, red, blue):
        numerator = 2.5 * (nir - red)
        denominator = nir + 6.0 * red - 7.5 * blue + 1.0

        return np.divide(
            numerator,
            denominator,
            out=np.full_like(nir, np.nan),
            where=denominator != 0,
        )

    return _process_rasters(
        [nir_path, red_path, blue_path],
        output_path,
        calculate,
        scale_factor=scale_factor,
    )


# ---------------------------------------------------------------------
# SAVI
# ---------------------------------------------------------------------

def savi(
    nir_path: str | Path,
    red_path: str | Path,
    output_path: str | Path,
    *,
    L: float = 0.5,
    scale_factor: float = 1.0,
) -> Path:
    """
    Calculate Soil Adjusted Vegetation Index (SAVI).

             (NIR - Red)
    SAVI = ---------------- * (1 + L)
           NIR + Red + L

    The traditional soil adjustment factor is L = 0.5.

    Huete (1988)

    Important
    ---------
    Reflectance must be on a 0-1 scale.

    For rasters stored as 0-10000, use:

        scale_factor=10000
    """

    def calculate(nir, red):
        denominator = nir + red + L

        return np.divide(
            (nir - red) * (1.0 + L),
            denominator,
            out=np.full_like(nir, np.nan),
            where=denominator != 0,
        )

    return _process_rasters(
        [nir_path, red_path],
        output_path,
        calculate,
        scale_factor=scale_factor,
    )


# ---------------------------------------------------------------------
# MSAVI2
# ---------------------------------------------------------------------

def msavi2(
    nir_path: str | Path,
    red_path: str | Path,
    output_path: str | Path,
    *,
    scale_factor: float = 1.0,
) -> Path:
    """
    Calculate Modified Soil Adjusted Vegetation Index 2 (MSAVI2).

                          _______________________________
             2*NIR + 1 - sqrt((2*NIR + 1)^2 - 8*(NIR-Red))
    MSAVI2 = ------------------------------------------------
                                  2

    Qi et al. (1994)

    Important
    ---------
    Reflectance must be on a 0-1 scale.

    For rasters stored as 0-10000, use:

        scale_factor=10000
    """

    def calculate(nir, red):
        term = 2.0 * nir + 1.0

        discriminant = (
            term**2
            - 8.0 * (nir - red)
        )

        # Negative values can occur due to invalid input pixels or
        # numerical problems. They are converted to NaN rather than
        # producing an invalid square root.
        sqrt_term = np.where(
            discriminant >= 0,
            np.sqrt(np.maximum(discriminant, 0)),
            np.nan,
        )

        return (term - sqrt_term) / 2.0

    return _process_rasters(
        [nir_path, red_path],
        output_path,
        calculate,
        scale_factor=scale_factor,
    )