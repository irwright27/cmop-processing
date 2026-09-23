from pathlib import Path
import re
from statistics import median

import Metashape


def find_raw_images(input_dir):
    """
    Find UAV image files recursively within an input directory.

    Parameters
    ----------
    input_dir : str or Path
        Root folder containing raw UAV imagery.

    Returns
    -------
    list[Path]
        Sorted paths to image files.
    """
    input_dir = Path(input_dir)

    if not input_dir.exists():
        raise FileNotFoundError(
            f"Input directory does not exist: {input_dir}"
        )

    if not input_dir.is_dir():
        raise NotADirectoryError(
            f"Input path is not a directory: {input_dir}"
        )

    valid_extensions = {
        ".tif",
        ".tiff",
        ".jpg",
        ".jpeg",
    }

    photos = sorted(
        path
        for path in input_dir.rglob("*")
        if path.is_file()
        and path.suffix.lower() in valid_extensions
    )

    if not photos:
        raise FileNotFoundError(
            f"No UAV image files found in: {input_dir}"
        )

    return photos


def create_metashape_document(project_path=None):
    """
    Create a new Metashape document.

    If project_path is provided, immediately establish the .psx project
    before any chunks are created. This prevents chunk references from
    becoming stale when the project is first saved.

    Parameters
    ----------
    project_path : str or Path, optional
        Path to the Metashape .psx project.

    Returns
    -------
    Metashape.Document
        New Metashape document.
    """
    doc = Metashape.Document()

    if project_path is not None:
        project_path = Path(project_path)
        project_path.parent.mkdir(parents=True, exist_ok=True)
        doc.save(str(project_path))

    return doc



def open_metashape_document(project_file):
    """
    Open an existing Metashape .psx project for continued processing.

    Existing camera enabled/disabled states are preserved.
    """
    project_file = Path(project_file)

    if not project_file.exists():
        raise FileNotFoundError(
            f"Metashape project does not exist: {project_file}"
        )

    doc = Metashape.Document()
    doc.open(str(project_file))

    if getattr(doc, "read_only", False):
        raise RuntimeError(
            "Metashape project opened in read-only mode. Close the same "
            "project in the Metashape GUI or any other Python session, "
            "then reopen it."
        )

    if not doc.chunks:
        raise RuntimeError(
            f"Metashape project contains no chunks: {project_file}"
        )

    return doc, doc.chunk


def create_chunk(doc, label=None):
    """
    Create a new chunk in a Metashape document.
    """
    chunk = doc.addChunk()

    if label is not None:
        chunk.label = label

    return chunk


def load_multicamera_images(chunk, photos):
    """
    Load MicaSense images into a Metashape chunk as a multi-camera system.
    """
    photo_paths = [str(photo) for photo in photos]

    chunk.addPhotos(
        photo_paths,
        layout=Metashape.MultiplaneLayout,
    )

    return chunk


def set_primary_channel_panchro(chunk):
    """
    Set the primary channel of an Altum-PT chunk to the panchromatic sensor.
    """
    for index, sensor in enumerate(chunk.sensors):
        if "panchro" in sensor.label.lower():
            chunk.primary_channel = index
            return index

    raise ValueError(
        "No panchromatic sensor was found in the chunk."
    )


def locate_reflectance_panels(chunk):
    """
    Locate MicaSense reflectance calibration panels using their QR codes.
    """
    chunk.locateReflectancePanels()
    return chunk


def load_reflectance_panel_calibration(chunk, calibration_file):
    """
    Load the reflectance calibration CSV for the detected MicaSense panel.
    """
    calibration_file = Path(calibration_file)

    if not calibration_file.exists():
        raise FileNotFoundError(
            f"Calibration file does not exist: {calibration_file}"
        )

    chunk.loadReflectancePanelCalibration(
        str(calibration_file)
    )

    return chunk


def _meta_float(meta, key):
    """
    Read a metadata value as float.

    Returns None when the key is absent or cannot be converted.
    """
    try:
        value = meta[key]
    except (KeyError, TypeError):
        return None

    if value is None:
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _capture_number(camera_label):
    """
    Extract the numeric capture number from an Altum-PT camera label.

    Examples
    --------
    IMG_0000_1 -> 0
    IMG_0486_6 -> 486
    """
    capture_id = _capture_id(camera_label)
    match = re.search(r"(\d+)$", capture_id)

    if match is None:
        return None

    return int(match.group(1))


def correct_panel_dls_from_early_flight(
    chunk,
    n_flight_captures=10,
    substitution_threshold=0.8,
):
    """
    Correct anomalously low panel DLS irradiance values band-by-band.

    For each calibration-panel image and each spectral sensor independently,
    this function compares the panel DLS irradiance with nearby flight DLS
    measurements from the same sensor.

    Reference-capture selection
    ---------------------------
    - Panel near the start of the flight:
      use the first ``n_flight_captures`` non-calibration captures after it.

    - Panel near the end of the flight:
      use the last ``n_flight_captures`` non-calibration captures before it.

    - Panel in the middle of the flight:
      use nearby captures around it, split as evenly as possible between
      before and after. With the default of 10, this is 5 before + 5 after.

    If

        panel_irradiance < substitution_threshold * reference_median

    the panel irradiance is replaced in Metashape's in-memory metadata with
    the median reference irradiance for that band.

    Each Altum-PT band is treated independently. Source TIFF files are not
    modified.

    Parameters
    ----------
    chunk : Metashape.Chunk
        Altum-PT chunk after reflectance panels have been located.

    n_flight_captures : int, default 10
        Total number of nearby non-calibration captures used per panel/band.

    substitution_threshold : float, default 0.8
        Trigger threshold expressed as a fraction of the nearby-flight
        median. For example, 0.8 means substitution occurs when the panel
        DLS value is less than 80% of the reference median.

    Returns
    -------
    list[dict]
        One diagnostic record per calibration-panel camera/band.
    """
    if n_flight_captures < 1:
        raise ValueError("n_flight_captures must be at least 1.")

    if not 0 < substitution_threshold <= 1:
        raise ValueError(
            "substitution_threshold must be greater than 0 and <= 1."
        )

    irradiance_keys = (
        "Camera/Irradiance",
        "DLS/SpectralIrradiance",
        "MicaSense/SpectralIrradiance",
        "MicaSense/Irradiance",
    )

    def is_calibration_camera(camera):
        group_label = (
            camera.group.label.lower()
            if camera.group is not None
            else ""
        )
        return "calibration" in group_label

    panel_cameras = [
        camera
        for camera in chunk.cameras
        if camera.photo is not None and is_calibration_camera(camera)
    ]

    if not panel_cameras:
        raise RuntimeError(
            "No reflectance-panel cameras were found in a calibration "
            "image group. Run locate_reflectance_panels() first."
        )

    panel_capture_ids = {
        _capture_id(camera.label)
        for camera in panel_cameras
    }

    diagnostics = []

    for panel_camera in panel_cameras:
        sensor = panel_camera.sensor
        sensor_label = str(sensor.label)
        panel_number = _capture_number(panel_camera.label)

        if panel_number is None:
            raise RuntimeError(
                f"Could not parse panel capture number from "
                f"{panel_camera.label}."
            )

        panel_meta = panel_camera.photo.meta

        panel_key = None
        panel_value = None

        for key in irradiance_keys:
            value = _meta_float(panel_meta, key)

            if value is not None:
                panel_key = key
                panel_value = value
                break

        if panel_key is None:
            print(
                f"{sensor_label}: no usable panel DLS irradiance found "
                "-- skipping DLS substitution for this band."
            )
            diagnostics.append(
                {
                    "sensor": sensor_label,
                    "panel_camera": panel_camera.label,
                    "panel_irradiance": None,
                    "reference_median": None,
                    "threshold_value": None,
                    "substituted": False,
                    "reason": "no_panel_irradiance",
                }
            )
            continue

        # ---------------------------------------------------------
        # Collect all usable non-calibration captures for this sensor.
        # ---------------------------------------------------------
        before = []
        after = []

        for camera in chunk.cameras:
            if not camera.enabled:
                continue

            if camera.photo is None:
                continue

            if is_calibration_camera(camera):
                continue

            if _capture_id(camera.label) in panel_capture_ids:
                continue

            try:
                same_sensor = camera.sensor.key == sensor.key
            except Exception:
                same_sensor = camera.sensor.label == sensor.label

            if not same_sensor:
                continue

            capture_number = _capture_number(camera.label)

            if capture_number is None:
                continue

            value = _meta_float(camera.photo.meta, panel_key)

            if value is None:
                for key in irradiance_keys:
                    value = _meta_float(camera.photo.meta, key)

                    if value is not None:
                        break

            if value is None or value <= 0:
                continue

            item = (capture_number, camera.label, value)

            if capture_number < panel_number:
                before.append(item)
            elif capture_number > panel_number:
                after.append(item)

        before.sort(key=lambda item: item[0])
        after.sort(key=lambda item: item[0])

        # ---------------------------------------------------------
        # Choose nearby reference captures.
        #
        # Start panel:
        #   not enough captures before -> use first N after
        #
        # End panel:
        #   not enough captures after -> use last N before
        #
        # Middle panel:
        #   use an even split around the panel where possible
        # ---------------------------------------------------------
        half_before = n_flight_captures // 2
        half_after = n_flight_captures - half_before

        if len(before) < half_before and len(after) >= n_flight_captures:
            reference_items = after[:n_flight_captures]
            reference_mode = "after"

        elif len(after) < half_after and len(before) >= n_flight_captures:
            reference_items = before[-n_flight_captures:]
            reference_mode = "before"

        else:
            selected_before = before[-min(half_before, len(before)):]
            selected_after = after[:min(half_after, len(after))]

            reference_items = selected_before + selected_after

            # If one side could not supply its share, top up from the other
            # side with the nearest remaining captures.
            if len(reference_items) < n_flight_captures:
                used_labels = {item[1] for item in reference_items}

                remaining = [
                    item
                    for item in before + after
                    if item[1] not in used_labels
                ]

                remaining.sort(
                    key=lambda item: abs(item[0] - panel_number)
                )

                needed = n_flight_captures - len(reference_items)
                reference_items.extend(remaining[:needed])

            reference_items.sort(key=lambda item: item[0])
            reference_mode = "around"

        if len(reference_items) < n_flight_captures:
            raise RuntimeError(
                f"{sensor_label}: found only {len(reference_items)} usable "
                f"nearby DLS values for panel {panel_camera.label}; "
                f"{n_flight_captures} are required."
            )

        reference_values = [
            value
            for _, _, value in reference_items
        ]

        reference_labels = [
            label
            for _, label, _ in reference_items
        ]

        reference_median = median(reference_values)
        threshold_value = substitution_threshold * reference_median
        substitute = panel_value < threshold_value

        updated_keys = []

        if substitute:
            for key in irradiance_keys:
                if _meta_float(panel_meta, key) is None:
                    continue

                panel_meta[key] = str(reference_median)
                updated_keys.append(key)

            print(
                f"{sensor_label} / {panel_camera.label}: panel DLS "
                f"{panel_value:.6g} is below "
                f"{substitution_threshold:.0%} of {reference_mode} "
                f"reference median {reference_median:.6g} "
                f"-> substituting {reference_median:.6g}"
            )
        else:
            print(
                f"{sensor_label} / {panel_camera.label}: panel DLS "
                f"{panel_value:.6g} is >= "
                f"{substitution_threshold:.0%} of {reference_mode} "
                f"reference median {reference_median:.6g} "
                "-> keeping original panel value"
            )

        diagnostics.append(
            {
                "sensor": sensor_label,
                "panel_camera": panel_camera.label,
                "metadata_key": panel_key,
                "panel_irradiance": panel_value,
                "reference_mode": reference_mode,
                "reference_labels": reference_labels,
                "reference_values": reference_values,
                "reference_median": reference_median,
                "threshold_value": threshold_value,
                "substituted": substitute,
                "replacement_value": (
                    reference_median if substitute else panel_value
                ),
                "updated_keys": updated_keys,
            }
        )

    return diagnostics

def calibrate_reflectance(
    chunk,
    cal_panel=True,
    cal_sun_sensor=True,
    dls_reference_captures=10,
    dls_substitution_threshold=0.8,
):
    """
    Calibrate Altum-PT imagery using reflectance panels, the sun sensor,
    or both.

    When both panel and sun-sensor calibration are enabled, panel DLS values
    are checked for anomalously low irradiance before calibration. The DLS
    correction is not needed for panel-only or sun-sensor-only calibration.
    """
    if not cal_panel and not cal_sun_sensor:
        raise ValueError(
            "At least one of cal_panel or cal_sun_sensor must be True."
        )

    dls_diagnostics = []

    if cal_panel and cal_sun_sensor:
        dls_diagnostics = correct_panel_dls_from_early_flight(
            chunk,
            n_flight_captures=dls_reference_captures,
            substitution_threshold=dls_substitution_threshold,
        )

    chunk.calibrateReflectance(
        use_reflectance_panels=cal_panel,
        use_sun_sensor=cal_sun_sensor,
    )

    return dls_diagnostics


def _tie_point_count(chunk):
    """
    Return the number of tie points currently stored in the chunk.
    """
    try:
        if chunk.tie_points is None:
            return 0
        return len(chunk.tie_points.points)
    except Exception:
        return 0


def _aligned_enabled_count(chunk):
    """
    Return the number of enabled cameras that currently have transforms.
    """
    return sum(
        1
        for camera in chunk.cameras
        if camera.enabled and camera.transform is not None
    )


def _enabled_camera_count(chunk):
    """
    Return the number of enabled cameras.
    """
    return sum(1 for camera in chunk.cameras if camera.enabled)


def align_photos(chunk):
    """
    Match and align Altum-PT photos using the SOP settings.
    """
    chunk.matchPhotos(
        downscale=1,
        generic_preselection=True,
        reference_preselection=True,
        reference_preselection_mode=Metashape.ReferencePreselectionSource,
        keypoint_limit=40000,
        tiepoint_limit=4000,
        filter_stationary_points=True,
        guided_matching=False,
    )

    chunk.alignCameras(
        adaptive_fitting=False
    )

    return chunk


def align_from_existing_tie_points(chunk):
    """
    Align cameras using tie points already stored in the project.

    This is useful when a previous matching pass succeeded but camera
    alignment did not. Matching is not repeated.
    """
    tie_points = _tie_point_count(chunk)

    if tie_points == 0:
        raise RuntimeError(
            "No existing tie points are available for camera alignment."
        )

    chunk.alignCameras(
        reset_alignment=True,
        adaptive_fitting=False,
    )

    aligned = _aligned_enabled_count(chunk)

    if aligned == 0:
        raise RuntimeError(
            "Metashape still aligned zero enabled cameras using the "
            f"existing {tie_points:,} tie points."
        )

    return chunk


def _normalize_crs_string(crs):
    crs = str(crs)
    if crs.upper().startswith("EPSG:") and not crs.upper().startswith("EPSG::"):
        code = crs.split(":")[-1]
        return f"EPSG::{code}"
    return crs


def make_ortho_projection(crs="EPSG::32610"):
    """
    Create a Metashape OrthoProjection from a CRS string.
    """
    projection = Metashape.OrthoProjection()
    projection.crs = Metashape.CoordinateSystem(
        _normalize_crs_string(crs)
    )
    return projection


def infer_flight_name(input_dir):
    """
    Infer the flight name from a typical .../<flight>/raw/<set> layout.
    """
    input_dir = Path(input_dir)

    if input_dir.parent.name.lower() == "raw":
        return input_dir.parent.parent.name

    if input_dir.name.lower() == "raw":
        return input_dir.parent.name

    for parent in input_dir.parents:
        if parent.name.lower() == "raw":
            return parent.parent.name

    return input_dir.name


def save_project(doc, project_path=None):
    """
    Save the Metashape project.

    On the first save, a project path is required.
    On subsequent saves, save the already-open project in place.
    """
    if not doc.path:
        if project_path is None:
            raise ValueError(
                "project_path is required for the first project save."
            )

        project_path = Path(project_path)
        project_path.parent.mkdir(parents=True, exist_ok=True)

        doc.save(str(project_path))

    else:
        doc.save()

    return Path(doc.path)


def _capture_id(camera_label):
    """
    Convert IMG_0123_4 -> IMG_0123 for Altum-PT image grouping.
    """
    match = re.match(r"^(.*)_([1-7])$", camera_label)
    if match:
        return match.group(1)
    return camera_label


def _robust_spread(values):
    """
    Robust standard-deviation-like spread based on median absolute deviation.
    """
    if not values:
        return 0.0

    center = median(values)
    mad = median(abs(value - center) for value in values)
    return 1.4826 * mad


def _angle_difference(a, b):
    """
    Smallest signed angular difference in degrees.
    """
    return (a - b + 180.0) % 360.0 - 180.0


def disable_calibration_images(chunk):
    """
    Disable images that Metashape has placed in a calibration-image group.

    Images are disabled rather than deleted so that the original imagery
    remains in the project.
    """
    disabled = []

    for camera in chunk.cameras:
        group_label = camera.group.label.lower() if camera.group else ""

        if "calibration" in group_label:
            camera.enabled = False
            disabled.append(camera.label)

    return disabled


def filter_nonmapping_captures(
    chunk,
    altitude_min_tolerance=15.0,
    heading_min_tolerance=15.0,
    edge_fraction=0.20,
    min_transit_run=3,
    mad_multiplier=5.0,
):
    """
    Disable takeoff, landing, and transit captures while retaining
    normal mapping rows and turnaround images.

    Filtering logic
    ---------------
    1. Strong altitude outliers are disabled anywhere in the flight.

    2. Transit is identified using heading AND temporal position:
       - the capture must have a heading-axis outlier,
       - it must occur near the beginning or end of the flight,
       - and it must be part of a consecutive run of heading outliers.

    Turnaround captures are retained because isolated heading deviations
    occurring throughout the interior of the mission are not treated as
    transit.

    Opposing mapping headings are treated as equivalent. For example,
    headings of 20 and 200 degrees represent the same flight-line axis.

    Parameters
    ----------
    chunk : Metashape.Chunk
        Chunk containing Altum-PT imagery.

    altitude_min_tolerance : float, default 15
        Minimum altitude deviation in meters.

    heading_min_tolerance : float, default 15
        Minimum deviation from the dominant flight-line axis in degrees.

    edge_fraction : float, default 0.20
        Fraction of the capture sequence at each end considered eligible
        for transit filtering. 0.20 means the first and last 20%.

    min_transit_run : int, default 3
        Minimum number of consecutive heading outliers required to classify
        a run as transit.

    mad_multiplier : float, default 5
        Multiplier applied to robust MAD-based spread estimates.

    Returns
    -------
    dict
        Filtering summary.
    """

    def axis_angle_difference(a, b):
        """
        Smallest angular difference when headings 180 degrees apart
        are considered equivalent.

        Result is in the range [-90, 90).
        """
        return (a - b + 90.0) % 180.0 - 90.0

    def capture_number(capture_id):
        """
        IMG_0123 -> 123
        """
        match = re.search(r"(\d+)$", capture_id)

        if match:
            return int(match.group(1))

        return 0

    # ---------------------------------------------------------
    # Group the seven Altum-PT image planes into captures
    # ---------------------------------------------------------

    captures = {}

    for camera in chunk.cameras:

        if not camera.enabled:
            continue

        cid = _capture_id(camera.label)

        captures.setdefault(
            cid,
            [],
        ).append(camera)

    representatives = []

    for cid, cameras in captures.items():

        representative = None

        for camera in cameras:

            if camera.reference.location is not None:
                representative = camera
                break

        if representative is None:
            continue

        location = representative.reference.location
        meta = representative.photo.meta

        try:
            yaw = meta["DLS/Yaw"]
        except KeyError:
            continue

        if yaw is None:
            continue

        # Metashape exposes the DLS value in radians for these images.
        yaw_degrees = (
            float(yaw)
            * 180.0
            / 3.141592653589793
        )

        representatives.append(
            {
                "capture_id": cid,
                "capture_number": capture_number(cid),
                "cameras": cameras,
                "altitude": float(location.z),
                "yaw": yaw_degrees,
            }
        )

    if len(representatives) < 10:

        return {
            "disabled_capture_count": 0,
            "disabled_camera_count": 0,
            "disabled_capture_ids": [],
            "reason": (
                "Too few captures with GPS and DLS yaw metadata "
                "to filter safely."
            ),
        }

    # Sort in flight order.
    representatives.sort(
        key=lambda item: item["capture_number"]
    )

    n = len(representatives)

    # ---------------------------------------------------------
    # Altitude statistics
    # ---------------------------------------------------------

    altitudes = [
        item["altitude"]
        for item in representatives
    ]

    altitude_center = median(altitudes)

    altitude_tolerance = max(
        altitude_min_tolerance,
        mad_multiplier * _robust_spread(altitudes),
    )

    # ---------------------------------------------------------
    # Determine dominant mapping heading from central 60%
    #
    # Beginning/end are intentionally excluded because those are
    # the areas where transit is expected.
    # ---------------------------------------------------------

    middle_start = int(n * 0.20)
    middle_end = int(n * 0.80)

    middle = representatives[
        middle_start:middle_end
    ]

    middle_yaws = [
        item["yaw"] % 180.0
        for item in middle
    ]

    heading_reference = middle_yaws[0]

    heading_offsets = [
        axis_angle_difference(
            yaw,
            heading_reference,
        )
        for yaw in middle_yaws
    ]

    heading_center = (
        heading_reference
        + median(heading_offsets)
    ) % 180.0

    heading_deviations = [
        abs(
            axis_angle_difference(
                yaw,
                heading_center,
            )
        )
        for yaw in middle_yaws
    ]

    heading_tolerance = max(
        heading_min_tolerance,
        mad_multiplier
        * _robust_spread(heading_deviations),
    )

    # ---------------------------------------------------------
    # Classify each capture
    # ---------------------------------------------------------

    edge_count = max(
        1,
        int(n * edge_fraction),
    )

    for i, item in enumerate(representatives):

        item["heading_deviation"] = abs(
            axis_angle_difference(
                item["yaw"],
                heading_center,
            )
        )

        item["heading_outlier"] = (
            item["heading_deviation"]
            > heading_tolerance
        )

        item["edge_capture"] = (
            i < edge_count
            or i >= n - edge_count
        )

        item["altitude_outlier"] = (
            abs(
                item["altitude"]
                - altitude_center
            )
            > altitude_tolerance
        )

    # ---------------------------------------------------------
    # Find consecutive runs of heading outliers at flight edges
    # ---------------------------------------------------------

    transit_indices = set()

    for start, stop in (
        (0, edge_count),
        (n - edge_count, n),
    ):

        run = []

        for i in range(start, stop):

            item = representatives[i]

            if item["heading_outlier"]:

                run.append(i)

            else:

                if len(run) >= min_transit_run:
                    transit_indices.update(run)

                run = []

        if len(run) >= min_transit_run:
            transit_indices.update(run)

    # ---------------------------------------------------------
    # Disable captures
    # ---------------------------------------------------------

    disabled_capture_ids = []
    disabled_camera_count = 0

    altitude_disabled_ids = []
    transit_disabled_ids = []

    for i, item in enumerate(representatives):

        altitude_outlier = item[
            "altitude_outlier"
        ]

        transit_outlier = (
            i in transit_indices
        )

        if altitude_outlier or transit_outlier:

            disabled_capture_ids.append(
                item["capture_id"]
            )

            if altitude_outlier:
                altitude_disabled_ids.append(
                    item["capture_id"]
                )

            if transit_outlier:
                transit_disabled_ids.append(
                    item["capture_id"]
                )

            for camera in item["cameras"]:

                camera.enabled = False
                disabled_camera_count += 1

    return {
        "disabled_capture_count": len(
            disabled_capture_ids
        ),
        "disabled_camera_count": disabled_camera_count,
        "disabled_capture_ids": disabled_capture_ids,

        "altitude_disabled_ids": altitude_disabled_ids,
        "transit_disabled_ids": transit_disabled_ids,

        "captures_evaluated": n,

        "altitude_center": altitude_center,
        "altitude_tolerance": altitude_tolerance,

        "heading_center": heading_center,
        "heading_tolerance": heading_tolerance,

        "edge_fraction": edge_fraction,
        "edge_capture_count": edge_count,
        "min_transit_run": min_transit_run,
    }


def retry_unaligned_cameras(chunk):
    """
    Retry enabled cameras that failed the first alignment attempt.
    """
    unaligned = [
        camera
        for camera in chunk.cameras
        if camera.enabled and camera.transform is None
    ]

    if unaligned:
        chunk.alignCameras(
            cameras=unaligned,
            reset_alignment=False,
            adaptive_fitting=False,
        )

    remaining = [
        camera
        for camera in chunk.cameras
        if camera.enabled and camera.transform is None
    ]

    return remaining


def optimize_cameras(chunk):
    """
    Optimize cameras using the SOP camera-model settings.
    """
    chunk.optimizeCameras(
        fit_f=True,
        fit_cx=True,
        fit_cy=True,
        fit_b1=False,
        fit_b2=False,
        fit_k1=True,
        fit_k2=True,
        fit_k3=True,
        fit_k4=False,
        fit_p1=True,
        fit_p2=True,
        fit_corrections=False,
        adaptive_fitting=False,
        tiepoint_covariance=False,
    )

    return chunk


def build_point_cloud(chunk):
    """
    Build depth maps and the dense point cloud using the SOP settings.

    SOP:
    - Quality: Medium
    - Depth filtering: Aggressive
    - Reuse depth maps: off
    - Calculate point colors: on
    - Calculate point confidence: off
    """
    chunk.buildDepthMaps(
        downscale=4,
        filter_mode=Metashape.AggressiveFiltering,
        reuse_depth=False,
    )

    chunk.buildPointCloud(
        source_data=Metashape.DepthMapsData,
        point_colors=True,
        point_confidence=False,
    )

    if chunk.point_cloud is None:
        raise RuntimeError("Point cloud was not created.")

    chunk.point_cloud.label = "Point Cloud"

    return chunk.point_cloud


def build_dsm(chunk, projection):
    """
    Build the DSM from all point-cloud classes.
    """
    chunk.buildDem(
        source_data=Metashape.PointCloudData,
        interpolation=Metashape.EnabledInterpolation,
        projection=projection,
    )

    if chunk.elevation is None:
        raise RuntimeError("DSM was not created.")

    chunk.elevation.label = "DSM"
    return chunk.elevation


def build_orthomosaic(chunk, projection):
    """
    Build the orthomosaic using the currently active DSM as the surface.
    """
    chunk.buildOrthomosaic(
        surface_data=Metashape.ElevationData,
        blending_mode=Metashape.MosaicBlending,
        fill_holes=True,
        ghosting_filter=False,
        cull_faces=False,
        refine_seamlines=False,
        projection=projection,
    )

    if chunk.orthomosaic is None:
        raise RuntimeError("Orthomosaic was not created.")

    chunk.orthomosaic.label = "Orthomosaic"
    return chunk.orthomosaic


def classify_ground_points(chunk):
    """
    Classify ground points using the exact SOP settings.

    SOP settings:
    - From: Any class
    - To: Ground + Low Points
    - Keep existing ground points: off
    - Max angle: 10 degrees
    - Max distance: 1 m
    - Max terrain slope: 10 degrees
    - Cell size: 10 m
    - Return number: Any Return
    - Erosion radius: 0 m

    Metashape's classifyGroundPoints operation assigns the detected terrain
    points to the Ground class and internally handles low-point rejection.
    """
    if chunk.point_cloud is None:
        raise RuntimeError(
            "A point cloud must exist before ground classification."
        )

    chunk.point_cloud.classifyGroundPoints(
        max_angle=10.0,
        max_distance=1.0,
        max_terrain_slope=10.0,
        cell_size=10.0,
        erosion_radius=0.0,
        return_number=0,
        keep_existing=False,
    )

    return chunk.point_cloud


def build_dem(chunk, projection):
    """
    Build a terrain DEM from Ground-class points only.
    """
    chunk.buildDem(
        source_data=Metashape.PointCloudData,
        interpolation=Metashape.EnabledInterpolation,
        projection=projection,
        classes=[Metashape.PointClass.Ground],
    )

    if chunk.elevation is None:
        raise RuntimeError("Ground DEM was not created.")

    chunk.elevation.label = "DEM"
    return chunk.elevation


def build_chm(chunk, dsm, dem, projection):
    """
    Build a canopy height model as DSM minus DEM.

    CHM = DSM - DEM
    """
    chunk.transformRaster(
        source_data=Metashape.ElevationData,
        asset=dsm.key,
        subtract=True,
        operand_asset=dem.key,
        projection=projection,
    )

    if chunk.elevation is None:
        raise RuntimeError("CHM was not created.")

    chm = chunk.elevation
    chm.label = "CHM"

    return chm


def make_tiff_compression():
    """
    Create TIFF export settings matching the SOP.
    """
    compression = Metashape.ImageCompression()
    compression.tiff_big = True
    compression.tiff_tiled = True
    compression.tiff_overviews = True
    compression.tiff_compression = (
        Metashape.ImageCompression.TiffCompressionLZW
    )
    return compression


def export_elevation(
    chunk,
    elevation,
    output_path,
    projection,
):
    """
    Export a Metashape elevation asset as GeoTIFF.
    """
    output_path = Path(output_path)

    chunk.exportRaster(
        path=str(output_path),
        source_data=Metashape.ElevationData,
        asset=elevation.key,
        projection=projection,
        image_format=Metashape.ImageFormatTIFF,
        raster_transform=Metashape.RasterTransformNone,
        image_compression=make_tiff_compression(),
        nodata_value=-32767,
        save_alpha=False,
    )

    return output_path


def export_point_cloud(
    chunk,
    output_path,
    crs="EPSG::32610",
):
    """
    Export the dense point cloud as LAS with colors and classifications.
    """
    output_path = Path(output_path)
    output_crs = Metashape.CoordinateSystem(
        _normalize_crs_string(crs)
    )

    chunk.exportPointCloud(
        path=str(output_path),
        source_data=Metashape.PointCloudData,
        format=Metashape.PointCloudFormatLAS,
        crs=output_crs,
        save_point_color=True,
        save_point_classification=True,
        save_point_confidence=False,
        save_point_normal=False,
    )

    return output_path


ALTUM_PT_EXPORT_FORMULAS = {
    "B1_blue": (
        "B1 * (B3 / "
        "(0.2 * B1 + 0.2 * B2 + 0.2 * B4 + 0.2 * B5 + 0.2 * B6)"
        ")/32768"
    ),
    "B2_green": (
        "B2 * (B3 / "
        "(0.2 * B1 + 0.2 * B2 + 0.2 * B4 + 0.2 * B5 + 0.2 * B6)"
        ")/32768"
    ),
    "B4_red": (
        "B4 * (B3 / "
        "(0.2 * B1 + 0.2 * B2 + 0.2 * B4 + 0.2 * B5 + 0.2 * B6)"
        ")/32768"
    ),
    "B5_rededge": (
        "B5 * (B3 / "
        "(0.2 * B1 + 0.2 * B2 + 0.2 * B4 + 0.2 * B5 + 0.2 * B6)"
        ")/32768"
    ),
    "B6_nir": (
        "B6 * (B3 / "
        "(0.2 * B1 + 0.2 * B2 + 0.2 * B4 + 0.2 * B5 + 0.2 * B6)"
        ")/32768"
    ),
    "B7_C": "(B7 / 100) - 273.15",
    "B7_K": "B7 / 100",
}



def export_rgb_orthomosaic(
    chunk,
    output_path,
    projection,
):
    """
    Export a pan-sharpened RGB orthomosaic GeoTIFF.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    denominator = (
        "(0.2 * B1 + 0.2 * B2 + 0.2 * B4 + "
        "0.2 * B5 + 0.2 * B6)"
    )

    red = f"B4 * (B3 / {denominator}) / 32768"
    green = f"B2 * (B3 / {denominator}) / 32768"
    blue = f"B1 * (B3 / {denominator}) / 32768"

    chunk.raster_transform.formula = [red, green, blue]
    chunk.raster_transform.enabled = True

    try:
        chunk.exportRaster(
            path=str(output_path),
            source_data=Metashape.OrthomosaicData,
            projection=projection,
            image_format=Metashape.ImageFormatTIFF,
            raster_transform=Metashape.RasterTransformValue,
            image_compression=make_tiff_compression(),
            nodata_value=-32767,
            save_alpha=False,
        )
    finally:
        chunk.raster_transform.enabled = False

    return output_path


def export_altum_pt_bands(
    chunk,
    output_dir,
    prefix,
    projection,
    overwrite=False,
):
    """
    Export Altum-PT analysis rasters using the SOP equations.

    Existing non-empty TIFFs are preserved by default so interrupted export
    runs can resume without rewriting completed files.

    B3 Panchro is used for sharpening but is not exported separately.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs = {}

    try:
        for band_name, formula in ALTUM_PT_EXPORT_FORMULAS.items():
            output_path = output_dir / f"{prefix}_{band_name}.tif"

            if _file_ready(output_path) and not overwrite:
                print(
                    f"  Existing output found -- skipping: "
                    f"{output_path.name}"
                )
                outputs[band_name] = output_path
                continue

            chunk.raster_transform.formula = [formula]
            chunk.raster_transform.enabled = True

            chunk.exportRaster(
                path=str(output_path),
                source_data=Metashape.OrthomosaicData,
                projection=projection,
                image_format=Metashape.ImageFormatTIFF,
                raster_transform=Metashape.RasterTransformValue,
                image_compression=make_tiff_compression(),
                nodata_value=-32767,
                save_alpha=False,
            )

            outputs[band_name] = output_path

    finally:
        chunk.raster_transform.enabled = False

    return outputs


def _find_asset_by_label(assets, label):
    """
    Return the first Metashape asset whose label matches case-insensitively.
    """
    target = str(label).strip().casefold()

    for asset in assets:
        asset_label = str(getattr(asset, "label", "")).strip().casefold()

        if asset_label == target:
            return asset

    return None


def get_elevation_asset(chunk, label):
    """
    Find an elevation asset such as DSM, DEM, or CHM by label.
    """
    return _find_asset_by_label(
        getattr(chunk, "elevations", []),
        label,
    )


def get_orthomosaic_asset(chunk, label="Orthomosaic"):
    """
    Find an orthomosaic asset by label.
    """
    return _find_asset_by_label(
        getattr(chunk, "orthomosaics", []),
        label,
    )


def _get_cmop_marker(chunk, key):
    """
    Read a CMOP processing marker from chunk metadata.
    """
    full_key = f"cmop_processing/{key}"

    try:
        value = chunk.meta[full_key]
    except Exception:
        return None

    if value is None:
        return None

    return str(value)


def _set_cmop_marker(chunk, key, value="complete"):
    """
    Store a CMOP processing marker in chunk metadata.

    Metadata markers allow process_uav() to distinguish completed steps
    whose state is not otherwise represented by a dedicated Metashape
    asset, such as camera optimization and ground classification.
    """
    full_key = f"cmop_processing/{key}"

    try:
        chunk.meta[full_key] = str(value)
    except Exception:
        # Asset-based resume logic still works even if a particular
        # Metashape build does not permit writing this metadata field.
        pass


def _file_ready(path):
    """
    Return True when an expected output file exists and is non-empty.
    """
    path = Path(path)

    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def _export_if_missing(
    output_path,
    exporter,
    overwrite=False,
):
    """
    Run an exporter only when its output is missing, unless overwrite=True.
    """
    output_path = Path(output_path)

    if _file_ready(output_path) and not overwrite:
        print(f"  Existing output found -- skipping: {output_path.name}")
        return output_path

    return exporter()


def project_processing_state(chunk):
    """
    Inspect a chunk and summarize the processing products already present.

    The state is used by process_uav() to resume from the first incomplete
    stage instead of blindly repeating expensive Metashape operations.
    """
    enabled = _enabled_camera_count(chunk)
    aligned = _aligned_enabled_count(chunk)
    tie_points = _tie_point_count(chunk)

    dsm = get_elevation_asset(chunk, "DSM")
    dem = get_elevation_asset(chunk, "DEM")
    chm = get_elevation_asset(chunk, "CHM")
    orthomosaic = get_orthomosaic_asset(chunk, "Orthomosaic")

    point_cloud = getattr(chunk, "point_cloud", None)

    return {
        "enabled_cameras": enabled,
        "aligned_cameras": aligned,
        "tie_points": tie_points,
        "alignment_complete": enabled > 0 and aligned > 0,
        "point_cloud": point_cloud,
        "point_cloud_complete": point_cloud is not None,
        "DSM": dsm,
        "DSM_complete": dsm is not None,
        "orthomosaic": orthomosaic,
        "orthomosaic_complete": orthomosaic is not None,
        "DEM": dem,
        "DEM_complete": dem is not None,
        "CHM": chm,
        "CHM_complete": chm is not None,
        "optimized_marker": _get_cmop_marker(chunk, "optimized"),
        "ground_classified_marker": _get_cmop_marker(
            chunk,
            "ground_classified",
        ),
        "reflectance_marker": _get_cmop_marker(
            chunk,
            "reflectance_calibrated",
        ),
    }


def alignment_summary(chunk):
    """
    Return a simple camera-alignment summary.
    """
    enabled = [camera for camera in chunk.cameras if camera.enabled]
    aligned = [
        camera
        for camera in enabled
        if camera.transform is not None
    ]
    unaligned = [
        camera
        for camera in enabled
        if camera.transform is None
    ]

    return {
        "enabled_cameras": len(enabled),
        "aligned_cameras": len(aligned),
        "unaligned_cameras": len(unaligned),
        "unaligned_labels": [camera.label for camera in unaligned],
    }


def process_uav(
    input_dir=None,
    output_dir=None,
    calibration_file=None,
    output_crs="EPSG::32610",
    project_file=None,
    project_name=None,
    overwrite_exports=False,
    cal_panel=True,
    cal_sun_sensor=True,
):
    """
    Run or resume the complete Altum-PT Metashape processing workflow.

    The function inspects an existing project and skips expensive stages
    that are already complete.

    Resume behavior
    ---------------
    Alignment
        - If enabled cameras are already aligned, alignment is skipped.
        - If zero cameras are aligned but saved tie points exist,
          alignCameras() is attempted directly without rematching.
        - If no usable tie points exist, photos are matched and aligned.

    Point cloud
        Reused when a point cloud already exists.

    DSM / orthomosaic / DEM / CHM
        Reused by asset label when the corresponding Metashape asset exists.

    Ground classification and optimization
        CMOP metadata markers are written after successful completion.
        Existing downstream assets are also treated as evidence that earlier
        required stages were completed.

    Exports
        Every expected TIFF and LAS output is checked individually.
        Existing non-empty files are skipped unless overwrite_exports=True.
        Missing exports are recreated without rebuilding upstream products.

    Existing project mode
    ---------------------
    Pass ``project_file`` to continue from a prepared .psx project.
    Existing camera enabled/disabled states are preserved.

    New project mode
    ----------------
    Leave ``project_file=None`` and provide ``input_dir``.

    Reflectance calibration
    -----------------------
    ``cal_panel=True, cal_sun_sensor=True`` uses panel + DLS.
    ``cal_panel=True, cal_sun_sensor=False`` uses panel only.
    ``cal_panel=False, cal_sun_sensor=True`` uses DLS only.
    At least one calibration source must be enabled.

    Naming
    ------
    - project_name, when provided, is the output prefix.
    - Existing project + no project_name -> existing .psx stem.
    - New project + no project_name -> inferred from input_dir.
    """
    if not Metashape.License().valid:
        raise RuntimeError(
            "Metashape Professional license was not detected."
        )

    if output_dir is None:
        raise ValueError("output_dir is required.")

    if not cal_panel and not cal_sun_sensor:
        raise ValueError(
            "At least one of cal_panel or cal_sun_sensor must be True."
        )

    if cal_panel and calibration_file is None:
        raise ValueError(
            "calibration_file is required when cal_panel=True."
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    calibration_file = (
        Path(calibration_file) if calibration_file is not None else None
    )
    projection = make_ortho_projection(output_crs)

    # ---------------------------------------------------------
    # Open prepared project OR create a new project.
    # ---------------------------------------------------------
    if project_file is not None:
        project_path = Path(project_file)

        if project_name is None:
            prefix = project_path.stem
        else:
            prefix = str(project_name)

        print(f"Project: {project_path}")
        print(f"Output prefix: {prefix}")
        print(f"Output: {output_dir}")
        print(f"Output CRS: {projection.crs.name}")

        print("\nOpening existing Metashape project...")
        doc, chunk = open_metashape_document(project_path)

        enabled_count = _enabled_camera_count(chunk)
        disabled_count = len(chunk.cameras) - enabled_count

        print(
            f"Loaded chunk '{chunk.label}' with "
            f"{enabled_count} enabled and {disabled_count} disabled "
            "camera planes."
        )
        print(
            "Existing manual camera enable/disable states will be preserved."
        )

    else:
        if input_dir is None:
            raise ValueError(
                "input_dir is required when project_file is not provided."
            )

        input_dir = Path(input_dir)

        if project_name is None:
            prefix = infer_flight_name(input_dir)
        else:
            prefix = str(project_name)

        project_path = output_dir / f"{prefix}.psx"

        if project_path.exists():
            raise FileExistsError(
                f"A project already exists at {project_path}. "
                "Pass it with project_file=... to resume it rather than "
                "creating a new project over it."
            )

        print(f"Project name: {prefix}")
        print(f"Input: {input_dir}")
        print(f"Output: {output_dir}")
        print(f"Output CRS: {projection.crs.name}")

        print("\nFinding raw imagery...")
        photos = find_raw_images(input_dir)
        print(f"Found {len(photos)} image files.")

        print("\nCreating Metashape project...")
        doc = create_metashape_document(project_path)
        chunk = create_chunk(doc, label=prefix)
        load_multicamera_images(chunk, photos)
        save_project(doc)

    # Always make sure the desired primary channel is selected.
    print("\nChecking Altum-PT primary channel...")
    set_primary_channel_panchro(chunk)
    save_project(doc)

    state = project_processing_state(chunk)

    print("\nDetected project state:")
    print(
        f"  Aligned enabled cameras: "
        f"{state['aligned_cameras']}/{state['enabled_cameras']}"
    )
    print(f"  Tie points: {state['tie_points']:,}")
    print(f"  Point cloud: {state['point_cloud_complete']}")
    print(f"  DSM: {state['DSM_complete']}")
    print(f"  Orthomosaic: {state['orthomosaic_complete']}")
    print(f"  DEM: {state['DEM_complete']}")
    print(f"  CHM: {state['CHM_complete']}")

    # ---------------------------------------------------------
    # Reflectance calibration.
    #
    # If downstream products already exist, do not recalibrate them on a
    # resume run. Otherwise use a marker to avoid repeating calibration.
    # ---------------------------------------------------------
    downstream_exists = (
        state["alignment_complete"]
        or state["point_cloud_complete"]
        or state["DSM_complete"]
        or state["orthomosaic_complete"]
        or state["DEM_complete"]
        or state["CHM_complete"]
    )

    if state["reflectance_marker"] is not None or downstream_exists:
        print("\nReflectance calibration already completed -- skipping.")
    else:
        print(
            "\nRequested reflectance calibration: "
            f"panel={cal_panel}, sun_sensor={cal_sun_sensor}"
        )

        if cal_panel:
            print("Locating and loading reflectance panel...")
            locate_reflectance_panels(chunk)
            load_reflectance_panel_calibration(
                chunk,
                calibration_file,
            )

        if cal_panel and cal_sun_sensor:
            print(
                "Checking panel DLS values band-by-band against 10 nearby "
                "flight captures..."
            )
            print(
                "Substitution threshold: panel DLS < 80% of nearby-flight "
                "median"
            )

        if cal_panel and cal_sun_sensor:
            mode_label = "panel + sun sensor"
        elif cal_panel:
            mode_label = "panel only"
        else:
            mode_label = "sun sensor only"

        print(f"Calibrating reflectance with {mode_label}...")
        calibrate_reflectance(
            chunk,
            cal_panel=cal_panel,
            cal_sun_sensor=cal_sun_sensor,
            dls_reference_captures=10,
            dls_substitution_threshold=0.8,
        )

        if cal_panel:
            calibration_disabled = disable_calibration_images(chunk)
            print(
                f"Disabled {len(calibration_disabled)} calibration-image "
                "planes before alignment."
            )

        calibration_mode = (
            f"panel={cal_panel};sun_sensor={cal_sun_sensor}"
        )
        _set_cmop_marker(
            chunk,
            "reflectance_calibrated",
            calibration_mode,
        )
        save_project(doc)

    # ---------------------------------------------------------
    # Alignment.
    # ---------------------------------------------------------
    state = project_processing_state(chunk)

    if state["alignment_complete"]:
        print(
            "\nAlignment already exists -- skipping matching/alignment."
        )
    else:
        print("\nCamera alignment is incomplete.")

        if state["tie_points"] > 0:
            print(
                f"Found {state['tie_points']:,} saved tie points. "
                "Attempting alignCameras() without rematching..."
            )
            try:
                align_from_existing_tie_points(chunk)
            except RuntimeError as exc:
                print(f"Existing-tie-point alignment failed: {exc}")
                print(
                    "Rebuilding matches once, then attempting a fresh "
                    "alignment..."
                )
                align_photos(chunk)
        else:
            print("No saved tie points found. Matching and aligning...")
            align_photos(chunk)

        save_project(doc)

    align_qc = alignment_summary(chunk)

    if align_qc["aligned_cameras"] == 0:
        raise RuntimeError(
            "Alignment stage ended with zero aligned enabled cameras. "
            "Processing stopped before optimization."
        )

    print(
        f"Alignment available for {align_qc['aligned_cameras']} of "
        f"{align_qc['enabled_cameras']} enabled cameras."
    )

    # ---------------------------------------------------------
    # Camera optimization.
    #
    # A point cloud or any later raster is evidence that optimization was
    # already passed on an older project that predates CMOP markers.
    # ---------------------------------------------------------
    state = project_processing_state(chunk)

    optimization_implied = (
        state["point_cloud_complete"]
        or state["DSM_complete"]
        or state["orthomosaic_complete"]
        or state["DEM_complete"]
        or state["CHM_complete"]
    )

    if (
        state["optimized_marker"] == "complete"
        or optimization_implied
    ):
        print("\nCamera optimization already completed -- skipping.")
    else:
        print("\nOptimizing cameras...")
        optimize_cameras(chunk)
        _set_cmop_marker(chunk, "optimized", "complete")
        save_project(doc)

    # ---------------------------------------------------------
    # Point cloud.
    # ---------------------------------------------------------
    state = project_processing_state(chunk)

    if state["point_cloud_complete"]:
        point_cloud = state["point_cloud"]
        print("\nPoint cloud already exists -- skipping build.")
    else:
        print("\nBuilding depth maps and point cloud...")
        point_cloud = build_point_cloud(chunk)
        save_project(doc)

    # ---------------------------------------------------------
    # DSM.
    # ---------------------------------------------------------
    state = project_processing_state(chunk)

    if state["DSM_complete"]:
        dsm = state["DSM"]
        print("\nDSM already exists -- skipping build.")
    else:
        print("\nBuilding DSM...")
        dsm = build_dsm(chunk, projection)
        save_project(doc)

    # ---------------------------------------------------------
    # Orthomosaic.
    # ---------------------------------------------------------
    state = project_processing_state(chunk)

    if state["orthomosaic_complete"]:
        orthomosaic = state["orthomosaic"]
        print("\nOrthomosaic already exists -- skipping build.")
    else:
        print("\nBuilding orthomosaic from DSM...")
        # Make the DSM active before building from elevation data.
        try:
            chunk.elevation = dsm
        except Exception:
            pass

        orthomosaic = build_orthomosaic(chunk, projection)
        save_project(doc)

    # ---------------------------------------------------------
    # Ground classification.
    #
    # An existing DEM/CHM is treated as proof that classification already
    # occurred, which makes older projects resumable even without markers.
    # ---------------------------------------------------------
    state = project_processing_state(chunk)

    ground_implied = (
        state["DEM_complete"]
        or state["CHM_complete"]
    )

    if (
        state["ground_classified_marker"] == "complete"
        or ground_implied
    ):
        print("\nGround classification already completed -- skipping.")
    else:
        print("\nClassifying ground points...")
        classify_ground_points(chunk)
        _set_cmop_marker(
            chunk,
            "ground_classified",
            "complete",
        )
        save_project(doc)

    # ---------------------------------------------------------
    # DEM.
    # ---------------------------------------------------------
    state = project_processing_state(chunk)

    if state["DEM_complete"]:
        dem = state["DEM"]
        print("\nDEM already exists -- skipping build.")
    else:
        print("\nBuilding ground DEM...")
        dem = build_dem(chunk, projection)
        save_project(doc)

    # ---------------------------------------------------------
    # CHM.
    # ---------------------------------------------------------
    state = project_processing_state(chunk)

    if state["CHM_complete"]:
        chm = state["CHM"]
        print("\nCHM already exists -- skipping build.")
    else:
        print("\nBuilding CHM = DSM - DEM...")
        chm = build_chm(
            chunk,
            dsm=dsm,
            dem=dem,
            projection=projection,
        )
        save_project(doc)

    # ---------------------------------------------------------
    # Exports. Every output is checked independently.
    # ---------------------------------------------------------
    print("\nChecking/exporting terrain, RGB, and point-cloud products...")

    dsm_path = output_dir / f"{prefix}_DSM.tif"
    dem_path = output_dir / f"{prefix}_DEM.tif"
    chm_path = output_dir / f"{prefix}_CHM.tif"
    rgb_path = output_dir / f"{prefix}_RGB.tif"
    point_cloud_path = output_dir / f"{prefix}_pointcloud.las"

    dsm_path = _export_if_missing(
        dsm_path,
        lambda: export_elevation(
            chunk,
            dsm,
            dsm_path,
            projection,
        ),
        overwrite=overwrite_exports,
    )

    dem_path = _export_if_missing(
        dem_path,
        lambda: export_elevation(
            chunk,
            dem,
            dem_path,
            projection,
        ),
        overwrite=overwrite_exports,
    )

    chm_path = _export_if_missing(
        chm_path,
        lambda: export_elevation(
            chunk,
            chm,
            chm_path,
            projection,
        ),
        overwrite=overwrite_exports,
    )

    rgb_path = _export_if_missing(
        rgb_path,
        lambda: export_rgb_orthomosaic(
            chunk,
            rgb_path,
            projection,
        ),
        overwrite=overwrite_exports,
    )

    point_cloud_path = _export_if_missing(
        point_cloud_path,
        lambda: export_point_cloud(
            chunk,
            point_cloud_path,
            crs=output_crs,
        ),
        overwrite=overwrite_exports,
    )

    print("\nChecking/exporting Altum-PT spectral and thermal TIFFs...")
    band_paths = export_altum_pt_bands(
        chunk,
        output_dir=output_dir,
        prefix=prefix,
        projection=projection,
        overwrite=overwrite_exports,
    )

    save_project(doc)

    outputs = {
        "project": Path(doc.path),
        "RGB": rgb_path,
        "DSM": dsm_path,
        "DEM": dem_path,
        "CHM": chm_path,
        "pointcloud": point_cloud_path,
        **band_paths,
    }

    print("\nProcessing complete.")
    print("Outputs:")

    for name, path in outputs.items():
        print(f"  {name}: {path}")

    return {
        "project_name": prefix,
        "output_crs": output_crs,
        "outputs": outputs,
        "alignment": align_qc,
        "document": doc,
        "chunk": chunk,
    }

