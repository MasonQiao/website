import argparse

import matplotlib.pyplot as plt
import nd2
import numpy as np
from csbdeep.utils import normalize
from matplotlib.patches import Patch
from scipy import ndimage as ndi
from skimage.measure import EllipseModel, regionprops
from skimage.morphology import convex_hull_image

from cell_tracking import (
    DEFAULT_SEGMENTATION_SCALE,
    LEGACY_SEGMENTATION_VERSION,
    SEGMENTATION_VERSION,
    projection_cell_segmentation,
    segment_cell_tracks,
)

# Analysis parameters
DEFAULT_SCALE = DEFAULT_SEGMENTATION_SCALE
DEFAULT_SEGMENTATION_MODE = SEGMENTATION_VERSION
DEFAULT_DAPI_CHANNEL = 0
DEFAULT_CELL_CHANNEL = DEFAULT_DAPI_CHANNEL
DEFAULT_SEGMENTATION_CHANNEL = 1
DEFAULT_PNC_CHANNEL = 1
CELL_BORDER_ITERATIONS = 2
MIN_CELL_MEDIAN_AREA_FRACTION = 0.5
MAX_CELL_MEDIAN_AREA_FRACTION = 2.0
MIN_CELL_DAPI_MEDIAN_FRACTION = 0.5

PNC_BRIGHT_PIXEL_PERCENTILE = 95
PNC_THRESHOLD_MULTIPLIER = 1.25
PNC_MIN_CELL_AREA_FRACTION = 1 / 1000

# Nucleolus-aware rescue detection. The broad, dim nucleolus is detected from
# TxRed, while PNC candidates are always measured on the original TxRed layer.
ENABLE_NUCLEOLUS_RESCUE = True
NUCLEOLUS_INNER_THRESHOLD_MULTIPLIER = 0.85
NUCLEOLUS_OUTER_THRESHOLD_MULTIPLIER = 1.05
NUCLEOLUS_OUTER_BAND_PIXELS = 25
NUCLEOLUS_Z_PROPAGATION = 1

NUCLEOLUS_SMOOTH_SIGMA = 3
NUCLEOLUS_BACKGROUND_SIGMA = 30
NUCLEOLUS_PEAK_PERCENTILE = 65
NUCLEOLUS_PEAK_WINDOW = 15
NUCLEOLUS_GROW_PEAK_FRACTION = 0.2
NUCLEOLUS_MIN_CELL_AREA_FRACTION = 0.002
NUCLEOLUS_MAX_CELL_AREA_FRACTION = 0.12
NUCLEOLUS_MIN_SOLIDITY = 0.65
NUCLEOLUS_MAX_ECCENTRICITY = 0.96
NUCLEOLUS_INTERIOR_MARGIN_FRACTION = 0.06
NUCLEOLUS_MIN_INTERIOR_MARGIN_PIXELS = 6
NUCLEOLUS_MAX_RADIUS_FRACTION = 0.45
NUCLEOLUS_MIN_RADIUS_PIXELS = 18
NUCLEOLUS_CONVEX_DILATION_PIXELS = 3
NUCLEOLUS_ROI_PADDING_PIXELS = 100

PNC_LOCAL_SPOT_SIGMA = 1.2
PNC_LOCAL_BACKGROUND_SIGMA = 8
PNC_LOCAL_CONTRAST_PERCENTILE = 99.5
PNC_LOCAL_CONTRAST_MULTIPLIER = 2.0
PNC_RESCUE_MAX_CELL_AREA_FRACTION = 0.01
PNC_RESCUE_MIN_CORE_CELL_AREA_FRACTION = 1 / 5000

# Conservative TxRed smear detection. Smears are broad/patchy features that
# become strongest near the top of the z-stack, unlike ordinary cell signal
# that persists through the stack. A retained smear component may overlap a
# cell or touch it after a one-pixel dilation; either case excludes that cell.
ENABLE_SMEAR_EXCLUSION = True
SMEAR_TOP_Z_LAYERS = 2
SMEAR_SMOOTH_SIGMA = 3
SMEAR_GROWTH_PERCENTILE = 98
SMEAR_SEED_PERCENTILE = 99.5
SMEAR_MIN_CELL_AREA_FRACTION = 0.05
SMEAR_MIN_SEED_PIXELS = 3
SMEAR_TOUCH_DILATION_PIXELS = 1
SMEAR_MIN_CELL_OVERLAP_FRACTION = 0.01
SMEAR_MIN_CELL_BOUNDARY_CONTACT_FRACTION = 0.02
SMEAR_MIN_TOP_TAIL_RATIO = 2.0
SMEAR_MIN_TOP_TO_BOTTOM_RATIO = 1.5
SMEAR_MIN_Z_CORRELATION = 0.8
SMEAR_WEAK_MIN_CELL_AREA_FRACTION = 0.01
SMEAR_WEAK_MIN_ECCENTRICITY = 0.75
SMEAR_WEAK_MIN_Z_CORRELATION = 0.95
SMEAR_WEAK_MIN_CELL_OVERLAP_FRACTION = 0.005
SMEAR_WEAK_INTERIOR_CLEARANCE_PIXELS = 1
SMEAR_TRANSIENT_MIN_LOCAL_PEAK_RATIO = 1.10
SMEAR_TRANSIENT_MIN_ECCENTRICITY = 0.8

BORDER_CELL_MARGIN = 2
MAX_MISSING_ELLIPSE_FRACTION = 0.15
MAX_ELLIPSE_FIT_RMSE = 0.35
EXCLUDE_UNCERTAIN_BORDER_CELLS = True


def _cell_ids_from_labels(labels):
    return {int(cell_id) for cell_id in np.unique(labels[labels > 0])}


def _select_analysis_stacks(
    data,
    cell_channel=DEFAULT_CELL_CHANNEL,
    segmentation_channel=DEFAULT_SEGMENTATION_CHANNEL,
    pnc_channel=DEFAULT_PNC_CHANNEL,
):
    """Return DAPI, segmentation, and PNC stacks from Z,C,Y,X data."""
    data = np.asarray(data)
    if data.ndim != 4:
        raise ValueError("ND2 data must have axes Z, C, Y, X.")

    channel_count = data.shape[1]
    selections = {
        "DAPI/cell": cell_channel,
        "segmentation": segmentation_channel,
        "PNC/TxRed": pnc_channel,
    }
    for role, channel_index in selections.items():
        if not isinstance(channel_index, (int, np.integer)):
            raise ValueError(f"The {role} channel index must be an integer.")
        if not 0 <= int(channel_index) < channel_count:
            raise ValueError(
                f"The {role} channel index {channel_index} is outside the "
                f"available range 0-{channel_count - 1}."
            )

    return (
        data[:, int(cell_channel), :, :],
        data[:, int(segmentation_channel), :, :],
        data[:, int(pnc_channel), :, :],
    )


def _cell_size_outlier_ids(
    labels,
    cell_ids,
    min_median_area_fraction=MIN_CELL_MEDIAN_AREA_FRACTION,
    max_median_area_fraction=MAX_CELL_MEDIAN_AREA_FRACTION,
    track_metadata=None,
):
    if not cell_ids:
        return set(), set(), None, None, None

    if track_metadata is not None:
        cell_areas = {
            cell_id: float(track_metadata[cell_id]["median_area_pixels"])
            for cell_id in cell_ids
            if cell_id in track_metadata
        }
    elif np.asarray(labels).ndim == 3:
        cell_areas = {}
        for cell_id in cell_ids:
            areas = [
                int(np.count_nonzero(layer == cell_id))
                for layer in labels
            ]
            positive_areas = [area for area in areas if area]
            if positive_areas:
                cell_areas[cell_id] = float(np.median(positive_areas))
    else:
        label_areas = np.bincount(labels.ravel())
        cell_areas = {
            cell_id: int(label_areas[cell_id]) for cell_id in cell_ids
        }
    if not cell_areas:
        return set(), set(), None, None, None
    median_cell_area = float(np.median(list(cell_areas.values())))
    minimum_cell_area = median_cell_area * min_median_area_fraction
    maximum_cell_area = median_cell_area * max_median_area_fraction
    small_cell_ids = {
        cell_id for cell_id, area in cell_areas.items() if area < minimum_cell_area
    }
    large_cell_ids = {
        cell_id for cell_id, area in cell_areas.items() if area > maximum_cell_area
    }
    return (
        small_cell_ids,
        large_cell_ids,
        median_cell_area,
        minimum_cell_area,
        maximum_cell_area,
    )


def _low_dapi_cell_ids(
    cell_img,
    labels,
    cell_ids,
    min_median_fraction=MIN_CELL_DAPI_MEDIAN_FRACTION,
    track_metadata=None,
):
    """Return exceptionally dim labels that are likely segmentation ghosts."""
    if not cell_ids:
        return set(), None, {}

    if np.asarray(labels).ndim == 3:
        cell_stack = np.asarray(cell_img)
        cell_medians = {}
        object_slices = ndi.find_objects(labels)
        for cell_id in cell_ids:
            track_slice = (
                object_slices[cell_id - 1]
                if cell_id - 1 < len(object_slices)
                else None
            )
            if track_slice is None:
                continue
            _, y_slice, x_slice = track_slice
            if track_metadata and cell_id in track_metadata:
                observed_layers = track_metadata[cell_id].get(
                    "observed_z_layers",
                    [],
                )
                z_indices = [value - 1 for value in observed_layers]
            else:
                z_indices = range(len(labels))
            layer_medians = [
                float(
                    np.median(
                        cell_stack[z_index, y_slice, x_slice][
                            labels[z_index, y_slice, x_slice] == cell_id
                        ]
                    )
                )
                for z_index in z_indices
                if np.any(labels[z_index, y_slice, x_slice] == cell_id)
            ]
            if layer_medians:
                cell_medians[cell_id] = float(np.median(layer_medians))
    else:
        cell_medians = {
            cell_id: float(np.median(cell_img[labels == cell_id]))
            for cell_id in cell_ids
        }
    if not cell_medians:
        return set(), None, {}
    field_median = float(np.median(list(cell_medians.values())))
    minimum_median = field_median * min_median_fraction
    low_dapi_ids = {
        cell_id
        for cell_id, cell_median in cell_medians.items()
        if cell_median < minimum_median
    }
    return low_dapi_ids, field_median, cell_medians


def _border_band(shape, margin):
    margin = max(1, int(margin))
    band = np.zeros(shape, dtype=bool)
    band[:margin, :] = True
    band[-margin:, :] = True
    band[:, :margin] = True
    band[:, -margin:] = True
    return band


def _label_border(labels, label_ids, iterations):
    border = np.zeros(labels.shape, dtype=bool)
    for label_id in label_ids:
        mask = labels == label_id
        eroded = ndi.binary_erosion(mask, border_value=0)
        border |= mask & ~eroded

    if iterations:
        border = ndi.binary_dilation(border, iterations=iterations)

    return border


def _perimeter_fit_points(mask, border_band):
    perimeter = mask & ~ndi.binary_erosion(mask, border_value=0)

    # Ignore the image-edge perimeter because it can be an artificial cut line.
    fit_perimeter = perimeter & ~border_band
    y_coords, x_coords = np.nonzero(fit_perimeter)
    if x_coords.size < 12:
        return None

    return np.column_stack((x_coords, y_coords)).astype(float)


def _ellipse_normalized_rmse(points, ellipse):
    x_values = points[:, 0]
    y_values = points[:, 1]
    cos_t = np.cos(ellipse["theta"])
    sin_t = np.sin(ellipse["theta"])
    dx = x_values - ellipse["cx"]
    dy = y_values - ellipse["cy"]
    u = cos_t * dx + sin_t * dy
    v = -sin_t * dx + cos_t * dy
    residuals = (
        np.sqrt((u / ellipse["semi_major"]) ** 2 + (v / ellipse["semi_minor"]) ** 2)
        - 1.0
    )
    return float(np.sqrt(np.mean(residuals**2)))


def _fit_ellipse(mask, border_band, max_fit_rmse):
    points = _perimeter_fit_points(mask, border_band)
    if points is None:
        return None

    height, width = mask.shape
    max_axis = max(height, width) * 1.5

    try:
        model = EllipseModel.from_estimate(points)
    except ValueError:
        return None

    if model is None:
        return None

    cx, cy = model.center
    semi_a, semi_b = model.axis_lengths
    theta = model.theta
    ellipse_values = np.array([cx, cy, semi_a, semi_b, theta])
    if not np.all(np.isfinite(ellipse_values)):
        return None
    if semi_a <= 0 or semi_b <= 0 or semi_a > max_axis or semi_b > max_axis:
        return None

    if semi_b > semi_a:
        semi_a, semi_b = semi_b, semi_a
        theta += np.pi / 2.0

    ellipse = {
        "cx": float(cx),
        "cy": float(cy),
        "semi_major": float(semi_a),
        "semi_minor": float(semi_b),
        "theta": float(theta),
    }
    rmse = _ellipse_normalized_rmse(points, ellipse)
    if rmse > max_fit_rmse:
        return None

    return ellipse


def _ellipse_missing_fraction(mask, ellipse):
    visible_area = np.count_nonzero(mask)
    fitted_ellipse_area = np.pi * ellipse["semi_major"] * ellipse["semi_minor"]
    if fitted_ellipse_area <= 0:
        return 0.0

    visible_fraction = visible_area / fitted_ellipse_area
    return max(0.0, 1.0 - visible_fraction)


def get_non_intact_ellipse_ids(
    labels,
    border_margin=BORDER_CELL_MARGIN,
    max_missing_ellipse_fraction=MAX_MISSING_ELLIPSE_FRACTION,
    max_fit_rmse=MAX_ELLIPSE_FIT_RMSE,
    exclude_uncertain_border_cells=EXCLUDE_UNCERTAIN_BORDER_CELLS,
):
    border_band = _border_band(labels.shape, border_margin)
    non_intact_ids = set()

    border_cell_ids = _cell_ids_from_labels(labels[border_band])
    for cell_id in border_cell_ids:
        mask = labels == cell_id
        if not np.any(mask & border_band):
            continue

        ellipse = _fit_ellipse(
            mask,
            border_band,
            max_fit_rmse=max_fit_rmse,
        )
        if ellipse is None:
            if exclude_uncertain_border_cells:
                non_intact_ids.add(cell_id)
            continue

        missing_fraction = _ellipse_missing_fraction(mask, ellipse)
        if missing_fraction > max_missing_ellipse_fraction:
            non_intact_ids.add(cell_id)

    return non_intact_ids


def _non_intact_ids_across_z(label_stack):
    """Exclude a track if any observed cross-section is cut by the field edge."""
    non_intact_ids = set()
    for labels in np.asarray(label_stack):
        non_intact_ids.update(get_non_intact_ellipse_ids(labels))
    return non_intact_ids


def _smear_contact_cell_ids(
    smear_mask,
    labels,
    touch_dilation_pixels,
    min_cell_overlap_fraction,
    min_cell_boundary_contact_fraction,
):
    """Map a patchy smear mask to cells by overlap or boundary contact."""
    if not np.any(smear_mask):
        return set()

    contact_mask = smear_mask
    if touch_dilation_pixels:
        contact_mask = ndi.binary_dilation(
            smear_mask,
            iterations=touch_dilation_pixels,
        )

    smear_cell_ids = set()
    for cell_id in _cell_ids_from_labels(labels[contact_mask]):
        cell_mask = labels == cell_id
        cell_area = int(np.count_nonzero(cell_mask))
        overlap_fraction = (
            np.count_nonzero(smear_mask & cell_mask) / cell_area
        )
        cell_boundary = cell_mask & ~ndi.binary_erosion(
            cell_mask,
            border_value=0,
        )
        boundary_area = int(np.count_nonzero(cell_boundary))
        boundary_contact_fraction = (
            np.count_nonzero(contact_mask & cell_boundary) / boundary_area
            if boundary_area
            else 0.0
        )
        if (
            overlap_fraction >= min_cell_overlap_fraction
            or boundary_contact_fraction
            >= min_cell_boundary_contact_fraction
        ):
            smear_cell_ids.add(cell_id)

    return smear_cell_ids


def _detect_transient_txred_smears(
    pnc_stack,
    labels,
    smooth_sigma,
    growth_percentile,
    seed_percentile,
    min_cell_area_fraction,
    min_seed_pixels,
    min_local_peak_ratio,
    min_eccentricity,
    touch_dilation_pixels,
    min_cell_overlap_fraction,
    min_cell_boundary_contact_fraction,
):
    """Detect broad, patchy TxRed events that peak inside the z-stack."""
    empty_mask = np.zeros(labels.shape, dtype=bool)
    empty_response = np.zeros(labels.shape, dtype=float)
    empty_mask_stack = np.zeros((len(pnc_stack),) + labels.shape, dtype=bool)
    if len(pnc_stack) < 3 or not np.any(labels > 0):
        return {
            "mask": empty_mask,
            "cell_ids": set(),
            "response": empty_response,
            "component_count": 0,
            "z_layers": [],
            "detections": [],
            "mask_stack": empty_mask_stack,
        }

    background_mask = labels == 0
    if not np.any(background_mask):
        return {
            "mask": empty_mask,
            "cell_ids": set(),
            "response": empty_response,
            "component_count": 0,
            "z_layers": [],
            "detections": [],
            "mask_stack": empty_mask_stack,
        }

    label_areas = np.bincount(labels.ravel())[1:]
    positive_label_areas = label_areas[label_areas > 0]
    median_cell_area = float(np.median(positive_label_areas))
    minimum_smear_area = max(
        1,
        int(np.ceil(median_cell_area * min_cell_area_fraction)),
    )

    stack_values = pnc_stack.astype(float, copy=False)
    temporal_reference = np.median(stack_values, axis=0)
    transient_mask = empty_mask.copy()
    transient_mask_stack = empty_mask_stack.copy()
    maximum_response = empty_response.copy()
    detections = []

    # End layers cannot demonstrate a peak relative to two neighbors. They are
    # handled by the existing endpoint branch when a smear builds toward the
    # top of the stack.
    for z_index in range(1, len(pnc_stack) - 1):
        layer_response = ndi.gaussian_filter(
            stack_values[z_index] - temporal_reference,
            smooth_sigma,
        )
        maximum_response = np.maximum(maximum_response, layer_response)
        background_response = layer_response[background_mask]
        growth_threshold = float(
            np.percentile(background_response, growth_percentile)
        )
        seed_threshold = float(
            np.percentile(background_response, seed_percentile)
        )

        # One closing joins tiny gaps between streak fragments. It does not
        # fill holes or turn the candidate into a solid object.
        growth_mask = ndi.binary_closing(
            layer_response >= growth_threshold,
            iterations=1,
        )
        component_labels, component_count = ndi.label(growth_mask)
        component_sizes = np.bincount(component_labels.ravel())
        seed_counts = np.bincount(
            component_labels[layer_response >= seed_threshold].ravel(),
            minlength=component_count + 1,
        )
        candidate_ids = np.flatnonzero(
            (component_sizes >= minimum_smear_area)
            & (seed_counts >= min_seed_pixels)
        )

        for component_id in candidate_ids[candidate_ids != 0]:
            component_mask = component_labels == component_id
            layer_means = np.mean(
                stack_values[:, component_mask],
                axis=1,
            )
            if int(np.argmax(layer_means)) != z_index:
                continue

            adjacent_reference = max(
                layer_means[z_index - 1],
                layer_means[z_index + 1],
                1.0,
            )
            local_peak_ratio = (
                layer_means[z_index] / adjacent_reference
            )
            if local_peak_ratio < min_local_peak_ratio:
                continue

            properties = regionprops(component_mask.astype(np.uint8))
            if not properties:
                continue
            eccentricity = float(properties[0].eccentricity)
            if eccentricity < min_eccentricity:
                continue

            transient_mask |= component_mask
            transient_mask_stack[z_index] |= component_mask
            detections.append(
                {
                    "z_index": z_index,
                    "z_layer": z_index + 1,
                    "area_pixels": int(component_sizes[component_id]),
                    "local_peak_ratio": float(local_peak_ratio),
                    "eccentricity": eccentricity,
                }
            )

    cell_ids = _smear_contact_cell_ids(
        transient_mask,
        labels,
        touch_dilation_pixels,
        min_cell_overlap_fraction,
        min_cell_boundary_contact_fraction,
    )
    return {
        "mask": transient_mask,
        "cell_ids": cell_ids,
        "response": maximum_response,
        "component_count": len(detections),
        "z_layers": sorted(
            {detection["z_layer"] for detection in detections}
        ),
        "detections": detections,
        "mask_stack": transient_mask_stack,
    }


def _detect_txred_smears(
    pnc_stack,
    labels,
    top_z_layers=SMEAR_TOP_Z_LAYERS,
    smooth_sigma=SMEAR_SMOOTH_SIGMA,
    growth_percentile=SMEAR_GROWTH_PERCENTILE,
    seed_percentile=SMEAR_SEED_PERCENTILE,
    min_cell_area_fraction=SMEAR_MIN_CELL_AREA_FRACTION,
    min_seed_pixels=SMEAR_MIN_SEED_PIXELS,
    touch_dilation_pixels=SMEAR_TOUCH_DILATION_PIXELS,
    min_cell_overlap_fraction=SMEAR_MIN_CELL_OVERLAP_FRACTION,
    min_cell_boundary_contact_fraction=(
        SMEAR_MIN_CELL_BOUNDARY_CONTACT_FRACTION
    ),
    min_top_tail_ratio=SMEAR_MIN_TOP_TAIL_RATIO,
    min_top_to_bottom_ratio=SMEAR_MIN_TOP_TO_BOTTOM_RATIO,
    min_z_correlation=SMEAR_MIN_Z_CORRELATION,
    weak_min_cell_area_fraction=SMEAR_WEAK_MIN_CELL_AREA_FRACTION,
    weak_min_eccentricity=SMEAR_WEAK_MIN_ECCENTRICITY,
    weak_min_z_correlation=SMEAR_WEAK_MIN_Z_CORRELATION,
    weak_min_cell_overlap_fraction=(
        SMEAR_WEAK_MIN_CELL_OVERLAP_FRACTION
    ),
    weak_interior_clearance_pixels=(
        SMEAR_WEAK_INTERIOR_CLEARANCE_PIXELS
    ),
    transient_min_local_peak_ratio=(
        SMEAR_TRANSIENT_MIN_LOCAL_PEAK_RATIO
    ),
    transient_min_eccentricity=SMEAR_TRANSIENT_MIN_ECCENTRICITY,
):
    """Detect broad endpoint or transient TxRed smears and map cell contact."""
    empty_mask = np.zeros(labels.shape, dtype=bool)
    empty_mask_stack = np.zeros((len(pnc_stack),) + labels.shape, dtype=bool)
    if len(pnc_stack) < 3 or not np.any(labels > 0):
        return {
            "mask": empty_mask,
            "endpoint_mask": empty_mask.copy(),
            "strong_endpoint_mask": empty_mask.copy(),
            "weak_endpoint_mask": empty_mask.copy(),
            "endpoint_mask_stack": empty_mask_stack.copy(),
            "strong_endpoint_mask_stack": empty_mask_stack.copy(),
            "weak_endpoint_mask_stack": empty_mask_stack.copy(),
            "cell_ids": set(),
            "response": np.zeros(labels.shape, dtype=float),
            "growth_threshold": None,
            "seed_threshold": None,
            "component_count": 0,
            "top_tail_ratio": None,
            "transient_mask": empty_mask.copy(),
            "transient_response": np.zeros(labels.shape, dtype=float),
            "transient_component_count": 0,
            "transient_cell_ids": set(),
            "transient_z_layers": [],
            "transient_detections": [],
            "transient_mask_stack": empty_mask_stack.copy(),
        }

    transient_detection = _detect_transient_txred_smears(
        pnc_stack,
        labels,
        smooth_sigma,
        growth_percentile,
        seed_percentile,
        min_cell_area_fraction,
        min_seed_pixels,
        transient_min_local_peak_ratio,
        transient_min_eccentricity,
        touch_dilation_pixels,
        min_cell_overlap_fraction,
        min_cell_boundary_contact_fraction,
    )

    top_z_layers = min(max(1, int(top_z_layers)), len(pnc_stack) - 1)
    earlier_stack = pnc_stack[:-top_z_layers].astype(float, copy=False)
    top_stack = pnc_stack[-top_z_layers:].astype(float, copy=False)
    top_projection = np.max(top_stack, axis=0)
    persistent_reference = np.median(earlier_stack, axis=0)
    smoothed_last_layer = ndi.gaussian_filter(
        pnc_stack[-1].astype(float, copy=False),
        smooth_sigma,
    )
    last_layer_p95 = float(np.percentile(smoothed_last_layer, 95))
    last_layer_p999 = float(np.percentile(smoothed_last_layer, 99.9))
    top_tail_ratio = last_layer_p999 / max(last_layer_p95, 1.0)
    response = ndi.gaussian_filter(
        top_projection - persistent_reference,
        smooth_sigma,
    )

    if top_tail_ratio < min_top_tail_ratio:
        return {
            "mask": transient_detection["mask"],
            "endpoint_mask": empty_mask.copy(),
            "strong_endpoint_mask": empty_mask.copy(),
            "weak_endpoint_mask": empty_mask.copy(),
            "endpoint_mask_stack": empty_mask_stack.copy(),
            "strong_endpoint_mask_stack": empty_mask_stack.copy(),
            "weak_endpoint_mask_stack": empty_mask_stack.copy(),
            "cell_ids": transient_detection["cell_ids"],
            "response": np.maximum(
                response,
                transient_detection["response"],
            ),
            "growth_threshold": None,
            "seed_threshold": None,
            "component_count": transient_detection["component_count"],
            "top_tail_ratio": top_tail_ratio,
            "transient_mask": transient_detection["mask"],
            "transient_response": transient_detection["response"],
            "transient_component_count": transient_detection[
                "component_count"
            ],
            "transient_cell_ids": transient_detection["cell_ids"],
            "transient_z_layers": transient_detection["z_layers"],
            "transient_detections": transient_detection["detections"],
            "transient_mask_stack": transient_detection["mask_stack"],
        }

    background_mask = labels == 0
    background_response = response[background_mask]
    if not background_response.size:
        return {
            "mask": transient_detection["mask"],
            "endpoint_mask": empty_mask.copy(),
            "strong_endpoint_mask": empty_mask.copy(),
            "weak_endpoint_mask": empty_mask.copy(),
            "endpoint_mask_stack": empty_mask_stack.copy(),
            "strong_endpoint_mask_stack": empty_mask_stack.copy(),
            "weak_endpoint_mask_stack": empty_mask_stack.copy(),
            "cell_ids": transient_detection["cell_ids"],
            "response": np.maximum(
                response,
                transient_detection["response"],
            ),
            "growth_threshold": None,
            "seed_threshold": None,
            "component_count": transient_detection["component_count"],
            "top_tail_ratio": top_tail_ratio,
            "transient_mask": transient_detection["mask"],
            "transient_response": transient_detection["response"],
            "transient_component_count": transient_detection[
                "component_count"
            ],
            "transient_cell_ids": transient_detection["cell_ids"],
            "transient_z_layers": transient_detection["z_layers"],
            "transient_detections": transient_detection["detections"],
            "transient_mask_stack": transient_detection["mask_stack"],
        }

    growth_threshold = float(
        np.percentile(background_response, growth_percentile)
    )
    seed_threshold = float(
        np.percentile(background_response, seed_percentile)
    )
    growth_mask = response >= growth_threshold
    growth_mask = ndi.binary_closing(growth_mask, iterations=1)
    component_labels, component_count = ndi.label(growth_mask)
    component_sizes = np.bincount(component_labels.ravel())
    seed_counts = np.bincount(
        component_labels[response >= seed_threshold].ravel(),
        minlength=component_count + 1,
    )

    label_areas = np.bincount(labels.ravel())[1:]
    positive_label_areas = label_areas[label_areas > 0]
    median_cell_area = float(np.median(positive_label_areas))
    minimum_smear_area = max(
        1,
        int(np.ceil(median_cell_area * min_cell_area_fraction)),
    )
    component_properties = {
        prop.label: prop for prop in regionprops(component_labels)
    }
    candidate_ids = np.flatnonzero(
        (component_sizes >= minimum_smear_area)
        & (seed_counts >= min_seed_pixels)
    )
    candidate_ids = candidate_ids[candidate_ids != 0]

    # A bright focus-dependent feature can resemble a smear in the final-layer
    # subtraction. Real smears in the reviewed stacks intensify broadly and
    # progressively toward the top of z. Validate that behavior per component
    # before mapping the component to cells.
    z_positions = np.arange(len(pnc_stack), dtype=float)
    z_statistics = {}

    def component_z_statistics(component_id):
        if component_id in z_statistics:
            return z_statistics[component_id]

        component_values = pnc_stack[
            :, component_labels == component_id
        ].astype(float, copy=False)
        layer_p95 = np.percentile(component_values, 95, axis=1)
        top_to_bottom_ratio = layer_p95[-1] / max(layer_p95[0], 1.0)
        layer_means = np.mean(component_values, axis=1)
        if np.ptp(layer_means) > 0:
            z_correlation = float(
                np.corrcoef(z_positions, layer_means)[0, 1]
            )
        else:
            z_correlation = 0.0
        if not np.isfinite(z_correlation):
            z_correlation = 0.0

        z_statistics[component_id] = (
            top_to_bottom_ratio,
            z_correlation,
        )
        return z_statistics[component_id]

    keep_ids = []
    for component_id in candidate_ids:
        top_to_bottom_ratio, z_correlation = component_z_statistics(
            component_id
        )

        if (
            top_to_bottom_ratio >= min_top_to_bottom_ratio
            and z_correlation >= min_z_correlation
        ):
            keep_ids.append(int(component_id))

    keep_ids = np.asarray(keep_ids, dtype=int)
    strong_smear_mask = np.isin(component_labels, keep_ids)

    # Subtle internal smear streaks can be smaller than the broad-component
    # cutoff. Search for them only when this field already contains a validated
    # broad smear, and keep the rule narrow so compact PNC-like dots do not
    # become smear calls.
    weak_component_ids = set()
    weak_cell_ids = set()
    if keep_ids.size:
        minimum_weak_area = max(
            1,
            int(
                np.ceil(
                    median_cell_area * weak_min_cell_area_fraction
                )
            ),
        )
        for component_id, prop in component_properties.items():
            if not (
                minimum_weak_area <= prop.area < minimum_smear_area
                and prop.eccentricity >= weak_min_eccentricity
            ):
                continue

            _, z_correlation = component_z_statistics(component_id)
            if z_correlation < weak_min_z_correlation:
                continue

            component_mask = component_labels == component_id
            if weak_interior_clearance_pixels:
                clearance_mask = ndi.binary_dilation(
                    component_mask,
                    iterations=weak_interior_clearance_pixels,
                )
            else:
                clearance_mask = component_mask

            for cell_id in _cell_ids_from_labels(labels[component_mask]):
                cell_mask = labels == cell_id
                cell_area = int(np.count_nonzero(cell_mask))
                overlap_fraction = (
                    np.count_nonzero(component_mask & cell_mask)
                    / cell_area
                )
                if overlap_fraction < weak_min_cell_overlap_fraction:
                    continue

                cell_boundary = cell_mask & ~ndi.binary_erosion(
                    cell_mask,
                    border_value=0,
                )
                if np.any(clearance_mask & cell_boundary):
                    continue

                weak_component_ids.add(component_id)
                weak_cell_ids.add(cell_id)

    weak_smear_mask = np.isin(
        component_labels,
        list(weak_component_ids),
    )
    endpoint_smear_mask = strong_smear_mask | weak_smear_mask
    endpoint_source_layers = (
        np.argmax(top_stack, axis=0) + len(pnc_stack) - top_z_layers
    )
    strong_endpoint_mask_stack = empty_mask_stack.copy()
    weak_endpoint_mask_stack = empty_mask_stack.copy()
    for z_index in range(len(pnc_stack) - top_z_layers, len(pnc_stack)):
        source_mask = endpoint_source_layers == z_index
        strong_endpoint_mask_stack[z_index] = strong_smear_mask & source_mask
        weak_endpoint_mask_stack[z_index] = weak_smear_mask & source_mask
    endpoint_mask_stack = (
        strong_endpoint_mask_stack | weak_endpoint_mask_stack
    )
    smear_mask = endpoint_smear_mask | transient_detection["mask"]
    smear_cell_ids = set(weak_cell_ids)
    smear_cell_ids |= _smear_contact_cell_ids(
        strong_smear_mask,
        labels,
        touch_dilation_pixels,
        min_cell_overlap_fraction,
        min_cell_boundary_contact_fraction,
    )
    smear_cell_ids |= transient_detection["cell_ids"]
    return {
        "mask": smear_mask,
        "endpoint_mask": endpoint_smear_mask,
        "strong_endpoint_mask": strong_smear_mask,
        "weak_endpoint_mask": weak_smear_mask,
        "endpoint_mask_stack": endpoint_mask_stack,
        "strong_endpoint_mask_stack": strong_endpoint_mask_stack,
        "weak_endpoint_mask_stack": weak_endpoint_mask_stack,
        "cell_ids": smear_cell_ids,
        "response": np.maximum(
            response,
            transient_detection["response"],
        ),
        "growth_threshold": growth_threshold,
        "seed_threshold": seed_threshold,
        "component_count": int(
            len(keep_ids)
            + len(weak_component_ids)
            + transient_detection["component_count"]
        ),
        "top_tail_ratio": top_tail_ratio,
        "transient_mask": transient_detection["mask"],
        "transient_response": transient_detection["response"],
        "transient_component_count": transient_detection[
            "component_count"
        ],
        "transient_cell_ids": transient_detection["cell_ids"],
        "transient_z_layers": transient_detection["z_layers"],
        "transient_detections": transient_detection["detections"],
        "transient_mask_stack": transient_detection["mask_stack"],
    }


def _weak_smear_contact_cell_ids(smear_mask, labels):
    """Apply the existing weak-smear overlap and interior rules to one layer."""
    component_labels, component_count = ndi.label(smear_mask)
    cell_ids = set()
    for component_id in range(1, component_count + 1):
        component_mask = component_labels == component_id
        clearance_mask = ndi.binary_dilation(
            component_mask,
            iterations=SMEAR_WEAK_INTERIOR_CLEARANCE_PIXELS,
        )
        for cell_id in _cell_ids_from_labels(labels[component_mask]):
            cell_mask = labels == cell_id
            cell_area = int(np.count_nonzero(cell_mask))
            if not cell_area:
                continue
            overlap_fraction = (
                np.count_nonzero(component_mask & cell_mask) / cell_area
            )
            if overlap_fraction < SMEAR_WEAK_MIN_CELL_OVERLAP_FRACTION:
                continue
            cell_boundary = cell_mask & ~ndi.binary_erosion(
                cell_mask,
                border_value=0,
            )
            if np.any(clearance_mask & cell_boundary):
                continue
            cell_ids.add(cell_id)
    return cell_ids


def _map_smears_to_z_specific_cells(smear_detection, cell_label_stack):
    """Map endpoint and transient smear masks on their actual z-layers."""
    cell_label_stack = np.asarray(cell_label_stack)
    endpoint_ids = set()
    first_endpoint = max(0, len(cell_label_stack) - SMEAR_TOP_Z_LAYERS)
    strong_endpoint_stack = smear_detection.get("strong_endpoint_mask_stack")
    weak_endpoint_stack = smear_detection.get("weak_endpoint_mask_stack")
    for z_index in range(first_endpoint, len(cell_label_stack)):
        layer_labels = cell_label_stack[z_index]
        strong_mask = (
            strong_endpoint_stack[z_index]
            if strong_endpoint_stack is not None
            else smear_detection["strong_endpoint_mask"]
        )
        weak_mask = (
            weak_endpoint_stack[z_index]
            if weak_endpoint_stack is not None
            else smear_detection["weak_endpoint_mask"]
        )
        endpoint_ids.update(
            _smear_contact_cell_ids(
                strong_mask,
                layer_labels,
                SMEAR_TOUCH_DILATION_PIXELS,
                SMEAR_MIN_CELL_OVERLAP_FRACTION,
                SMEAR_MIN_CELL_BOUNDARY_CONTACT_FRACTION,
            )
        )
        endpoint_ids.update(
            _weak_smear_contact_cell_ids(
                weak_mask,
                layer_labels,
            )
        )

    transient_ids = set()
    transient_stack = smear_detection.get("transient_mask_stack")
    if transient_stack is not None:
        for z_index, transient_mask in enumerate(transient_stack):
            transient_ids.update(
                _smear_contact_cell_ids(
                    transient_mask,
                    cell_label_stack[z_index],
                    SMEAR_TOUCH_DILATION_PIXELS,
                    SMEAR_MIN_CELL_OVERLAP_FRACTION,
                    SMEAR_MIN_CELL_BOUNDARY_CONTACT_FRACTION,
                )
            )
    mapped = dict(smear_detection)
    mapped["cell_ids"] = endpoint_ids | transient_ids
    mapped["transient_cell_ids"] = transient_ids
    return mapped


def _renumber_large_components(component_labels, min_area_pixels, first_label):
    component_sizes = np.bincount(component_labels.ravel())
    keep_ids = np.flatnonzero(component_sizes >= min_area_pixels)
    keep_ids = keep_ids[keep_ids != 0]

    remap = np.zeros(component_sizes.size, dtype=np.int32)
    remap[keep_ids] = first_label + np.arange(keep_ids.size)
    return remap[component_labels], first_label + keep_ids.size


def _bright_pixel_reference(pixel_intensities, percentile):
    """Estimate bright non-PNC signal, assuming PNC dots occupy the upper tail."""
    return float(np.percentile(pixel_intensities, percentile))


def _pnc_intensity_threshold(bright_reference, multiplier, offset=None):
    if offset is not None:
        return bright_reference + offset
    return bright_reference * multiplier


def _segment_pncs_by_cell(
    pnc_img,
    labels,
    bright_pixel_percentile=PNC_BRIGHT_PIXEL_PERCENTILE,
    pnc_threshold_multiplier=PNC_THRESHOLD_MULTIPLIER,
    pnc_threshold_offset=None,
):
    pnc_labels = np.zeros(pnc_img.shape, dtype=np.int32)
    next_pnc_id = 1

    for cell_id, cell_slice in enumerate(ndi.find_objects(labels), start=1):
        if cell_slice is None:
            continue

        cell_labels = labels[cell_slice]
        cell_mask = cell_labels == cell_id
        cell_pnc_img = pnc_img[cell_slice]
        bright_reference = _bright_pixel_reference(
            cell_pnc_img[cell_mask],
            bright_pixel_percentile,
        )
        min_area_pixels = max(
            1,
            int(np.ceil(np.count_nonzero(cell_mask) * PNC_MIN_CELL_AREA_FRACTION)),
        )
        threshold = _pnc_intensity_threshold(
            bright_reference,
            pnc_threshold_multiplier,
            offset=pnc_threshold_offset,
        )
        candidates = cell_mask & (cell_pnc_img > threshold)
        component_labels, _ = ndi.label(candidates)

        kept_components, next_pnc_id = _renumber_large_components(
            component_labels,
            min_area_pixels,
            next_pnc_id,
        )
        pnc_roi = pnc_labels[cell_slice]
        pnc_roi[cell_mask] = kept_components[cell_mask]

    return pnc_labels


def _segment_pncs_across_z(
    pnc_stack,
    labels,
    bright_pixel_percentile=PNC_BRIGHT_PIXEL_PERCENTILE,
    pnc_threshold_multiplier=PNC_THRESHOLD_MULTIPLIER,
    pnc_threshold_offset=None,
):
    labels = np.asarray(labels)
    dynamic_labels = labels.ndim == 3
    image_shape = labels.shape[1:] if dynamic_labels else labels.shape
    if dynamic_labels and len(labels) != len(pnc_stack):
        raise ValueError("The cell label stack must match the PNC z-stack.")
    projected_pnc_mask = np.zeros(image_shape, dtype=bool)
    cells_with_pnc_ids = set()

    for z_index, pnc_img in enumerate(pnc_stack):
        layer_labels = labels[z_index] if dynamic_labels else labels
        layer_pnc_labels = _segment_pncs_by_cell(
            pnc_img,
            layer_labels,
            bright_pixel_percentile=bright_pixel_percentile,
            pnc_threshold_multiplier=pnc_threshold_multiplier,
            pnc_threshold_offset=pnc_threshold_offset,
        )
        layer_pnc_mask = layer_pnc_labels > 0
        projected_pnc_mask |= layer_pnc_mask
        cells_with_pnc_ids.update(
            _cell_ids_from_labels(layer_labels[layer_pnc_mask])
        )

    projected_pnc_labels, _ = ndi.label(projected_pnc_mask)
    return projected_pnc_labels, cells_with_pnc_ids


def _normalized_gaussian_filter(image, mask, sigma):
    """Smooth values inside a mask without treating outside pixels as zero."""
    weights = ndi.gaussian_filter(mask.astype(float), sigma)
    weighted_image = ndi.gaussian_filter(image.astype(float) * mask, sigma)
    return weighted_image / np.maximum(weights, 1e-6)


def _expanded_object_slice(object_slice, shape, padding):
    return tuple(
        slice(
            max(0, axis_slice.start - padding),
            min(axis_size, axis_slice.stop + padding),
        )
        for axis_slice, axis_size in zip(object_slice, shape)
    )


def _detect_txred_nucleoli(pnc_img, cell_mask):
    """Return broad, dim, compact TxRed regions that can represent nucleoli."""
    cell_area = int(np.count_nonzero(cell_mask))
    if cell_area == 0:
        return np.zeros(cell_mask.shape, dtype=bool)

    equivalent_radius = np.sqrt(cell_area / np.pi)
    interior_margin = max(
        NUCLEOLUS_MIN_INTERIOR_MARGIN_PIXELS,
        int(round(equivalent_radius * NUCLEOLUS_INTERIOR_MARGIN_FRACTION)),
    )
    distance_from_cell_border = ndi.distance_transform_edt(cell_mask)
    cell_interior = cell_mask & (distance_from_cell_border >= interior_margin)
    if not np.any(cell_interior):
        return np.zeros(cell_mask.shape, dtype=bool)

    nucleolus_scale = _normalized_gaussian_filter(
        pnc_img,
        cell_mask,
        NUCLEOLUS_SMOOTH_SIGMA,
    )
    local_background = _normalized_gaussian_filter(
        pnc_img,
        cell_mask,
        NUCLEOLUS_BACKGROUND_SIGMA,
    )
    darkness = local_background - nucleolus_scale
    peak_threshold = np.percentile(
        darkness[cell_interior],
        NUCLEOLUS_PEAK_PERCENTILE,
    )
    local_maxima = darkness == ndi.maximum_filter(
        darkness,
        size=NUCLEOLUS_PEAK_WINDOW,
    )
    peak_coordinates = np.argwhere(
        cell_interior & local_maxima & (darkness >= peak_threshold)
    )
    peak_coordinates = sorted(
        peak_coordinates,
        key=lambda coordinate: darkness[tuple(coordinate)],
        reverse=True,
    )

    max_nucleolus_radius = max(
        NUCLEOLUS_MIN_RADIUS_PIXELS,
        int(round(equivalent_radius * NUCLEOLUS_MAX_RADIUS_FRACTION)),
    )
    y_grid, x_grid = np.indices(cell_mask.shape)
    accepted_regions = []
    accepted_peaks = []

    for peak_y, peak_x in peak_coordinates:
        if any(
            np.hypot(peak_y - other_y, peak_x - other_x) < 10
            for other_y, other_x in accepted_peaks
        ):
            continue

        peak_value = float(darkness[peak_y, peak_x])
        search_disk = (
            (y_grid - peak_y) ** 2 + (x_grid - peak_x) ** 2
            <= max_nucleolus_radius**2
        )
        grow_threshold = max(
            0.0,
            peak_value * NUCLEOLUS_GROW_PEAK_FRACTION,
        )
        growable = (
            cell_interior & search_disk & (darkness >= grow_threshold)
        )
        grown_labels, _ = ndi.label(growable)
        grown_id = int(grown_labels[peak_y, peak_x])
        if grown_id == 0:
            continue

        region = grown_labels == grown_id
        region = ndi.binary_closing(region, iterations=3)
        region = ndi.binary_fill_holes(region)
        properties = regionprops(region.astype(np.uint8))
        if not properties:
            continue

        prop = properties[0]
        area_fraction = prop.area / cell_area
        if not (
            NUCLEOLUS_MIN_CELL_AREA_FRACTION
            <= area_fraction
            <= NUCLEOLUS_MAX_CELL_AREA_FRACTION
        ):
            continue
        if prop.solidity < NUCLEOLUS_MIN_SOLIDITY:
            continue
        if prop.eccentricity > NUCLEOLUS_MAX_ECCENTRICITY:
            continue
        # The growth mask is already restricted to ``cell_interior``. Allow a
        # nucleolus to reach that inset boundary so modest segmentation errors
        # between touching cells do not erase otherwise valid geometry.
        if np.min(distance_from_cell_border[region]) < interior_margin - 0.25:
            continue

        # A bright PNC can cut a notch into the detected dark region. The
        # convex hull restores the underlying elliptical nucleolus boundary.
        region = convex_hull_image(region)
        region = ndi.binary_dilation(
            region,
            iterations=NUCLEOLUS_CONVEX_DILATION_PIXELS,
        )
        region &= cell_mask
        accepted_regions.append(region)
        accepted_peaks.append((int(peak_y), int(peak_x)))

    nucleolus_mask = np.zeros(cell_mask.shape, dtype=bool)
    for region in accepted_regions:
        nucleolus_mask |= region
    return nucleolus_mask


def _nucleolus_rescue_candidates(
    pnc_img,
    cell_mask,
    nucleolus_mask,
    bright_reference,
    inner_threshold_multiplier,
    outer_threshold_multiplier,
    outer_band_pixels,
    min_core_cell_area_fraction=(
        PNC_RESCUE_MIN_CORE_CELL_AREA_FRACTION
    ),
):
    if not np.any(nucleolus_mask):
        empty = np.zeros(cell_mask.shape, dtype=bool)
        return empty, empty, empty, []

    inner_region = nucleolus_mask & cell_mask
    distance_from_nucleolus = ndi.distance_transform_edt(~nucleolus_mask)
    outer_region = (
        cell_mask
        & ~nucleolus_mask
        & (distance_from_nucleolus <= outer_band_pixels)
    )

    threshold_image = np.full(
        cell_mask.shape,
        np.inf,
        dtype=float,
    )
    threshold_image[outer_region] = (
        bright_reference * outer_threshold_multiplier
    )
    threshold_image[inner_region] = (
        bright_reference * inner_threshold_multiplier
    )
    candidates = cell_mask & (pnc_img > threshold_image)
    component_labels, component_count = ndi.label(candidates)
    component_sizes = np.bincount(component_labels.ravel())

    min_area_pixels = max(
        1,
        int(np.ceil(np.count_nonzero(cell_mask) * PNC_MIN_CELL_AREA_FRACTION)),
    )
    max_area_pixels = max(
        min_area_pixels,
        int(
            np.ceil(
                np.count_nonzero(cell_mask)
                * PNC_RESCUE_MAX_CELL_AREA_FRACTION
            )
        ),
    )
    min_core_pixels = max(
        1,
        int(
            np.ceil(
                np.count_nonzero(cell_mask)
                * min_core_cell_area_fraction
            )
        ),
    )

    spot_scale = _normalized_gaussian_filter(
        pnc_img,
        cell_mask,
        PNC_LOCAL_SPOT_SIGMA,
    )
    spot_background = _normalized_gaussian_filter(
        pnc_img,
        cell_mask,
        PNC_LOCAL_BACKGROUND_SIGMA,
    )
    local_contrast = spot_scale - spot_background
    minimum_local_contrast = max(
        0.0,
        float(
            np.percentile(
                local_contrast[cell_mask],
                PNC_LOCAL_CONTRAST_PERCENTILE,
            )
            * PNC_LOCAL_CONTRAST_MULTIPLIER
        ),
    )

    kept_mask = np.zeros(cell_mask.shape, dtype=bool)
    details = []
    for component_id in range(1, component_count + 1):
        component_area = int(component_sizes[component_id])
        if not min_area_pixels <= component_area <= max_area_pixels:
            continue

        component = component_labels == component_id
        peak_local_contrast = float(np.max(local_contrast[component]))
        if peak_local_contrast < minimum_local_contrast:
            continue
        strong_core_pixels = int(
            np.count_nonzero(
                component
                & (local_contrast >= minimum_local_contrast)
            )
        )
        if strong_core_pixels < min_core_pixels:
            continue

        inside_pixels = int(np.count_nonzero(component & inner_region))
        outside_pixels = int(np.count_nonzero(component & outer_region))
        if inside_pixels and outside_pixels:
            zone = "both"
        elif inside_pixels:
            zone = "interior"
        else:
            zone = "exterior"

        center_y, center_x = ndi.center_of_mass(component)
        kept_mask |= component
        details.append(
            {
                "area_pixels": component_area,
                "center_y": float(center_y),
                "center_x": float(center_x),
                "zone": zone,
                "peak_local_contrast": peak_local_contrast,
                "minimum_local_contrast": minimum_local_contrast,
                "strong_core_pixels": strong_core_pixels,
                "minimum_core_pixels": min_core_pixels,
            }
        )

    return kept_mask, inner_region, outer_region, details


def _segment_nucleolus_rescues_across_z(
    pnc_stack,
    labels,
    baseline_positive_cell_ids,
    bright_pixel_percentile=PNC_BRIGHT_PIXEL_PERCENTILE,
    inner_threshold_multiplier=NUCLEOLUS_INNER_THRESHOLD_MULTIPLIER,
    outer_threshold_multiplier=NUCLEOLUS_OUTER_THRESHOLD_MULTIPLIER,
    outer_band_pixels=NUCLEOLUS_OUTER_BAND_PIXELS,
):
    labels = np.asarray(labels)
    if labels.ndim == 2:
        label_stack = np.repeat(labels[np.newaxis, ...], len(pnc_stack), axis=0)
    elif labels.ndim == 3 and len(labels) == len(pnc_stack):
        label_stack = labels
    else:
        raise ValueError("Cell labels must be 2D or match the PNC z-stack.")

    projected_rescue_mask = np.zeros(label_stack.shape[1:], dtype=bool)
    projected_nucleolus_mask = np.zeros(label_stack.shape[1:], dtype=bool)
    projected_outer_region = np.zeros(label_stack.shape[1:], dtype=bool)
    rescued_cell_ids = set()
    detections = []

    cells_to_check = (
        _cell_ids_from_labels(label_stack) - set(baseline_positive_cell_ids)
    )
    object_slices = ndi.find_objects(label_stack)

    for cell_id in sorted(cells_to_check):
        track_slice = (
            object_slices[cell_id - 1]
            if cell_id - 1 < len(object_slices)
            else None
        )
        if track_slice is None:
            continue
        _, y_slice, x_slice = track_slice
        object_slice = (
            y_slice,
            x_slice,
        )
        cell_slice = _expanded_object_slice(
            object_slice,
            label_stack.shape[1:],
            NUCLEOLUS_ROI_PADDING_PIXELS,
        )
        cell_mask_stack = label_stack[(slice(None),) + cell_slice] == cell_id
        cell_stack = pnc_stack[(slice(None),) + cell_slice].astype(
            float,
            copy=False,
        )
        layer_nucleolus_masks = np.asarray(
            [
                _detect_txred_nucleoli(pnc_img, cell_mask_stack[z_index])
                for z_index, pnc_img in enumerate(cell_stack)
            ]
        )
        projected_nucleolus_mask[cell_slice] |= np.any(
            layer_nucleolus_masks,
            axis=0,
        )

        cell_was_rescued = False
        for z_index, pnc_img in enumerate(cell_stack):
            cell_mask = cell_mask_stack[z_index]
            if not np.any(cell_mask):
                continue
            first_z = max(0, z_index - NUCLEOLUS_Z_PROPAGATION)
            last_z = min(
                len(cell_stack),
                z_index + NUCLEOLUS_Z_PROPAGATION + 1,
            )
            propagated_nucleolus_mask = np.any(
                layer_nucleolus_masks[first_z:last_z],
                axis=0,
            ) & cell_mask
            bright_reference = _bright_pixel_reference(
                pnc_img[cell_mask],
                bright_pixel_percentile,
            )
            (
                layer_rescue_mask,
                _inner_region,
                outer_region,
                layer_details,
            ) = _nucleolus_rescue_candidates(
                pnc_img,
                cell_mask,
                propagated_nucleolus_mask,
                bright_reference,
                inner_threshold_multiplier,
                outer_threshold_multiplier,
                outer_band_pixels,
            )
            projected_outer_region[cell_slice] |= outer_region
            if not layer_details:
                continue

            projected_rescue_mask[cell_slice] |= layer_rescue_mask
            cell_was_rescued = True
            for detail in layer_details:
                detail = detail.copy()
                detail.update(
                    {
                        "cell_id": cell_id,
                        "z_index": z_index,
                        "z_layer": z_index + 1,
                        "center_y": detail["center_y"] + cell_slice[0].start,
                        "center_x": detail["center_x"] + cell_slice[1].start,
                    }
                )
                detections.append(detail)

        if cell_was_rescued:
            rescued_cell_ids.add(cell_id)

    return {
        "mask": projected_rescue_mask,
        "cell_ids": rescued_cell_ids,
        "nucleolus_mask": projected_nucleolus_mask,
        "outer_region": projected_outer_region,
        "detections": detections,
    }


def analyze_pnc(
    file_path,
    model,
    scale=DEFAULT_SCALE,
    segmentation_mode=DEFAULT_SEGMENTATION_MODE,
    cell_channel=DEFAULT_CELL_CHANNEL,
    pnc_channel=DEFAULT_PNC_CHANNEL,
    segmentation_channel=DEFAULT_SEGMENTATION_CHANNEL,
    border_iterations=CELL_BORDER_ITERATIONS,
    min_cell_median_area_fraction=MIN_CELL_MEDIAN_AREA_FRACTION,
    max_cell_median_area_fraction=MAX_CELL_MEDIAN_AREA_FRACTION,
    min_cell_dapi_median_fraction=MIN_CELL_DAPI_MEDIAN_FRACTION,
    bright_pixel_percentile=PNC_BRIGHT_PIXEL_PERCENTILE,
    pnc_threshold_multiplier=PNC_THRESHOLD_MULTIPLIER,
    pnc_threshold_offset=None,
    enable_nucleolus_rescue=ENABLE_NUCLEOLUS_RESCUE,
    nucleolus_inner_threshold_multiplier=(
        NUCLEOLUS_INNER_THRESHOLD_MULTIPLIER
    ),
    nucleolus_outer_threshold_multiplier=(
        NUCLEOLUS_OUTER_THRESHOLD_MULTIPLIER
    ),
    nucleolus_outer_band_pixels=NUCLEOLUS_OUTER_BAND_PIXELS,
    enable_smear_exclusion=ENABLE_SMEAR_EXCLUSION,
    histogram_cell_id=None,
    histogram_z_index=None,
):
    if not 0 <= min_cell_median_area_fraction <= 1:
        raise ValueError("min_cell_median_area_fraction must be between 0 and 1.")
    if max_cell_median_area_fraction < 1:
        raise ValueError("max_cell_median_area_fraction must be at least 1.")
    if max_cell_median_area_fraction < min_cell_median_area_fraction:
        raise ValueError(
            "max_cell_median_area_fraction must not be below the minimum fraction."
        )
    if not 0 <= min_cell_dapi_median_fraction <= 1:
        raise ValueError(
            "min_cell_dapi_median_fraction must be between 0 and 1."
        )
    if not 0 < bright_pixel_percentile < 100:
        raise ValueError("bright_pixel_percentile must be between 0 and 100.")
    if pnc_threshold_offset is None and pnc_threshold_multiplier < 1:
        raise ValueError("pnc_threshold_multiplier must be at least 1.")
    if pnc_threshold_offset is not None and pnc_threshold_offset < 0:
        raise ValueError("pnc_threshold_offset must be non-negative.")
    if nucleolus_inner_threshold_multiplier < 0:
        raise ValueError(
            "nucleolus_inner_threshold_multiplier must be non-negative."
        )
    if nucleolus_outer_threshold_multiplier < 0:
        raise ValueError(
            "nucleolus_outer_threshold_multiplier must be non-negative."
        )
    if (
        nucleolus_inner_threshold_multiplier
        > nucleolus_outer_threshold_multiplier
    ):
        raise ValueError(
            "The nucleolus interior threshold multiplier must not exceed "
            "the exterior multiplier."
        )
    if nucleolus_outer_band_pixels < 0:
        raise ValueError("nucleolus_outer_band_pixels must be non-negative.")

    data = nd2.imread(file_path)
    print(f"File shape: {data.shape}")
    cell_stack, segmentation_stack, pnc_stack = _select_analysis_stacks(
        data,
        cell_channel=cell_channel,
        segmentation_channel=segmentation_channel,
        pnc_channel=pnc_channel,
    )

    cell_img = np.max(cell_stack, axis=0)
    segmentation_img = np.max(segmentation_stack, axis=0)
    pnc_img = np.max(pnc_stack, axis=0)

    if segmentation_mode == SEGMENTATION_VERSION:
        segmentation = segment_cell_tracks(
            segmentation_stack,
            model,
            scale=scale,
        )
    elif segmentation_mode == LEGACY_SEGMENTATION_VERSION:
        segmentation = projection_cell_segmentation(
            segmentation_stack,
            model,
            scale=scale,
        )
    else:
        raise ValueError(
            "segmentation_mode must be 'z_track_v1' or 'projection_v1'."
        )
    labels = segmentation["canonical_labels"]
    cell_label_stack = segmentation["label_stack"]
    track_metadata = segmentation["tracks"]
    # Raw per-layer StarDist IDs are no longer needed after stable tracks have
    # been constructed; release that duplicate stack before TxRed processing.
    segmentation.pop("raw_layer_labels", None)
    z_aware_segmentation = segmentation_mode == SEGMENTATION_VERSION
    all_cell_ids = _cell_ids_from_labels(labels)
    segmentation_excluded_cell_ids = set(
        segmentation.get("persistent_merge_track_ids", [])
    )
    non_intact_cell_ids = (
        _non_intact_ids_across_z(cell_label_stack)
        if z_aware_segmentation
        else get_non_intact_ellipse_ids(labels)
    )
    intact_cell_ids = (
        all_cell_ids
        - non_intact_cell_ids
        - segmentation_excluded_cell_ids
    )
    (
        low_dapi_cell_ids,
        median_cell_dapi,
        cell_dapi_medians,
    ) = _low_dapi_cell_ids(
        cell_stack if z_aware_segmentation else cell_img,
        cell_label_stack if z_aware_segmentation else labels,
        intact_cell_ids,
        min_median_fraction=min_cell_dapi_median_fraction,
        track_metadata=(track_metadata if z_aware_segmentation else None),
    )
    intact_quality_cell_ids = intact_cell_ids - low_dapi_cell_ids
    if enable_smear_exclusion:
        smear_detection = _detect_txred_smears(pnc_stack, labels)
        if z_aware_segmentation:
            smear_detection = _map_smears_to_z_specific_cells(
                smear_detection,
                cell_label_stack,
            )
    else:
        smear_detection = {
            "mask": np.zeros(labels.shape, dtype=bool),
            "endpoint_mask": np.zeros(labels.shape, dtype=bool),
            "strong_endpoint_mask": np.zeros(labels.shape, dtype=bool),
            "weak_endpoint_mask": np.zeros(labels.shape, dtype=bool),
            "cell_ids": set(),
            "response": np.zeros(labels.shape, dtype=float),
            "growth_threshold": None,
            "seed_threshold": None,
            "component_count": 0,
            "top_tail_ratio": None,
            "transient_mask": np.zeros(labels.shape, dtype=bool),
            "transient_response": np.zeros(labels.shape, dtype=float),
            "transient_component_count": 0,
            "transient_cell_ids": set(),
            "transient_z_layers": [],
            "transient_detections": [],
            "transient_mask_stack": np.zeros(
                cell_label_stack.shape,
                dtype=bool,
            ),
        }
    detected_smear_cell_ids = smear_detection["cell_ids"] & all_cell_ids
    smear_excluded_cell_ids = (
        detected_smear_cell_ids & intact_quality_cell_ids
    )
    intact_non_smear_cell_ids = (
        intact_quality_cell_ids - smear_excluded_cell_ids
    )
    (
        small_cell_ids,
        large_cell_ids,
        median_cell_area,
        minimum_cell_area,
        maximum_cell_area,
    ) = _cell_size_outlier_ids(
        cell_label_stack if z_aware_segmentation else labels,
        intact_non_smear_cell_ids,
        min_median_area_fraction=min_cell_median_area_fraction,
        max_median_area_fraction=max_cell_median_area_fraction,
        track_metadata=(track_metadata if z_aware_segmentation else None),
    )
    for cell_id in large_cell_ids:
        if cell_id in track_metadata:
            track_metadata[cell_id]["quality_status"] = (
                "suspected_persistent_merge_large_area"
            )
    valid_cell_ids = intact_non_smear_cell_ids - small_cell_ids - large_cell_ids

    analysis_labels = cell_label_stack if z_aware_segmentation else labels
    pnc_detection_labels = np.where(
        np.isin(analysis_labels, list(valid_cell_ids)),
        analysis_labels,
        0,
    )
    baseline_pnc_labels, baseline_cells_with_pnc_ids = _segment_pncs_across_z(
        pnc_stack,
        pnc_detection_labels,
        bright_pixel_percentile=bright_pixel_percentile,
        pnc_threshold_multiplier=pnc_threshold_multiplier,
        pnc_threshold_offset=pnc_threshold_offset,
    )
    if enable_nucleolus_rescue:
        nucleolus_rescue = _segment_nucleolus_rescues_across_z(
            pnc_stack,
            pnc_detection_labels,
            baseline_cells_with_pnc_ids,
            bright_pixel_percentile=bright_pixel_percentile,
            inner_threshold_multiplier=(
                nucleolus_inner_threshold_multiplier
            ),
            outer_threshold_multiplier=(
                nucleolus_outer_threshold_multiplier
            ),
            outer_band_pixels=nucleolus_outer_band_pixels,
        )
    else:
        empty_mask = np.zeros(labels.shape, dtype=bool)
        nucleolus_rescue = {
            "mask": empty_mask.copy(),
            "cell_ids": set(),
            "nucleolus_mask": empty_mask.copy(),
            "outer_region": empty_mask.copy(),
            "detections": [],
        }

    nucleolus_rescued_cell_ids = nucleolus_rescue["cell_ids"]
    cells_with_pnc_ids = (
        baseline_cells_with_pnc_ids | nucleolus_rescued_cell_ids
    )
    pnc_labels, _ = ndi.label(
        (baseline_pnc_labels > 0) | nucleolus_rescue["mask"]
    )
    valid_cells_with_pnc_ids = cells_with_pnc_ids & valid_cell_ids

    if histogram_cell_id is None:
        histogram_candidates = (
            valid_cells_with_pnc_ids or valid_cell_ids or all_cell_ids
        )
        histogram_cell_id = min(histogram_candidates, default=None)
    elif histogram_cell_id not in all_cell_ids:
        available_ids = ", ".join(str(cell_id) for cell_id in sorted(all_cell_ids))
        raise ValueError(
            f"Cell ID {histogram_cell_id} was not found. "
            f"Available cell IDs: {available_ids or 'none'}"
        )

    if histogram_z_index is not None and not 0 <= histogram_z_index < len(pnc_stack):
        raise ValueError(
            f"PNC z-index {histogram_z_index} is outside the available range "
            f"0-{len(pnc_stack) - 1}."
        )

    fixed_histogram_z_index = histogram_z_index

    def _z_index_for_histogram_cell(cell_id):
        if fixed_histogram_z_index is not None:
            return fixed_histogram_z_index

        layer_maximums = np.full(len(pnc_stack), -np.inf, dtype=float)
        for z_index in range(len(pnc_stack)):
            cell_mask = cell_label_stack[z_index] == cell_id
            if np.any(cell_mask):
                layer_maximums[z_index] = np.max(
                    pnc_stack[z_index][cell_mask]
                )
        return int(np.argmax(layer_maximums))

    if histogram_cell_id is None:
        histogram_z_index = None
    else:
        histogram_z_index = _z_index_for_histogram_cell(histogram_cell_id)

    total_cells = len(valid_cell_ids)
    cells_with_pnc = len(valid_cells_with_pnc_ids)
    percent_cells_with_pnc = cells_with_pnc / total_cells * 100 if total_cells else 0

    raw_cells = normalize(cell_img)
    raw_pncs = normalize(pnc_img)
    combined_raw = np.dstack(
        (
            np.maximum(raw_cells, raw_pncs),
            raw_cells,
            raw_cells,
        )
    )
    combined_raw = np.clip(combined_raw, 0, 1)

    valid_without_pnc_ids = valid_cell_ids - valid_cells_with_pnc_ids
    baseline_positive_cell_ids = (
        valid_cells_with_pnc_ids - nucleolus_rescued_cell_ids
    )
    baseline_pnc_border = _label_border(
        labels,
        baseline_positive_cell_ids,
        iterations=border_iterations,
    )
    nucleolus_rescued_border = _label_border(
        labels,
        nucleolus_rescued_cell_ids,
        iterations=border_iterations,
    )
    without_pnc_border = _label_border(
        labels,
        valid_without_pnc_ids,
        iterations=border_iterations,
    )
    excluded_border = _label_border(
        labels,
        non_intact_cell_ids,
        iterations=border_iterations,
    )
    low_dapi_cell_border = _label_border(
        labels,
        low_dapi_cell_ids,
        iterations=border_iterations,
    )
    small_cell_border = _label_border(
        labels,
        small_cell_ids,
        iterations=border_iterations,
    )
    large_cell_border = _label_border(
        labels,
        large_cell_ids,
        iterations=border_iterations,
    )
    smear_cell_border = _label_border(
        labels,
        detected_smear_cell_ids,
        iterations=border_iterations,
    )
    segmentation_excluded_border = _label_border(
        labels,
        segmentation_excluded_cell_ids,
        iterations=border_iterations,
    )

    combined_annotated = combined_raw.copy()
    combined_annotated[without_pnc_border] = [0, 1, 1]
    combined_annotated[baseline_pnc_border] = [0, 1, 0]
    combined_annotated[nucleolus_rescued_border] = [1, 1, 0]
    combined_annotated[low_dapi_cell_border] = [0, 0.4, 1]
    combined_annotated[small_cell_border] = [1, 0.5, 0]
    combined_annotated[large_cell_border] = [0.5, 0, 1]
    combined_annotated[excluded_border] = [1, 0, 0]
    combined_annotated[segmentation_excluded_border] = [1, 0.75, 0]
    # Draw smear contact last so a cell that is also non-intact still remains
    # visibly classified as smear-positive.
    combined_annotated[smear_cell_border] = [1, 0, 1]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    axes[0, 0].imshow(cell_img, cmap="gray")
    axes[0, 0].set_title("Raw cells (maximum z-projection)")
    segmented_cells_ax = axes[1, 0]
    segmented_cells_ax.imshow(labels, cmap="nipy_spectral")
    indicator_overlay = np.zeros((*labels.shape, 4), dtype=float)
    indicator_artist = segmented_cells_ax.imshow(indicator_overlay)
    indicator_text = segmented_cells_ax.text(
        0,
        0,
        "",
        color="black",
        fontsize=10,
        fontweight="bold",
        ha="center",
        va="center",
        visible=False,
        bbox={
            "boxstyle": "circle,pad=0.25",
            "facecolor": "yellow",
            "edgecolor": "black",
        },
    )

    def _draw_histogram_cell_indicator(cell_id):
        indicator_overlay.fill(0)
        if cell_id is None:
            indicator_artist.set_data(indicator_overlay)
            indicator_text.set_visible(False)
            segmented_cells_ax.set_title("Segmented cells (click to select)")
            return

        cell_mask = labels == cell_id
        indicator_border = _label_border(labels, {cell_id}, iterations=0)
        indicator_overlay[indicator_border] = [1, 1, 0, 1]
        indicator_artist.set_data(indicator_overlay)

        center_y, center_x = ndi.center_of_mass(cell_mask)
        indicator_text.set_position((center_x, center_y))
        indicator_text.set_text(str(cell_id))
        indicator_text.set_visible(True)
        segmented_cells_ax.set_title(
            "Segmented cells (click to select)\n"
            f"Histogram cell {cell_id} = yellow outline"
        )

    _draw_histogram_cell_indicator(histogram_cell_id)

    axes[0, 1].imshow(pnc_img, cmap="gray")
    axes[0, 1].set_title("Raw PNCs (maximum z-projection)")
    axes[1, 1].imshow(pnc_labels, cmap="nipy_spectral")
    rescue_overlay = np.zeros((*labels.shape, 4), dtype=float)
    rescue_overlay[nucleolus_rescue["mask"]] = [1, 1, 0, 1]
    axes[1, 1].imshow(rescue_overlay)
    if pnc_threshold_offset is None:
        pnc_threshold_rule = (
            f"{pnc_threshold_multiplier:g}x cell P{bright_pixel_percentile:g}"
        )
    else:
        pnc_threshold_rule = (
            f"cell P{bright_pixel_percentile:g} + {pnc_threshold_offset:g}"
        )
    segmented_pnc_title = (
        "Segmented PNCs (maximum z-projection)\n"
        f">{pnc_threshold_rule}, "
        f">= {PNC_MIN_CELL_AREA_FRACTION:g} cell area"
    )
    if enable_nucleolus_rescue:
        segmented_pnc_title += (
            "\n"
            "nucleolus rescue: "
            f"{nucleolus_inner_threshold_multiplier:g}x inside, "
            f"{nucleolus_outer_threshold_multiplier:g}x outside"
        )
    axes[1, 1].set_title(segmented_pnc_title)

    axes[0, 2].imshow(combined_annotated)
    axes[0, 2].set_title(
        f"Annotated: {cells_with_pnc}/{total_cells}, {percent_cells_with_pnc:.1f}%"
    )

    legend_handles = [
        Patch(
            facecolor=(0, 1, 0),
            edgecolor="none",
            label="PNC-positive at baseline threshold",
        ),
    ]
    if enable_nucleolus_rescue:
        legend_handles.append(
            Patch(
                facecolor=(1, 1, 0),
                edgecolor="none",
                label="PNC-positive via nucleolus rescue",
            )
        )
    legend_handles.extend(
        [
        Patch(facecolor=(0, 1, 1), edgecolor="none", label="PNC-negative cell"),
        Patch(
            facecolor=(1, 0.5, 0),
            edgecolor="none",
            label=f"Excluded <{min_cell_median_area_fraction:.0%} median cell area",
        ),
        Patch(
            facecolor=(0.5, 0, 1),
            edgecolor="none",
            label=f"Excluded >{max_cell_median_area_fraction:g}x median cell area",
        ),
        Patch(
            facecolor=(1, 0, 1),
            edgecolor="none",
            label="Detected: touches TxRed smear",
        ),
        Patch(
            facecolor=(0, 0.4, 1),
            edgecolor="none",
            label=(
                "Excluded low-DAPI segmentation artifact "
                f"(<{min_cell_dapi_median_fraction:.0%} field median)"
            ),
        ),
        Patch(
            facecolor=(1, 0.75, 0),
            edgecolor="none",
            label="Excluded unresolved tracked merge",
        ),
        Patch(facecolor=(1, 0, 0), edgecolor="none", label="Excluded non-intact cell"),
        ]
    )

    histogram_ax = axes[1, 2]

    def _draw_cell_histogram(cell_id, z_index):
        histogram_ax.clear()
        histogram_legend_handles = legend_handles.copy()
        histogram_median = None
        histogram_bright_reference = None
        histogram_threshold = None
        histogram_nucleolus_inner_threshold = None
        histogram_nucleolus_outer_threshold = None

        if cell_id is None:
            histogram_ax.text(
                0.5,
                0.5,
                "No segmented cell available",
                ha="center",
                va="center",
                transform=histogram_ax.transAxes,
            )
            histogram_ax.set_title("PNC pixel intensities")
        else:
            cell_mask = cell_label_stack[z_index] == cell_id
            if not np.any(cell_mask):
                available_layers = np.flatnonzero(
                    np.any(cell_label_stack == cell_id, axis=(1, 2))
                )
                if available_layers.size:
                    z_index = int(
                        available_layers[
                            np.argmin(np.abs(available_layers - z_index))
                        ]
                    )
                    cell_mask = cell_label_stack[z_index] == cell_id
            cell_intensities = pnc_stack[z_index][cell_mask]
            histogram_median = float(np.median(cell_intensities))
            histogram_bright_reference = _bright_pixel_reference(
                cell_intensities,
                bright_pixel_percentile,
            )
            histogram_threshold = _pnc_intensity_threshold(
                histogram_bright_reference,
                pnc_threshold_multiplier,
                offset=pnc_threshold_offset,
            )
            histogram_ax.hist(
                cell_intensities,
                bins=50,
                color="magenta",
                alpha=0.75,
            )
            median_line = histogram_ax.axvline(
                histogram_median,
                color="black",
                linestyle=":",
                linewidth=2,
                label=f"Median: {histogram_median:g}",
            )
            bright_reference_line = histogram_ax.axvline(
                histogram_bright_reference,
                color="orange",
                linestyle="-.",
                linewidth=2,
                label=(
                    f"Bright P{bright_pixel_percentile:g}: "
                    f"{histogram_bright_reference:g}"
                ),
            )
            threshold_line = histogram_ax.axvline(
                histogram_threshold,
                color="red",
                linestyle="--",
                linewidth=2,
                label=(
                    f"PNC threshold ({pnc_threshold_rule}): {histogram_threshold:g}"
                ),
            )
            histogram_legend_handles.extend(
                [median_line, bright_reference_line, threshold_line]
            )
            if enable_nucleolus_rescue:
                histogram_nucleolus_inner_threshold = (
                    histogram_bright_reference
                    * nucleolus_inner_threshold_multiplier
                )
                histogram_nucleolus_outer_threshold = (
                    histogram_bright_reference
                    * nucleolus_outer_threshold_multiplier
                )
                inner_threshold_line = histogram_ax.axvline(
                    histogram_nucleolus_inner_threshold,
                    color="purple",
                    linestyle="--",
                    linewidth=1.5,
                    label=(
                        "Nucleolus interior threshold: "
                        f"{histogram_nucleolus_inner_threshold:g}"
                    ),
                )
                outer_threshold_line = histogram_ax.axvline(
                    histogram_nucleolus_outer_threshold,
                    color="blue",
                    linestyle="--",
                    linewidth=1.5,
                    label=(
                        "Nucleolus exterior threshold: "
                        f"{histogram_nucleolus_outer_threshold:g}"
                    ),
                )
                histogram_legend_handles.extend(
                    [inner_threshold_line, outer_threshold_line]
                )
            histogram_ax.set_title(
                f"PNC pixel intensities: cell {cell_id}\n"
                f"(z-layer {z_index + 1}/{len(pnc_stack)})"
            )
            histogram_ax.set_xlabel("Raw pixel intensity")
            histogram_ax.set_ylabel("Pixel count")

        histogram_ax.legend(
            handles=histogram_legend_handles,
            loc="upper right",
            frameon=False,
            title="Legend",
        )
        return (
            histogram_median,
            histogram_bright_reference,
            histogram_threshold,
            histogram_nucleolus_inner_threshold,
            histogram_nucleolus_outer_threshold,
        )

    (
        histogram_median,
        histogram_bright_reference,
        histogram_threshold,
        histogram_nucleolus_inner_threshold,
        histogram_nucleolus_outer_threshold,
    ) = _draw_cell_histogram(histogram_cell_id, histogram_z_index)

    for ax in axes.flat:
        if ax is not histogram_ax:
            ax.axis("off")

    result = {
        "fig": fig,
        "cell_image": cell_img,
        "cell_stack": cell_stack,
        "segmentation_image": segmentation_img,
        "segmentation_stack": segmentation_stack,
        "pnc_image": pnc_img,
        "pnc_stack": pnc_stack,
        "z_layers": len(pnc_stack),
        "analysis_scale": scale,
        "segmentation_scale": segmentation["segmentation_scale"],
        "segmentation_version": segmentation["segmentation_version"],
        "segmentation_mode": segmentation_mode,
        "label_stack_fingerprint": segmentation[
            "label_stack_fingerprint"
        ],
        "track_metadata": track_metadata,
        "discarded_single_layer_tracks": segmentation[
            "discarded_single_layer_tracks"
        ],
        "cell_channel": cell_channel,
        "dapi_channel": cell_channel,
        "segmentation_channel": segmentation_channel,
        "pnc_channel": pnc_channel,
        "bright_pixel_percentile": bright_pixel_percentile,
        "pnc_threshold_multiplier": pnc_threshold_multiplier,
        "cells_with_pnc": cells_with_pnc,
        "total_cells": total_cells,
        "percent_cells_with_pnc": percent_cells_with_pnc,
        "all_cell_ids": sorted(all_cell_ids),
        "valid_cell_ids": sorted(valid_cell_ids),
        "cells_with_pnc_ids": sorted(valid_cells_with_pnc_ids),
        "baseline_cells_with_pnc_ids": sorted(
            baseline_cells_with_pnc_ids & valid_cell_ids
        ),
        "nucleolus_rescued_cells": len(nucleolus_rescued_cell_ids),
        "nucleolus_rescued_cell_ids": sorted(nucleolus_rescued_cell_ids),
        "nucleolus_rescue_detections": nucleolus_rescue["detections"],
        "cell_labels": labels,
        "cell_label_stack": cell_label_stack,
        "pnc_labels": pnc_labels,
        "baseline_pnc_labels": baseline_pnc_labels,
        "nucleolus_rescue_mask": nucleolus_rescue["mask"],
        "nucleolus_mask": nucleolus_rescue["nucleolus_mask"],
        "nucleolus_outer_region": nucleolus_rescue["outer_region"],
        "nucleolus_rescue_enabled": enable_nucleolus_rescue,
        "nucleolus_inner_threshold_multiplier": (
            nucleolus_inner_threshold_multiplier
        ),
        "nucleolus_outer_threshold_multiplier": (
            nucleolus_outer_threshold_multiplier
        ),
        "nucleolus_outer_band_pixels": nucleolus_outer_band_pixels,
        "nucleolus_min_core_cell_area_fraction": (
            PNC_RESCUE_MIN_CORE_CELL_AREA_FRACTION
        ),
        "smear_exclusion_enabled": enable_smear_exclusion,
        "smear_mask": smear_detection["mask"],
        "smear_response": smear_detection["response"],
        "smear_cell_ids": sorted(detected_smear_cell_ids),
        "smear_excluded_cell_ids": sorted(smear_excluded_cell_ids),
        "excluded_smear_cells": len(smear_excluded_cell_ids),
        "smear_growth_threshold": smear_detection["growth_threshold"],
        "smear_seed_threshold": smear_detection["seed_threshold"],
        "smear_component_count": smear_detection["component_count"],
        "smear_top_tail_ratio": smear_detection["top_tail_ratio"],
        "transient_smear_mask": smear_detection["transient_mask"],
        "transient_smear_response": smear_detection[
            "transient_response"
        ],
        "transient_smear_component_count": smear_detection[
            "transient_component_count"
        ],
        "transient_smear_cell_ids": sorted(
            smear_detection["transient_cell_ids"] & all_cell_ids
        ),
        "transient_smear_z_layers": smear_detection[
            "transient_z_layers"
        ],
        "transient_smear_detections": smear_detection[
            "transient_detections"
        ],
        "excluded_non_intact_cells": len(non_intact_cell_ids),
        "non_intact_cell_ids": sorted(non_intact_cell_ids),
        "segmentation_excluded_cell_ids": sorted(
            segmentation_excluded_cell_ids
        ),
        "excluded_unresolved_merge_cells": len(
            segmentation_excluded_cell_ids
        ),
        "excluded_low_dapi_cells": len(low_dapi_cell_ids),
        "low_dapi_cell_ids": sorted(low_dapi_cell_ids),
        "min_cell_dapi_median_fraction": min_cell_dapi_median_fraction,
        "median_cell_dapi": median_cell_dapi,
        "cell_dapi_medians": cell_dapi_medians,
        "excluded_small_cells": len(small_cell_ids),
        "small_cell_ids": sorted(small_cell_ids),
        "excluded_large_cells": len(large_cell_ids),
        "large_cell_ids": sorted(large_cell_ids),
        "median_cell_area_pixels": median_cell_area,
        "minimum_cell_area_pixels": minimum_cell_area,
        "maximum_cell_area_pixels": maximum_cell_area,
        "pnc_threshold_mode": (
            "offset" if pnc_threshold_offset is not None else "multiplier"
        ),
        "pnc_threshold_offset": pnc_threshold_offset,
        "histogram_cell_id": histogram_cell_id,
        "histogram_z_index": histogram_z_index,
        "histogram_median": histogram_median,
        "histogram_bright_reference": histogram_bright_reference,
        "histogram_threshold": histogram_threshold,
        "histogram_nucleolus_inner_threshold": (
            histogram_nucleolus_inner_threshold
        ),
        "histogram_nucleolus_outer_threshold": (
            histogram_nucleolus_outer_threshold
        ),
    }

    def _on_segmented_cell_click(event):
        if event.inaxes is not segmented_cells_ax:
            return
        if event.xdata is None or event.ydata is None:
            return

        x_index = int(np.floor(event.xdata + 0.5))
        y_index = int(np.floor(event.ydata + 0.5))
        if not (0 <= y_index < labels.shape[0] and 0 <= x_index < labels.shape[1]):
            return

        selected_cell_id = int(labels[y_index, x_index])
        if selected_cell_id <= 0:
            return

        selected_z_index = _z_index_for_histogram_cell(selected_cell_id)
        (
            selected_median,
            selected_bright_reference,
            selected_threshold,
            selected_nucleolus_inner_threshold,
            selected_nucleolus_outer_threshold,
        ) = _draw_cell_histogram(
            selected_cell_id,
            selected_z_index,
        )
        _draw_histogram_cell_indicator(selected_cell_id)
        result.update(
            histogram_cell_id=selected_cell_id,
            histogram_z_index=selected_z_index,
            histogram_median=selected_median,
            histogram_bright_reference=selected_bright_reference,
            histogram_threshold=selected_threshold,
            histogram_nucleolus_inner_threshold=(
                selected_nucleolus_inner_threshold
            ),
            histogram_nucleolus_outer_threshold=(
                selected_nucleolus_outer_threshold
            ),
        )
        fig.canvas.draw_idle()

    result["histogram_click_callback_id"] = fig.canvas.mpl_connect(
        "button_press_event",
        _on_segmented_cell_click,
    )
    return result


def _parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run PNC analysis with local per-cell PNC thresholding and show the plot."
        )
    )
    parser.add_argument("file_path", help="Path to the ND2 file to analyze.")
    parser.add_argument(
        "--segmentation-scale",
        "--scale",
        dest="scale",
        type=float,
        default=DEFAULT_SCALE,
        help=(
            "StarDist scale for each segmentation layer; --scale is retained as an "
            f"alias (default: {DEFAULT_SCALE:g})."
        ),
    )
    parser.add_argument(
        "--segmentation-mode",
        choices=(SEGMENTATION_VERSION, LEGACY_SEGMENTATION_VERSION),
        default=DEFAULT_SEGMENTATION_MODE,
        help=(
            "Use z-aware tracks by default or the historical fixed "
            "maximum-projection masks for reproduction."
        ),
    )
    parser.add_argument(
        "--cell-channel",
        type=int,
        default=DEFAULT_CELL_CHANNEL,
        help=(
            "DAPI channel used for display and DAPI quality checks "
            f"(default: {DEFAULT_CELL_CHANNEL})."
        ),
    )
    parser.add_argument(
        "--segmentation-channel",
        type=int,
        default=DEFAULT_SEGMENTATION_CHANNEL,
        help=(
            "Channel supplied to StarDist (default: "
            f"{DEFAULT_SEGMENTATION_CHANNEL}, TxRed). Use 0 for the previous "
            "DAPI segmentation."
        ),
    )
    parser.add_argument(
        "--pnc-channel",
        type=int,
        default=DEFAULT_PNC_CHANNEL,
    )
    parser.add_argument(
        "--min-cell-median-area-fraction",
        type=float,
        default=MIN_CELL_MEDIAN_AREA_FRACTION,
        help=(
            "Exclude intact cells below this fraction of the median intact-cell "
            f"area (default: {MIN_CELL_MEDIAN_AREA_FRACTION:g})."
        ),
    )
    parser.add_argument(
        "--max-cell-median-area-fraction",
        type=float,
        default=MAX_CELL_MEDIAN_AREA_FRACTION,
        help=(
            "Exclude intact cells above this multiple of the median intact-cell "
            f"area (default: {MAX_CELL_MEDIAN_AREA_FRACTION:g})."
        ),
    )
    parser.add_argument(
        "--min-cell-dapi-median-fraction",
        type=float,
        default=MIN_CELL_DAPI_MEDIAN_FRACTION,
        help=(
            "Exclude segmented objects whose median DAPI intensity is below "
            "this fraction of the field's median segmented-cell intensity "
            f"(default: {MIN_CELL_DAPI_MEDIAN_FRACTION:g})."
        ),
    )
    parser.add_argument(
        "--bright-pixel-percentile",
        type=float,
        default=PNC_BRIGHT_PIXEL_PERCENTILE,
        help=(
            "Percentile used to estimate each cell's bright non-PNC pixels "
            f"(default: {PNC_BRIGHT_PIXEL_PERCENTILE:g})."
        ),
    )
    parser.add_argument(
        "--pnc-threshold-multiplier",
        type=float,
        default=PNC_THRESHOLD_MULTIPLIER,
        help=(
            "Required intensity relative to the bright-pixel estimate when no "
            "offset is supplied "
            f"(default: {PNC_THRESHOLD_MULTIPLIER:g})."
        ),
    )
    parser.add_argument(
        "--pnc-threshold-offset",
        type=float,
        help=(
            "Add this value to the bright-pixel estimate instead of multiplying; "
            "for example, 1000 uses PNC threshold = bright reference + 1000."
        ),
    )
    parser.add_argument(
        "--no-nucleolus-rescue",
        dest="enable_nucleolus_rescue",
        action="store_false",
        default=ENABLE_NUCLEOLUS_RESCUE,
        help="Disable the TxRed nucleolus-aware rescue detector.",
    )
    parser.add_argument(
        "--nucleolus-inner-threshold-multiplier",
        type=float,
        default=NUCLEOLUS_INNER_THRESHOLD_MULTIPLIER,
        help=(
            "P95 multiplier used inside a detected TxRed nucleolus "
            f"(default: {NUCLEOLUS_INNER_THRESHOLD_MULTIPLIER:g})."
        ),
    )
    parser.add_argument(
        "--nucleolus-outer-threshold-multiplier",
        type=float,
        default=NUCLEOLUS_OUTER_THRESHOLD_MULTIPLIER,
        help=(
            "P95 multiplier used just outside a detected TxRed nucleolus "
            f"(default: {NUCLEOLUS_OUTER_THRESHOLD_MULTIPLIER:g})."
        ),
    )
    parser.add_argument(
        "--nucleolus-outer-band-pixels",
        type=int,
        default=NUCLEOLUS_OUTER_BAND_PIXELS,
        help=(
            "Euclidean width of the exterior nucleolus search band "
            f"(default: {NUCLEOLUS_OUTER_BAND_PIXELS} pixels)."
        ),
    )
    parser.add_argument(
        "--no-smear-exclusion",
        dest="enable_smear_exclusion",
        action="store_false",
        default=ENABLE_SMEAR_EXCLUSION,
        help="Disable exclusion of cells touching a detected TxRed smear.",
    )
    parser.add_argument(
        "--histogram-cell-id",
        type=int,
        help=(
            "Segmented cell ID to use for the PNC-intensity histogram. "
            "Defaults to a PNC-positive cell when available."
        ),
    )
    parser.add_argument(
        "--histogram-z-index",
        type=int,
        help=(
            "Zero-based PNC z-index to use for the histogram. Defaults to the "
            "selected cell's brightest PNC layer."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    from stardist.models import StarDist2D

    args = _parse_args()
    model = StarDist2D.from_pretrained("2D_versatile_fluo")
    result = analyze_pnc(
        args.file_path,
        model,
        scale=args.scale,
        segmentation_mode=args.segmentation_mode,
        cell_channel=args.cell_channel,
        segmentation_channel=args.segmentation_channel,
        pnc_channel=args.pnc_channel,
        min_cell_median_area_fraction=args.min_cell_median_area_fraction,
        max_cell_median_area_fraction=args.max_cell_median_area_fraction,
        min_cell_dapi_median_fraction=(
            args.min_cell_dapi_median_fraction
        ),
        bright_pixel_percentile=args.bright_pixel_percentile,
        pnc_threshold_multiplier=args.pnc_threshold_multiplier,
        pnc_threshold_offset=args.pnc_threshold_offset,
        enable_nucleolus_rescue=args.enable_nucleolus_rescue,
        nucleolus_inner_threshold_multiplier=(
            args.nucleolus_inner_threshold_multiplier
        ),
        nucleolus_outer_threshold_multiplier=(
            args.nucleolus_outer_threshold_multiplier
        ),
        nucleolus_outer_band_pixels=args.nucleolus_outer_band_pixels,
        enable_smear_exclusion=args.enable_smear_exclusion,
        histogram_cell_id=args.histogram_cell_id,
        histogram_z_index=args.histogram_z_index,
    )
    print(
        f"Cells with at least one PNC: {result['cells_with_pnc']}/"
        f"{result['total_cells']} ({result['percent_cells_with_pnc']:.1f}%)"
    )
    print(
        "StarDist segmentation channel: "
        f"{result['segmentation_channel']}"
    )
    print(f"Excluded non-intact cells: {result['excluded_non_intact_cells']}")
    print(
        "Excluded low-DAPI segmentation artifacts: "
        f"{result['low_dapi_cell_ids']}"
    )
    print(
        "Detected TxRed-smear cell IDs: "
        f"{result['smear_cell_ids']}"
    )
    print(
        "Excluded because of TxRed smear: "
        f"{result['smear_excluded_cell_ids']}"
    )
    print(f"Excluded small cells: {result['excluded_small_cells']}")
    print(f"Excluded large cells: {result['excluded_large_cells']}")
    print(
        "Nucleolus-rescued cell IDs: "
        f"{result['nucleolus_rescued_cell_ids']}"
    )
    plt.show()
