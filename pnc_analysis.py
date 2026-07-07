import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import nd2
import numpy as np
from csbdeep.utils import normalize
from scipy import ndimage as ndi
from skimage.measure import EllipseModel


# Analysis parameters
DEFAULT_SCALE = 0.1
DEFAULT_CELL_CHANNEL = 0
DEFAULT_PNC_CHANNEL = 1
CELL_BORDER_ITERATIONS = 2

PNC_THRESHOLD_MULTIPLIER = 2.0
PNC_MIN_CELL_AREA_FRACTION = 1 / 500

BORDER_CELL_MARGIN = 2
MAX_MISSING_ELLIPSE_FRACTION = 0.15
MAX_ELLIPSE_FIT_RMSE = 0.35
EXCLUDE_UNCERTAIN_BORDER_CELLS = True


def make_single_plot(image, cmap=None):
    fig, ax = plt.subplots(figsize=(5, 5), constrained_layout=True)
    ax.imshow(image, cmap=cmap)
    ax.axis("off")
    return fig


def _cell_ids_from_labels(labels):
    return set(int(cell_id) for cell_id in np.unique(labels[labels > 0]))


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

    for cell_id in _cell_ids_from_labels(labels):
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


def _renumber_large_components(component_labels, min_area_pixels, first_label):
    component_sizes = np.bincount(component_labels.ravel())
    keep_ids = np.flatnonzero(component_sizes >= min_area_pixels)
    keep_ids = keep_ids[keep_ids != 0]

    remap = np.zeros(component_sizes.size, dtype=np.int32)
    remap[keep_ids] = first_label + np.arange(keep_ids.size)
    return remap[component_labels], first_label + keep_ids.size


def _segment_pncs_by_cell(pnc_img, labels):
    pnc_labels = np.zeros(pnc_img.shape, dtype=np.int32)
    next_pnc_id = 1

    for cell_id, cell_slice in enumerate(ndi.find_objects(labels), start=1):
        if cell_slice is None:
            continue

        cell_labels = labels[cell_slice]
        cell_mask = cell_labels == cell_id
        cell_pnc_img = pnc_img[cell_slice]
        cell_median = np.median(cell_pnc_img[cell_mask])
        min_area_pixels = max(
            1,
            int(np.ceil(np.count_nonzero(cell_mask) * PNC_MIN_CELL_AREA_FRACTION)),
        )
        threshold = cell_median * PNC_THRESHOLD_MULTIPLIER
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


def analyze_pnc(
    file_path,
    model,
    scale=DEFAULT_SCALE,
    cell_channel=DEFAULT_CELL_CHANNEL,
    pnc_channel=DEFAULT_PNC_CHANNEL,
    border_iterations=CELL_BORDER_ITERATIONS,
):
    data = nd2.imread(file_path)
    cell_img = data[0, cell_channel, :, :]
    pnc_img = data[0, pnc_channel, :, :]

    labels, _ = model.predict_instances(normalize(cell_img), scale=scale)
    pnc_labels = _segment_pncs_by_cell(pnc_img, labels)
    pnc_mask = pnc_labels > 0
    cells_with_pnc_ids = _cell_ids_from_labels(labels[pnc_mask])

    all_cell_ids = _cell_ids_from_labels(labels)
    non_intact_cell_ids = get_non_intact_ellipse_ids(labels)
    valid_cell_ids = all_cell_ids - non_intact_cell_ids
    valid_cells_with_pnc_ids = cells_with_pnc_ids & valid_cell_ids

    total_cells = len(valid_cell_ids)
    cells_with_pnc = len(valid_cells_with_pnc_ids)
    percent_cells_with_pnc = (
        cells_with_pnc / total_cells * 100 if total_cells else 0
    )

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
    with_pnc_border = _label_border(
        labels,
        valid_cells_with_pnc_ids,
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

    combined_annotated = combined_raw.copy()
    combined_annotated[without_pnc_border] = [0, 1, 1]
    combined_annotated[with_pnc_border] = [0, 1, 0]
    combined_annotated[excluded_border] = [1, 0, 0]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    axes[0, 0].imshow(cell_img, cmap="gray")
    axes[0, 0].set_title("Raw cells")
    axes[1, 0].imshow(labels, cmap="nipy_spectral")
    axes[1, 0].set_title("Segmented cells")

    axes[0, 1].imshow(pnc_img, cmap="gray")
    axes[0, 1].set_title("Raw PNCs")
    axes[1, 1].imshow(pnc_labels, cmap="nipy_spectral")
    axes[1, 1].set_title(
        f"Segmented PNCs (>{PNC_THRESHOLD_MULTIPLIER:g}x, "
        f">= {PNC_MIN_CELL_AREA_FRACTION:g} cell area)"
    )

    axes[0, 2].imshow(combined_annotated)
    axes[0, 2].set_title(
        f"Annotated: {cells_with_pnc}/{total_cells}, "
        f"{percent_cells_with_pnc:.1f}%"
    )

    legend_handles = [
        Patch(facecolor=(0, 1, 0), edgecolor="none", label="PNC-positive cell"),
        Patch(facecolor=(0, 1, 1), edgecolor="none", label="PNC-negative cell"),
        Patch(facecolor=(1, 0, 0), edgecolor="none", label="Excluded non-intact cell"),
    ]
    axes[1, 2].legend(
        handles=legend_handles,
        loc="upper center",
        frameon=False,
        title="Annotated Image Legend",
    )

    for ax in axes.flat:
        ax.axis("off")

    subplot_figs = {
        "raw_cells": make_single_plot(cell_img, cmap="gray"),
        "raw_pncs": make_single_plot(pnc_img, cmap="gray"),
        "segmented_cells": make_single_plot(
            labels,
            cmap="nipy_spectral",
        ),
        "pnc_mask": make_single_plot(pnc_labels, cmap="nipy_spectral"),
        "combined_annotated": make_single_plot(combined_annotated),
    }

    return {
        "fig": fig,
        "subplot_figs": subplot_figs,
        "cells_with_pnc": cells_with_pnc,
        "total_cells": total_cells,
        "percent_cells_with_pnc": percent_cells_with_pnc,
        "excluded_non_intact_cells": len(non_intact_cell_ids),
    }
