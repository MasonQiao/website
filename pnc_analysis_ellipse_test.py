import argparse

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import nd2
import numpy as np
from csbdeep.utils import normalize
from scipy import ndimage as ndi
from skimage.measure import EllipseModel


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

    # Ignore the image-edge perimeter because that edge can be the artificial
    # cut line, not the real cell boundary.
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

    return {
        **ellipse,
        "rmse": rmse,
        "fit_points": int(points.shape[0]),
    }


def _ellipse_missing_fraction(mask, ellipse):
    visible_area = np.count_nonzero(mask)
    fitted_ellipse_area = np.pi * ellipse["semi_major"] * ellipse["semi_minor"]
    if fitted_ellipse_area <= 0:
        return 0.0

    visible_fraction = visible_area / fitted_ellipse_area
    return max(0.0, 1.0 - visible_fraction)


def get_non_intact_ellipse_ids(
    labels,
    border_margin=2,
    max_missing_ellipse_fraction=0.15,
    max_fit_rmse=0.35,
    exclude_uncertain_border_cells=True,
):
    border_band = _border_band(labels.shape, border_margin)
    non_intact_ids = set()
    checked_count = 0
    fit_failed_count = 0

    for cell_id in _cell_ids_from_labels(labels):
        mask = labels == cell_id
        if not np.any(mask & border_band):
            continue

        checked_count += 1
        ellipse = _fit_ellipse(
            mask,
            border_band,
            max_fit_rmse=max_fit_rmse,
        )
        if ellipse is None:
            fit_failed_count += 1
            if exclude_uncertain_border_cells:
                non_intact_ids.add(cell_id)
            continue

        missing_fraction = _ellipse_missing_fraction(mask, ellipse)
        excluded = missing_fraction > max_missing_ellipse_fraction
        if excluded:
            non_intact_ids.add(cell_id)

    return non_intact_ids, checked_count, fit_failed_count


def _parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Test PNC prevalence counting with ellipse-based exclusion of "
            "non-intact border nuclei."
        )
    )
    parser.add_argument("file_path", help="Path to the ND2 file to analyze.")
    parser.add_argument("--threshold", type=float, default=7500)
    parser.add_argument("--scale", type=float, default=0.1)
    parser.add_argument("--cell-channel", type=int, default=0)
    parser.add_argument("--pnc-channel", type=int, default=1)
    parser.add_argument("--border-iterations", type=int, default=2)
    parser.add_argument("--ellipse-border-margin", type=int, default=2)
    parser.add_argument(
        "--max-missing-ellipse-fraction",
        "--max-outside-ellipse-fraction",
        dest="max_missing_ellipse_fraction",
        type=float,
        default=0.15,
        help=(
            "Flag a border cell if this fraction of its fitted ellipse area "
            "appears missing from the visible segmented mask."
        ),
    )
    parser.add_argument("--max-ellipse-fit-rmse", type=float, default=0.35)
    parser.add_argument(
        "--keep-non-intact",
        action="store_true",
        help="Run the ellipse checks but do not remove flagged nuclei from counts.",
    )
    parser.add_argument(
        "--exclude-uncertain-border-cells",
        action="store_true",
        help="Exclude border cells when ellipse fitting fails.",
    )
    return parser.parse_args()


def main():
    args = _parse_args()

    from stardist.models import StarDist2D

    print("Loading StarDist model...")
    model = StarDist2D.from_pretrained("2D_versatile_fluo")

    print(f"Reading {args.file_path}...")
    data = nd2.imread(args.file_path)
    cell_img = data[0, args.cell_channel, :, :]
    pnc_img = data[0, args.pnc_channel, :, :]

    print("Segmenting nuclei/cells...")
    labels, _ = model.predict_instances(normalize(cell_img), scale=args.scale)
    pnc_mask = pnc_img > args.threshold
    pnc_labels, num_pncs = ndi.label(pnc_mask)

    print("Mapping PNCs to segmented nuclei/cells...")
    cells_with_pnc_ids = set()
    for pnc_id in range(1, num_pncs + 1):
        cell_ids = labels[pnc_labels == pnc_id]
        cell_ids = cell_ids[cell_ids > 0]
        if cell_ids.size:
            cells_with_pnc_ids.add(int(np.bincount(cell_ids).argmax()))

    all_cell_ids = _cell_ids_from_labels(labels)
    raw_total_cells = len(all_cell_ids)
    raw_cells_with_pnc = len(cells_with_pnc_ids & all_cell_ids)

    print("Checking border nuclei with ellipse fitting...")
    (
        non_intact_cell_ids,
        ellipse_checked_count,
        fit_failed_count,
    ) = get_non_intact_ellipse_ids(
        labels,
        border_margin=args.ellipse_border_margin,
        max_missing_ellipse_fraction=args.max_missing_ellipse_fraction,
        max_fit_rmse=args.max_ellipse_fit_rmse,
        exclude_uncertain_border_cells=args.exclude_uncertain_border_cells,
    )

    if args.keep_non_intact:
        excluded_cell_ids = set()
    else:
        excluded_cell_ids = non_intact_cell_ids

    valid_cell_ids = all_cell_ids - excluded_cell_ids
    valid_cells_with_pnc_ids = cells_with_pnc_ids & valid_cell_ids

    total_cells = len(valid_cell_ids)
    cells_with_pnc = len(valid_cells_with_pnc_ids)
    percent_cells_with_pnc = (
        cells_with_pnc / total_cells * 100 if total_cells else 0
    )

    print()
    print("PNC analysis summary")
    print("--------------------")
    print(
        f"Raw PNC-positive cells: {raw_cells_with_pnc}/{raw_total_cells} "
        f"({raw_cells_with_pnc / raw_total_cells * 100:.1f}%)"
        if raw_total_cells
        else "Raw PNC-positive cells: 0/0 (0.0%)"
    )
    print(f"Border cells checked by ellipse fit: {ellipse_checked_count}")
    print(f"Ellipse fits failed: {fit_failed_count}")
    print(f"Ellipse-flagged non-intact cells: {len(non_intact_cell_ids)}")
    print(f"Excluded from final count: {len(excluded_cell_ids)}")
    print(
        f"Final PNC-positive cells: {cells_with_pnc}/{total_cells} "
        f"({percent_cells_with_pnc:.1f}%)"
    )
    if non_intact_cell_ids:
        print(
            "Flagged cell IDs: "
            + ", ".join(str(cell_id) for cell_id in sorted(non_intact_cell_ids))
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
        iterations=args.border_iterations,
    )
    without_pnc_border = _label_border(
        labels,
        valid_without_pnc_ids,
        iterations=args.border_iterations,
    )
    excluded_border = _label_border(
        labels,
        excluded_cell_ids,
        iterations=args.border_iterations,
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
    axes[1, 1].imshow(pnc_mask, cmap="gray")
    axes[1, 1].set_title("Segmented PNCs")

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

    plt.show()


if __name__ == "__main__":
    main()
