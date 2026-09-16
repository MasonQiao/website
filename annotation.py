import csv
import io
import re
from datetime import datetime, timezone

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi
from scipy.optimize import linear_sum_assignment
from skimage.segmentation import find_boundaries


ANNOTATION_SCHEMA_VERSION = 2

FIJI_DISPLAY_STYLE = "fiji_bioformats"
HIGH_CONTRAST_DISPLAY_STYLE = "high_contrast"
REVIEW_DISPLAY_STYLES = (
    FIJI_DISPLAY_STYLE,
    HIGH_CONTRAST_DISPLAY_STYLE,
)

PNC_LABELS = ("unreviewed", "positive", "negative", "ambiguous")
SMEAR_LABELS = ("unreviewed", "positive", "negative", "ambiguous")
SEGMENTATION_LABELS = (
    "unreviewed",
    "valid",
    "merged_cells",
    "split_cell",
    "artifact",
    "border_or_non_intact",
    "ambiguous",
)

ANNOTATION_COLUMNS = (
    "annotation_schema_version",
    "image_name",
    "image_sha256",
    "cell_id",
    "centroid_x_px",
    "centroid_y_px",
    "bbox_x_min_px",
    "bbox_y_min_px",
    "bbox_x_max_exclusive_px",
    "bbox_y_max_exclusive_px",
    "cell_area_px",
    "z_layers_total",
    "segmentation_version",
    "segmentation_scale",
    "segmentation_channel",
    "dapi_channel",
    "pnc_channel",
    "label_stack_fingerprint",
    "observed_z_layers",
    "interpolated_z_layers",
    "median_cross_sectional_area_px",
    "migration_provenance",
    "algorithm_scale",
    "algorithm_bright_pixel_percentile",
    "algorithm_pnc_threshold_mode",
    "algorithm_pnc_threshold_multiplier",
    "algorithm_pnc_threshold_offset",
    "algorithm_nucleolus_rescue_enabled",
    "algorithm_nucleolus_inner_threshold_multiplier",
    "algorithm_nucleolus_outer_threshold_multiplier",
    "algorithm_nucleolus_outer_band_pixels",
    "algorithm_smear_exclusion_enabled",
    "algorithm_cell_status",
    "algorithm_pnc_call",
    "algorithm_pnc_source",
    "algorithm_smear_call",
    "ground_truth_pnc",
    "ground_truth_pnc_z_layers",
    "ground_truth_smear",
    "ground_truth_smear_z_layers",
    "ground_truth_segmentation",
    "notes",
    "reviewed_at_utc",
)


def empty_annotation():
    return {
        "pnc": "unreviewed",
        "pnc_z_layers": [],
        "smear": "unreviewed",
        "smear_z_layers": [],
        "segmentation": "unreviewed",
        "notes": "",
        "reviewed_at_utc": "",
        "migration_provenance": "",
    }


def annotation_is_complete(annotation):
    return all(
        annotation.get(field, "unreviewed") != "unreviewed"
        for field in ("pnc", "smear", "segmentation")
    )


def reviewed_at_now():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def cell_geometry(labels):
    """Return stable pixel-space geometry for every positive cell label."""
    cell_ids = [int(cell_id) for cell_id in np.unique(labels) if cell_id > 0]
    if not cell_ids:
        return {}

    areas = np.bincount(labels.ravel())
    slices = ndi.find_objects(labels)
    centers = ndi.center_of_mass(
        np.ones(labels.shape, dtype=np.uint8),
        labels,
        cell_ids,
    )
    geometry = {}
    for cell_id, center in zip(cell_ids, centers):
        cell_slice = slices[cell_id - 1]
        if cell_slice is None:
            continue
        y_slice, x_slice = cell_slice
        geometry[cell_id] = {
            "centroid_x_px": float(center[1]),
            "centroid_y_px": float(center[0]),
            "bbox_x_min_px": int(x_slice.start),
            "bbox_y_min_px": int(y_slice.start),
            "bbox_x_max_exclusive_px": int(x_slice.stop),
            "bbox_y_max_exclusive_px": int(y_slice.stop),
            "cell_area_px": int(areas[cell_id]),
        }
    return geometry


def _display_limits(image, lower_percentile=1.0, upper_percentile=99.8):
    values = np.asarray(image)
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return 0.0, 1.0

    lower, upper = np.percentile(
        finite_values,
        [lower_percentile, upper_percentile],
    )
    if upper <= lower:
        lower = float(np.min(finite_values))
        upper = float(np.max(finite_values))
    if upper <= lower:
        upper = lower + 1.0
    return float(lower), float(upper)


def normalize_uint8(image, limits=None):
    if limits is None:
        limits = _display_limits(image)
    lower, upper = limits
    scaled = (np.asarray(image, dtype=np.float32) - lower) / (upper - lower)
    return np.rint(np.clip(scaled, 0, 1) * 255).astype(np.uint8)


def _raw_display_limits(image):
    """Return the finite raw minimum and maximum used by Fiji autoscale."""
    values = np.asarray(image)
    if values.size == 0:
        return 0.0, 1.0

    if np.issubdtype(values.dtype, np.integer) or np.issubdtype(
        values.dtype,
        np.bool_,
    ):
        # Microscopy stacks are normally uint16. Avoid making a full-size
        # finite-value copy every time a review crop is rendered.
        lower = float(np.min(values))
        upper = float(np.max(values))
    else:
        finite_values = values[np.isfinite(values)]
        if finite_values.size == 0:
            return 0.0, 1.0
        lower = float(np.min(finite_values))
        upper = float(np.max(finite_values))
    if upper <= lower:
        upper = lower + 1.0
    return lower, upper


def _colorize(values, color):
    """Apply a Fiji-style RGB lookup-table color to uint8 intensities."""
    values = np.asarray(values, dtype=np.uint8)
    rgb = np.empty((*values.shape, 3), dtype=np.uint8)
    for channel_index, channel_maximum in enumerate(color):
        if channel_maximum == 0:
            rgb[..., channel_index] = 0
        elif channel_maximum == 255:
            rgb[..., channel_index] = values
        else:
            rgb[..., channel_index] = np.rint(
                values.astype(np.float32) * channel_maximum / 255
            ).astype(np.uint8)
    return rgb


def _validate_display_style(display_style):
    if display_style not in REVIEW_DISPLAY_STYLES:
        raise ValueError(
            f"display_style must be one of {REVIEW_DISPLAY_STYLES}, "
            f"not {display_style!r}."
        )


def _load_label_font(size=15):
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf", size=size)
    except OSError:
        return ImageFont.load_default()


def make_selector_assets(
    cell_image,
    labels,
    max_width=900,
    display_style=FIJI_DISPLAY_STYLE,
):
    """Create a neutral, clickable overview and its exactly aligned labels."""
    _validate_display_style(display_style)
    height, width = labels.shape
    scale = min(1.0, max_width / width)
    display_size = (
        max(1, int(round(width * scale))),
        max(1, int(round(height * scale))),
    )

    if display_style == FIJI_DISPLAY_STYLE:
        display_limits = _raw_display_limits(cell_image)
    else:
        display_limits = _display_limits(cell_image)
    grayscale = normalize_uint8(cell_image, limits=display_limits)
    cell_pil = Image.fromarray(grayscale).resize(
        display_size,
        resample=Image.Resampling.BILINEAR,
    )
    label_pil = Image.fromarray(labels.astype(np.int32)).resize(
        display_size,
        resample=Image.Resampling.NEAREST,
    )
    display_labels = np.asarray(label_pil, dtype=np.int32)

    if display_style == FIJI_DISPLAY_STYLE:
        base = _colorize(np.asarray(cell_pil), (0, 0, 255))
    else:
        base = np.repeat(np.asarray(cell_pil)[..., None], 3, axis=2)
    boundaries = find_boundaries(display_labels, mode="inner")
    base[boundaries] = [240, 240, 240]

    overview = Image.fromarray(base)
    draw = ImageDraw.Draw(overview)
    display_ids = [
        int(cell_id) for cell_id in np.unique(display_labels) if cell_id > 0
    ]
    centers = ndi.center_of_mass(
        np.ones(display_labels.shape, dtype=np.uint8),
        display_labels,
        display_ids,
    )
    font = _load_label_font(size=15)
    for cell_id, (center_y, center_x) in zip(display_ids, centers):
        draw.text(
            (float(center_x), float(center_y)),
            str(cell_id),
            anchor="mm",
            fill=(255, 255, 255),
            stroke_width=2,
            stroke_fill=(0, 0, 0),
            font=font,
        )

    return np.asarray(overview), display_labels


def highlight_selector_image(base_image, display_labels, selected_cell_id):
    highlighted = np.asarray(base_image).copy()
    selected_mask = display_labels == selected_cell_id
    if not np.any(selected_mask):
        return highlighted

    selected_boundary = find_boundaries(selected_mask, mode="inner")
    selected_boundary = ndi.binary_dilation(selected_boundary, iterations=1)
    highlighted[selected_boundary] = [255, 215, 0]
    return highlighted


def cell_id_at_click(display_labels, x, y, search_radius=8):
    """Map a selector-image click to a label, tolerating a small miss."""
    x = int(round(x))
    y = int(round(y))
    height, width = display_labels.shape
    if not (0 <= x < width and 0 <= y < height):
        return None

    direct_id = int(display_labels[y, x])
    if direct_id > 0:
        return direct_id

    y_min = max(0, y - search_radius)
    y_max = min(height, y + search_radius + 1)
    x_min = max(0, x - search_radius)
    x_max = min(width, x + search_radius + 1)
    nearby = display_labels[y_min:y_max, x_min:x_max]
    candidate_y, candidate_x = np.nonzero(nearby)
    if candidate_y.size == 0:
        return None

    squared_distances = (
        candidate_y + y_min - y
    ) ** 2 + (
        candidate_x + x_min - x
    ) ** 2
    nearest_index = int(np.argmin(squared_distances))
    return int(nearby[candidate_y[nearest_index], candidate_x[nearest_index]])


def _padded_cell_bounds(geometry, image_shape, padding):
    height, width = image_shape
    return (
        max(0, geometry["bbox_y_min_px"] - padding),
        min(height, geometry["bbox_y_max_exclusive_px"] + padding),
        max(0, geometry["bbox_x_min_px"] - padding),
        min(width, geometry["bbox_x_max_exclusive_px"] + padding),
    )


def make_cell_review_stack(
    cell_image,
    pnc_stack,
    labels,
    selected_cell_id,
    geometry,
    padding=55,
    display_style=FIJI_DISPLAY_STYLE,
):
    """Return synchronized all-z DAPI/TxRed crops with current-z boundaries."""
    _validate_display_style(display_style)
    pnc_stack = np.asarray(pnc_stack)
    cell_values = np.asarray(cell_image)
    label_values = np.asarray(labels)
    if cell_values.ndim == 2:
        cell_stack = np.repeat(
            cell_values[np.newaxis, ...],
            len(pnc_stack),
            axis=0,
        )
    elif cell_values.ndim == 3 and len(cell_values) == len(pnc_stack):
        cell_stack = cell_values
    else:
        raise ValueError("DAPI data must be 2D or match the TxRed z-stack.")
    if label_values.ndim == 2:
        label_stack = np.repeat(
            label_values[np.newaxis, ...],
            len(pnc_stack),
            axis=0,
        )
    elif label_values.ndim == 3 and len(label_values) == len(pnc_stack):
        label_stack = label_values
    else:
        raise ValueError("Cell labels must be 2D or match the TxRed z-stack.")

    track_coordinates = np.argwhere(label_stack == selected_cell_id)
    review_geometry = dict(geometry)
    if track_coordinates.size:
        review_geometry.update(
            bbox_y_min_px=int(np.min(track_coordinates[:, 1])),
            bbox_y_max_exclusive_px=int(np.max(track_coordinates[:, 1])) + 1,
            bbox_x_min_px=int(np.min(track_coordinates[:, 2])),
            bbox_x_max_exclusive_px=int(np.max(track_coordinates[:, 2])) + 1,
        )
    y_min, y_max, x_min, x_max = _padded_cell_bounds(
        review_geometry,
        label_stack.shape[1:],
        padding,
    )
    label_crop_stack = label_stack[:, y_min:y_max, x_min:x_max]
    cell_crop_stack = cell_stack[:, y_min:y_max, x_min:x_max]
    pnc_crop_stack = pnc_stack[:, y_min:y_max, x_min:x_max]

    if display_style == FIJI_DISPLAY_STYLE:
        # Bio-Formats Autoscale maps the minimum and maximum of the image or
        # stack to the available display range. Calculate these limits from
        # the complete inputs so zooming to a crop does not change brightness.
        dapi_limits = _raw_display_limits(cell_values)
        pnc_limits = _raw_display_limits(pnc_stack)
    else:
        dapi_limits = _display_limits(
            cell_crop_stack,
            lower_percentile=1.0,
            upper_percentile=99.9,
        )
        pnc_limits = _display_limits(
            pnc_crop_stack,
            lower_percentile=1.0,
            upper_percentile=99.9,
        )

    dapi_values = normalize_uint8(cell_crop_stack, limits=dapi_limits)
    pnc_values = normalize_uint8(pnc_crop_stack, limits=pnc_limits)

    if display_style == FIJI_DISPLAY_STYLE:
        # These are the channel colors stored in the supplied ND2 metadata.
        dapi_rgb = _colorize(dapi_values, (0, 0, 255))
        txred_stack_rgb = _colorize(pnc_values, (255, 0, 0))
    else:
        dapi_rgb = np.repeat(dapi_values[..., None], 3, axis=3)
        txred_stack_rgb = _colorize(pnc_values, (255, 0, 255))

    for z_index in range(len(label_crop_stack)):
        selected_boundary = find_boundaries(
            label_crop_stack[z_index] == selected_cell_id,
            mode="inner",
        )
        dapi_rgb[z_index][selected_boundary] = [255, 215, 0]
        txred_stack_rgb[z_index][selected_boundary] = [255, 215, 0]
    return dapi_rgb, txred_stack_rgb, (x_min, y_min, x_max, y_max)


def make_cell_review_images(
    cell_image,
    pnc_stack,
    labels,
    selected_cell_id,
    z_index,
    geometry,
    padding=55,
    display_style=FIJI_DISPLAY_STYLE,
):
    """Return aligned DAPI and one-layer TxRed crops for non-web consumers."""
    dapi_stack_rgb, txred_stack_rgb, bounds = make_cell_review_stack(
        cell_image,
        pnc_stack,
        labels,
        selected_cell_id,
        geometry,
        padding=padding,
        display_style=display_style,
    )
    return dapi_stack_rgb[z_index], txred_stack_rgb[z_index], bounds


def brightest_cell_z_index(pnc_stack, labels, cell_id):
    labels = np.asarray(labels)
    if labels.ndim == 2:
        labels = np.repeat(labels[np.newaxis, ...], len(pnc_stack), axis=0)
    layer_maxima = np.full(len(pnc_stack), -np.inf, dtype=float)
    for z_index in range(len(pnc_stack)):
        cell_mask = labels[z_index] == cell_id
        if np.any(cell_mask):
            layer_maxima[z_index] = np.max(pnc_stack[z_index][cell_mask])
    if not np.any(np.isfinite(layer_maxima)):
        return 0
    return int(np.argmax(layer_maxima))


def _cell_algorithm_status(result, cell_id):
    status = []
    status_fields = (
        ("segmentation_excluded_cell_ids", "excluded_unresolved_merge"),
        ("non_intact_cell_ids", "excluded_non_intact"),
        ("low_dapi_cell_ids", "excluded_low_dapi"),
        ("smear_excluded_cell_ids", "excluded_smear"),
        ("small_cell_ids", "excluded_small"),
        ("large_cell_ids", "excluded_large"),
    )
    for result_key, label in status_fields:
        if cell_id in set(result.get(result_key, [])):
            status.append(label)
    return ";".join(status) if status else "valid"


def _cell_pnc_source(result, cell_id):
    if cell_id in set(result.get("baseline_cells_with_pnc_ids", [])):
        return "baseline"
    if cell_id in set(result.get("nucleolus_rescued_cell_ids", [])):
        return "nucleolus_rescue"
    return "none"


def build_annotation_rows(
    result,
    image_name,
    image_sha256,
    annotations,
    geometry=None,
):
    labels = result["cell_labels"]
    if geometry is None:
        geometry = cell_geometry(labels)

    positive_ids = set(result.get("cells_with_pnc_ids", []))
    smear_ids = set(result.get("smear_cell_ids", []))
    rows = []
    for cell_id in sorted(geometry):
        annotation = empty_annotation()
        annotation.update(annotations.get(cell_id, {}))
        cell_values = geometry[cell_id]
        track_metadata = result.get("track_metadata", {}).get(cell_id, {})
        rows.append(
            {
                "annotation_schema_version": ANNOTATION_SCHEMA_VERSION,
                "image_name": image_name,
                "image_sha256": image_sha256,
                "cell_id": cell_id,
                "centroid_x_px": f"{cell_values['centroid_x_px']:.3f}",
                "centroid_y_px": f"{cell_values['centroid_y_px']:.3f}",
                "bbox_x_min_px": cell_values["bbox_x_min_px"],
                "bbox_y_min_px": cell_values["bbox_y_min_px"],
                "bbox_x_max_exclusive_px": cell_values[
                    "bbox_x_max_exclusive_px"
                ],
                "bbox_y_max_exclusive_px": cell_values[
                    "bbox_y_max_exclusive_px"
                ],
                "cell_area_px": cell_values["cell_area_px"],
                "z_layers_total": int(result["pnc_stack"].shape[0]),
                "segmentation_version": result.get(
                    "segmentation_version",
                    "",
                ),
                "segmentation_scale": result.get(
                    "segmentation_scale",
                    result.get("analysis_scale", ""),
                ),
                "segmentation_channel": result.get(
                    "segmentation_channel",
                    "",
                ),
                "dapi_channel": result.get(
                    "dapi_channel",
                    result.get("cell_channel", ""),
                ),
                "pnc_channel": result.get("pnc_channel", ""),
                "label_stack_fingerprint": result.get(
                    "label_stack_fingerprint",
                    "",
                ),
                "observed_z_layers": ";".join(
                    str(layer)
                    for layer in track_metadata.get("observed_z_layers", [])
                ),
                "interpolated_z_layers": ";".join(
                    str(layer)
                    for layer in track_metadata.get(
                        "interpolated_z_layers",
                        [],
                    )
                ),
                "median_cross_sectional_area_px": track_metadata.get(
                    "median_area_pixels",
                    cell_values["cell_area_px"],
                ),
                "migration_provenance": annotation.get(
                    "migration_provenance",
                    "",
                ),
                "algorithm_scale": result.get("analysis_scale", ""),
                "algorithm_bright_pixel_percentile": result.get(
                    "bright_pixel_percentile",
                    "",
                ),
                "algorithm_pnc_threshold_mode": result.get(
                    "pnc_threshold_mode",
                    "",
                ),
                "algorithm_pnc_threshold_multiplier": result.get(
                    "pnc_threshold_multiplier",
                    "",
                ),
                "algorithm_pnc_threshold_offset": (
                    ""
                    if result.get("pnc_threshold_offset") is None
                    else result["pnc_threshold_offset"]
                ),
                "algorithm_nucleolus_rescue_enabled": result.get(
                    "nucleolus_rescue_enabled",
                    "",
                ),
                "algorithm_nucleolus_inner_threshold_multiplier": result.get(
                    "nucleolus_inner_threshold_multiplier",
                    "",
                ),
                "algorithm_nucleolus_outer_threshold_multiplier": result.get(
                    "nucleolus_outer_threshold_multiplier",
                    "",
                ),
                "algorithm_nucleolus_outer_band_pixels": result.get(
                    "nucleolus_outer_band_pixels",
                    "",
                ),
                "algorithm_smear_exclusion_enabled": result.get(
                    "smear_exclusion_enabled",
                    "",
                ),
                "algorithm_cell_status": _cell_algorithm_status(result, cell_id),
                "algorithm_pnc_call": (
                    "positive" if cell_id in positive_ids else "negative"
                ),
                "algorithm_pnc_source": _cell_pnc_source(result, cell_id),
                "algorithm_smear_call": (
                    "positive" if cell_id in smear_ids else "negative"
                ),
                "ground_truth_pnc": annotation["pnc"],
                "ground_truth_pnc_z_layers": ";".join(
                    str(layer) for layer in annotation["pnc_z_layers"]
                ),
                "ground_truth_smear": annotation["smear"],
                "ground_truth_smear_z_layers": ";".join(
                    str(layer) for layer in annotation["smear_z_layers"]
                ),
                "ground_truth_segmentation": annotation["segmentation"],
                "notes": annotation["notes"],
                "reviewed_at_utc": annotation["reviewed_at_utc"],
            }
        )
    return rows


def annotations_to_csv(rows):
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=ANNOTATION_COLUMNS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode("utf-8-sig")


def _parse_layers(raw_layers, z_layers_total):
    if raw_layers is None or not str(raw_layers).strip():
        return []
    layers = []
    for value in re.split(r"[;,\s]+", str(raw_layers).strip()):
        if not value:
            continue
        try:
            layer = int(value)
        except ValueError as exc:
            raise ValueError(f"Invalid z-layer value: {value!r}.") from exc
        if not 1 <= layer <= z_layers_total:
            raise ValueError(
                f"Z-layer {layer} is outside the available range "
                f"1-{z_layers_total}."
            )
        layers.append(layer)
    return sorted(set(layers))


def _validated_label(value, allowed_values, field_name):
    value = (value or "unreviewed").strip().lower()
    if value not in allowed_values:
        choices = ", ".join(allowed_values)
        raise ValueError(
            f"Invalid {field_name} value {value!r}; expected one of: {choices}."
        )
    return value


def annotations_from_csv(
    csv_bytes,
    expected_image_sha256,
    valid_cell_ids,
    z_layers_total,
    expected_label_stack_fingerprint=None,
    legacy_labels=None,
    current_labels=None,
):
    """Restore schema-2 rows or conservatively migrate schema-1 rows."""
    try:
        text = csv_bytes.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError("The annotation CSV must use UTF-8 encoding.") from exc

    reader = csv.DictReader(io.StringIO(text))
    required_columns = {"cell_id", "image_sha256"}
    if not reader.fieldnames or not required_columns.issubset(reader.fieldnames):
        raise ValueError(
            "The annotation CSV is missing its cell_id or image_sha256 column."
        )

    rows = list(reader)
    matching_rows = [
        row
        for row in rows
        if (row.get("image_sha256") or "").strip()
        == expected_image_sha256
    ]
    if not matching_rows:
        raise ValueError(
            "This CSV does not contain annotations for the uploaded ND2 file."
        )

    schema_versions = set()
    for row in matching_rows:
        raw_version = (row.get("annotation_schema_version") or "1").strip()
        try:
            schema_versions.add(int(raw_version))
        except ValueError as exc:
            raise ValueError(
                f"Invalid annotation schema version: {raw_version!r}."
            ) from exc
    if len(schema_versions) != 1:
        raise ValueError("The CSV mixes incompatible annotation schema versions.")
    schema_version = next(iter(schema_versions))
    if schema_version < ANNOTATION_SCHEMA_VERSION:
        if legacy_labels is None or current_labels is None:
            raise ValueError(
                "This is a version-1 CSV. Reconstruct the recorded scale-0.1 "
                "projection labels before importing it."
            )
        return _migrate_legacy_annotations(
            matching_rows,
            legacy_labels,
            current_labels,
            valid_cell_ids,
            z_layers_total,
        )
    if schema_version != ANNOTATION_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported annotation schema version {schema_version}."
        )

    if expected_label_stack_fingerprint is not None:
        fingerprints = {
            (row.get("label_stack_fingerprint") or "").strip()
            for row in matching_rows
        }
        if fingerprints != {expected_label_stack_fingerprint}:
            raise ValueError(
                "The CSV image matches, but its z-track segmentation "
                "fingerprint does not match this analysis. Cell IDs were not "
                "restored."
            )

    valid_cell_ids = set(valid_cell_ids)
    annotations = {}
    skipped_ids = []
    for row in matching_rows:
        try:
            cell_id = int(row["cell_id"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid cell_id value: {row.get('cell_id')!r}.") from exc
        if cell_id not in valid_cell_ids:
            skipped_ids.append(cell_id)
            continue

        annotations[cell_id] = {
            "pnc": _validated_label(
                row.get("ground_truth_pnc"),
                PNC_LABELS,
                "ground_truth_pnc",
            ),
            "pnc_z_layers": _parse_layers(
                row.get("ground_truth_pnc_z_layers"),
                z_layers_total,
            ),
            "smear": _validated_label(
                row.get("ground_truth_smear"),
                SMEAR_LABELS,
                "ground_truth_smear",
            ),
            "smear_z_layers": _parse_layers(
                row.get("ground_truth_smear_z_layers"),
                z_layers_total,
            ),
            "segmentation": _validated_label(
                row.get("ground_truth_segmentation"),
                SEGMENTATION_LABELS,
                "ground_truth_segmentation",
            ),
            "notes": row.get("notes") or "",
            "reviewed_at_utc": row.get("reviewed_at_utc") or "",
            "migration_provenance": row.get("migration_provenance") or "",
        }
    return annotations, sorted(set(skipped_ids))


def _legacy_geometry_matches(row, legacy_geometry, cell_id):
    geometry = legacy_geometry.get(cell_id)
    if geometry is None:
        return False
    try:
        exact_fields = (
            "bbox_x_min_px",
            "bbox_y_min_px",
            "bbox_x_max_exclusive_px",
            "bbox_y_max_exclusive_px",
            "cell_area_px",
        )
        if any(
            int(row.get(field, "")) != int(geometry[field])
            for field in exact_fields
        ):
            return False
        return (
            abs(float(row["centroid_x_px"]) - geometry["centroid_x_px"])
            <= 0.002
            and abs(float(row["centroid_y_px"]) - geometry["centroid_y_px"])
            <= 0.002
        )
    except (KeyError, TypeError, ValueError):
        return False


def _label_iou_matrix(old_labels, old_ids, new_labels, new_ids):
    matrix = np.zeros((len(old_ids), len(new_ids)), dtype=float)
    old_areas = np.bincount(old_labels.ravel())
    new_areas = np.bincount(new_labels.ravel())
    old_index = {cell_id: index for index, cell_id in enumerate(old_ids)}
    new_index = {cell_id: index for index, cell_id in enumerate(new_ids)}
    positive_overlap = (old_labels > 0) & (new_labels > 0)
    if not np.any(positive_overlap):
        return matrix
    pair_base = int(np.max(new_labels)) + 1
    pair_codes, intersections = np.unique(
        old_labels[positive_overlap].astype(np.int64) * pair_base
        + new_labels[positive_overlap],
        return_counts=True,
    )
    for pair_code, intersection in zip(pair_codes, intersections):
        old_id = int(pair_code // pair_base)
        new_id = int(pair_code % pair_base)
        if old_id not in old_index or new_id not in new_index:
            continue
        union = (
            int(old_areas[old_id])
            + int(new_areas[new_id])
            - int(intersection)
        )
        matrix[old_index[old_id], new_index[new_id]] = (
            intersection / union if union else 0.0
        )
    return matrix


def _second_highest(values, excluded_index):
    remaining = np.delete(np.asarray(values, dtype=float), excluded_index)
    return float(np.max(remaining)) if remaining.size else 0.0


def _migrate_legacy_annotations(
    rows,
    legacy_labels,
    current_labels,
    valid_cell_ids,
    z_layers_total,
):
    legacy_labels = np.asarray(legacy_labels)
    current_labels = np.asarray(current_labels)
    if legacy_labels.ndim != 2 or current_labels.ndim != 2:
        raise ValueError("Legacy migration requires two 2D label images.")

    report = []
    candidate_rows = []
    legacy_geometry = cell_geometry(legacy_labels)
    for row in rows:
        try:
            old_id = int(row.get("cell_id", ""))
        except (TypeError, ValueError):
            report.append("invalid legacy cell ID")
            continue
        segmentation_label = (row.get("ground_truth_segmentation") or "").strip().lower()
        if segmentation_label != "valid":
            report.append(
                f"old cell {old_id}: segmentation was {segmentation_label or 'unreviewed'}"
            )
            continue
        if not _legacy_geometry_matches(row, legacy_geometry, old_id):
            report.append(
                f"old cell {old_id}: recorded geometry did not match reconstructed labels"
            )
            continue
        candidate_rows.append((old_id, row))

    old_ids = [old_id for old_id, _ in candidate_rows]
    new_ids = sorted(
        int(cell_id)
        for cell_id in set(valid_cell_ids)
        if np.any(current_labels == int(cell_id))
    )
    if not old_ids or not new_ids:
        return {}, report

    iou = _label_iou_matrix(legacy_labels, old_ids, current_labels, new_ids)
    row_indices, column_indices = linear_sum_assignment(1.0 - iou)
    assignments = dict(zip(row_indices, column_indices))
    migrated = {}
    for row_index, (old_id, row) in enumerate(candidate_rows):
        if row_index not in assignments:
            report.append(f"old cell {old_id}: no one-to-one track assignment")
            continue
        column_index = assignments[row_index]
        new_id = new_ids[column_index]
        overlap = float(iou[row_index, column_index])
        competing_new = _second_highest(iou[row_index], column_index)
        competing_old = _second_highest(iou[:, column_index], row_index)
        if overlap < 0.65:
            report.append(
                f"old cell {old_id}: best track IoU {overlap:.3f} was below 0.65"
            )
            continue
        if competing_new >= 0.25 or competing_old >= 0.25:
            report.append(
                f"old cell {old_id}: competing overlap was at least 0.25"
            )
            continue
        migrated[new_id] = {
            "pnc": _validated_label(
                row.get("ground_truth_pnc"),
                PNC_LABELS,
                "ground_truth_pnc",
            ),
            "pnc_z_layers": _parse_layers(
                row.get("ground_truth_pnc_z_layers"),
                z_layers_total,
            ),
            "smear": _validated_label(
                row.get("ground_truth_smear"),
                SMEAR_LABELS,
                "ground_truth_smear",
            ),
            "smear_z_layers": _parse_layers(
                row.get("ground_truth_smear_z_layers"),
                z_layers_total,
            ),
            "segmentation": "unreviewed",
            "notes": row.get("notes") or "",
            "reviewed_at_utc": row.get("reviewed_at_utc") or "",
            "migration_provenance": (
                f"schema1 old_cell={old_id};new_track={new_id};iou={overlap:.3f};"
                "projection_scale=0.1"
            ),
        }
    return migrated, report
