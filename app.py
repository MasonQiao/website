import hashlib
import io
import os
import tempfile
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates

from annotation import (
    FIJI_DISPLAY_STYLE,
    HIGH_CONTRAST_DISPLAY_STYLE,
    PNC_LABELS,
    SEGMENTATION_LABELS,
    SMEAR_LABELS,
    annotation_is_complete,
    annotations_from_csv,
    annotations_to_csv,
    brightest_cell_z_index,
    build_annotation_rows,
    cell_geometry,
    cell_id_at_click,
    empty_annotation,
    highlight_selector_image,
    make_cell_review_stack,
    make_selector_assets,
    reviewed_at_now,
)
from cell_tracking import LEGACY_RECONCILIATION_SCALE, segment_projection_labels
from live_z_slider import live_dual_z_stack_viewer
from testing import (
    DEFAULT_CELL_CHANNEL,
    DEFAULT_SEGMENTATION_CHANNEL,
    PNC_THRESHOLD_MULTIPLIER,
    analyze_pnc,
)

st.set_page_config(page_title="PNC Cell Analysis", layout="wide")

LABEL_TITLES = {
    "unreviewed": "Unreviewed",
    "positive": "Positive",
    "negative": "Negative",
    "ambiguous": "Ambiguous",
    "valid": "Valid cell",
    "merged_cells": "Merged cells",
    "split_cell": "Split cell",
    "artifact": "Segmentation artifact",
    "border_or_non_intact": "Border / non-intact",
}


@st.cache_resource
def load_model():
    from stardist.models import StarDist2D

    return StarDist2D.from_pretrained("2D_versatile_fluo")


def figure_to_png(fig):
    png = io.BytesIO()
    fig.savefig(png, format="png", dpi=300, bbox_inches="tight")
    png.seek(0)
    return png


def uploaded_file_sha256(uploaded_file):
    digest = hashlib.sha256()
    digest.update(uploaded_file.getbuffer())
    return digest.hexdigest()


def run_uploaded_analysis(
    uploaded_file,
    model,
    pnc_threshold_multiplier,
    segmentation_channel,
):
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".nd2") as tmp:
            tmp.write(uploaded_file.getbuffer())
            tmp_path = tmp.name
        return analyze_pnc(
            tmp_path,
            model,
            pnc_threshold_multiplier=pnc_threshold_multiplier,
            segmentation_channel=segmentation_channel,
        )
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


def get_analysis_result(
    uploaded_file,
    image_sha256,
    model,
    pnc_threshold_multiplier,
    segmentation_channel,
):
    analysis_key = (
        image_sha256,
        float(pnc_threshold_multiplier),
        int(segmentation_channel),
    )
    if st.session_state.get("analysis_key") == analysis_key:
        return st.session_state["analysis_result"], analysis_key

    previous_result = st.session_state.get("analysis_result")
    if previous_result is not None:
        plt.close(previous_result["fig"])

    with st.spinner("Analyzing every z-layer..."):
        result = run_uploaded_analysis(
            uploaded_file,
            model,
            pnc_threshold_multiplier,
            segmentation_channel,
        )
    st.session_state["analysis_key"] = analysis_key
    st.session_state["analysis_result"] = result
    return result, analysis_key


def get_image_annotations(annotation_key):
    if "annotations_by_image" not in st.session_state:
        st.session_state["annotations_by_image"] = {}
    return st.session_state["annotations_by_image"].setdefault(annotation_key, {})


def get_selector_assets(analysis_key, result, display_style):
    selector_key = (analysis_key, display_style)
    if st.session_state.get("selector_analysis_key") != selector_key:
        overview, overview_labels = make_selector_assets(
            result["cell_image"],
            result["cell_labels"],
            display_style=display_style,
        )
        st.session_state["selector_analysis_key"] = selector_key
        st.session_state["selector_overview"] = overview
        st.session_state["selector_labels"] = overview_labels
        st.session_state["cell_geometry"] = cell_geometry(result["cell_labels"])
    return (
        st.session_state["selector_overview"],
        st.session_state["selector_labels"],
        st.session_state["cell_geometry"],
    )


def cell_picker_label(cell_id, annotations):
    marker = (
        "reviewed" if annotation_is_complete(annotations.get(cell_id, {})) else "open"
    )
    return f"Cell {cell_id} — {marker}"


def next_incomplete_cell(cell_ids, current_cell_id, annotations):
    current_index = cell_ids.index(current_cell_id)
    ordered_ids = cell_ids[current_index + 1 :] + cell_ids[: current_index + 1]
    for cell_id in ordered_ids:
        if not annotation_is_complete(annotations.get(cell_id, {})):
            return cell_id
    return current_cell_id


def algorithm_summary(row):
    pnc_source = row["algorithm_pnc_source"].replace("_", " ")
    status = row["algorithm_cell_status"].replace("_", " ")
    return (
        f"PNC: {row['algorithm_pnc_call']} ({pnc_source}); "
        f"smear: {row['algorithm_smear_call']}; cell status: {status}."
    )


st.title("PNC Cell Analysis and Ground-Truth Review")
st.caption(
    "Analyze the full TxRed z-stack, inspect raw cell crops, and export one "
    "ground-truth row for every segmented cell."
)

with st.sidebar:
    st.header("Analysis settings")
    segmentation_source = st.selectbox(
        "StarDist segmentation source",
        options=("TxRed", "DAPI"),
        index=0,
        help=(
            "TxRed is the current default trial. DAPI remains available for "
            "comparison and for reopening annotations made with the previous "
            "segmentation source."
        ),
    )
    segmentation_channel = (
        DEFAULT_SEGMENTATION_CHANNEL
        if segmentation_source == "TxRed"
        else DEFAULT_CELL_CHANNEL
    )
    pnc_threshold_multiplier = st.number_input(
        "PNC threshold multiplier",
        min_value=1.0,
        value=float(PNC_THRESHOLD_MULTIPLIER),
        step=0.05,
        format="%.2f",
        help="The current all-z default is cell P95 × 1.25.",
    )
    blind_review = st.toggle(
        "Hide algorithm calls while reviewing",
        value=True,
        help=(
            "Keeps the human label independent of the algorithm. The calls are "
            "still included in the exported CSV."
        ),
    )
    st.header("Review settings")
    fiji_style_display = st.toggle(
        "Fiji-style image display",
        value=True,
        help=(
            "Uses the ND2 channel colors (DAPI blue and TxRed red) and "
            "Bio-Formats-style stack autoscaling. This changes display only; "
            "detection continues to use the original raw intensities."
        ),
    )

review_display_style = (
    FIJI_DISPLAY_STYLE if fiji_style_display else HIGH_CONTRAST_DISPLAY_STYLE
)

with st.spinner("Loading StarDist model..."):
    model = load_model()

uploaded_file = st.file_uploader("Upload an ND2 file", type=["nd2"])
if uploaded_file is None:
    st.info(
        "Upload one ND2 file to start. Annotations remain in this browser "
        "session until you export them as CSV."
    )
    st.stop()

image_sha256 = uploaded_file_sha256(uploaded_file)
file_stem = Path(uploaded_file.name).stem

try:
    result, analysis_key = get_analysis_result(
        uploaded_file,
        image_sha256,
        model,
        pnc_threshold_multiplier,
        segmentation_channel,
    )
except Exception as exc:
    st.error(f"Analysis failed: {exc}")
    st.stop()

segmentation_fingerprint = result["label_stack_fingerprint"]
annotation_key = f"{image_sha256}:{segmentation_fingerprint}"
image_token = f"{image_sha256[:8]}-{segmentation_fingerprint[:8]}"
annotations = get_image_annotations(annotation_key)
overview, overview_labels, geometry = get_selector_assets(
    analysis_key,
    result,
    review_display_style,
)
cell_ids = sorted(geometry)
if not cell_ids:
    st.warning("No segmented cells were found in this image.")
    st.stop()

rows = build_annotation_rows(
    result,
    uploaded_file.name,
    image_sha256,
    annotations,
    geometry=geometry,
)
rows_by_cell_id = {row["cell_id"]: row for row in rows}
reviewed_count = sum(
    annotation_is_complete(annotations.get(cell_id, {})) for cell_id in cell_ids
)

metric_columns = st.columns(5)
metric_columns[0].metric("Segmented cells", len(cell_ids))
metric_columns[1].metric("Eligible cells", result["total_cells"])
metric_columns[2].metric(
    "Algorithm PNC-positive",
    f"{result['cells_with_pnc']}/{result['total_cells']}",
)
metric_columns[3].metric(
    "Algorithm smear calls",
    len(result["smear_cell_ids"]),
)
metric_columns[4].metric(
    "Fully reviewed",
    f"{reviewed_count}/{len(cell_ids)}",
)
st.caption(
    f"Segmentation {result['segmentation_version']} from "
    f"{segmentation_source} channel {result['segmentation_channel']} at scale "
    f"{result['segmentation_scale']:g}; label-stack fingerprint "
    f"{segmentation_fingerprint[:12]}…; discarded single-layer artifacts: "
    f"{result['discarded_single_layer_tracks']}; automatic unresolved-merge "
    f"exclusions: {result['segmentation_excluded_cell_ids'] or 'none'}."
)

flash_key = f"annotation_flash_{image_token}"
flash_message = st.session_state.pop(flash_key, None)
if flash_message:
    level, message = flash_message
    getattr(st, level)(message)

selected_cells = st.session_state.setdefault("selected_cells", {})
selected_cell_id = selected_cells.get(annotation_key, cell_ids[0])
if selected_cell_id not in cell_ids:
    selected_cell_id = cell_ids[0]
    selected_cells[annotation_key] = selected_cell_id

picker_key = f"cell_picker_{image_token}"
pending_selection_key = f"pending_cell_selection_{image_token}"
if pending_selection_key in st.session_state:
    selected_cell_id = st.session_state.pop(pending_selection_key)
    selected_cells[annotation_key] = selected_cell_id
    st.session_state[picker_key] = selected_cell_id
elif picker_key not in st.session_state:
    st.session_state[picker_key] = selected_cell_id

selector_column, review_column = st.columns([1.15, 1], gap="large")

with selector_column:
    st.subheader("1. Select a cell")
    st.caption(
        "Click inside a numbered outline. The overview does not show the "
        "algorithm classification."
    )
    selector_image = highlight_selector_image(
        overview,
        overview_labels,
        selected_cell_id,
    )
    click_value = streamlit_image_coordinates(
        selector_image,
        width="stretch",
        key=f"cell_selector_{image_token}_{pnc_threshold_multiplier:g}",
        cursor="crosshair",
        png_compression_level=3,
    )
    if click_value:
        click_token = tuple(sorted(click_value.items()))
        last_click_key = f"last_cell_click_{image_token}"
        if st.session_state.get(last_click_key) != click_token:
            st.session_state[last_click_key] = click_token
            displayed_width = max(
                1,
                int(click_value.get("width", overview_labels.shape[1])),
            )
            displayed_height = max(
                1,
                int(click_value.get("height", overview_labels.shape[0])),
            )
            selector_x = click_value["x"] * overview_labels.shape[1] / displayed_width
            selector_y = click_value["y"] * overview_labels.shape[0] / displayed_height
            clicked_cell_id = cell_id_at_click(
                overview_labels,
                selector_x,
                selector_y,
            )
            if clicked_cell_id is not None and clicked_cell_id != selected_cell_id:
                selected_cells[annotation_key] = clicked_cell_id
                st.session_state[picker_key] = clicked_cell_id
                st.rerun()

    picked_cell_id = st.selectbox(
        "Selected cell",
        options=cell_ids,
        key=picker_key,
        format_func=lambda cell_id: cell_picker_label(cell_id, annotations),
    )
    if picked_cell_id != selected_cell_id:
        selected_cells[annotation_key] = picked_cell_id
        st.rerun()

    navigation_columns = st.columns(2)
    selected_index = cell_ids.index(selected_cell_id)
    if navigation_columns[0].button(
        "Previous cell",
        width="stretch",
    ):
        st.session_state[pending_selection_key] = cell_ids[
            (selected_index - 1) % len(cell_ids)
        ]
        st.rerun()
    if navigation_columns[1].button(
        "Next cell",
        width="stretch",
    ):
        st.session_state[pending_selection_key] = cell_ids[
            (selected_index + 1) % len(cell_ids)
        ]
        st.rerun()

    progress = reviewed_count / len(cell_ids)
    st.progress(
        progress,
        text=f"{reviewed_count} of {len(cell_ids)} cells fully reviewed",
    )

with review_column:
    st.subheader(f"2. Inspect cell {selected_cell_id}")
    selected_geometry = geometry[selected_cell_id]
    st.caption(
        "Centroid: "
        f"x={selected_geometry['centroid_x_px']:.1f}, "
        f"y={selected_geometry['centroid_y_px']:.1f} pixels"
    )

    z_count = int(result["pnc_stack"].shape[0])
    default_z_layer = (
        brightest_cell_z_index(
            result["pnc_stack"],
            result["cell_label_stack"],
            selected_cell_id,
        )
        + 1
    )
    dapi_crop_stack, txred_crop_stack, crop_bounds = make_cell_review_stack(
        result["cell_stack"],
        result["pnc_stack"],
        result["cell_label_stack"],
        selected_cell_id,
        selected_geometry,
        display_style=review_display_style,
    )
    live_dual_z_stack_viewer(
        (
            "Raw DAPI — Fiji blue LUT"
            if fiji_style_display
            else "Raw DAPI — high contrast"
        ),
        dapi_crop_stack,
        (
            "Raw TxRed — Fiji red LUT"
            if fiji_style_display
            else "Raw TxRed — high contrast magenta"
        ),
        txred_crop_stack,
        default_z_layer,
        key=f"review_z_{image_token}_{selected_cell_id}",
    )
    selected_track = result["track_metadata"].get(selected_cell_id, {})
    observed_layers = selected_track.get("observed_z_layers", [])
    interpolated_layers = selected_track.get("interpolated_z_layers", [])
    track_quality = selected_track.get("quality_status", "unknown")
    st.caption(
        "Track layers — observed: "
        f"{observed_layers or 'none'}; interpolated: "
        f"{interpolated_layers or 'none'}; median cross-section: "
        f"{selected_track.get('median_area_pixels', 0):.1f} px; "
        f"quality: {track_quality.replace('_', ' ')}."
    )
    st.caption(
        (
            "Fiji-style color and brightness are display-only; PNC and smear "
            "calculations still use the original raw pixels. "
            if fiji_style_display
            else "High contrast is display-only; analysis still uses raw pixels. "
        )
        + "Yellow is the selected segmentation boundary. Crop bounds: "
        f"x={crop_bounds[0]}:{crop_bounds[2]}, y={crop_bounds[1]}:{crop_bounds[3]}."
    )

    if not blind_review:
        st.info(algorithm_summary(rows_by_cell_id[selected_cell_id]))

    st.subheader("3. Record ground truth")
    current_annotation = empty_annotation()
    current_annotation.update(annotations.get(selected_cell_id, {}))
    form_revisions = st.session_state.setdefault("annotation_form_revisions", {})
    form_revision = form_revisions.get(annotation_key, 0)
    with st.form(
        key=f"annotation_form_{image_token}_{selected_cell_id}_{form_revision}"
    ):
        pnc_label = st.selectbox(
            "PNC ground truth",
            options=PNC_LABELS,
            index=PNC_LABELS.index(current_annotation["pnc"]),
            format_func=LABEL_TITLES.get,
        )
        pnc_z_layers = st.multiselect(
            "Most obvious PNC z-layer(s), if visible",
            options=list(range(1, z_count + 1)),
            default=current_annotation["pnc_z_layers"],
            help=(
                "These are positive-evidence layers only. An unlisted layer "
                "is not treated as PNC-negative."
            ),
        )
        smear_label = st.selectbox(
            "Smear ground truth",
            options=SMEAR_LABELS,
            index=SMEAR_LABELS.index(current_annotation["smear"]),
            format_func=LABEL_TITLES.get,
        )
        smear_z_layers = st.multiselect(
            "Smear z-layer(s), if visible",
            options=list(range(1, z_count + 1)),
            default=current_annotation["smear_z_layers"],
        )
        segmentation_label = st.selectbox(
            "Segmentation ground truth",
            options=SEGMENTATION_LABELS,
            index=SEGMENTATION_LABELS.index(current_annotation["segmentation"]),
            format_func=LABEL_TITLES.get,
        )
        notes = st.text_area(
            "Notes",
            value=current_annotation["notes"],
            placeholder=(
                "Example: small PNC just inside the nucleolus border; two cells "
                "were merged."
            ),
        )
        submit_columns = st.columns(2)
        save_annotation = submit_columns[0].form_submit_button(
            "Save",
            width="stretch",
        )
        save_and_next = submit_columns[1].form_submit_button(
            "Save and next open cell",
            width="stretch",
        )

    if save_annotation or save_and_next:
        annotations[selected_cell_id] = {
            "pnc": pnc_label,
            "pnc_z_layers": (sorted(pnc_z_layers) if pnc_label != "negative" else []),
            "smear": smear_label,
            "smear_z_layers": (
                sorted(smear_z_layers) if smear_label != "negative" else []
            ),
            "segmentation": segmentation_label,
            "notes": notes.strip(),
            "reviewed_at_utc": reviewed_at_now(),
            "migration_provenance": current_annotation.get(
                "migration_provenance",
                "",
            ),
        }
        warnings = []
        if pnc_label == "positive" and not pnc_z_layers:
            warnings.append("positive PNC without a z-layer")
        if smear_label == "positive" and not smear_z_layers:
            warnings.append("positive smear without a z-layer")
        message = f"Saved cell {selected_cell_id}."
        if warnings:
            message += " Please revisit: " + " and ".join(warnings) + "."
        st.session_state[flash_key] = (
            "warning" if warnings else "success",
            message,
        )
        if save_and_next:
            st.session_state[pending_selection_key] = next_incomplete_cell(
                cell_ids,
                selected_cell_id,
                annotations,
            )
        st.rerun()

st.divider()
st.subheader("Annotation dataset")
st.caption(
    "Export regularly. Session annotations are not a database and may be lost "
    "when the browser session or server restarts."
)

dataset_columns = st.columns([1, 1])
with dataset_columns[0]:
    csv_bytes = annotations_to_csv(rows)
    st.download_button(
        "Download all cell annotations",
        data=csv_bytes,
        file_name=f"{file_stem}_cell_annotations.csv",
        mime="text/csv",
        width="stretch",
    )

with dataset_columns[1]:
    imported_csv = st.file_uploader(
        "Resume from a previously exported CSV",
        type=["csv"],
        key=f"annotation_csv_upload_{image_token}",
    )
    if imported_csv is not None:
        imported_bytes = imported_csv.getvalue()
        import_token = hashlib.sha256(imported_bytes).hexdigest()
        processed_import_key = f"processed_import_{image_token}"
        if st.session_state.get(processed_import_key) != import_token:
            try:
                decoded_csv = imported_bytes.decode("utf-8-sig")
                csv_reader = csv.DictReader(io.StringIO(decoded_csv))
                first_row = next(csv_reader, None)
                schema_version = int(
                    (first_row or {}).get("annotation_schema_version") or 1
                )
                legacy_labels = None
                if schema_version == 1:
                    legacy_labels = result.get("legacy_projection_labels")
                    if legacy_labels is None:
                        with st.spinner(
                            "Reconstructing scale-0.1 projection labels for "
                            "conservative migration..."
                        ):
                            legacy_labels = segment_projection_labels(
                                result["cell_stack"],
                                model,
                                scale=LEGACY_RECONCILIATION_SCALE,
                            )
                        result["legacy_projection_labels"] = legacy_labels
                imported_annotations, skipped_ids = annotations_from_csv(
                    imported_bytes,
                    image_sha256,
                    cell_ids,
                    z_count,
                    expected_label_stack_fingerprint=(
                        segmentation_fingerprint
                    ),
                    legacy_labels=legacy_labels,
                    current_labels=result["cell_labels"],
                )
            except (UnicodeDecodeError, ValueError) as exc:
                st.error(str(exc))
            else:
                annotations.update(imported_annotations)
                st.session_state[processed_import_key] = import_token
                form_revisions[annotation_key] = form_revision + 1
                message = f"Imported {len(imported_annotations)} cell rows."
                if skipped_ids:
                    message += (
                        " Manual reconciliation required: "
                        + "; ".join(str(item) for item in skipped_ids)
                        + "."
                    )
                st.session_state[flash_key] = (
                    "warning" if skipped_ids else "success",
                    message,
                )
                st.rerun()

table_rows = []
for row in rows:
    annotation = annotations.get(row["cell_id"], {})
    table_rows.append(
        {
            "Cell": row["cell_id"],
            "Review": ("complete" if annotation_is_complete(annotation) else "open"),
            "PNC truth": row["ground_truth_pnc"],
            "PNC z": row["ground_truth_pnc_z_layers"],
            "Smear truth": row["ground_truth_smear"],
            "Smear z": row["ground_truth_smear_z_layers"],
            "Segmentation": row["ground_truth_segmentation"],
        }
    )
st.dataframe(table_rows, hide_index=True, width="stretch")

with st.expander("Algorithm diagnostics and plot", expanded=False):
    st.write(
        f"Baseline-positive cells: {result['baseline_cells_with_pnc_ids']}  "
        f"  \nNucleolus-rescued cells: {result['nucleolus_rescued_cell_ids']}  "
        f"  \nSmear cells: {result['smear_cell_ids']}"
    )
    render_diagnostic_plot = st.toggle(
        "Render full diagnostic plot",
        value=False,
        key=f"render_diagnostics_{image_token}",
    )
    if render_diagnostic_plot:
        st.pyplot(result["fig"], clear_figure=False)
        if "diagnostic_png" not in result:
            result["diagnostic_png"] = figure_to_png(result["fig"]).getvalue()
        st.download_button(
            "Download diagnostic plot",
            data=result["diagnostic_png"],
            file_name=f"{file_stem}_pnc_analysis.png",
            mime="image/png",
        )
