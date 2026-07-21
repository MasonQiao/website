import io
import os
import tempfile
from pathlib import Path

import streamlit as st

from pnc_analysis import PNC_THRESHOLD_MULTIPLIER, analyze_pnc


st.set_page_config(page_title="PNC Cell Analysis", layout="wide")


@st.cache_resource
def load_model():
    from stardist.models import StarDist2D

    return StarDist2D.from_pretrained("2D_versatile_fluo")


def figure_to_png(fig):
    png = io.BytesIO()
    fig.savefig(png, format="png", dpi=300, bbox_inches="tight")
    png.seek(0)
    return png


st.title("PNC Cell Analysis")
with st.spinner("Loading StarDist model..."):
    model = load_model()

pnc_threshold_multiplier = st.number_input(
    "PNC_THRESHOLD_MULTIPLIER",
    min_value=0.0,
    value=float(PNC_THRESHOLD_MULTIPLIER),
)

uploaded_file = st.file_uploader("Upload an ND2 file", type=["nd2"])
if uploaded_file is None:
    st.info("Upload an ND2 file to run the analysis.")
else:
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".nd2") as tmp:
            tmp.write(uploaded_file.getbuffer())
            tmp_path = tmp.name

        with st.spinner("Analyzing image..."):
            result = analyze_pnc(
                tmp_path,
                model,
                pnc_threshold_multiplier=pnc_threshold_multiplier,
            )

        st.metric(
            "Cells with at least one PNC",
            f"{result['cells_with_pnc']}/{result['total_cells']}",
            f"{result['percent_cells_with_pnc']:.1f}%",
        )

        file_stem = Path(uploaded_file.name).stem
        st.columns([4, 1, 4])[1].download_button(
            "Download plot",
            data=figure_to_png(result["fig"]),
            file_name=f"{file_stem}_pnc_analysis.png",
            mime="image/png",
        )

        st.pyplot(result["fig"], clear_figure=False)

        subplot_figs = result.get("subplot_figs", {})
        if subplot_figs:
            panel_names = {
                "raw_cells": "Raw cells",
                "raw_pncs": "Raw PNCs",
                "segmented_cells": "Segmented cells",
                "pnc_mask": "PNC mask",
                "combined_annotated": "Combined annotated",
            }
            columns = st.columns(len(panel_names))
            for column, (panel_key, panel_label) in zip(columns, panel_names.items()):
                if panel_key not in subplot_figs:
                    continue
                with column:
                    st.download_button(
                        panel_label,
                        data=figure_to_png(subplot_figs[panel_key]),
                        file_name=f"{file_stem}_{panel_key}.png",
                        mime="image/png",
                        key=f"download_{panel_key}",
                    )
    except Exception as exc:
        st.error(f"Analysis failed: {exc}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
