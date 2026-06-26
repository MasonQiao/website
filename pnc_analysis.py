import os

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/pnc_streamlit_matplotlib")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import nd2
import numpy as np
from csbdeep.utils import normalize
from scipy import ndimage as ndi


def make_single_plot(image, cmap=None):
    fig, ax = plt.subplots(figsize=(5, 5), constrained_layout=True)
    ax.imshow(image, cmap=cmap)
    ax.axis("off")
    return fig


def analyze_pnc(
    file_path,
    model,
    threshold=7500,
    scale=0.1,
    cell_channel=0,
    pnc_channel=1,
    border_iterations=2,
):
    data = nd2.imread(file_path)
    cell_img = data[0, cell_channel, :, :]
    pnc_img = data[0, pnc_channel, :, :]

    labels, _ = model.predict_instances(normalize(cell_img), scale=scale)
    pnc_mask = pnc_img > threshold
    pnc_labels, num_pncs = ndi.label(pnc_mask)

    cells_with_pnc_ids = set()
    for pnc_id in range(1, num_pncs + 1):
        cell_ids = labels[pnc_labels == pnc_id]
        cell_ids = cell_ids[cell_ids > 0]
        if cell_ids.size:
            cells_with_pnc_ids.add(np.bincount(cell_ids).argmax())

    total_cells = np.unique(labels[labels > 0]).size
    cells_with_pnc = len(cells_with_pnc_ids)
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

    with_pnc_mask = np.isin(labels, list(cells_with_pnc_ids))
    without_pnc_mask = (labels > 0) & ~with_pnc_mask
    with_pnc_border = with_pnc_mask & ~ndi.binary_erosion(with_pnc_mask)
    without_pnc_border = without_pnc_mask & ~ndi.binary_erosion(without_pnc_mask)
    with_pnc_border = ndi.binary_dilation(
        with_pnc_border, iterations=border_iterations
    )
    without_pnc_border = ndi.binary_dilation(
        without_pnc_border,
        iterations=border_iterations,
    )

    combined_annotated = combined_raw.copy()
    combined_annotated[without_pnc_border] = [0, 1, 1]
    combined_annotated[with_pnc_border] = [0, 1, 0]

    fig, axes = plt.subplots(1, 5, figsize=(18, 4), constrained_layout=True)
    axes[0].imshow(cell_img, cmap="gray")
    axes[0].set_title("Raw cells")
    axes[1].imshow(pnc_img, cmap="gray")
    axes[1].set_title("Raw PNCs")
    axes[2].imshow(labels, cmap="nipy_spectral")
    axes[2].set_title("Segmented cells")
    axes[3].imshow(pnc_mask, cmap="gray")
    axes[3].set_title("PNC mask")
    axes[4].imshow(combined_annotated)
    axes[4].set_title(
        f"Green PNC+ / cyan PNC-: "
        f"{cells_with_pnc}/{total_cells}, {percent_cells_with_pnc:.1f}%"
    )

    for ax in axes:
        ax.axis("off")

    subplot_figs = {
        "raw_cells": make_single_plot(cell_img, cmap="gray"),
        "raw_pncs": make_single_plot(pnc_img, cmap="gray"),
        "segmented_cells": make_single_plot(
            labels,
            cmap="nipy_spectral",
        ),
        "pnc_mask": make_single_plot(pnc_mask, cmap="gray"),
        "combined_annotated": make_single_plot(combined_annotated),
    }

    return {
        "fig": fig,
        "subplot_figs": subplot_figs,
        "cells_with_pnc": cells_with_pnc,
        "total_cells": total_cells,
        "percent_cells_with_pnc": percent_cells_with_pnc,
    }
