import base64
import io
from pathlib import Path

import numpy as np
import streamlit.components.v1 as components
from PIL import Image


_FRONTEND_DIRECTORY = Path(__file__).parent / "components" / "live_z_slider"
_live_z_slider_component = components.declare_component(
    "live_z_stack_viewer_v4",
    path=str(_FRONTEND_DIRECTORY),
)


def _image_data_url(image):
    buffer = io.BytesIO()
    Image.fromarray(np.asarray(image, dtype=np.uint8)).save(
        buffer,
        format="PNG",
        compress_level=3,
    )
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def live_z_stack_viewer(label, images, value, key=None):
    """Display a z-stack whose range input changes images during dragging."""
    images = np.asarray(images)
    if images.ndim != 4 or images.shape[-1] not in (3, 4):
        raise ValueError("images must have shape (z, height, width, 3 or 4).")
    if len(images) == 0:
        raise ValueError("images must contain at least one z-layer.")

    min_value = 1
    max_value = len(images)
    if value is None:
        value = min_value
    value = min(max(int(value), min_value), max_value)
    selected_value = _live_z_slider_component(
        label=str(label),
        sources=[_image_data_url(image) for image in images],
        secondary_sources=[],
        secondary_label="",
        image_height=int(images.shape[1]),
        image_width=int(images.shape[2]),
        storage_key=f"live-z-layer:{key or label}",
        min_value=min_value,
        max_value=max_value,
        value=value,
        key=key,
        default=value,
    )
    if selected_value is None:
        return value
    return min(max(int(selected_value), min_value), max_value)


def live_dual_z_stack_viewer(
    left_label,
    left_images,
    right_label,
    right_images,
    value,
    key=None,
):
    """Display two synchronized image stacks controlled by one live slider."""
    left_images = np.asarray(left_images)
    right_images = np.asarray(right_images)
    for images, name in (
        (left_images, "left_images"),
        (right_images, "right_images"),
    ):
        if images.ndim != 4 or images.shape[-1] not in (3, 4):
            raise ValueError(
                f"{name} must have shape (z, height, width, 3 or 4)."
            )
    if len(left_images) == 0 or len(left_images) != len(right_images):
        raise ValueError("Both image stacks must have the same non-zero z count.")
    if left_images.shape[1:3] != right_images.shape[1:3]:
        raise ValueError("Both image stacks must have matching image dimensions.")

    min_value = 1
    max_value = len(left_images)
    if value is None:
        value = min_value
    value = min(max(int(value), min_value), max_value)
    selected_value = _live_z_slider_component(
        label=str(left_label),
        sources=[_image_data_url(image) for image in left_images],
        secondary_label=str(right_label),
        secondary_sources=[
            _image_data_url(image) for image in right_images
        ],
        image_height=int(left_images.shape[1]),
        image_width=int(left_images.shape[2]),
        storage_key=f"live-z-layer:{key or f'{left_label}:{right_label}'}",
        min_value=min_value,
        max_value=max_value,
        value=value,
        key=key,
        default=value,
    )
    if selected_value is None:
        return value
    return min(max(int(selected_value), min_value), max_value)
