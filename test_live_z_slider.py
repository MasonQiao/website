import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np

from live_z_slider import live_dual_z_stack_viewer, live_z_stack_viewer


class LiveZStackViewerTest(unittest.TestCase):
    @patch("live_z_slider._live_z_slider_component")
    def test_encodes_each_layer_and_returns_component_value(self, component):
        component.return_value = 2
        images = np.zeros((3, 8, 9, 3), dtype=np.uint8)

        selected = live_z_stack_viewer(
            "Raw TxRed z-layer",
            images,
            value=1,
            key="fixture",
        )

        self.assertEqual(selected, 2)
        call_arguments = component.call_args.kwargs
        self.assertEqual(len(call_arguments["sources"]), 3)
        self.assertTrue(
            call_arguments["sources"][0].startswith("data:image/png;base64,")
        )
        self.assertEqual(call_arguments["min_value"], 1)
        self.assertEqual(call_arguments["max_value"], 3)
        self.assertEqual(call_arguments["image_height"], 8)
        self.assertEqual(call_arguments["image_width"], 9)
        self.assertEqual(call_arguments["storage_key"], "live-z-layer:fixture")

    def test_rejects_an_invalid_image_stack(self):
        with self.assertRaisesRegex(ValueError, "shape"):
            live_z_stack_viewer(
                "Raw TxRed z-layer",
                np.zeros((8, 9), dtype=np.uint8),
                value=1,
            )

    @patch("live_z_slider._live_z_slider_component")
    def test_none_initial_value_falls_back_to_first_layer(self, component):
        component.return_value = None
        images = np.zeros((3, 8, 9, 3), dtype=np.uint8)

        selected = live_z_stack_viewer(
            "Raw TxRed z-layer",
            images,
            value=None,
            key="fixture",
        )

        self.assertEqual(selected, 1)
        self.assertEqual(component.call_args.kwargs["value"], 1)
        self.assertEqual(component.call_args.kwargs["default"], 1)

    @patch("live_z_slider._live_z_slider_component")
    def test_dual_viewer_sends_synchronized_stacks_and_handles_none(self, component):
        component.return_value = None
        dapi = np.zeros((4, 8, 9, 3), dtype=np.uint8)
        txred = np.ones((4, 8, 9, 3), dtype=np.uint8)

        selected = live_dual_z_stack_viewer(
            "Raw DAPI",
            dapi,
            "Raw TxRed",
            txred,
            value=None,
            key="dual-fixture",
        )

        self.assertEqual(selected, 1)
        call_arguments = component.call_args.kwargs
        self.assertEqual(len(call_arguments["sources"]), 4)
        self.assertEqual(len(call_arguments["secondary_sources"]), 4)
        self.assertEqual(call_arguments["secondary_label"], "Raw TxRed")
        self.assertEqual(
            call_arguments["storage_key"],
            "live-z-layer:dual-fixture",
        )

    def test_dual_viewer_requires_matching_z_counts(self):
        with self.assertRaisesRegex(ValueError, "same non-zero z count"):
            live_dual_z_stack_viewer(
                "DAPI",
                np.zeros((2, 8, 9, 3), dtype=np.uint8),
                "TxRed",
                np.zeros((3, 8, 9, 3), dtype=np.uint8),
                value=1,
            )

    def test_frontend_updates_on_input_without_streamlit_rerun_or_scroll_anchor(self):
        component_dir = Path(__file__).parent / "components" / "live_z_slider"
        javascript = (component_dir / "main.js").read_text(encoding="utf-8")
        stylesheet = (component_dir / "style.css").read_text(encoding="utf-8")

        self.assertIn('slider.addEventListener("input"', javascript)
        self.assertNotIn("streamlit:setComponentValue", javascript)
        self.assertIn("overflow-anchor: none", stylesheet)


if __name__ == "__main__":
    unittest.main()
