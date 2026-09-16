import unittest

import numpy as np

from testing import (
    DEFAULT_CELL_CHANNEL,
    DEFAULT_PNC_CHANNEL,
    DEFAULT_SEGMENTATION_CHANNEL,
    _cell_size_outlier_ids,
    _low_dapi_cell_ids,
    _map_smears_to_z_specific_cells,
    _select_analysis_stacks,
    _segment_pncs_by_cell,
)


class ZSpecificAnalysisTest(unittest.TestCase):
    def test_txred_is_default_segmentation_but_dapi_remains_quality_stack(self):
        data = np.zeros((2, 3, 4, 5), dtype=np.uint16)
        data[:, 0] = 10
        data[:, 1] = 20
        data[:, 2] = 30

        dapi, segmentation, pnc = _select_analysis_stacks(data)

        self.assertEqual(DEFAULT_CELL_CHANNEL, 0)
        self.assertEqual(DEFAULT_SEGMENTATION_CHANNEL, 1)
        self.assertEqual(DEFAULT_PNC_CHANNEL, 1)
        self.assertTrue(np.all(dapi == 10))
        self.assertTrue(np.all(segmentation == 20))
        self.assertTrue(np.all(pnc == 20))

    def test_analysis_channel_selection_rejects_an_invalid_channel(self):
        with self.assertRaisesRegex(ValueError, "outside the available range"):
            _select_analysis_stacks(
                np.zeros((2, 3, 4, 5), dtype=np.uint16),
                segmentation_channel=3,
            )

    def test_pnc_minimum_area_is_recomputed_from_each_layer_mask(self):
        image = np.zeros((60, 60), dtype=float)
        large_labels = np.zeros((60, 60), dtype=np.uint16)
        large_labels[5:45, 5:55] = 1
        image[large_labels == 1] = 10
        image[20, 20] = 100
        small_labels = np.zeros_like(large_labels)
        small_labels[15:25, 15:25] = 1

        large_result = _segment_pncs_by_cell(image, large_labels)
        small_result = _segment_pncs_by_cell(image, small_labels)

        self.assertFalse(np.any(large_result > 0))
        self.assertTrue(np.any(small_result > 0))

    def test_area_filter_uses_median_cross_section_not_motion_union(self):
        labels = np.zeros((3, 30, 50), dtype=np.uint16)
        labels[0, 5:15, 3:13] = 1
        labels[1, 5:15, 13:23] = 1
        labels[2, 5:15, 23:33] = 1
        labels[:, 18:28, 38:48] = 2

        small, large, median, minimum, maximum = _cell_size_outlier_ids(
            labels,
            {1, 2},
        )

        self.assertEqual(small, set())
        self.assertEqual(large, set())
        self.assertEqual(median, 100.0)
        self.assertEqual((minimum, maximum), (50.0, 200.0))
        self.assertEqual(np.count_nonzero(np.any(labels == 1, axis=0)), 300)

    def test_dapi_quality_is_median_of_observed_layer_medians(self):
        labels = np.zeros((3, 12, 24), dtype=np.uint16)
        labels[0, 2:8, 2:8] = 1
        labels[1, 2:8, 2:8] = 1
        labels[0, 2:8, 14:20] = 2
        labels[1, 2:8, 14:20] = 2
        dapi = np.zeros_like(labels, dtype=float)
        dapi[0][labels[0] == 1] = 10
        dapi[1][labels[1] == 1] = 30
        dapi[0][labels[0] == 2] = 100
        dapi[1][labels[1] == 2] = 100
        metadata = {
            1: {"observed_z_layers": [1, 2]},
            2: {"observed_z_layers": [1, 2]},
        }

        low_ids, field_median, medians = _low_dapi_cell_ids(
            dapi,
            labels,
            {1, 2},
            track_metadata=metadata,
        )

        self.assertEqual(medians, {1: 20.0, 2: 100.0})
        self.assertEqual(field_median, 60.0)
        self.assertEqual(low_ids, {1})

    def test_endpoint_smear_uses_its_source_layer_cell_mask(self):
        label_stack = np.zeros((3, 20, 30), dtype=np.uint16)
        label_stack[0, 5:15, 3:13] = 1
        label_stack[1, 5:15, 17:27] = 1
        label_stack[2, 5:15, 17:27] = 1
        strong_stack = np.zeros_like(label_stack, dtype=bool)
        # This location belongs to the cell only on the earlier layer. The
        # endpoint event itself is sourced by z-layer 3, where the track moved.
        strong_stack[2, 7:11, 5:9] = True
        empty = np.zeros(label_stack.shape[1:], dtype=bool)
        detection = {
            "strong_endpoint_mask": np.any(strong_stack, axis=0),
            "weak_endpoint_mask": empty,
            "strong_endpoint_mask_stack": strong_stack,
            "weak_endpoint_mask_stack": np.zeros_like(strong_stack),
            "transient_mask_stack": np.zeros_like(strong_stack),
            "cell_ids": set(),
            "transient_cell_ids": set(),
        }

        mapped = _map_smears_to_z_specific_cells(detection, label_stack)

        self.assertEqual(mapped["cell_ids"], set())

    def test_transient_smear_uses_exact_detection_layer(self):
        label_stack = np.zeros((4, 20, 30), dtype=np.uint16)
        label_stack[1, 5:15, 3:13] = 1
        label_stack[2, 5:15, 17:27] = 1
        transient_stack = np.zeros_like(label_stack, dtype=bool)
        transient_stack[2, 7:11, 5:9] = True
        empty = np.zeros(label_stack.shape[1:], dtype=bool)
        detection = {
            "strong_endpoint_mask": empty,
            "weak_endpoint_mask": empty,
            "strong_endpoint_mask_stack": np.zeros_like(transient_stack),
            "weak_endpoint_mask_stack": np.zeros_like(transient_stack),
            "transient_mask_stack": transient_stack,
            "cell_ids": set(),
            "transient_cell_ids": set(),
        }

        mapped = _map_smears_to_z_specific_cells(detection, label_stack)

        self.assertEqual(mapped["cell_ids"], set())
        self.assertEqual(mapped["transient_cell_ids"], set())


if __name__ == "__main__":
    unittest.main()
