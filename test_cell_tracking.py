import unittest

import numpy as np

from cell_tracking import (
    LEGACY_SEGMENTATION_VERSION,
    SEGMENTATION_VERSION,
    projection_cell_segmentation,
    track_label_layers,
)
from testing import (
    PNC_BRIGHT_PIXEL_PERCENTILE,
    PNC_MIN_CELL_AREA_FRACTION,
    PNC_THRESHOLD_MULTIPLIER,
    _segment_pncs_across_z,
)


def ellipse_mask(shape, center_y, center_x, radius_y=4, radius_x=5):
    y_grid, x_grid = np.indices(shape)
    return (
        ((y_grid - center_y) / radius_y) ** 2
        + ((x_grid - center_x) / radius_x) ** 2
        <= 1
    )


def label_layer(shape, objects):
    labels = np.zeros(shape, dtype=np.int32)
    for label_id, center_y, center_x, radius_y, radius_x in objects:
        labels[
            ellipse_mask(shape, center_y, center_x, radius_y, radius_x)
        ] = label_id
    return labels


class CellTrackingTest(unittest.TestCase):
    def test_legacy_projection_mode_repeats_the_exact_projection_labels(self):
        expected = label_layer((30, 40), [(9, 15, 20, 5, 6)])

        class FakeModel:
            def predict_instances(self, image, scale):
                self.scale = scale
                return expected, {}

        model = FakeModel()
        result = projection_cell_segmentation(
            np.zeros((3, 30, 40), dtype=np.uint16),
            model,
            scale=0.1,
        )

        self.assertEqual(
            result["segmentation_version"],
            LEGACY_SEGMENTATION_VERSION,
        )
        self.assertEqual(model.scale, 0.1)
        np.testing.assert_array_equal(result["canonical_labels"], expected)
        np.testing.assert_array_equal(result["label_stack"][2], expected)

    def test_moving_neighbors_get_deterministic_row_major_ids(self):
        shape = (60, 80)
        layers = np.stack(
            [
                label_layer(
                    shape,
                    [
                        (91 + z, 18 + z, 20 + z, 4, 5),
                        (7 + z, 39 - z, 57 - z, 5, 5),
                    ],
                )
                for z in range(4)
            ]
        )
        result = track_label_layers(layers)

        self.assertEqual(result["segmentation_version"], SEGMENTATION_VERSION)
        self.assertEqual(set(result["tracks"]), {1, 2})
        self.assertLess(
            result["tracks"][1]["median_centroid_y"],
            result["tracks"][2]["median_centroid_y"],
        )
        self.assertEqual(
            result["tracks"][1]["observed_z_layers"],
            [1, 2, 3, 4],
        )
        self.assertTrue(result["label_stack_fingerprint"])

        relabeled = np.zeros_like(layers)
        for z_index, layer in enumerate(layers):
            positive_ids = np.unique(layer[layer > 0])
            for replacement, original in zip((500, 300), positive_ids):
                relabeled[z_index][layer == original] = replacement
        repeated = track_label_layers(relabeled)
        np.testing.assert_array_equal(
            repeated["label_stack"],
            result["label_stack"],
        )
        self.assertEqual(
            repeated["label_stack_fingerprint"],
            result["label_stack_fingerprint"],
        )

    def test_one_layer_dropout_is_interpolated_but_single_layer_artifact_is_dropped(self):
        shape = (48, 64)
        layers = []
        for z in range(5):
            objects = []
            if z != 2:
                objects.append((10 + z, 20, 18 + 2 * z, 4, 5))
            if z == 1:
                objects.append((99, 38, 50, 2, 2))
            layers.append(label_layer(shape, objects))
        result = track_label_layers(np.stack(layers))

        self.assertEqual(set(result["tracks"]), {1})
        self.assertEqual(result["tracks"][1]["interpolated_z_layers"], [3])
        self.assertTrue(np.any(result["label_stack"][2] == 1))
        self.assertEqual(result["discarded_single_layer_tracks"], 1)

    def test_temporary_merge_is_partitioned_by_predicted_center(self):
        shape = (50, 70)
        layers = np.stack(
            [
                label_layer(shape, [(1, 25, 24, 4, 5), (2, 25, 44, 4, 5)]),
                label_layer(shape, [(3, 25, 27, 4, 5), (4, 25, 41, 4, 5)]),
                label_layer(shape, [(8, 25, 34, 5, 9)]),
                label_layer(shape, [(5, 25, 30, 4, 5), (6, 25, 38, 4, 5)]),
            ]
        )
        result = track_label_layers(layers)
        merged_layer = result["label_stack"][2]

        self.assertEqual(set(result["tracks"]), {1, 2})
        self.assertEqual(int(merged_layer[25, 29]), 1)
        self.assertEqual(int(merged_layer[25, 39]), 2)
        self.assertEqual(int(merged_layer[25, 34]), 1)
        self.assertEqual(result["tracks"][1]["merge_z_layers"], [3])
        self.assertEqual(result["tracks"][2]["quality_status"], "valid")

        pnc_stack = np.zeros((4,) + shape, dtype=float)
        pnc_stack[result["label_stack"] > 0] = 10
        pnc_stack[2, 25, 39] = 100
        _, positive_ids = _segment_pncs_across_z(
            pnc_stack,
            result["label_stack"],
        )
        self.assertEqual(positive_ids, {2})

    def test_persistent_many_to_one_is_flagged(self):
        shape = (50, 70)
        layers = np.stack(
            [
                label_layer(shape, [(1, 25, 24, 4, 5), (2, 25, 44, 4, 5)]),
                label_layer(shape, [(3, 25, 27, 4, 5), (4, 25, 41, 4, 5)]),
                label_layer(shape, [(8, 25, 34, 5, 9)]),
                label_layer(shape, [(9, 25, 34, 5, 9)]),
            ]
        )
        result = track_label_layers(layers)

        self.assertEqual(result["persistent_merge_track_ids"], [1, 2])
        self.assertEqual(
            result["tracks"][1]["quality_status"],
            "persistent_unresolved_many_to_one",
        )

    def test_pnc_uses_the_owner_on_the_actual_layer(self):
        shape = (40, 60)
        label_stack = np.stack(
            [
                label_layer(shape, [(1, 20, 18, 6, 7), (2, 20, 42, 6, 7)]),
                label_layer(shape, [(1, 20, 28, 6, 7), (2, 20, 48, 6, 7)]),
            ]
        )
        pnc_stack = np.zeros((2,) + shape, dtype=np.float32)
        pnc_stack[label_stack > 0] = 10
        pnc_stack[1, 20, 28] = 100

        _, positive_ids = _segment_pncs_across_z(
            pnc_stack,
            label_stack,
            bright_pixel_percentile=PNC_BRIGHT_PIXEL_PERCENTILE,
            pnc_threshold_multiplier=PNC_THRESHOLD_MULTIPLIER,
        )

        self.assertEqual(positive_ids, {1})
        self.assertEqual(PNC_THRESHOLD_MULTIPLIER, 1.25)
        self.assertEqual(PNC_MIN_CELL_AREA_FRACTION, 1 / 1000)


if __name__ == "__main__":
    unittest.main()
