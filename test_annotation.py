import unittest

import numpy as np

from annotation import (
    HIGH_CONTRAST_DISPLAY_STYLE,
    annotation_is_complete,
    annotations_from_csv,
    annotations_to_csv,
    build_annotation_rows,
    cell_geometry,
    cell_id_at_click,
    make_cell_review_images,
    make_cell_review_stack,
    make_selector_assets,
)


class AnnotationHelpersTest(unittest.TestCase):
    def setUp(self):
        self.labels = np.zeros((24, 32), dtype=np.int32)
        self.labels[3:11, 4:13] = 1
        self.labels[12:22, 18:29] = 2
        self.cell_image = np.arange(24 * 32, dtype=np.uint16).reshape(24, 32)
        self.pnc_stack = np.stack(
            [self.cell_image, self.cell_image * 2, self.cell_image * 3],
            axis=0,
        )
        self.result = {
            "cell_labels": self.labels,
            "cell_image": self.cell_image,
            "pnc_stack": self.pnc_stack,
            "analysis_scale": 0.1,
            "segmentation_version": "z_track_v1",
            "segmentation_scale": 0.15,
            "segmentation_channel": 1,
            "dapi_channel": 0,
            "pnc_channel": 1,
            "label_stack_fingerprint": "track-fingerprint",
            "track_metadata": {
                1: {
                    "observed_z_layers": [1, 2, 3],
                    "interpolated_z_layers": [],
                    "median_area_pixels": 72.0,
                },
                2: {
                    "observed_z_layers": [1, 2, 3],
                    "interpolated_z_layers": [],
                    "median_area_pixels": 110.0,
                },
            },
            "cells_with_pnc_ids": [1],
            "baseline_cells_with_pnc_ids": [],
            "nucleolus_rescued_cell_ids": [1],
            "smear_cell_ids": [2],
            "non_intact_cell_ids": [],
            "low_dapi_cell_ids": [],
            "smear_excluded_cell_ids": [2],
            "small_cell_ids": [],
            "large_cell_ids": [],
        }

    def test_geometry_and_click_mapping(self):
        geometry = cell_geometry(self.labels)
        self.assertEqual(set(geometry), {1, 2})
        self.assertEqual(geometry[1]["cell_area_px"], 72)
        self.assertEqual(geometry[2]["bbox_x_min_px"], 18)

        _, display_labels = make_selector_assets(
            self.cell_image,
            self.labels,
            max_width=32,
        )
        self.assertEqual(cell_id_at_click(display_labels, 6, 6), 1)
        self.assertEqual(cell_id_at_click(display_labels, 20, 15), 2)
        self.assertIsNone(cell_id_at_click(display_labels, 31, 0, search_radius=1))

    def test_review_crops_remain_aligned(self):
        geometry = cell_geometry(self.labels)
        dapi, txred, bounds = make_cell_review_images(
            self.cell_image,
            self.pnc_stack,
            self.labels,
            selected_cell_id=1,
            z_index=1,
            geometry=geometry[1],
            padding=2,
        )
        self.assertEqual(dapi.shape, txred.shape)
        self.assertEqual(dapi.shape[2], 3)
        self.assertEqual(bounds, (2, 1, 15, 13))

    def test_fiji_review_uses_channel_luts_and_full_image_range(self):
        geometry = cell_geometry(self.labels)
        dapi, txred, _ = make_cell_review_images(
            self.cell_image,
            self.pnc_stack,
            self.labels,
            selected_cell_id=1,
            z_index=1,
            geometry=geometry[1],
            padding=2,
        )

        # Crop pixel (0, 0) comes from source pixel (y=1, x=2), outside the
        # yellow cell boundary. Fiji-style display maps the full-image range
        # into the ND2 channel colors: DAPI blue and TxRed red.
        expected_dapi = int(round(34 / 767 * 255))
        expected_txred = int(round(68 / 2301 * 255))
        np.testing.assert_array_equal(dapi[0, 0], [0, 0, expected_dapi])
        np.testing.assert_array_equal(txred[0, 0], [expected_txred, 0, 0])

    def test_high_contrast_review_mode_remains_available(self):
        geometry = cell_geometry(self.labels)
        dapi, txred, _ = make_cell_review_images(
            self.cell_image,
            self.pnc_stack,
            self.labels,
            selected_cell_id=1,
            z_index=1,
            geometry=geometry[1],
            padding=2,
            display_style=HIGH_CONTRAST_DISPLAY_STYLE,
        )

        self.assertEqual(int(dapi[0, 0, 0]), int(dapi[0, 0, 1]))
        self.assertEqual(int(dapi[0, 0, 1]), int(dapi[0, 0, 2]))
        self.assertEqual(int(txred[0, 0, 1]), 0)
        self.assertEqual(int(txred[0, 0, 0]), int(txred[0, 0, 2]))

    def test_csv_round_trip_preserves_labels_and_provenance(self):
        annotations = {
            1: {
                "pnc": "positive",
                "pnc_z_layers": [2, 3],
                "smear": "negative",
                "smear_z_layers": [],
                "segmentation": "valid",
                "notes": "inside nucleolus",
                "reviewed_at_utc": "2026-08-15T12:00:00+00:00",
            }
        }
        self.assertTrue(annotation_is_complete(annotations[1]))

        rows = build_annotation_rows(
            self.result,
            "fixture.nd2",
            "abc123",
            annotations,
        )
        self.assertEqual(rows[0]["algorithm_pnc_source"], "nucleolus_rescue")
        self.assertEqual(rows[1]["algorithm_cell_status"], "excluded_smear")
        self.assertEqual(rows[0]["centroid_x_px"], "8.000")
        self.assertEqual(rows[0]["segmentation_channel"], 1)
        self.assertEqual(rows[0]["dapi_channel"], 0)

        csv_bytes = annotations_to_csv(rows)
        imported, skipped = annotations_from_csv(
            csv_bytes,
            expected_image_sha256="abc123",
            valid_cell_ids={1, 2},
            z_layers_total=3,
            expected_label_stack_fingerprint="track-fingerprint",
        )
        self.assertEqual(skipped, [])
        self.assertEqual(imported[1]["pnc"], "positive")
        self.assertEqual(imported[1]["pnc_z_layers"], [2, 3])
        self.assertEqual(imported[1]["notes"], "inside nucleolus")

    def test_schema_two_import_rejects_a_segmentation_fingerprint_mismatch(self):
        rows = build_annotation_rows(
            self.result,
            "fixture.nd2",
            "abc123",
            {},
        )
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            annotations_from_csv(
                annotations_to_csv(rows),
                expected_image_sha256="abc123",
                valid_cell_ids={1, 2},
                z_layers_total=3,
                expected_label_stack_fingerprint="different",
            )

    def test_legacy_valid_unique_match_migrates_but_resets_segmentation(self):
        annotations = {
            1: {
                "pnc": "positive",
                "pnc_z_layers": [2],
                "smear": "negative",
                "smear_z_layers": [],
                "segmentation": "valid",
                "notes": "legacy evidence",
                "reviewed_at_utc": "2026-08-15T12:00:00+00:00",
            }
        }
        rows = build_annotation_rows(
            self.result,
            "fixture.nd2",
            "abc123",
            annotations,
        )
        for row in rows:
            row["annotation_schema_version"] = 1
        imported, report = annotations_from_csv(
            annotations_to_csv(rows),
            expected_image_sha256="abc123",
            valid_cell_ids={1, 2},
            z_layers_total=3,
            expected_label_stack_fingerprint="new-fingerprint",
            legacy_labels=self.labels,
            current_labels=self.labels,
        )

        self.assertEqual(imported[1]["pnc"], "positive")
        self.assertEqual(imported[1]["pnc_z_layers"], [2])
        self.assertEqual(imported[1]["segmentation"], "unreviewed")
        self.assertIn("old_cell=1", imported[1]["migration_provenance"])
        self.assertTrue(any("old cell 2" in item for item in report))

    def test_legacy_competing_overlap_is_left_entirely_unreviewed(self):
        old_labels = np.zeros((20, 20), dtype=np.int32)
        old_labels[5:15, 5:15] = 1
        current_labels = np.zeros_like(old_labels)
        current_labels[5:15, 5:12] = 1
        current_labels[5:15, 12:15] = 2
        old_result = dict(self.result)
        old_result["cell_labels"] = old_labels
        old_result["track_metadata"] = {}
        rows = build_annotation_rows(
            old_result,
            "fixture.nd2",
            "abc123",
            {
                1: {
                    "pnc": "positive",
                    "pnc_z_layers": [1],
                    "smear": "negative",
                    "smear_z_layers": [],
                    "segmentation": "valid",
                    "notes": "must not transfer",
                    "reviewed_at_utc": "",
                }
            },
        )
        rows[0]["annotation_schema_version"] = 1
        imported, report = annotations_from_csv(
            annotations_to_csv(rows),
            expected_image_sha256="abc123",
            valid_cell_ids={1, 2},
            z_layers_total=3,
            legacy_labels=old_labels,
            current_labels=current_labels,
        )

        self.assertEqual(imported, {})
        self.assertTrue(any("competing overlap" in item for item in report))

    def test_review_stacks_share_dynamic_layer_boundaries(self):
        cell_stack = np.stack([self.cell_image, self.cell_image + 10])
        pnc_stack = np.stack([self.cell_image, self.cell_image + 20])
        label_stack = np.zeros((2,) + self.labels.shape, dtype=np.int32)
        label_stack[0, 3:11, 4:13] = 1
        label_stack[1, 3:11, 7:16] = 1
        geometry = cell_geometry(self.labels)[1]

        dapi, txred, _ = make_cell_review_stack(
            cell_stack,
            pnc_stack,
            label_stack,
            selected_cell_id=1,
            geometry=geometry,
            padding=2,
        )

        self.assertEqual(dapi.shape, txred.shape)
        self.assertEqual(dapi.shape[0], 2)
        yellow_first = np.all(dapi[0] == [255, 215, 0], axis=-1)
        yellow_second = np.all(dapi[1] == [255, 215, 0], axis=-1)
        self.assertFalse(np.array_equal(yellow_first, yellow_second))

    def test_import_rejects_a_different_image(self):
        rows = build_annotation_rows(
            self.result,
            "fixture.nd2",
            "abc123",
            {},
        )
        with self.assertRaisesRegex(ValueError, "uploaded ND2"):
            annotations_from_csv(
                annotations_to_csv(rows),
                expected_image_sha256="different",
                valid_cell_ids={1, 2},
                z_layers_total=3,
            )


if __name__ == "__main__":
    unittest.main()
