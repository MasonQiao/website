"""Z-aware StarDist instance segmentation and deterministic track construction."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

import numpy as np
from csbdeep.utils import normalize
from scipy import ndimage as ndi
from scipy.optimize import linear_sum_assignment


SEGMENTATION_VERSION = "z_track_v1"
LEGACY_SEGMENTATION_VERSION = "projection_v1"
DEFAULT_SEGMENTATION_SCALE = 0.15
LEGACY_RECONCILIATION_SCALE = 0.1

MAX_NORMALIZED_DISPLACEMENT = 0.75
MIN_AREA_RATIO = 0.4
MAX_AREA_RATIO = 2.5
DISTANCE_COST_WEIGHT = 0.65
IOU_COST_WEIGHT = 0.35
MAX_MISSING_LAYERS = 1


@dataclass(frozen=True)
class _Instance:
    z_index: int
    label_id: int
    area: int
    centroid_y: float
    centroid_x: float
    radius: float
    bounds: tuple[int, int, int, int]


@dataclass
class _Observation:
    instance: _Instance
    center_y: float
    center_x: float
    shared: bool = False


@dataclass
class _Track:
    temporary_id: int
    observations: dict[int, _Observation] = field(default_factory=dict)
    merge_layers: set[int] = field(default_factory=set)
    maximum_consecutive_merge_layers: int = 0

    @property
    def last_z(self):
        return max(self.observations)

    @property
    def last_observation(self):
        return self.observations[self.last_z]

    @property
    def is_established(self):
        layers = sorted(self.observations)
        return any(b - a == 1 for a, b in zip(layers, layers[1:]))


def _instances_from_labels(labels, z_index):
    labels = np.asarray(labels)
    instance_ids = np.unique(labels)
    instance_ids = instance_ids[instance_ids > 0]
    if not instance_ids.size:
        return []
    areas = np.bincount(labels.ravel())
    centers = ndi.center_of_mass(
        np.ones(labels.shape, dtype=np.uint8),
        labels,
        instance_ids,
    )
    object_slices = ndi.find_objects(labels)
    instances = []
    for label_id, center in zip(instance_ids, centers):
        label_id = int(label_id)
        object_slice = object_slices[label_id - 1]
        if object_slice is None:
            continue
        y_slice, x_slice = object_slice
        area = int(areas[label_id])
        instances.append(
            _Instance(
                z_index=int(z_index),
                label_id=label_id,
                area=area,
                centroid_y=float(center[0]),
                centroid_x=float(center[1]),
                radius=float(np.sqrt(area / np.pi)),
                bounds=(
                    int(y_slice.start),
                    int(y_slice.stop),
                    int(x_slice.start),
                    int(x_slice.stop),
                ),
            )
        )
    return instances


def _instance_mask(labels, instance):
    y_min, y_max, x_min, x_max = instance.bounds
    return labels[y_min:y_max, x_min:x_max] == instance.label_id


def _instance_iou(first, second, layer_labels):
    first_labels = layer_labels[first.z_index]
    second_labels = layer_labels[second.z_index]
    y_min = min(first.bounds[0], second.bounds[0])
    y_max = max(first.bounds[1], second.bounds[1])
    x_min = min(first.bounds[2], second.bounds[2])
    x_max = max(first.bounds[3], second.bounds[3])
    first_mask = first_labels[y_min:y_max, x_min:x_max] == first.label_id
    second_mask = second_labels[y_min:y_max, x_min:x_max] == second.label_id
    intersection = int(np.count_nonzero(first_mask & second_mask))
    union = first.area + second.area - intersection
    return intersection / union if union else 0.0


def _predict_center(track, z_index):
    observations = sorted(track.observations.items())
    usable = [
        (z, observation)
        for z, observation in observations
        if not observation.shared
    ]
    if not usable:
        usable = observations
    last_z, last = usable[-1]
    if len(usable) < 2:
        return last.center_y, last.center_x
    previous_z, previous = usable[-2]
    delta_z = max(1, last_z - previous_z)
    multiplier = (z_index - last_z) / delta_z
    return (
        last.center_y + (last.center_y - previous.center_y) * multiplier,
        last.center_x + (last.center_x - previous.center_x) * multiplier,
    )


def _reference_area(track):
    non_shared = [
        observation.instance.area
        for _, observation in sorted(track.observations.items(), reverse=True)
        if not observation.shared
    ]
    return non_shared[0] if non_shared else track.last_observation.instance.area


def _association_values(track, instance, layer_labels):
    previous = track.last_observation.instance
    radius_sum = previous.radius + instance.radius
    predicted_y, predicted_x = _predict_center(track, instance.z_index)
    distance = float(
        np.hypot(
            instance.centroid_y - predicted_y,
            instance.centroid_x - predicted_x,
        )
    )
    normalized_distance = distance / max(radius_sum, 1e-6)
    area_ratio = instance.area / max(_reference_area(track), 1)
    eligible = (
        normalized_distance <= MAX_NORMALIZED_DISPLACEMENT
        and MIN_AREA_RATIO <= area_ratio <= MAX_AREA_RATIO
    )
    iou = (
        _instance_iou(previous, instance, layer_labels)
        if eligible
        else 0.0
    )
    cost = (
        DISTANCE_COST_WEIGHT * normalized_distance
        + IOU_COST_WEIGHT * (1.0 - iou)
    )
    return eligible, cost, normalized_distance, iou


def _maximum_consecutive_run(values):
    values = sorted(values)
    if not values:
        return 0
    maximum = current = 1
    for first, second in zip(values, values[1:]):
        if second == first + 1:
            current += 1
            maximum = max(maximum, current)
        else:
            current = 1
    return maximum


def _add_observation(track, instance, shared=False, center=None):
    if center is None:
        center = (instance.centroid_y, instance.centroid_x)
    track.observations[instance.z_index] = _Observation(
        instance=instance,
        center_y=float(center[0]),
        center_x=float(center[1]),
        shared=bool(shared),
    )
    if shared:
        track.merge_layers.add(instance.z_index)
        track.maximum_consecutive_merge_layers = _maximum_consecutive_run(
            track.merge_layers
        )


def _track_instances(layer_labels):
    tracks = []
    next_temporary_id = 1

    for z_index, labels in enumerate(layer_labels):
        instances = _instances_from_labels(labels, z_index)
        if z_index == 0:
            for instance in instances:
                track = _Track(next_temporary_id)
                next_temporary_id += 1
                _add_observation(track, instance)
                tracks.append(track)
            continue

        active_tracks = [
            track
            for track in tracks
            if 1 <= z_index - track.last_z <= MAX_MISSING_LAYERS + 1
        ]
        handled_tracks = set()
        handled_instances = set()

        # A single current instance can temporarily represent several tracks.
        # Only tracks already confirmed on adjacent layers are allowed into
        # this branch; newly appearing close objects still use one-to-one
        # Hungarian assignment.
        for instance_index, instance in enumerate(instances):
            candidates = []
            for track in active_tracks:
                if track.temporary_id in handled_tracks:
                    continue
                if not track.is_established or track.last_z != z_index - 1:
                    continue
                eligible, cost, _, _ = _association_values(
                    track,
                    instance,
                    layer_labels,
                )
                if eligible:
                    candidates.append((track, cost))
            if len(candidates) < 2:
                continue
            candidates.sort(key=lambda item: (item[1], item[0].temporary_id))
            combined_area = sum(
                _reference_area(item[0]) for item in candidates
            )
            if instance.area < 0.65 * combined_area:
                continue
            for track, _ in candidates:
                predicted_center = _predict_center(track, z_index)
                _add_observation(
                    track,
                    instance,
                    shared=True,
                    center=predicted_center,
                )
                handled_tracks.add(track.temporary_id)
            handled_instances.add(instance_index)

        remaining_tracks = [
            track
            for track in active_tracks
            if track.temporary_id not in handled_tracks
        ]
        remaining_instance_indices = [
            index
            for index in range(len(instances))
            if index not in handled_instances
        ]

        matched_track_ids = set()
        matched_instance_indices = set()
        if remaining_tracks and remaining_instance_indices:
            costs = np.full(
                (len(remaining_tracks), len(remaining_instance_indices)),
                1e6,
                dtype=float,
            )
            eligibility = np.zeros(costs.shape, dtype=bool)
            for track_index, track in enumerate(remaining_tracks):
                for column, instance_index in enumerate(
                    remaining_instance_indices
                ):
                    eligible, cost, _, _ = _association_values(
                        track,
                        instances[instance_index],
                        layer_labels,
                    )
                    if eligible:
                        eligibility[track_index, column] = True
                        costs[track_index, column] = cost

            row_indices, column_indices = linear_sum_assignment(costs)
            for row_index, column_index in zip(row_indices, column_indices):
                if not eligibility[row_index, column_index]:
                    continue
                track = remaining_tracks[row_index]
                instance_index = remaining_instance_indices[column_index]
                _add_observation(track, instances[instance_index])
                matched_track_ids.add(track.temporary_id)
                matched_instance_indices.add(instance_index)

        for instance_index in remaining_instance_indices:
            if instance_index in matched_instance_indices:
                continue
            track = _Track(next_temporary_id)
            next_temporary_id += 1
            _add_observation(track, instances[instance_index])
            tracks.append(track)

    return tracks


def _instance_candidate(layer_labels, observation):
    instance = observation.instance
    y_min, y_max, x_min, x_max = instance.bounds
    local_mask = (
        layer_labels[instance.z_index, y_min:y_max, x_min:x_max]
        == instance.label_id
    )
    return (y_min, x_min, local_mask)


def _candidate_coordinates(candidate):
    y_min, x_min, local_mask = candidate
    y_coords, x_coords = np.nonzero(local_mask)
    return y_coords + y_min, x_coords + x_min


def _translated_candidate(candidate, shape, delta_y, delta_x):
    y_coords, x_coords = _candidate_coordinates(candidate)
    shifted_y = y_coords + int(round(delta_y))
    shifted_x = x_coords + int(round(delta_x))
    inside = (
        (shifted_y >= 0)
        & (shifted_y < shape[0])
        & (shifted_x >= 0)
        & (shifted_x < shape[1])
    )
    shifted_y = shifted_y[inside]
    shifted_x = shifted_x[inside]
    if not shifted_y.size:
        return (0, 0, np.zeros((0, 0), dtype=bool))
    y_min = int(np.min(shifted_y))
    y_max = int(np.max(shifted_y)) + 1
    x_min = int(np.min(shifted_x))
    x_max = int(np.max(shifted_x)) + 1
    local_mask = np.zeros((y_max - y_min, x_max - x_min), dtype=bool)
    local_mask[shifted_y - y_min, shifted_x - x_min] = True
    return (y_min, x_min, local_mask)


def _candidate_masks_for_track(track, layer_labels):
    candidates = {}
    centers = {}
    statuses = {}
    for z_index, observation in track.observations.items():
        candidates[z_index] = _instance_candidate(layer_labels, observation)
        centers[z_index] = (observation.center_y, observation.center_x)
        statuses[z_index] = "observed"

    observed_layers = sorted(track.observations)
    for first_z, second_z in zip(observed_layers, observed_layers[1:]):
        if second_z - first_z != 2:
            continue
        gap_z = first_z + 1
        first = track.observations[first_z]
        second = track.observations[second_z]
        target_center = (
            (first.center_y + second.center_y) / 2.0,
            (first.center_x + second.center_x) / 2.0,
        )
        first_candidate = _instance_candidate(layer_labels, first)
        # Both observations are one layer from the gap. Resolve that temporal
        # tie deterministically in favor of the earlier observed mask.
        source, source_candidate = first, first_candidate
        candidates[gap_z] = _translated_candidate(
            source_candidate,
            layer_labels.shape[1:],
            target_center[0] - source.center_y,
            target_center[1] - source.center_x,
        )
        centers[gap_z] = target_center
        statuses[gap_z] = "interpolated"
    return candidates, centers, statuses


def _median_track_center(track):
    observations = [
        observation
        for observation in track.observations.values()
        if not observation.shared
    ]
    if not observations:
        observations = list(track.observations.values())
    return (
        float(np.median([item.center_y for item in observations])),
        float(np.median([item.center_x for item in observations])),
    )


def _confirm_tracks(tracks):
    confirmed = []
    for track in tracks:
        layers = sorted(track.observations)
        if any(second - first == 1 for first, second in zip(layers, layers[1:])):
            confirmed.append(track)
    return confirmed


def _paint_non_overlapping_stack(confirmed_tracks, layer_labels):
    ordered_tracks = sorted(
        confirmed_tracks,
        key=lambda track: (*_median_track_center(track), track.temporary_id),
    )
    id_by_temporary_id = {
        track.temporary_id: final_id
        for final_id, track in enumerate(ordered_tracks, start=1)
    }
    candidate_data = {}
    for track in ordered_tracks:
        candidate_data[track.temporary_id] = _candidate_masks_for_track(
            track,
            layer_labels,
        )

    if len(ordered_tracks) > np.iinfo(np.uint16).max:
        raise ValueError("Too many cell tracks for the uint16 label stack.")
    label_stack = np.zeros(layer_labels.shape, dtype=np.uint16)
    for z_index in range(len(layer_labels)):
        best_distance = np.full(layer_labels.shape[1:], np.inf, dtype=np.float32)
        owners = np.zeros(layer_labels.shape[1:], dtype=np.uint16)
        for track in ordered_tracks:
            final_id = id_by_temporary_id[track.temporary_id]
            masks, centers, _ = candidate_data[track.temporary_id]
            candidate = masks.get(z_index)
            if candidate is None:
                continue
            center_y, center_x = centers[z_index]
            y_coords, x_coords = _candidate_coordinates(candidate)
            if not y_coords.size:
                continue
            distances = (
                (y_coords - center_y) ** 2 + (x_coords - center_x) ** 2
            ).astype(np.float32)
            current = best_distance[y_coords, x_coords]
            current_owner = owners[y_coords, x_coords]
            better = (distances < current) | (
                (distances == current)
                & ((current_owner == 0) | (final_id < current_owner))
            )
            if np.any(better):
                chosen_y = y_coords[better]
                chosen_x = x_coords[better]
                best_distance[chosen_y, chosen_x] = distances[better]
                owners[chosen_y, chosen_x] = final_id
        label_stack[z_index] = owners
    return label_stack, ordered_tracks, id_by_temporary_id, candidate_data


def canonical_labels_from_stack(label_stack, track_centers=None):
    """Return a non-overlapping per-pixel modal track image."""
    label_stack = np.asarray(label_stack)
    canonical = np.zeros(label_stack.shape[1:], dtype=np.uint16)
    best_count = np.zeros(label_stack.shape[1:], dtype=np.uint8)
    best_distance = np.full(label_stack.shape[1:], np.inf, dtype=np.float32)
    track_ids = np.unique(label_stack)
    track_ids = track_ids[track_ids > 0]
    object_slices = ndi.find_objects(label_stack)
    for track_id in track_ids:
        track_slice = object_slices[int(track_id) - 1]
        if track_slice is None:
            continue
        _, y_slice, x_slice = track_slice
        y_min, y_max = int(y_slice.start), int(y_slice.stop)
        x_min, x_max = int(x_slice.start), int(x_slice.stop)
        crop = label_stack[:, y_min:y_max, x_min:x_max]
        counts = np.sum(crop == track_id, axis=0).astype(np.uint8)
        if track_centers and int(track_id) in track_centers:
            center_y, center_x = track_centers[int(track_id)]
        else:
            center = ndi.center_of_mass(crop == track_id)
            center_y = float(center[1] + y_min)
            center_x = float(center[2] + x_min)
        y_grid, x_grid = np.indices(counts.shape)
        distances = (
            (y_grid + y_min - center_y) ** 2
            + (x_grid + x_min - center_x) ** 2
        ).astype(np.float32)
        current_counts = best_count[y_min:y_max, x_min:x_max]
        current_distances = best_distance[y_min:y_max, x_min:x_max]
        current_ids = canonical[y_min:y_max, x_min:x_max]
        positive = counts > 0
        better = positive & (
            (counts > current_counts)
            | (
                (counts == current_counts)
                & (
                    (distances < current_distances)
                    | (
                        (distances == current_distances)
                        & ((current_ids == 0) | (track_id < current_ids))
                    )
                )
            )
        )
        current_counts[better] = counts[better]
        current_distances[better] = distances[better]
        current_ids[better] = int(track_id)
    return canonical


def label_stack_fingerprint(label_stack, version, scale):
    digest = hashlib.sha256()
    metadata = json.dumps(
        {
            "version": str(version),
            "scale": float(scale),
            "shape": list(np.asarray(label_stack).shape),
            "dtype": str(np.asarray(label_stack).dtype),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest.update(metadata)
    digest.update(np.ascontiguousarray(label_stack).tobytes())
    return digest.hexdigest()


def track_label_layers(layer_labels, scale=DEFAULT_SEGMENTATION_SCALE):
    """Track pre-segmented layers; useful for tests and model-independent use."""
    layer_labels = np.asarray(layer_labels, dtype=np.uint16)
    if layer_labels.ndim != 3:
        raise ValueError("layer_labels must have shape (z, y, x).")
    tracks = _track_instances(layer_labels)
    confirmed_tracks = _confirm_tracks(tracks)
    discarded_single_layer_tracks = len(tracks) - len(confirmed_tracks)
    (
        label_stack,
        ordered_tracks,
        id_by_temporary_id,
        candidate_data,
    ) = _paint_non_overlapping_stack(confirmed_tracks, layer_labels)

    centers = {
        id_by_temporary_id[track.temporary_id]: _median_track_center(track)
        for track in ordered_tracks
    }
    canonical_labels = canonical_labels_from_stack(label_stack, centers)
    track_metadata = {}
    per_layer_counts = [
        np.bincount(
            layer.ravel(),
            minlength=len(ordered_tracks) + 1,
        )
        for layer in label_stack
    ]
    for track in ordered_tracks:
        final_id = id_by_temporary_id[track.temporary_id]
        _, candidate_centers, statuses = candidate_data[track.temporary_id]
        observed_layers = sorted(track.observations)
        interpolated_layers = sorted(
            z_index
            for z_index, status in statuses.items()
            if status == "interpolated"
        )
        per_layer_areas = {
            z_index + 1: int(per_layer_counts[z_index][final_id])
            for z_index in sorted(statuses)
            if per_layer_counts[z_index][final_id] > 0
        }
        observed_areas = [
            per_layer_areas[z_index + 1]
            for z_index in observed_layers
            if z_index + 1 in per_layer_areas
        ]
        unresolved_merge = (
            track.maximum_consecutive_merge_layers >= 2
            and max(track.merge_layers) == max(track.observations)
        ) if track.merge_layers else False
        quality_status = (
            "persistent_unresolved_many_to_one"
            if unresolved_merge
            else "valid"
        )
        track_metadata[final_id] = {
            "track_id": final_id,
            "observed_z_layers": [value + 1 for value in observed_layers],
            "interpolated_z_layers": [
                value + 1 for value in interpolated_layers
            ],
            "median_area_pixels": (
                float(np.median(observed_areas)) if observed_areas else 0.0
            ),
            "per_layer_area_pixels": per_layer_areas,
            "median_centroid_y": centers[final_id][0],
            "median_centroid_x": centers[final_id][1],
            "centroid_path": [
                {
                    "z_layer": z_index + 1,
                    "centroid_y": float(candidate_centers[z_index][0]),
                    "centroid_x": float(candidate_centers[z_index][1]),
                    "status": statuses[z_index],
                }
                for z_index in sorted(statuses)
            ],
            "merge_z_layers": [value + 1 for value in sorted(track.merge_layers)],
            "quality_status": quality_status,
        }

    return {
        "segmentation_version": SEGMENTATION_VERSION,
        "segmentation_scale": float(scale),
        "raw_layer_labels": layer_labels,
        "label_stack": label_stack,
        "canonical_labels": canonical_labels,
        "tracks": track_metadata,
        "discarded_single_layer_tracks": discarded_single_layer_tracks,
        "persistent_merge_track_ids": sorted(
            track_id
            for track_id, metadata in track_metadata.items()
            if metadata["quality_status"]
            == "persistent_unresolved_many_to_one"
        ),
        "label_stack_fingerprint": label_stack_fingerprint(
            label_stack,
            SEGMENTATION_VERSION,
            scale,
        ),
    }


def segment_cell_tracks(cell_stack, model, scale=DEFAULT_SEGMENTATION_SCALE):
    """Run StarDist on each layer of the selected channel and track instances."""
    cell_stack = np.asarray(cell_stack)
    if cell_stack.ndim != 3:
        raise ValueError("cell_stack must have shape (z, y, x).")
    raw_layer_labels = []
    for cell_layer in cell_stack:
        labels, _ = model.predict_instances(normalize(cell_layer), scale=scale)
        raw_layer_labels.append(np.asarray(labels, dtype=np.uint16))
    return track_label_layers(np.asarray(raw_layer_labels), scale=scale)


def segment_projection_labels(cell_stack, model, scale=DEFAULT_SEGMENTATION_SCALE):
    """Return only the historical maximum-projection StarDist labels."""
    cell_stack = np.asarray(cell_stack)
    cell_projection = np.max(cell_stack, axis=0)
    labels, _ = model.predict_instances(normalize(cell_projection), scale=scale)
    return np.asarray(labels, dtype=np.uint16)


def projection_cell_segmentation(
    cell_stack,
    model,
    scale=DEFAULT_SEGMENTATION_SCALE,
):
    """Reproduce the historical maximum-projection mask across every z."""
    cell_stack = np.asarray(cell_stack)
    labels = segment_projection_labels(cell_stack, model, scale=scale)
    label_stack = np.repeat(labels[np.newaxis, ...], len(cell_stack), axis=0)
    tracks = {}
    for track_id in np.unique(labels[labels > 0]):
        track_id = int(track_id)
        y_coords, x_coords = np.nonzero(labels == track_id)
        area = int(y_coords.size)
        tracks[track_id] = {
            "track_id": track_id,
            "observed_z_layers": list(range(1, len(cell_stack) + 1)),
            "interpolated_z_layers": [],
            "median_area_pixels": float(area),
            "per_layer_area_pixels": {
                z_layer: area for z_layer in range(1, len(cell_stack) + 1)
            },
            "median_centroid_y": float(np.mean(y_coords)),
            "median_centroid_x": float(np.mean(x_coords)),
            "centroid_path": [
                {
                    "z_layer": z_layer,
                    "centroid_y": float(np.mean(y_coords)),
                    "centroid_x": float(np.mean(x_coords)),
                    "status": "observed",
                }
                for z_layer in range(1, len(cell_stack) + 1)
            ],
            "merge_z_layers": [],
            "quality_status": "legacy_projection",
        }
    return {
        "segmentation_version": LEGACY_SEGMENTATION_VERSION,
        "segmentation_scale": float(scale),
        "raw_layer_labels": label_stack.copy(),
        "label_stack": label_stack,
        "canonical_labels": labels,
        "tracks": tracks,
        "discarded_single_layer_tracks": 0,
        "persistent_merge_track_ids": [],
        "label_stack_fingerprint": label_stack_fingerprint(
            label_stack,
            LEGACY_SEGMENTATION_VERSION,
            scale,
        ),
    }
