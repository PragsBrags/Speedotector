from __future__ import annotations

from dataclasses import dataclass, field
from math import dist

from detection.vehicle import VehicleDetectionResult
from ingestion.zones import centroid


@dataclass
class Track:
    track_id: int
    class_name: str
    bbox_history: list[tuple[float, float, float, float]]
    centroid_history: list[tuple[float, float]]
    first_frame: int
    last_frame: int
    active: bool = True
    missed_frames: int = 0

    @property
    def latest_bbox(self) -> tuple[float, float, float, float]:
        return self.bbox_history[-1]

    @property
    def latest_centroid(self) -> tuple[float, float]:
        return self.centroid_history[-1]

    def update(
        self, bbox: tuple[float, float, float, float], frame_number: int
    ) -> None:
        self.bbox_history.append(bbox)
        self.centroid_history.append(centroid(bbox))
        self.last_frame = frame_number
        self.active = True
        self.missed_frames = 0


@dataclass
class CentroidTracker:
    max_distance: float = 80.0
    max_missed_frames: int = 5
    _next_track_id: int = 1
    tracks: dict[int, Track] = field(default_factory=dict)

    def update(
        self,
        detections: list[VehicleDetectionResult],
        frame_number: int,
    ) -> list[Track]:
        unmatched_track_ids = {
            track_id
            for track_id, track in self.tracks.items()
            if track.active or track.missed_frames <= self.max_missed_frames
        }
        assignments: dict[int, VehicleDetectionResult] = {}

        for detection in detections:
            detection_center = centroid(detection.bbox)
            best_track_id = None
            best_distance = self.max_distance

            for track_id in list(unmatched_track_ids):
                track = self.tracks[track_id]
                if track.class_name != detection.class_name:
                    continue
                distance = dist(track.latest_centroid, detection_center)
                if distance <= best_distance:
                    best_distance = distance
                    best_track_id = track_id

            if best_track_id is None:
                track = self._create_track(detection, frame_number)
                assignments[track.track_id] = detection
                continue

            self.tracks[best_track_id].update(detection.bbox, frame_number)
            assignments[best_track_id] = detection
            unmatched_track_ids.remove(best_track_id)

        for track_id in unmatched_track_ids:
            track = self.tracks[track_id]
            track.missed_frames += 1
            if track.missed_frames > self.max_missed_frames:
                track.active = False

        return [track for track in self.tracks.values() if track.active]

    def _create_track(
        self, detection: VehicleDetectionResult, frame_number: int
    ) -> Track:
        track = Track(
            track_id=self._next_track_id,
            class_name=detection.class_name,
            bbox_history=[detection.bbox],
            centroid_history=[centroid(detection.bbox)],
            first_frame=frame_number,
            last_frame=frame_number,
        )
        self.tracks[track.track_id] = track
        self._next_track_id += 1
        return track
