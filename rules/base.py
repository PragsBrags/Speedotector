from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from detection.vehicle import VehicleDetectionResult
from ingestion.zones import Zone
from rules.traffic_signal import SignalState
from tracking.centroid_tracker import Track


@dataclass(frozen=True)
class FrameContext:
    frame_number: int
    timestamp_seconds: float | None
    frame: object | None
    zones: list[Zone]
    vehicle_detections: list[VehicleDetectionResult]
    tracks: list[Track]
    signal_state: SignalState | None = None

    def zones_by_type(self, zone_type: str) -> list[Zone]:
        return [zone for zone in self.zones if zone.type == zone_type]


@dataclass(frozen=True)
class ViolationEvent:
    violation_type: str
    frame_number: int
    timestamp_seconds: float | None
    zone_id: str
    track_id: int
    vehicle_bbox: tuple[float, float, float, float]
    signal_state: str | None
    reason: str


class ViolationRule(Protocol):
    def evaluate(self, frame_context: FrameContext) -> list[ViolationEvent]:
        ...
