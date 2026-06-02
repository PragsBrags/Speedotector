from __future__ import annotations

from ingestion.zones import ZoneType, point_in_polygon
from rules.base import FrameContext, ViolationEvent


class RestrictedZoneRule:
    violation_type = "restricted_zone_violation"

    def __init__(self):
        self._emitted_track_zone_pairs: set[tuple[int, str]] = set()

    def evaluate(self, frame_context: FrameContext) -> list[ViolationEvent]:
        events: list[ViolationEvent] = []
        forbidden_zones = frame_context.zones_by_type(ZoneType.FORBIDDEN_AREA.value)

        for track in frame_context.tracks:
            for zone in forbidden_zones:
                key = (track.track_id, zone.id)
                if key in self._emitted_track_zone_pairs:
                    continue
                if not point_in_polygon(track.latest_centroid, zone.points):
                    continue

                self._emitted_track_zone_pairs.add(key)
                events.append(
                    ViolationEvent(
                        violation_type=self.violation_type,
                        frame_number=frame_context.frame_number,
                        timestamp_seconds=frame_context.timestamp_seconds,
                        zone_id=zone.id,
                        track_id=track.track_id,
                        vehicle_bbox=track.latest_bbox,
                        signal_state=(
                            frame_context.signal_state.state
                            if frame_context.signal_state
                            else None
                        ),
                        reason="vehicle centroid entered forbidden zone",
                    )
                )

        return events
