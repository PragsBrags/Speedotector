from __future__ import annotations

from ingestion.zones import ZoneType, bbox_intersects_zone
from rules.base import FrameContext, ViolationEvent


class ZebraCrossingRule:
    violation_type = "zebra_crossing_violation"

    def __init__(self):
        self._emitted_track_zone_pairs: set[tuple[int, str]] = set()

    def evaluate(self, frame_context: FrameContext) -> list[ViolationEvent]:
        if frame_context.signal_state is None or frame_context.signal_state.state != "red":
            return []

        events: list[ViolationEvent] = []
        crosswalk_zones = frame_context.zones_by_type(ZoneType.CROSSWALK.value)

        for track in frame_context.tracks:
            for zone in crosswalk_zones:
                key = (track.track_id, zone.id)
                if key in self._emitted_track_zone_pairs:
                    continue
                if not bbox_intersects_zone(track.latest_bbox, zone.points):
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
                        signal_state=frame_context.signal_state.state,
                        reason="vehicle occupied crosswalk while signal was red",
                    )
                )

        return events
