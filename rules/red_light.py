from __future__ import annotations

from ingestion.zones import ZoneType, line_crossed
from rules.base import FrameContext, ViolationEvent


class RedLightRule:
    violation_type = "red_light_violation"

    def __init__(self):
        self._emitted_track_ids: set[int] = set()

    def evaluate(self, frame_context: FrameContext) -> list[ViolationEvent]:
        if frame_context.signal_state is None or frame_context.signal_state.state != "red":
            return []

        events: list[ViolationEvent] = []
        stop_lines = frame_context.zones_by_type(ZoneType.STOP_LINE.value)

        for track in frame_context.tracks:
            if track.track_id in self._emitted_track_ids:
                continue
            if len(track.centroid_history) < 2:
                continue

            previous_point = track.centroid_history[-2]
            current_point = track.centroid_history[-1]
            for stop_line in stop_lines:
                if not line_crossed(previous_point, current_point, stop_line.points):
                    continue

                self._emitted_track_ids.add(track.track_id)
                events.append(
                    ViolationEvent(
                        violation_type=self.violation_type,
                        frame_number=frame_context.frame_number,
                        timestamp_seconds=frame_context.timestamp_seconds,
                        zone_id=stop_line.id,
                        track_id=track.track_id,
                        vehicle_bbox=track.latest_bbox,
                        signal_state=frame_context.signal_state.state,
                        reason="vehicle track crossed stop line while signal was red",
                    )
                )

        return events
