from __future__ import annotations

import cv2

from ingestion.zones import Zone
from rules.base import ViolationEvent


def draw_evidence_overlay(
    frame,
    event: ViolationEvent,
    zones: list[Zone],
    plate_bbox: tuple[float, float, float, float] | None = None,
    plate_text: str | None = None,
):
    annotated = frame.copy()

    for zone in zones:
        color = (0, 255, 255) if zone.id == event.zone_id else (180, 180, 180)
        points = [(int(x), int(y)) for x, y in zone.points]
        for start, end in zip(points, points[1:] + points[:1], strict=False):
            cv2.line(annotated, start, end, color, 2)
        if zone.label:
            cv2.putText(
                annotated,
                zone.label,
                points[0],
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )

    x1, y1, x2, y2 = [int(value) for value in event.vehicle_bbox]
    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)

    if plate_bbox is not None:
        px1, py1, px2, py2 = [int(value) for value in plate_bbox]
        cv2.rectangle(annotated, (px1, py1), (px2, py2), (255, 0, 0), 2)

    label = f"{event.violation_type} track={event.track_id}"
    if event.signal_state:
        label += f" signal={event.signal_state}"
    if plate_text:
        label += f" plate={plate_text}"

    cv2.putText(
        annotated,
        label,
        (max(0, x1), max(24, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2,
    )
    return annotated
