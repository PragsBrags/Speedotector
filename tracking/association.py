from __future__ import annotations

from math import dist
from typing import Protocol

from ingestion.zones import centroid


class PlateLike(Protocol):
    coords: tuple[float, float, float, float]


def associate_plate_to_vehicle(
    plate_detections: list[PlateLike],
    vehicle_bbox: tuple[float, float, float, float],
) -> PlateLike | None:
    if not plate_detections:
        return None

    x1, y1, x2, y2 = vehicle_bbox
    vehicle_center = centroid(vehicle_bbox)
    inside_vehicle = []

    for plate in plate_detections:
        plate_center = centroid(plate.coords)
        if x1 <= plate_center[0] <= x2 and y1 <= plate_center[1] <= y2:
            inside_vehicle.append(plate)

    candidates = inside_vehicle or plate_detections
    return min(
        candidates,
        key=lambda plate: dist(centroid(plate.coords), vehicle_center),
    )
