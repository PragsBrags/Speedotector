from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class ZoneType(StrEnum):
    TRAFFIC_LIGHT = "traffic_light"
    STOP_LINE = "stop_line"
    FORBIDDEN_AREA = "forbidden_area"
    CROSSWALK = "crosswalk"
    LANE = "lane"
    PLATE_ROI = "plate_roi"


@dataclass(frozen=True)
class Zone:
    id: str
    type: str
    points: list[tuple[int, int]]
    label: str | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Zone:
        points = payload.get("points")
        if points is None and "bbox" in payload:
            x, y, width, height = payload["bbox"]
            points = [
                (x, y),
                (x + width, y),
                (x + width, y + height),
                (x, y + height),
            ]

        if not points:
            raise ValueError("Zone requires points or bbox")

        return cls(
            id=str(payload["id"]),
            type=str(payload["type"]),
            points=[(int(x), int(y)) for x, y in points],
            label=payload.get("label"),
        )

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        xs = [point[0] for point in self.points]
        ys = [point[1] for point in self.points]
        return min(xs), min(ys), max(xs), max(ys)

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "id": self.id,
            "type": self.type,
            "points": [[x, y] for x, y in self.points],
        }
        if self.label is not None:
            payload["label"] = self.label
        return payload


def centroid(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2, (y1 + y2) / 2


def point_in_polygon(
    point: tuple[float, float], polygon: list[tuple[int, int]]
) -> bool:
    x, y = point
    inside = False
    j = len(polygon) - 1

    for i, current in enumerate(polygon):
        xi, yi = current
        xj, yj = polygon[j]

        if _point_on_segment(point, (xi, yi), (xj, yj)):
            return True

        intersects = (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (
            yj - yi
        ) + xi
        if intersects:
            inside = not inside
        j = i

    return inside


def bbox_intersects_zone(
    bbox: tuple[float, float, float, float], polygon: list[tuple[int, int]]
) -> bool:
    x1, y1, x2, y2 = bbox
    bbox_points = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]

    if any(point_in_polygon(point, polygon) for point in bbox_points):
        return True
    if any(x1 <= x <= x2 and y1 <= y <= y2 for x, y in polygon):
        return True

    polygon_edges = list(zip(polygon, polygon[1:] + polygon[:1], strict=False))
    bbox_edges = list(zip(bbox_points, bbox_points[1:] + bbox_points[:1], strict=False))
    return any(
        _segments_intersect(a1, a2, b1, b2)
        for a1, a2 in polygon_edges
        for b1, b2 in bbox_edges
    )


def line_crossed(
    previous_point: tuple[float, float],
    current_point: tuple[float, float],
    line: list[tuple[int, int]],
) -> bool:
    if len(line) < 2:
        return False
    return _segments_intersect(previous_point, current_point, line[0], line[1])


def _point_on_segment(
    point: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
) -> bool:
    px, py = point
    sx, sy = start
    ex, ey = end
    cross = (py - sy) * (ex - sx) - (px - sx) * (ey - sy)
    if abs(cross) > 1e-9:
        return False
    return min(sx, ex) <= px <= max(sx, ex) and min(sy, ey) <= py <= max(sy, ey)


def _orientation(
    a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]
) -> float:
    return (b[1] - a[1]) * (c[0] - b[0]) - (b[0] - a[0]) * (c[1] - b[1])


def _segments_intersect(
    a1: tuple[float, float],
    a2: tuple[float, float],
    b1: tuple[float, float],
    b2: tuple[float, float],
) -> bool:
    o1 = _orientation(a1, a2, b1)
    o2 = _orientation(a1, a2, b2)
    o3 = _orientation(b1, b2, a1)
    o4 = _orientation(b1, b2, a2)

    if o1 == 0 and _point_on_segment(b1, a1, a2):
        return True
    if o2 == 0 and _point_on_segment(b2, a1, a2):
        return True
    if o3 == 0 and _point_on_segment(a1, b1, b2):
        return True
    if o4 == 0 and _point_on_segment(a2, b1, b2):
        return True

    return (o1 > 0) != (o2 > 0) and (o3 > 0) != (o4 > 0)
