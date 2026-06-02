from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any

from ingestion.roi import clamp_roi
from ingestion.zones import Zone, ZoneType

ZONE_TYPES = {
    ZoneType.TRAFFIC_LIGHT.value: {
        "label": "Traffic light ROI",
        "drawing_mode": "rect",
        "color": "#ef4444",
        "fill": "rgba(239, 68, 68, 0.18)",
    },
    ZoneType.STOP_LINE.value: {
        "label": "Stop line",
        "drawing_mode": "line",
        "color": "#f97316",
        "fill": "rgba(249, 115, 22, 0.12)",
    },
    ZoneType.FORBIDDEN_AREA.value: {
        "label": "Forbidden area",
        "drawing_mode": "rect",
        "color": "#a855f7",
        "fill": "rgba(168, 85, 247, 0.16)",
    },
    ZoneType.CROSSWALK.value: {
        "label": "Crosswalk",
        "drawing_mode": "rect",
        "color": "#0ea5e9",
        "fill": "rgba(14, 165, 233, 0.16)",
    },
    ZoneType.LANE.value: {
        "label": "Lane",
        "drawing_mode": "rect",
        "color": "#22c55e",
        "fill": "rgba(34, 197, 94, 0.14)",
    },
    ZoneType.PLATE_ROI.value: {
        "label": "Plate ROI",
        "drawing_mode": "rect",
        "color": "#64748b",
        "fill": "rgba(100, 116, 139, 0.14)",
    },
}

RULE_OPTIONS = {
    "restricted_zone_violation": "Restricted zone",
    "red_light_violation": "Red light",
    "zebra_crossing_violation": "Crosswalk encroachment",
}


@dataclass(frozen=True)
class ZoneValidation:
    valid: bool
    messages: list[str]


def canvas_object_to_zone(
    canvas_object: dict[str, Any],
    zone_type: str,
    zone_id: str,
    label: str | None,
    scale: float,
    frame_width: int,
    frame_height: int,
) -> Zone:
    if zone_type == ZoneType.STOP_LINE.value:
        points = _line_points(canvas_object, scale, frame_width, frame_height)
    else:
        points = _rect_points(canvas_object, scale, frame_width, frame_height)

    return Zone(id=zone_id, type=zone_type, points=points, label=label)


def zones_to_canvas_objects(zones: list[Zone], scale: float) -> list[dict[str, Any]]:
    objects = []
    for zone in zones:
        style = ZONE_TYPES.get(zone.type, ZONE_TYPES[ZoneType.LANE.value])
        if zone.type == ZoneType.STOP_LINE.value and len(zone.points) >= 2:
            (x1, y1), (x2, y2) = zone.points[:2]
            objects.append(
                {
                    "type": "line",
                    "x1": x1 * scale,
                    "y1": y1 * scale,
                    "x2": x2 * scale,
                    "y2": y2 * scale,
                    "stroke": style["color"],
                    "strokeWidth": 4,
                }
            )
            continue

        x1, y1, x2, y2 = zone.bbox
        objects.append(
            {
                "type": "rect",
                "left": x1 * scale,
                "top": y1 * scale,
                "width": (x2 - x1) * scale,
                "height": (y2 - y1) * scale,
                "fill": style["fill"],
                "stroke": style["color"],
                "strokeWidth": 3,
            }
        )

    return objects


def validate_zone_setup(zones: list[Zone], enabled_rules: list[str]) -> ZoneValidation:
    messages: list[str] = []
    zone_types = {zone.type for zone in zones}

    if "restricted_zone_violation" in enabled_rules and ZoneType.FORBIDDEN_AREA.value not in zone_types:
        messages.append("Restricted-zone detection needs at least one forbidden area.")
    if "red_light_violation" in enabled_rules:
        if ZoneType.TRAFFIC_LIGHT.value not in zone_types:
            messages.append("Red-light detection needs a traffic light ROI.")
        if ZoneType.STOP_LINE.value not in zone_types:
            messages.append("Red-light detection needs a stop line.")
    if "zebra_crossing_violation" in enabled_rules and ZoneType.CROSSWALK.value not in zone_types:
        messages.append("Crosswalk encroachment needs at least one crosswalk zone.")

    if not enabled_rules:
        messages.append("Choose at least one violation type.")

    return ZoneValidation(valid=not messages, messages=messages)


def zone_config_json(
    video_path: str,
    zones: list[Zone],
    enabled_rules: list[str],
    pre_event_frames: int,
    post_event_frames: int,
) -> str:
    payload = {
        "video_path": video_path,
        "zones": [zone.as_dict() for zone in zones],
        "enabled_rules": enabled_rules,
        "events": {
            "pre_event_frames": pre_event_frames,
            "post_event_frames": post_event_frames,
        },
    }
    return json.dumps(payload, indent=2)


def zone_records(zones: list[Zone]) -> list[dict[str, Any]]:
    records = []
    for zone in zones:
        x1, y1, x2, y2 = zone.bbox
        records.append(
            {
                "id": zone.id,
                "type": zone.type,
                "label": zone.label or "",
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "points": zone.points,
            }
        )
    return records


def zone_validation_dict(validation: ZoneValidation) -> dict[str, Any]:
    return asdict(validation)


def _rect_points(
    canvas_object: dict[str, Any],
    scale: float,
    frame_width: int,
    frame_height: int,
) -> list[tuple[int, int]]:
    left = canvas_object.get("left", 0) / scale
    top = canvas_object.get("top", 0) / scale
    width = canvas_object.get("width", 1) * canvas_object.get("scaleX", 1) / scale
    height = canvas_object.get("height", 1) * canvas_object.get("scaleY", 1) / scale
    x, y, width, height = clamp_roi(left, top, width, height, frame_width, frame_height)
    return [(x, y), (x + width, y), (x + width, y + height), (x, y + height)]


def _line_points(
    canvas_object: dict[str, Any],
    scale: float,
    frame_width: int,
    frame_height: int,
) -> list[tuple[int, int]]:
    if "x1" in canvas_object and "x2" in canvas_object:
        x1 = canvas_object.get("x1", 0) / scale
        y1 = canvas_object.get("y1", 0) / scale
        x2 = canvas_object.get("x2", 1) / scale
        y2 = canvas_object.get("y2", 1) / scale
    else:
        left = canvas_object.get("left", 0) / scale
        top = canvas_object.get("top", 0) / scale
        width = canvas_object.get("width", 1) * canvas_object.get("scaleX", 1) / scale
        height = canvas_object.get("height", 1) * canvas_object.get("scaleY", 1) / scale
        x1 = left
        y1 = top
        x2 = left + width
        y2 = top + height

    start = clamp_roi(x1, y1, 1, 1, frame_width, frame_height)[:2]
    end = clamp_roi(x2, y2, 1, 1, frame_width, frame_height)[:2]
    return [start, end]
