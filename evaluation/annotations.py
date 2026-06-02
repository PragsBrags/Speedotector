from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ingestion.zones import Zone


@dataclass(frozen=True)
class ViolationAnnotation:
    id: str
    type: str
    frame_start: int
    frame_end: int
    plate_text: str | None = None
    vehicle_bbox: tuple[float, float, float, float] | None = None


@dataclass(frozen=True)
class VideoAnnotation:
    video_id: str
    video_path: str
    fps: float | None
    zones: list[Zone]
    violations: list[ViolationAnnotation]


def load_annotation(path: str | Path) -> VideoAnnotation:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return VideoAnnotation(
        video_id=str(payload["video_id"]),
        video_path=str(payload["video_path"]),
        fps=payload.get("fps"),
        zones=[Zone.from_dict(zone) for zone in payload.get("zones", [])],
        violations=[
            _violation_from_dict(violation)
            for violation in payload.get("violations", [])
        ],
    )


def _violation_from_dict(payload: dict[str, Any]) -> ViolationAnnotation:
    bbox = payload.get("vehicle_bbox")
    return ViolationAnnotation(
        id=str(payload["id"]),
        type=str(payload["type"]),
        frame_start=int(payload["frame_start"]),
        frame_end=int(payload["frame_end"]),
        plate_text=payload.get("plate_text"),
        vehicle_bbox=tuple(bbox) if bbox is not None else None,
    )
