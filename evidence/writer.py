from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2

from evidence.overlays import draw_evidence_overlay
from ingestion.zones import Zone
from rules.base import ViolationEvent


@dataclass(frozen=True)
class EvidenceRecord:
    violation_id: str
    violation_type: str
    frame_number: int
    timestamp_seconds: float | None
    zone_id: str
    track_id: int
    evidence_frame_path: str
    plate_crop_path: str | None
    plate_text: str | None
    detector_confidence: float | None
    ocr_confidence: float | None
    metadata: dict[str, Any]


class EvidenceWriter:
    def __init__(self, output_root: str | Path = "outputs", run_id: str | None = None):
        self.run_id = run_id or uuid.uuid4().hex[:12]
        self.run_dir = Path(output_root) / self.run_id
        self.evidence_dir = self.run_dir / "evidence"
        self.evidence_dir.mkdir(parents=True, exist_ok=True)
        self.records: list[EvidenceRecord] = []

    def write(
        self,
        event: ViolationEvent,
        frame,
        zones: list[Zone],
        plate_crop=None,
        plate_bbox: tuple[float, float, float, float] | None = None,
        plate_text: str | None = None,
        detector_confidence: float | None = None,
        ocr_confidence: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> EvidenceRecord:
        violation_id = f"violation_{len(self.records) + 1:04d}"
        evidence_frame_path = self.evidence_dir / f"{violation_id}_frame.jpg"
        plate_crop_path = (
            self.evidence_dir / f"{violation_id}_plate.jpg"
            if plate_crop is not None
            else None
        )

        annotated = draw_evidence_overlay(
            frame,
            event,
            zones,
            plate_bbox=plate_bbox,
            plate_text=plate_text,
        )
        cv2.imwrite(str(evidence_frame_path), annotated)
        if plate_crop_path is not None:
            cv2.imwrite(str(plate_crop_path), plate_crop)

        record = EvidenceRecord(
            violation_id=violation_id,
            violation_type=event.violation_type,
            frame_number=event.frame_number,
            timestamp_seconds=event.timestamp_seconds,
            zone_id=event.zone_id,
            track_id=event.track_id,
            evidence_frame_path=str(evidence_frame_path),
            plate_crop_path=str(plate_crop_path) if plate_crop_path else None,
            plate_text=plate_text,
            detector_confidence=detector_confidence,
            ocr_confidence=ocr_confidence,
            metadata=metadata or {},
        )
        self.records.append(record)
        return record

    def write_results(self, filename: str = "results.json") -> Path:
        path = self.run_dir / filename
        path.write_text(
            json.dumps([asdict(record) for record in self.records], indent=2),
            encoding="utf-8",
        )
        return path
