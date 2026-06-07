from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2

from detection.vehicle import VehicleDetector
from evidence.writer import EvidenceRecord, EvidenceWriter
from ingestion.zones import Zone, ZoneType
from rules.base import FrameContext, ViolationEvent
from rules.red_light import RedLightRule
from rules.restricted_zone import RestrictedZoneRule
from rules.traffic_signal import SignalState, TrafficSignalDetector
from rules.zebra_crossing import ZebraCrossingRule
from tracking.association import associate_plate_to_vehicle
from tracking.centroid_tracker import CentroidTracker


@dataclass(frozen=True)
class ScoredFrame:
    frame_number: int
    timestamp_seconds: float | None
    frame: Any
    mask: Any


@dataclass
class PendingEvidence:
    event: ViolationEvent
    close_frame: int
    best_frame: Any
    best_frame_number: int
    best_timestamp_seconds: float | None
    best_score: float
    best_motion_area: float
    best_sharpness: float


@dataclass(frozen=True)
class PlateCandidate:
    coords: tuple[float, float, float, float]
    confidence: float


def process_zone_violations(
    video_path: str,
    zones: list[Zone],
    enabled_rules: list[str],
    plate_model_path: str | None = None,
    vehicle_detector=None,
    plate_detector=None,
    ocr=None,
    tracker: CentroidTracker | None = None,
    signal_detector: TrafficSignalDetector | None = None,
    evidence_writer: EvidenceWriter | None = None,
    progress_callback=None,
    max_frames: int | None = None,
    min_vehicle_confidence: float = 0.25,
    run_plate_ocr: bool = True,
    pre_event_frames: int = 10,
    post_event_frames: int = 10,
    save_to_db: bool = False,
    manual_signal_intervals: list[tuple[int, int]] | None = None,
):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    vehicle_detector = vehicle_detector or VehicleDetector()
    tracker = tracker or CentroidTracker()
    signal_detector = signal_detector or TrafficSignalDetector()
    evidence_writer = evidence_writer or EvidenceWriter()
    rule_instances = _build_rules(enabled_rules)
    traffic_light_roi = _first_zone_bbox(zones, ZoneType.TRAFFIC_LIGHT.value)
    manual_signal_intervals = manual_signal_intervals or []
    background_subtractor = cv2.createBackgroundSubtractorMOG2(
        history=100, varThreshold=16
    )
    recent_frames: deque[ScoredFrame] = deque(maxlen=max(pre_event_frames, 0) + 1)
    pending_evidence: list[PendingEvidence] = []
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    events = []
    frame_number = -1
    db = None
    video_record = None

    try:
        if save_to_db:
            from db.database import SessionLocal, create_tables
            from db.repository import save_video, save_zone

            create_tables()
            db = SessionLocal()
            video_record = save_video(db, video_path)
            for zone in zones:
                save_zone(db, zone, video_id=video_record.id)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_number += 1
            if max_frames is not None and frame_number >= max_frames:
                break

            timestamp_seconds = frame_number / fps if fps and fps > 0 else None
            motion_mask = background_subtractor.apply(frame)
            recent_frames.append(
                ScoredFrame(
                    frame_number=frame_number,
                    timestamp_seconds=timestamp_seconds,
                    frame=frame.copy(),
                    mask=motion_mask.copy(),
                )
            )

            vehicle_detections = vehicle_detector.detect(
                frame, min_confidence=min_vehicle_confidence
            )
            tracks = tracker.update(vehicle_detections, frame_number)
            signal_state = _signal_state_for_frame(
                frame=frame,
                frame_number=frame_number,
                traffic_light_roi=traffic_light_roi,
                signal_detector=signal_detector,
                manual_signal_intervals=manual_signal_intervals,
            )

            context = FrameContext(
                frame_number=frame_number,
                timestamp_seconds=timestamp_seconds,
                frame=frame,
                zones=zones,
                vehicle_detections=vehicle_detections,
                tracks=tracks,
                signal_state=signal_state,
            )

            for rule in rule_instances:
                for event in rule.evaluate(context):
                    pending_evidence.append(
                        _open_pending_evidence(
                            event=event,
                            recent_frames=recent_frames,
                            post_event_frames=post_event_frames,
                        )
                    )

            for pending in list(pending_evidence):
                _maybe_update_pending(pending, recent_frames[-1])
                if frame_number >= pending.close_frame:
                    events.append(
                        _finalize_pending_evidence(
                            pending=pending,
                            zones=zones,
                            plate_model_path=plate_model_path,
                            plate_detector=plate_detector,
                            ocr=ocr,
                            run_plate_ocr=run_plate_ocr,
                            db=db,
                            video_id=video_record.id if video_record else None,
                            evidence_writer=evidence_writer,
                        )
                    )
                    pending_evidence.remove(pending)

            if progress_callback:
                progress_callback(
                    {
                        "frame_number": frame_number,
                        "vehicle_count": len(vehicle_detections),
                        "active_tracks": len(tracks),
                        "signal_state": signal_state.state if signal_state else None,
                        "events": len(events),
                    }
                )

        for pending in pending_evidence:
            events.append(
                _finalize_pending_evidence(
                    pending=pending,
                    zones=zones,
                    plate_model_path=plate_model_path,
                    plate_detector=plate_detector,
                    ocr=ocr,
                    run_plate_ocr=run_plate_ocr,
                    db=db,
                    video_id=video_record.id if video_record else None,
                    evidence_writer=evidence_writer,
                )
            )
    finally:
        cap.release()
        if db is not None:
            db.close()

    results_path = evidence_writer.write_results()
    return {
        "events": events,
        "results_path": str(results_path),
        "run_dir": str(Path(results_path).parent),
        "processed_frames": frame_number + 1,
    }


def _build_rules(enabled_rules: list[str]):
    rule_map = {
        "restricted_zone_violation": RestrictedZoneRule,
        "red_light_violation": RedLightRule,
        "zebra_crossing_violation": ZebraCrossingRule,
    }
    return [rule_map[rule_name]() for rule_name in enabled_rules if rule_name in rule_map]


def _first_zone_bbox(zones: list[Zone], zone_type: str):
    for zone in zones:
        if zone.type == zone_type:
            x1, y1, x2, y2 = zone.bbox
            return x1, y1, x2 - x1, y2 - y1
    return None


def _signal_state_for_frame(
    *,
    frame,
    frame_number: int,
    traffic_light_roi,
    signal_detector: TrafficSignalDetector,
    manual_signal_intervals: list[tuple[int, int]],
) -> SignalState | None:
    for start, end in manual_signal_intervals:
        if start <= frame_number <= end:
            return SignalState("red", 1.0, 1.0, 0.0, 0.0, frame_number)

    if manual_signal_intervals:
        return SignalState("green", 1.0, 0.0, 0.0, 1.0, frame_number)

    if traffic_light_roi is not None:
        return signal_detector.detect(frame, traffic_light_roi, frame_number)

    return None


def _open_pending_evidence(
    *,
    event: ViolationEvent,
    recent_frames: deque[ScoredFrame],
    post_event_frames: int,
) -> PendingEvidence:
    best_scored_frame = recent_frames[-1]
    best_score = -1.0
    best_motion_area = 0.0
    best_sharpness = 0.0
    for scored_frame in recent_frames:
        score, motion_area, sharpness = _frame_score(
            scored_frame.frame, scored_frame.mask, event.vehicle_bbox
        )
        if score > best_score:
            best_scored_frame = scored_frame
            best_score = score
            best_motion_area = motion_area
            best_sharpness = sharpness

    return PendingEvidence(
        event=event,
        close_frame=event.frame_number + max(post_event_frames, 0),
        best_frame=best_scored_frame.frame.copy(),
        best_frame_number=best_scored_frame.frame_number,
        best_timestamp_seconds=best_scored_frame.timestamp_seconds,
        best_score=best_score,
        best_motion_area=best_motion_area,
        best_sharpness=best_sharpness,
    )


def _maybe_update_pending(pending: PendingEvidence, scored_frame: ScoredFrame) -> None:
    score, motion_area, sharpness = _frame_score(
        scored_frame.frame, scored_frame.mask, pending.event.vehicle_bbox
    )
    if score <= pending.best_score:
        return

    pending.best_frame = scored_frame.frame.copy()
    pending.best_frame_number = scored_frame.frame_number
    pending.best_timestamp_seconds = scored_frame.timestamp_seconds
    pending.best_score = score
    pending.best_motion_area = motion_area
    pending.best_sharpness = sharpness


def _finalize_pending_evidence(
    *,
    pending: PendingEvidence,
    zones: list[Zone],
    plate_model_path: str | None,
    plate_detector,
    ocr,
    run_plate_ocr: bool,
    db,
    video_id: int | None,
    evidence_writer: EvidenceWriter,
) -> dict[str, Any]:
    plate_result = _read_plate(
        pending.best_frame,
        plate_model_path=plate_model_path,
        plate_detector=plate_detector,
        ocr=ocr,
        run_plate_ocr=run_plate_ocr,
        vehicle_bbox=pending.event.vehicle_bbox,
    )
    record = evidence_writer.write(
        pending.event,
        pending.best_frame,
        zones,
        plate_crop=plate_result["plate_crop"],
        plate_bbox=plate_result["plate_bbox"],
        plate_text=plate_result["plate_text"],
        detector_confidence=plate_result["detector_confidence"],
        ocr_confidence=plate_result["ocr_confidence"],
        metadata={
            "reason": pending.event.reason,
            "evidence_frame_number": pending.best_frame_number,
            "evidence_timestamp_seconds": pending.best_timestamp_seconds,
            "motion_area": pending.best_motion_area,
            "sharpness": pending.best_sharpness,
            "score": pending.best_score,
        },
    )

    violation_id = None
    if db is not None and video_id is not None:
        from db.repository import save_violation

        violation = save_violation(
            db,
            video_id,
            pending.event,
            plate_text=plate_result["plate_text"],
            plate_bbox=plate_result["plate_bbox"],
            detector_confidence=plate_result["detector_confidence"],
            ocr_confidence=plate_result["ocr_confidence"],
            signal_confidence=1.0 if pending.event.signal_state else None,
            evidence_frame_path=record.evidence_frame_path,
            plate_crop_path=record.plate_crop_path,
        )
        violation_id = violation.id

    return _event_payload(
        pending=pending,
        plate_result=plate_result,
        record=record,
        violation_id=violation_id,
    )


def _event_payload(
    *,
    pending: PendingEvidence,
    plate_result: dict[str, Any],
    record: EvidenceRecord,
    violation_id: int | None,
) -> dict[str, Any]:
    return {
        "violation_type": pending.event.violation_type,
        "frame_number": pending.event.frame_number,
        "timestamp_seconds": pending.event.timestamp_seconds,
        "evidence_frame_number": pending.best_frame_number,
        "evidence_timestamp_seconds": pending.best_timestamp_seconds,
        "zone_id": pending.event.zone_id,
        "track_id": pending.event.track_id,
        "vehicle_bbox": pending.event.vehicle_bbox,
        "signal_state": pending.event.signal_state,
        "reason": pending.event.reason,
        "motion_area": pending.best_motion_area,
        "sharpness": pending.best_sharpness,
        "score": pending.best_score,
        "plate_text": plate_result["plate_text"],
        "plate_bbox": plate_result["plate_bbox"],
        "detector_confidence": plate_result["detector_confidence"],
        "ocr_confidence": plate_result["ocr_confidence"],
        "evidence_frame_path": record.evidence_frame_path,
        "plate_crop_path": record.plate_crop_path,
        "db_violation_id": violation_id,
    }


def _frame_score(frame, motion_mask, vehicle_bbox) -> tuple[float, float, float]:
    crop, mask_crop = _crop_frame_and_mask(frame, motion_mask, vehicle_bbox)
    if crop.size == 0:
        return 0.0, 0.0, 0.0

    contours, _ = cv2.findContours(mask_crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    motion_area = float(sum(cv2.contourArea(contour) for contour in contours))
    sharpness = float(cv2.Laplacian(crop, cv2.CV_64F).var())
    return motion_area * sharpness, motion_area, sharpness


def _crop_frame_and_mask(frame, motion_mask, bbox):
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(width - 1, int(x1)))
    y1 = max(0, min(height - 1, int(y1)))
    x2 = max(x1 + 1, min(width, int(x2)))
    y2 = max(y1 + 1, min(height, int(y2)))
    return frame[y1:y2, x1:x2], motion_mask[y1:y2, x1:x2]


def _read_plate(
    frame,
    plate_model_path: str | None,
    plate_detector,
    ocr,
    run_plate_ocr: bool,
    vehicle_bbox: tuple[float, float, float, float] | None = None,
):
    empty_result = {
        "plate_crop": None,
        "plate_bbox": None,
        "plate_text": None,
        "detector_confidence": None,
        "ocr_confidence": None,
    }
    if not run_plate_ocr:
        return empty_result

    if plate_detector is None or ocr is None:
        if plate_model_path is None:
            return empty_result
        from ocr.licensePlate import LicensePlateDetection, PaddleInference

        plate_detector = plate_detector or LicensePlateDetection(plate_model_path)
        ocr = ocr or PaddleInference()

    candidates = _plate_candidates(frame, plate_detector)
    if not candidates:
        return empty_result

    detection = associate_plate_to_vehicle(candidates, vehicle_bbox) if vehicle_bbox else None
    detection = detection or max(candidates, key=lambda candidate: candidate.confidence)
    plate_crop = plate_detector.crop_into_plate(frame, *detection.coords)
    ocr_result = ocr.ocr_inference(plate_crop)
    return {
        "plate_crop": plate_crop,
        "plate_bbox": detection.coords,
        "plate_text": ocr_result.text if ocr_result else None,
        "detector_confidence": detection.confidence,
        "ocr_confidence": ocr_result.confidence if ocr_result else None,
    }


def _plate_candidates(frame, plate_detector) -> list[PlateCandidate]:
    if hasattr(plate_detector, "license_candidates"):
        return list(plate_detector.license_candidates(frame))

    if hasattr(plate_detector, "model"):
        candidates: list[PlateCandidate] = []
        results = plate_detector.model([frame], stream=True)
        for result in results:
            for box in result.boxes:
                confidence = float(box.conf)
                x1, y1, x2, y2 = box.xyxy.numpy()[0]
                candidates.append(
                    PlateCandidate(
                        coords=(float(x1), float(y1), float(x2), float(y2)),
                        confidence=confidence,
                    )
                )
        return candidates

    detection = plate_detector.license_coordinates(frame)
    if detection is None:
        return []
    return [PlateCandidate(coords=detection.coords, confidence=detection.confidence)]
