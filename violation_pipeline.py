from __future__ import annotations

from pathlib import Path

import cv2

from detection.vehicle import VehicleDetector
from evidence.writer import EvidenceWriter
from ingestion.zones import Zone, ZoneType
from rules.base import FrameContext
from rules.red_light import RedLightRule
from rules.restricted_zone import RestrictedZoneRule
from rules.traffic_signal import TrafficSignalDetector
from rules.zebra_crossing import ZebraCrossingRule
from tracking.centroid_tracker import CentroidTracker


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
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    events = []
    frame_number = -1

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_number += 1
            if max_frames is not None and frame_number >= max_frames:
                break

            timestamp_seconds = frame_number / fps if fps and fps > 0 else None
            vehicle_detections = vehicle_detector.detect(
                frame, min_confidence=min_vehicle_confidence
            )
            tracks = tracker.update(vehicle_detections, frame_number)
            signal_state = None
            if traffic_light_roi is not None:
                signal_state = signal_detector.detect(
                    frame, traffic_light_roi, frame_number
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
                    plate_result = _read_plate(
                        frame,
                        plate_model_path=plate_model_path,
                        plate_detector=plate_detector,
                        ocr=ocr,
                        run_plate_ocr=run_plate_ocr,
                    )
                    record = evidence_writer.write(
                        event,
                        frame,
                        zones,
                        plate_crop=plate_result["plate_crop"],
                        plate_bbox=plate_result["plate_bbox"],
                        plate_text=plate_result["plate_text"],
                        detector_confidence=plate_result["detector_confidence"],
                        ocr_confidence=plate_result["ocr_confidence"],
                        metadata={"reason": event.reason},
                    )
                    events.append(
                        {
                            "violation_type": event.violation_type,
                            "frame_number": event.frame_number,
                            "timestamp_seconds": event.timestamp_seconds,
                            "zone_id": event.zone_id,
                            "track_id": event.track_id,
                            "vehicle_bbox": event.vehicle_bbox,
                            "signal_state": event.signal_state,
                            "reason": event.reason,
                            "plate_text": plate_result["plate_text"],
                            "plate_bbox": plate_result["plate_bbox"],
                            "detector_confidence": plate_result["detector_confidence"],
                            "ocr_confidence": plate_result["ocr_confidence"],
                            "evidence_frame_path": record.evidence_frame_path,
                            "plate_crop_path": record.plate_crop_path,
                        }
                    )

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
    finally:
        cap.release()

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


def _read_plate(
    frame,
    plate_model_path: str | None,
    plate_detector,
    ocr,
    run_plate_ocr: bool,
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

    detection = plate_detector.license_coordinates(frame)
    if detection is None:
        return empty_result

    plate_crop = plate_detector.crop_into_plate(frame, *detection.coords)
    ocr_result = ocr.ocr_inference(plate_crop)
    return {
        "plate_crop": plate_crop,
        "plate_bbox": detection.coords,
        "plate_text": ocr_result.text if ocr_result else None,
        "detector_confidence": detection.confidence,
        "ocr_confidence": ocr_result.confidence if ocr_result else None,
    }
