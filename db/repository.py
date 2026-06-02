import json
import os

from db.models import Detection, Video, Violation, ZoneRecord
from ingestion.zones import Zone
from rules.base import ViolationEvent


def save_video(db, video_path):
    video = Video(
        file_name=os.path.basename(video_path),
        file_path=video_path,
    )
    db.add(video)
    db.commit()
    db.refresh(video)
    return video


def save_detection(
    db,
    video_id,
    plate_text,
    coords,
    plate_img,
    detector_confidence=None,
    ocr_confidence=None,
):
    x1, y1, x2, y2 = coords
    crop_height, crop_width = plate_img.shape[:2]

    detection = Detection(
        video_id=video_id,
        plate_text=plate_text,
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
        crop_width=crop_width,
        crop_height=crop_height,
        detector_confidence=detector_confidence,
        ocr_confidence=ocr_confidence,
    )
    db.add(detection)
    db.commit()
    db.refresh(detection)
    return detection


def save_zone(db, zone: Zone, video_id=None):
    record = ZoneRecord(
        video_id=video_id,
        zone_key=zone.id,
        zone_type=zone.type,
        label=zone.label,
        points_json=json.dumps(zone.points),
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


def save_violation(
    db,
    video_id,
    event: ViolationEvent,
    plate_text=None,
    plate_bbox=None,
    detector_confidence=None,
    ocr_confidence=None,
    signal_confidence=None,
    evidence_frame_path=None,
    plate_crop_path=None,
):
    vehicle_x1, vehicle_y1, vehicle_x2, vehicle_y2 = event.vehicle_bbox
    plate_x1 = plate_y1 = plate_x2 = plate_y2 = None
    if plate_bbox is not None:
        plate_x1, plate_y1, plate_x2, plate_y2 = plate_bbox

    violation = Violation(
        video_id=video_id,
        violation_type=event.violation_type,
        frame_number=event.frame_number,
        timestamp_seconds=event.timestamp_seconds,
        zone_id=event.zone_id,
        track_id=event.track_id,
        vehicle_x1=vehicle_x1,
        vehicle_y1=vehicle_y1,
        vehicle_x2=vehicle_x2,
        vehicle_y2=vehicle_y2,
        plate_text=plate_text,
        plate_x1=plate_x1,
        plate_y1=plate_y1,
        plate_x2=plate_x2,
        plate_y2=plate_y2,
        detector_confidence=detector_confidence,
        ocr_confidence=ocr_confidence,
        signal_state=event.signal_state,
        signal_confidence=signal_confidence,
        evidence_frame_path=evidence_frame_path,
        plate_crop_path=plate_crop_path,
    )
    db.add(violation)
    db.commit()
    db.refresh(violation)
    return violation
