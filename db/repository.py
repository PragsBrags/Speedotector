import os

from db.models import Detection, Video


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
