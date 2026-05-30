import os

import cv2

from db.database import SessionLocal, create_tables
from db.models import Detection, Video
from ingestion.video_feed import selected_frames
from ocr.licensePlate import LicensePlateDetection, PaddleInference


def save_video(db, video_path):
    video = Video(
        file_name=os.path.basename(video_path),
        file_path=video_path,
    )
    db.add(video)
    db.commit()
    db.refresh(video)
    return video


def save_detection(db, video_id, plate_text, coords, plate_img):
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
    )
    db.add(detection)
    db.commit()
    db.refresh(detection)
    return detection


def process_video(
    video_path, model_path=None, roi=None, save_to_db=True, progress_callback=None
):
    script_dir = os.path.dirname(os.path.abspath(__file__))

    if model_path is None:
        model_path = os.path.join(script_dir, "models", "license_plate.pt")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Could not open video.")

    db = None
    video = None
    results = []

    try:
        detector = LicensePlateDetection(model_path)
        ocr = PaddleInference()

        if save_to_db:
            create_tables()
            db = SessionLocal()
            video = save_video(db, video_path)

        for index, selected_frame in enumerate(selected_frames(cap, roi=roi)):
            frame_image = selected_frame["image"]
            roi_x, roi_y, _, _ = selected_frame["roi"]

            roi_coords = detector.license_coordinates(frame_image)
            if roi_coords is None:
                continue

            x1, y1, x2, y2 = roi_coords
            plate_img = detector.crop_into_plate(frame_image, x1, y1, x2, y2)
            plate_text = ocr.ocr_inference(plate_img)

            if not plate_text:
                continue

            coords = (x1 + roi_x, y1 + roi_y, x2 + roi_x, y2 + roi_y)
            detection_id = None
            if save_to_db and db is not None and video is not None:
                detection = save_detection(db, video.id, plate_text, coords, plate_img)
                detection_id = detection.id

            result = {
                "frame_index": index,
                "plate_text": plate_text,
                "coords": coords,
                "roi_coords": roi_coords,
                "source_frame_number": selected_frame["frame_number"],
                "roi": selected_frame["roi"],
                "crop_shape": plate_img.shape,
                "detection_id": detection_id,
                "plate_img": plate_img,
            }

            results.append(result)

            if progress_callback:
                progress_callback(result)

    finally:
        cap.release()
        if db is not None:
            db.close()

    return results
