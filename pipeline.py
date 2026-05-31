import os

import cv2

from ingestion.video_feed import selected_frames


def process_video(
    video_path,
    model_path=None,
    roi=None,
    save_to_db=True,
    progress_callback=None,
    include_images=False,
    detector=None,
    ocr=None,
    min_detection_confidence=0.0,
):
    script_dir = os.path.dirname(os.path.abspath(__file__))

    if model_path is None:
        model_path = os.path.join(script_dir, "models", "license_plate.pt")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    db = None
    video = None
    results = []

    try:
        if detector is None or ocr is None:
            from ocr.licensePlate import LicensePlateDetection, PaddleInference

            detector = detector or LicensePlateDetection(model_path)
            ocr = ocr or PaddleInference()

        if save_to_db:
            from db.database import SessionLocal, create_tables
            from db.repository import save_video

            create_tables()
            db = SessionLocal()
            video = save_video(db, video_path)

        for index, selected_frame in enumerate(selected_frames(cap, roi=roi)):
            frame_image = selected_frame["image"]
            roi_x, roi_y, _, _ = selected_frame["roi"]
            frame_metadata = {
                key: value
                for key, value in selected_frame.items()
                if key not in {"image", "frame_number", "roi"}
            }

            detection_result = detector.license_coordinates(
                frame_image,
                min_confidence=min_detection_confidence,
            )
            if detection_result is None:
                continue

            roi_coords = detection_result.coords
            detector_confidence = detection_result.confidence
            x1, y1, x2, y2 = roi_coords
            plate_img = detector.crop_into_plate(frame_image, x1, y1, x2, y2)
            ocr_result = ocr.ocr_inference(plate_img)

            if not ocr_result:
                continue

            plate_text = ocr_result.text
            coords = (x1 + roi_x, y1 + roi_y, x2 + roi_x, y2 + roi_y)
            detection_id = None
            if save_to_db and db is not None and video is not None:
                from db.repository import save_detection

                detection = save_detection(
                    db,
                    video.id,
                    plate_text,
                    coords,
                    plate_img,
                    detector_confidence=detector_confidence,
                    ocr_confidence=ocr_result.confidence,
                )
                detection_id = detection.id

            result = {
                "frame_index": index,
                "plate_text": plate_text,
                "coords": coords,
                "roi_coords": roi_coords,
                "source_frame_number": selected_frame["frame_number"],
                "roi": selected_frame["roi"],
                "crop_shape": tuple(int(value) for value in plate_img.shape),
                "detector_confidence": detector_confidence,
                "ocr_confidence": ocr_result.confidence,
                "ocr_segments": ocr_result.segments,
                "frame_metadata": frame_metadata,
                "detection_id": detection_id,
            }
            if include_images:
                result["plate_img"] = plate_img

            results.append(result)

            if progress_callback:
                progress_callback(result)

    finally:
        cap.release()
        if db is not None:
            db.close()

    return results
