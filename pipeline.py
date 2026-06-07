import os

import cv2

from ingestion.video_feed import selected_frames


def process_video(
    video_path,
    model_path=None,
    vehicle_model_path=None,
    roi=None,
    save_to_db=True,
    progress_callback=None,
    include_images=False,
    vehicle_detector=None,
    plate_detector=None,
    ocr=None,
    min_detection_confidence=0.0,
):
    script_dir = os.path.dirname(os.path.abspath(__file__))

    if model_path is None:
        model_path = os.path.join(script_dir, "models", "license_plate.pt")

    if vehicle_model_path is None:
        vehicle_model_path = "yolo26n.pt"

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    db = None
    video = None
    results = []

    try:
        if vehicle_detector is None:
            from ocr.vehicleDetection import VehicleDetection

            vehicle_detector = VehicleDetection(vehicle_model_path)

        if plate_detector is None or ocr is None:
            from ocr.licensePlate import LicensePlateDetection, PaddleInference

            plate_detector = plate_detector or LicensePlateDetection(model_path)
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

            vehicles = vehicle_detector.vehicle_coordinates(frame_image)

            if not vehicles:
                continue

            for vehicle_class, vehicle_confidence, vx1, vy1, vx2, vy2 in vehicles:
                vehicle_x_offset = max(0, int(vx1))
                vehicle_y_offset = max(0, int(vy1))

                vehicle_crop = vehicle_detector.crop_vehicle(
                    frame_image,
                    vx1,
                    vy1,
                    vx2,
                    vy2,
                )

                if vehicle_crop.size == 0:
                    continue

                detection_result = plate_detector.license_coordinates(
                    vehicle_crop,
                    min_confidence=min_detection_confidence,
                )

                if detection_result is None:
                    continue

                roi_plate_coords = detection_result.coords
                detector_confidence = detection_result.confidence
                x1, y1, x2, y2 = roi_plate_coords

                plate_img = plate_detector.crop_into_plate(
                    vehicle_crop,
                    x1,
                    y1,
                    x2,
                    y2,
                )

                ocr_result = ocr.ocr_inference(plate_img)

                if not ocr_result:
                    continue

                plate_text = ocr_result.text

                # Main-branch equivalent coordinate math:
                # ROI offset + vehicle offset + plate offset inside vehicle.
                full_frame_coords = (
                    roi_x + vehicle_x_offset + x1,
                    roi_y + vehicle_y_offset + y1,
                    roi_x + vehicle_x_offset + x2,
                    roi_y + vehicle_y_offset + y2,
                )

                detection_id = None

                if save_to_db and db is not None and video is not None:
                    from db.repository import save_detection

                    detection = save_detection(
                        db,
                        video.id,
                        plate_text,
                        full_frame_coords,
                        plate_img,
                        detector_confidence=detector_confidence,
                        ocr_confidence=ocr_result.confidence,
                    )
                    detection_id = detection.id

                result = {
                    "frame_index": index,
                    "plate_text": plate_text,
                    "coords": full_frame_coords,
                    "roi_coords": roi_plate_coords,
                    "source_frame_number": selected_frame["frame_number"],
                    "roi": selected_frame["roi"],
                    "vehicle_class": vehicle_class,
                    "vehicle_confidence": vehicle_confidence,
                    "vehicle_coords": (
                        roi_x + vx1,
                        roi_y + vy1,
                        roi_x + vx2,
                        roi_y + vy2,
                    ),
                    "crop_shape": tuple(int(value) for value in plate_img.shape),
                    "detector_confidence": detector_confidence,
                    "ocr_confidence": ocr_result.confidence,
                    "ocr_segments": ocr_result.segments,
                    "frame_metadata": frame_metadata,
                    "detection_id": detection_id,
                }

                if include_images:
                    result["plate_img"] = plate_img
                    result["vehicle_img"] = vehicle_crop

                results.append(result)

                if progress_callback:
                    progress_callback(result)

    finally:
        cap.release()
        if db is not None:
            db.close()

    return results