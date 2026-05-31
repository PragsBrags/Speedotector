import os
from pathlib import Path

import cv2

from ingestion.video_feed import frame
from ocr.licensePlate import LicensePlateDetection, PaddleInference
from db.database import SessionLocal, create_tables
from db.models import Detection, Video


BASE_DIR = Path(__file__).resolve().parent


def resolve_path(env_name, default_name, docker_path=None):
    env_value = os.getenv(env_name)

    if env_value:
        path = Path(env_value)
        if path.exists():
            return str(path)

    if docker_path and Path(docker_path).exists():
        return docker_path

    local_path = BASE_DIR / default_name
    if local_path.exists():
        return str(local_path)

    raise FileNotFoundError(
        f"{env_name} not found.\n"
        f"Checked:\n"
        f"- ENV: {env_value}\n"
        f"- Docker: {docker_path}\n"
        f"- Local: {local_path}"
    )


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
        plate_text=str(plate_text),
        x1=int(x1),
        y1=int(y1),
        x2=int(x2),
        y2=int(y2),
        crop_width=int(crop_width),
        crop_height=int(crop_height),
    )

    db.add(detection)
    db.commit()
    db.refresh(detection)
    return detection


def main():
    model_path = resolve_path(
        env_name="MODEL_PATH",
        default_name="models/license_plate.pt",
        docker_path="/app/models/license_plate.pt"
    )

    video_path = resolve_path(
        env_name="VIDEO_PATH",
        default_name="test_video.mp4",
        docker_path="/app/test_video.mp4"
    )

    print("MODEL PATH:", model_path)
    print("VIDEO PATH:", video_path)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video at: {video_path}")

    print("Video opened successfully.")

    create_tables()
    db = SessionLocal()

    try:
        video = save_video(db, video_path)
        print(f"Video saved to database with id: {video.id}")

        LPD = LicensePlateDetection(model_path)
        print("License Plate Detection model loaded successfully.")

        PI = PaddleInference()
        print("Paddle OCR model loaded successfully.")

        for frame_data in frame(cap):
            coords = LPD.license_coordinates(frame_data)

            if coords is None:
                continue

            x1, y1, x2, y2 = coords

            os.makedirs("outputs", exist_ok=True)

            cv2.imwrite("outputs/latest_frame.jpg", frame_data)
            cv2.imwrite("outputs/latest_plate.jpg", plate_img)

            plate_img = LPD.crop_into_plate(frame_data, x1, y1, x2, y2)

            if plate_img is None or plate_img.size == 0:
                print("Invalid plate crop, skipped.")
                continue

            print(f"Plate crop size: {plate_img.shape[1]}x{plate_img.shape[0]}px")

            result = PI.ocr_inference(plate_img)

            if result:
                print("OCR result:", result)

                detection = save_detection(
                    db=db,
                    video_id=video.id,
                    plate_text=result,
                    coords=coords,
                    plate_img=plate_img
                )

                print(f"Detection saved to database with id: {detection.id}")
            else:
                print("OCR returned None")

    finally:
        db.close()
        cap.release()
        print("Resources released.")


if __name__ == "__main__":
    main()