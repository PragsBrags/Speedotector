import os
from ingestion.video_feed import frame
from ingestion.redlight import TrafficLightDetection
from ocr.licensePlate import LicensePlateDetection, PaddleInference
import cv2
from db.database import SessionLocal, create_tables
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

script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, 'models/license_plate.pt')
video_path = os.path.join(script_dir, 'test2.mp4')

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()
if cap:
    print("Video opened successfully.")

create_tables()
db = SessionLocal()
video = save_video(db, video_path)
print(f"Video saved to database with id: {video.id}")

LPD = LicensePlateDetection(model_path)
if LPD :
    print("License Plate Detection model loaded successfully.")
PI = PaddleInference()
if PI:
    print("Paddle OCR model loaded successfully.")

TLD = TrafficLightDetection(red_ratio_threshold = 0.03)
try:
    for frames, full_frame, traffic_light_rects in frame(cap):
        if not TLD.is_red(full_frame, traffic_light_rects):
            continue

        coords = LPD.license_coordinates(frames)
        if coords is None:
            continue  # Skip this frame if no plate detected

        x1, y1, x2, y2 = coords

        plate_img = LPD.crop_into_plate(frames, x1, y1, x2, y2)
        print(f"Plate crop size: {plate_img.shape[1]}x{plate_img.shape[0]}px")
        res = PI.ocr_inference(plate_img)
        if res:
            print("OCR result:\n",res)
            detection = save_detection(db, video.id, res, coords, plate_img)
            print(f"Detection saved to database with id: {detection.id}")
        else:
            print("OCR returned None")
finally:
    db.close()
    cap.release()
