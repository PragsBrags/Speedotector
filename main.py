import os
from ingestion.video_feed import frame
from ocr.vehicleDetection import VehicleDetection
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
vehicle_model_path = 'yolo26n.pt'
model_path = os.path.join(script_dir, 'models/license_plate.pt')
video_path = os.path.join(script_dir, 'test_video.mp4')

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

VD = VehicleDetection(vehicle_model_path)
if VD :
    print("Vehicle Detection model loaded successfully.")
LPD = LicensePlateDetection(model_path)
if LPD :
    print("License Plate Detection model loaded successfully.")
PI = PaddleInference()
if PI:
    print("Paddle OCR model loaded successfully.")

try:
    for frames in frame(cap):
        vehicles = VD.vehicle_coordinates(frames)
        if not vehicles:
            continue

        for vehicle_class, vehicle_conf, vx1, vy1, vx2, vy2 in vehicles:
            vehicle_x_offset = max(0, int(vx1))
            vehicle_y_offset = max(0, int(vy1))
            vehicle_crop = VD.crop_vehicle(frames, vx1, vy1, vx2, vy2)
            if vehicle_crop.size == 0:
                continue

            coords = LPD.license_coordinates(vehicle_crop)
            if coords is None:
                continue

            x1, y1, x2, y2 = coords

            plate_img = LPD.crop_into_plate(vehicle_crop, x1, y1, x2, y2)
            print(f"Plate crop size: {plate_img.shape[1]}x{plate_img.shape[0]}px")
            res = PI.ocr_inference(plate_img)
            if res:
                frame_coords = (
                    vehicle_x_offset + x1,
                    vehicle_y_offset + y1,
                    vehicle_x_offset + x2,
                    vehicle_y_offset + y2,
                )
                print(f"OCR result for {vehicle_class} ({vehicle_conf:.2%}):\n",res)
                detection = save_detection(db, video.id, res, frame_coords, plate_img)
                print(f"Detection saved to database with id: {detection.id}")
            else:
                print("OCR returned None")
finally:
    db.close()
    cap.release()
