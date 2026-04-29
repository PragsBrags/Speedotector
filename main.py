import os
from ingestion.video_feed import frame
from ocr.licensePlate import LicensePlateDetection, PaddleInference
from ocr.vehicleDetection import VehicleDetection
from ultralytics import YOLO
import cv2

script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, 'models/license_plate.pt')

cap = cv2.VideoCapture(0)

VD = VehicleDetection('yolon')
LPD = LicensePlateDetection(model_path)
PI = PaddleInference()

if __name__ == "__main__" :
    for frames in frame(cap):
        detections = VD.vehicle_class(frames)
        for class_name, conf, (x1, y1, x2, y2) in detections:
            vehicle_img = frames[int(y1):int(y2), int(x1):int(x2)]
            coords = LPD.license_coordinates(vehicle_img)
            if coords is None:
                continue  # Skip this frame if no plate detected

            x1, y1, x2, y2 = coords
            plate_img = LPD.crop_into_plate(vehicle_img, x1, y1, x2, y2)
            res = PI.ocr_inference(plate_img)
            print(res)

    