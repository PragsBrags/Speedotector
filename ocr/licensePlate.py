import os
from ultralytics import YOLO
import cv2
import numpy as np
from paddleocr import PaddleOCR
class LicensePlateDetection :
    def __init__ (self, yolo_path) :
        self.model = YOLO(yolo_path)

    def license_coordinates (self, frame) :
        results = self.model([frame], stream=True)
        for result in results :
            for box in result.boxes :
                x1, y1, x2, y2 = box.xyxy.numpy()[0]
                return x1,y1,x2,y2
        return None

    def crop_into_plate (self, frame, x1, y1, x2, y2) :
        plate_img = frame[int(y1):int(y2), int(x1):int(x2)]
        return plate_img
    
class PaddleInference :

    def __init__(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
        self.pipeline = PaddleOCR()

    def ocr_inference (self, plate_img) :
        output = self.pipeline.predict(plate_img)
        for res in output :
            print(res)
            return res
