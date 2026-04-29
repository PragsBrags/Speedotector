from ultralytics import YOLO
import cv2

class VehicleDetection :
    def __init__ (self, yolo_model) :
        self.model = YOLO(yolo_model)
    
    def vehicle_class(self, frame):
        results = self.model([frame], stream=False)
        detections = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy.numpy()[0]
                cls = int(box.cls)
                conf = float(box.conf)
                class_name = self.model.names[cls]
                if conf > 0.8 :
                    detections.append((class_name, conf, (x1, y1, x2, y2)))
        return detections

