from __future__ import annotations

from dataclasses import dataclass

VEHICLE_CLASSES = {"car", "motorcycle", "bus", "truck"}


@dataclass(frozen=True)
class VehicleDetectionResult:
    bbox: tuple[float, float, float, float]
    class_name: str
    confidence: float


class VehicleDetector:
    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        vehicle_classes: set[str] | None = None,
    ):
        from ultralytics import YOLO

        self.model = YOLO(model_path)
        self.vehicle_classes = vehicle_classes or VEHICLE_CLASSES

    def detect(
        self, frame, min_confidence: float = 0.25
    ) -> list[VehicleDetectionResult]:
        detections: list[VehicleDetectionResult] = []
        results = self.model([frame], stream=True)

        for result in results:
            names = getattr(result, "names", {})
            for box in result.boxes:
                confidence = float(box.conf)
                if confidence < min_confidence:
                    continue

                class_id = int(box.cls)
                class_name = names.get(class_id, str(class_id))
                if class_name not in self.vehicle_classes:
                    continue

                x1, y1, x2, y2 = box.xyxy.numpy()[0]
                detections.append(
                    VehicleDetectionResult(
                        bbox=(float(x1), float(y1), float(x2), float(y2)),
                        class_name=class_name,
                        confidence=confidence,
                    )
                )

        return detections
