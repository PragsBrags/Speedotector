from __future__ import annotations

from dataclasses import dataclass

from ocr.licensePlate import LicensePlateDetection, PlateDetectionResult


@dataclass(frozen=True)
class PlateDetectionCandidate:
    coords: tuple[float, float, float, float]
    confidence: float


class PlateDetector:
    def __init__(self, model_path: str):
        self.detector = LicensePlateDetection(model_path)

    def best_plate(
        self,
        frame,
        min_confidence: float = 0.0,
    ) -> PlateDetectionResult | None:
        return self.detector.license_coordinates(frame, min_confidence=min_confidence)

    def crop(
        self,
        frame,
        coords: tuple[float, float, float, float],
    ):
        return self.detector.crop_into_plate(frame, *coords)
