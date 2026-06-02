from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np

SignalLabel = Literal["red", "yellow", "green", "unknown"]


@dataclass(frozen=True)
class SignalState:
    state: SignalLabel
    confidence: float
    red_ratio: float
    yellow_ratio: float
    green_ratio: float
    frame_number: int


class TrafficSignalDetector:
    def __init__(self, red_confirm_frames: int = 3, min_ratio: float = 0.02):
        self.red_confirm_frames = red_confirm_frames
        self.min_ratio = min_ratio
        self._red_streak = 0

    def detect(self, frame, roi: tuple[int, int, int, int], frame_number: int):
        x, y, width, height = roi
        crop = frame[y : y + height, x : x + width]
        if crop.size == 0:
            return SignalState("unknown", 0.0, 0.0, 0.0, 0.0, frame_number)

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        total_pixels = float(crop.shape[0] * crop.shape[1])

        red_mask_1 = cv2.inRange(hsv, np.array([0, 80, 80]), np.array([10, 255, 255]))
        red_mask_2 = cv2.inRange(
            hsv, np.array([170, 80, 80]), np.array([180, 255, 255])
        )
        yellow_mask = cv2.inRange(
            hsv, np.array([15, 80, 80]), np.array([35, 255, 255])
        )
        green_mask = cv2.inRange(
            hsv, np.array([40, 60, 60]), np.array([90, 255, 255])
        )

        red_ratio = (cv2.countNonZero(red_mask_1) + cv2.countNonZero(red_mask_2)) / (
            total_pixels
        )
        yellow_ratio = cv2.countNonZero(yellow_mask) / total_pixels
        green_ratio = cv2.countNonZero(green_mask) / total_pixels

        raw_state, confidence = max(
            [
                ("red", red_ratio),
                ("yellow", yellow_ratio),
                ("green", green_ratio),
            ],
            key=lambda item: item[1],
        )
        if confidence < self.min_ratio:
            raw_state = "unknown"
            confidence = 0.0

        if raw_state == "red":
            self._red_streak += 1
        else:
            self._red_streak = 0

        state: SignalLabel = raw_state
        if raw_state == "red" and self._red_streak < self.red_confirm_frames:
            state = "unknown"

        return SignalState(
            state=state,
            confidence=float(confidence),
            red_ratio=float(red_ratio),
            yellow_ratio=float(yellow_ratio),
            green_ratio=float(green_ratio),
            frame_number=frame_number,
        )
