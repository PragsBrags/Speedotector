from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class EvaluationMetrics:
    total_frames: int
    selected_frames: int
    frame_selection_ratio: float
    frame_reduction_rate: float
    runtime_seconds: float
    speedup: float | None
    violation_true_positives: int
    violation_false_positives: int
    violation_false_negatives: int
    violation_recall: float
    plate_detection_recall: float
    ocr_accuracy: float
    end_to_end_identification_accuracy: float

    def as_dict(self):
        return asdict(self)


def compute_metrics(
    *,
    total_frames: int,
    selected_frames: int,
    runtime_seconds: float,
    full_frame_runtime_seconds: float | None = None,
    violation_true_positives: int = 0,
    violation_false_positives: int = 0,
    violation_false_negatives: int = 0,
    plate_detections: int = 0,
    ground_truth_violations: int = 0,
    correct_plate_texts: int = 0,
    detected_plate_texts: int = 0,
    correct_violation_and_plate: int = 0,
) -> EvaluationMetrics:
    frame_selection_ratio = _safe_div(selected_frames, total_frames)
    violation_denominator = violation_true_positives + violation_false_negatives

    return EvaluationMetrics(
        total_frames=total_frames,
        selected_frames=selected_frames,
        frame_selection_ratio=frame_selection_ratio,
        frame_reduction_rate=1 - frame_selection_ratio,
        runtime_seconds=runtime_seconds,
        speedup=(
            _safe_div(full_frame_runtime_seconds, runtime_seconds)
            if full_frame_runtime_seconds is not None
            else None
        ),
        violation_true_positives=violation_true_positives,
        violation_false_positives=violation_false_positives,
        violation_false_negatives=violation_false_negatives,
        violation_recall=_safe_div(violation_true_positives, violation_denominator),
        plate_detection_recall=_safe_div(plate_detections, ground_truth_violations),
        ocr_accuracy=_safe_div(correct_plate_texts, detected_plate_texts),
        end_to_end_identification_accuracy=_safe_div(
            correct_violation_and_plate, ground_truth_violations
        ),
    )


def _safe_div(numerator: float | int, denominator: float | int) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)
