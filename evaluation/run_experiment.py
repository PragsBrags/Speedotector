from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from evaluation.annotations import VideoAnnotation, load_annotation
from evaluation.baselines import (
    event_window_indices,
    fixed_skip_indices,
    full_frame_indices,
    representative_event_indices,
)
from evaluation.metrics import compute_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation", required=True)
    parser.add_argument("--total-frames", type=int, required=True)
    parser.add_argument("--full-frame-runtime-seconds", type=float, default=None)
    parser.add_argument("--runtime-seconds", type=float, default=0.0)
    parser.add_argument("--fixed-skip-step", type=int, default=10)
    parser.add_argument("--pre-event-frames", type=int, default=10)
    parser.add_argument("--post-event-frames", type=int, default=10)
    parser.add_argument("--motion-frames")
    parser.add_argument("--scored-frames")
    parser.add_argument("--predictions")
    parser.add_argument("--output", default="outputs/results.json")
    args = parser.parse_args()

    annotation = load_annotation(args.annotation)
    method_predictions = _load_predictions(args.predictions)
    motion_frames = _load_frame_list(args.motion_frames)
    scored_frames = _load_scored_frames(args.scored_frames)

    rows = []
    for method, selected in _method_selections(
        annotation=annotation,
        total_frames=args.total_frames,
        fixed_skip_step=args.fixed_skip_step,
        pre_event_frames=args.pre_event_frames,
        post_event_frames=args.post_event_frames,
        motion_frames=motion_frames,
        scored_frames=scored_frames,
    ).items():
        predictions = method_predictions.get(method, [])
        counts = _score_predictions(annotation, predictions)
        metrics = compute_metrics(
            total_frames=args.total_frames,
            selected_frames=len(selected),
            runtime_seconds=_runtime_for_method(
                method_predictions,
                method,
                default_runtime=args.runtime_seconds,
            ),
            full_frame_runtime_seconds=args.full_frame_runtime_seconds,
            violation_true_positives=counts["violation_true_positives"],
            violation_false_positives=counts["violation_false_positives"],
            violation_false_negatives=counts["violation_false_negatives"],
            plate_detections=counts["plate_detections"],
            ground_truth_violations=len(annotation.violations),
            correct_plate_texts=counts["correct_plate_texts"],
            detected_plate_texts=counts["detected_plate_texts"],
            correct_violation_and_plate=counts["correct_violation_and_plate"],
        ).as_dict()
        metrics["method"] = method
        rows.append(metrics)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"results": rows}, indent=2), encoding="utf-8")


def _method_selections(
    *,
    annotation: VideoAnnotation,
    total_frames: int,
    fixed_skip_step: int,
    pre_event_frames: int,
    post_event_frames: int,
    motion_frames: list[int],
    scored_frames: dict[int, float],
) -> dict[str, list[int]]:
    event_frames = [violation.frame_start for violation in annotation.violations]
    event_window = event_window_indices(
        event_frames,
        total_frames,
        pre_event_frames,
        post_event_frames,
    )
    representative_events = representative_event_indices(
        scored_frames,
        event_frames,
        total_frames,
        pre_event_frames,
        post_event_frames,
    )

    return {
        "full_frame_alpr": full_frame_indices(total_frames),
        f"fixed_skip_{fixed_skip_step}": fixed_skip_indices(
            total_frames, fixed_skip_step
        ),
        "motion_triggered_alpr": sorted(set(motion_frames)),
        "event_triggered_alpr": event_window,
        "proposed_event_triggered_motion_selection": representative_events,
    }


def _load_predictions(path: str | None) -> dict[str, Any]:
    if not path:
        return {}

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    methods = payload.get("methods", payload.get("results", []))
    predictions = {}
    for method in methods:
        name = str(method["method"])
        predictions[name] = method.get("events", [])
        if "runtime_seconds" in method:
            predictions[f"{name}__runtime_seconds"] = float(method["runtime_seconds"])
    return predictions


def _load_frame_list(path: str | None) -> list[int]:
    if not path:
        return []

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("frames", [])
    return [int(frame) for frame in payload]


def _load_scored_frames(path: str | None) -> dict[int, float]:
    if not path:
        return {}

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    frames = payload.get("frames", payload)
    return {int(frame): float(score) for frame, score in frames.items()}


def _runtime_for_method(
    method_predictions: dict[str, Any], method: str, default_runtime: float
) -> float:
    return float(method_predictions.get(f"{method}__runtime_seconds", default_runtime))


def _score_predictions(
    annotation: VideoAnnotation, predictions: list[dict[str, Any]]
) -> dict[str, int]:
    matched_annotation_ids = set()
    true_positives = 0
    false_positives = 0
    plate_detections = 0
    detected_plate_texts = 0
    correct_plate_texts = 0
    correct_violation_and_plate = 0

    for prediction in predictions:
        match = _matching_violation(annotation, prediction, matched_annotation_ids)
        plate_text = prediction.get("plate_text")
        if plate_text:
            plate_detections += 1
            detected_plate_texts += 1

        if match is None:
            false_positives += 1
            continue

        matched_annotation_ids.add(match.id)
        true_positives += 1
        if plate_text and match.plate_text and plate_text == match.plate_text:
            correct_plate_texts += 1
            correct_violation_and_plate += 1

    return {
        "violation_true_positives": true_positives,
        "violation_false_positives": false_positives,
        "violation_false_negatives": len(annotation.violations) - true_positives,
        "plate_detections": plate_detections,
        "detected_plate_texts": detected_plate_texts,
        "correct_plate_texts": correct_plate_texts,
        "correct_violation_and_plate": correct_violation_and_plate,
    }


def _matching_violation(
    annotation: VideoAnnotation,
    prediction: dict[str, Any],
    matched_annotation_ids: set[str],
):
    prediction_type = prediction.get("violation_type") or prediction.get("type")
    frame_number = int(
        prediction.get("frame_number", prediction.get("evidence_frame_number", -1))
    )
    for violation in annotation.violations:
        if violation.id in matched_annotation_ids:
            continue
        if prediction_type and prediction_type != violation.type:
            continue
        if violation.frame_start <= frame_number <= violation.frame_end:
            return violation
    return None


if __name__ == "__main__":
    main()
