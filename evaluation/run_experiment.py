from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluation.annotations import load_annotation
from evaluation.baselines import (
    event_window_indices,
    fixed_skip_indices,
    full_frame_indices,
)
from evaluation.metrics import compute_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation", required=True)
    parser.add_argument("--total-frames", type=int, required=True)
    parser.add_argument("--runtime-seconds", type=float, default=0.0)
    parser.add_argument("--output", default="outputs/results.json")
    args = parser.parse_args()

    annotation = load_annotation(args.annotation)
    event_frames = [violation.frame_start for violation in annotation.violations]
    methods = {
        "full_frame_alpr": full_frame_indices(args.total_frames),
        "fixed_skip_5": fixed_skip_indices(args.total_frames, 5),
        "fixed_skip_10": fixed_skip_indices(args.total_frames, 10),
        "fixed_skip_15": fixed_skip_indices(args.total_frames, 15),
        "fixed_skip_30": fixed_skip_indices(args.total_frames, 30),
        "event_window_alpr": event_window_indices(event_frames, args.total_frames),
    }

    results = []
    for method, selected in methods.items():
        metrics = compute_metrics(
            total_frames=args.total_frames,
            selected_frames=len(selected),
            runtime_seconds=args.runtime_seconds,
            violation_true_positives=0,
            violation_false_positives=0,
            violation_false_negatives=len(annotation.violations),
            ground_truth_violations=len(annotation.violations),
        ).as_dict()
        metrics["method"] = method
        results.append(metrics)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"results": results}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
