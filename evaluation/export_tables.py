from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_results(path: str | Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "results" in payload:
        return list(payload["results"])
    if isinstance(payload, list):
        return list(payload)
    raise ValueError("Expected a list of result rows or an object with a results key")


def render_markdown_tables(results: list[dict[str, Any]]) -> str:
    return "\n\n".join(
        [
            _render_table(
                "Table II. Efficiency Comparison",
                results,
                [
                    "method",
                    "total_frames",
                    "selected_frames",
                    "frame_selection_ratio",
                    "frame_reduction_rate",
                    "runtime_seconds",
                    "speedup",
                ],
            ),
            _render_table(
                "Table III. Accuracy Comparison",
                results,
                [
                    "method",
                    "violation_recall",
                    "plate_detection_recall",
                    "ocr_accuracy",
                    "end_to_end_identification_accuracy",
                ],
            ),
            _render_table(
                "Table IV. Ablation Study",
                results,
                [
                    "method",
                    "violation_true_positives",
                    "violation_false_positives",
                    "violation_false_negatives",
                ],
            ),
        ]
    )


def _render_table(title: str, rows: list[dict[str, Any]], columns: list[str]) -> str:
    output = [title, "", "| " + " | ".join(columns) + " |"]
    output.append("| " + " | ".join("---" for _ in columns) + " |")
    for row in rows:
        output.append("| " + " | ".join(_format_value(row.get(col)) for col in columns) + " |")
    return "\n".join(output)


def _format_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--format", choices=["markdown"], default="markdown")
    args = parser.parse_args()

    results = load_results(args.results)
    print(render_markdown_tables(results))


if __name__ == "__main__":
    main()
