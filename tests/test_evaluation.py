import json

from evaluation.baselines import (
    event_window_indices,
    fixed_skip_indices,
    representative_event_indices,
)
from evaluation.export_tables import load_results, render_markdown_tables
from evaluation.metrics import compute_metrics


def test_baseline_frame_selection_helpers():
    assert fixed_skip_indices(12, 5) == [0, 5, 10]
    assert event_window_indices([5], total_frames=12, pre_event_frames=2, post_event_frames=2) == [
        3,
        4,
        5,
        6,
        7,
    ]
    assert representative_event_indices({3: 1.0, 4: 5.0, 5: 2.0}, [4], 10, 1, 1) == [
        4
    ]


def test_compute_metrics_formulas():
    metrics = compute_metrics(
        total_frames=100,
        selected_frames=25,
        runtime_seconds=2.0,
        full_frame_runtime_seconds=10.0,
        violation_true_positives=8,
        violation_false_positives=1,
        violation_false_negatives=2,
        plate_detections=7,
        ground_truth_violations=10,
        correct_plate_texts=6,
        detected_plate_texts=8,
        correct_violation_and_plate=5,
    )

    assert metrics.frame_selection_ratio == 0.25
    assert metrics.frame_reduction_rate == 0.75
    assert metrics.speedup == 5.0
    assert metrics.violation_recall == 0.8
    assert metrics.plate_detection_recall == 0.7
    assert metrics.ocr_accuracy == 0.75
    assert metrics.end_to_end_identification_accuracy == 0.5


def test_export_tables_reads_results_and_renders_markdown(tmp_path):
    result_path = tmp_path / "results.json"
    result_path.write_text(
        json.dumps(
            {
                "results": [
                    {
                        "method": "event_window_alpr",
                        "total_frames": 100,
                        "selected_frames": 10,
                        "frame_selection_ratio": 0.1,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    output = render_markdown_tables(load_results(result_path))

    assert "Table II. Efficiency Comparison" in output
    assert "event_window_alpr" in output
