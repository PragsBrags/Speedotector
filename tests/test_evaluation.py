import json

from evaluation.baselines import (
    event_window_indices,
    fixed_skip_indices,
    representative_event_indices,
)
from evaluation.export_tables import load_results, render_markdown_tables
from evaluation.metrics import compute_metrics
from evaluation.run_experiment import main as run_experiment_main


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


def test_run_experiment_writes_paper_methods_with_prediction_metrics(
    monkeypatch, tmp_path
):
    annotation_path = tmp_path / "annotation.json"
    annotation_path.write_text(
        json.dumps(
            {
                "video_id": "video-1",
                "video_path": "video.mp4",
                "fps": 10,
                "zones": [],
                "violations": [
                    {
                        "id": "v1",
                        "type": "restricted_zone_violation",
                        "frame_start": 10,
                        "frame_end": 12,
                        "plate_text": "BA12PA3456",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    predictions_path = tmp_path / "predictions.json"
    predictions_path.write_text(
        json.dumps(
            {
                "methods": [
                    {
                        "method": "proposed_event_triggered_motion_selection",
                        "runtime_seconds": 2.0,
                        "events": [
                            {
                                "violation_type": "restricted_zone_violation",
                                "frame_number": 11,
                                "plate_text": "BA12PA3456",
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    scored_frames_path = tmp_path / "scores.json"
    scored_frames_path.write_text(
        json.dumps({"frames": {"9": 1.0, "10": 5.0, "11": 3.0}}),
        encoding="utf-8",
    )
    motion_frames_path = tmp_path / "motion.json"
    motion_frames_path.write_text(json.dumps({"frames": [9, 10, 11]}), encoding="utf-8")
    output_path = tmp_path / "results.json"

    monkeypatch.setattr(
        "sys.argv",
        [
            "run_experiment.py",
            "--annotation",
            str(annotation_path),
            "--total-frames",
            "20",
            "--motion-frames",
            str(motion_frames_path),
            "--scored-frames",
            str(scored_frames_path),
            "--predictions",
            str(predictions_path),
            "--output",
            str(output_path),
        ],
    )

    run_experiment_main()

    rows = json.loads(output_path.read_text(encoding="utf-8"))["results"]
    methods = {row["method"]: row for row in rows}

    assert set(methods) == {
        "full_frame_alpr",
        "fixed_skip_10",
        "motion_triggered_alpr",
        "event_triggered_alpr",
        "proposed_event_triggered_motion_selection",
    }
    proposed = methods["proposed_event_triggered_motion_selection"]
    assert proposed["selected_frames"] == 1
    assert proposed["violation_recall"] == 1.0
    assert proposed["ocr_accuracy"] == 1.0
    assert proposed["end_to_end_identification_accuracy"] == 1.0
