import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import violation_pipeline
from detection.vehicle import VehicleDetectionResult
from ingestion.zones import Zone
from rules.traffic_signal import SignalState, TrafficSignalDetector
from violation_pipeline import PlateCandidate, _signal_state_for_frame


class FakeCap:
    def __init__(self, frames):
        self.frames = frames
        self.index = 0

    def isOpened(self):
        return self.index <= len(self.frames)

    def read(self):
        if self.index >= len(self.frames):
            return False, None
        frame = self.frames[self.index]
        self.index += 1
        return True, frame

    def get(self, prop):
        if prop == violation_pipeline.cv2.CAP_PROP_FPS:
            return 10
        return len(self.frames)

    def release(self):
        pass


class FakeVehicleDetector:
    def detect(self, frame, min_confidence=0.25):
        return [
            VehicleDetectionResult(
                bbox=(5, 5, 25, 25),
                class_name="car",
                confidence=min_confidence,
            )
        ]


class FakePlateDetector:
    def license_candidates(self, frame):
        return [
            PlateCandidate(coords=(40, 40, 55, 55), confidence=0.99),
            PlateCandidate(coords=(8, 8, 18, 18), confidence=0.75),
        ]

    def crop_into_plate(self, frame, x1, y1, x2, y2):
        return np.full((8, 10, 3), int(x1), dtype=np.uint8)


class FakeOCR:
    def ocr_inference(self, plate_img):
        return SimpleNamespace(text="BA12PA3456", confidence=0.88)


class FakeEvidenceWriter:
    def __init__(self, root):
        self.root = Path(root)
        self.records = []

    def write(self, event, frame, zones, **kwargs):
        record = SimpleNamespace(
            evidence_frame_path=str(self.root / f"frame_{len(self.records)}.jpg"),
            plate_crop_path=str(self.root / f"plate_{len(self.records)}.jpg"),
        )
        self.records.append((event, frame.copy(), zones, kwargs, record))
        return record

    def write_results(self):
        path = self.root / "results.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("[]", encoding="utf-8")
        return path


def test_manual_signal_intervals_override_visual_detector():
    signal = _signal_state_for_frame(
        frame=np.zeros((4, 4, 3), dtype=np.uint8),
        frame_number=12,
        traffic_light_roi=None,
        signal_detector=TrafficSignalDetector(),
        manual_signal_intervals=[(10, 20)],
    )

    assert isinstance(signal, SignalState)
    assert signal.state == "red"

    signal = _signal_state_for_frame(
        frame=np.zeros((4, 4, 3), dtype=np.uint8),
        frame_number=22,
        traffic_light_roi=None,
        signal_detector=TrafficSignalDetector(),
        manual_signal_intervals=[(10, 20)],
    )

    assert signal.state == "green"


def test_zone_pipeline_selects_best_event_window_frame_and_associated_plate(
    monkeypatch, tmp_path
):
    frames = [
        np.full((40, 60, 3), 1, dtype=np.uint8),
        np.full((40, 60, 3), 2, dtype=np.uint8),
        np.full((40, 60, 3), 5, dtype=np.uint8),
    ]
    writer = FakeEvidenceWriter(tmp_path)

    monkeypatch.setattr(
        violation_pipeline.cv2,
        "VideoCapture",
        lambda _path: FakeCap(frames),
    )
    monkeypatch.setattr(
        violation_pipeline,
        "_frame_score",
        lambda frame, _mask, _bbox: (float(frame[0, 0, 0]), 10.0, float(frame[0, 0, 0])),
    )

    result = violation_pipeline.process_zone_violations(
        "fake.mp4",
        zones=[
            Zone(
                id="forbidden_1",
                type="forbidden_area",
                points=[(0, 0), (35, 0), (35, 35), (0, 35)],
            )
        ],
        enabled_rules=["restricted_zone_violation"],
        vehicle_detector=FakeVehicleDetector(),
        plate_detector=FakePlateDetector(),
        ocr=FakeOCR(),
        evidence_writer=writer,
        post_event_frames=2,
        save_to_db=False,
    )

    assert len(result["events"]) == 1
    event = result["events"][0]
    assert event["frame_number"] == 0
    assert event["evidence_frame_number"] == 2
    assert event["plate_bbox"] == (8, 8, 18, 18)
    assert event["plate_text"] == "BA12PA3456"
    assert event["db_violation_id"] is None


def test_zone_pipeline_persists_when_enabled(monkeypatch, tmp_path):
    calls = {"zones": 0, "violations": 0}

    class FakeSession:
        def close(self):
            calls["closed"] = True

    def create_tables():
        calls["create_tables"] = True

    def session_local():
        return FakeSession()

    def save_video(db, video_path):
        assert isinstance(db, FakeSession)
        return SimpleNamespace(id=42, file_path=video_path)

    def save_zone(db, zone, video_id=None):
        calls["zones"] += 1
        assert video_id == 42
        return zone

    def save_violation(db, video_id, event, **kwargs):
        calls["violations"] += 1
        assert video_id == 42
        assert kwargs["plate_text"] == "BA12PA3456"
        return SimpleNamespace(id=7)

    monkeypatch.setitem(
        sys.modules,
        "db.database",
        SimpleNamespace(create_tables=create_tables, SessionLocal=session_local),
    )
    monkeypatch.setitem(
        sys.modules,
        "db.repository",
        SimpleNamespace(
            save_video=save_video,
            save_zone=save_zone,
            save_violation=save_violation,
        ),
    )
    monkeypatch.setattr(
        violation_pipeline.cv2,
        "VideoCapture",
        lambda _path: FakeCap([np.full((40, 60, 3), 5, dtype=np.uint8)]),
    )
    monkeypatch.setattr(
        violation_pipeline,
        "_frame_score",
        lambda frame, _mask, _bbox: (5.0, 10.0, 0.5),
    )

    result = violation_pipeline.process_zone_violations(
        "fake.mp4",
        zones=[
            Zone(
                id="forbidden_1",
                type="forbidden_area",
                points=[(0, 0), (35, 0), (35, 35), (0, 35)],
            )
        ],
        enabled_rules=["restricted_zone_violation"],
        vehicle_detector=FakeVehicleDetector(),
        plate_detector=FakePlateDetector(),
        ocr=FakeOCR(),
        evidence_writer=FakeEvidenceWriter(tmp_path),
        post_event_frames=0,
        save_to_db=True,
    )

    assert calls == {
        "create_tables": True,
        "zones": 1,
        "violations": 1,
        "closed": True,
    }
    assert result["events"][0]["db_violation_id"] == 7
