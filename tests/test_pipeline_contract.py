from types import SimpleNamespace

import numpy as np

import pipeline


class FakeCap:
    def isOpened(self):
        return True

    def release(self):
        pass


class FakeDetector:
    def license_coordinates(self, frame, min_confidence=0.0):
        assert frame.shape == (20, 30, 3)
        assert min_confidence == 0.25
        return SimpleNamespace(coords=(1, 2, 11, 12), confidence=0.9)

    def crop_into_plate(self, frame, x1, y1, x2, y2):
        assert (x1, y1, x2, y2) == (1, 2, 11, 12)
        return np.zeros((10, 20, 3), dtype=np.uint8)


class FakeOCR:
    def ocr_inference(self, plate_img):
        assert plate_img.shape == (10, 20, 3)
        return SimpleNamespace(
            text="BA 12 PA 3456",
            confidence=0.82,
            segments=["BA", "12", "PA", "3456"],
            segment_confidences=[0.8, 0.85, 0.81, 0.82],
        )


def fake_selected_frames(_cap, roi=None):
    assert roi == (5, 7, 30, 20)
    yield {
        "image": np.zeros((20, 30, 3), dtype=np.uint8),
        "frame_number": 4,
        "roi": (5, 7, 30, 20),
        "motion_area": 125.0,
        "sharpness": 3.0,
        "score": 375.0,
        "timestamp_seconds": 0.4,
        "selection_reason": "motion_cooldown",
    }


def setup_pipeline_fakes(monkeypatch):
    monkeypatch.setattr(pipeline.cv2, "VideoCapture", lambda _path: FakeCap())
    monkeypatch.setattr(pipeline, "selected_frames", fake_selected_frames)


def test_process_video_excludes_images_by_default(monkeypatch):
    setup_pipeline_fakes(monkeypatch)

    results = pipeline.process_video(
        "fake.mp4",
        roi=(5, 7, 30, 20),
        save_to_db=False,
        detector=FakeDetector(),
        ocr=FakeOCR(),
        min_detection_confidence=0.25,
    )

    assert len(results) == 1
    result = results[0]
    assert "plate_img" not in result
    assert result["plate_text"] == "BA 12 PA 3456"
    assert result["coords"] == (6, 9, 16, 19)
    assert result["roi_coords"] == (1, 2, 11, 12)
    assert result["source_frame_number"] == 4
    assert result["detector_confidence"] == 0.9
    assert result["ocr_confidence"] == 0.82
    assert result["ocr_segments"] == ["BA", "12", "PA", "3456"]
    assert result["crop_shape"] == (10, 20, 3)
    assert result["frame_metadata"] == {
        "motion_area": 125.0,
        "sharpness": 3.0,
        "score": 375.0,
        "timestamp_seconds": 0.4,
        "selection_reason": "motion_cooldown",
    }


def test_process_video_includes_images_when_requested(monkeypatch):
    setup_pipeline_fakes(monkeypatch)

    results = pipeline.process_video(
        "fake.mp4",
        roi=(5, 7, 30, 20),
        save_to_db=False,
        include_images=True,
        detector=FakeDetector(),
        ocr=FakeOCR(),
        min_detection_confidence=0.25,
    )

    assert results[0]["plate_img"].shape == (10, 20, 3)
