from types import SimpleNamespace

import numpy as np

import pipeline


class FakeCap:
    def isOpened(self):
        return True

    def release(self):
        pass


class FakeVehicleDetector:
    def vehicle_coordinates(self, frame, conf_threshold=0.35):
        assert frame.shape == (20, 30, 3)
        assert conf_threshold == 0.35
        return [("car", 0.95, 2, 3, 22, 18)]

    def crop_vehicle(self, frame, x1, y1, x2, y2):
        assert (x1, y1, x2, y2) == (2, 3, 22, 18)
        return np.zeros((15, 20, 3), dtype=np.uint8)


class FakePlateDetector:
    def license_coordinates(self, frame, min_confidence=0.0):
        assert frame.shape == (15, 20, 3)
        assert min_confidence == 0.25
        return SimpleNamespace(coords=(1, 2, 11, 12), confidence=0.9)

    def crop_into_plate(self, frame, x1, y1, x2, y2):
        assert frame.shape == (15, 20, 3)
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
        vehicle_detector=FakeVehicleDetector(),
        plate_detector=FakePlateDetector(),
        ocr=FakeOCR(),
        min_detection_confidence=0.25,
    )

    assert len(results) == 1

    result = results[0]

    assert "plate_img" not in result
    assert "vehicle_img" not in result

    assert result["plate_text"] == "BA 12 PA 3456"

    # Full-frame coords = ROI offset + vehicle offset + plate offset.
    #
    # ROI:       x=5, y=7
    # Vehicle:   x=2, y=3
    # Plate box: x1=1, y1=2, x2=11, y2=12
    #
    # Final:
    # x1 = 5 + 2 + 1  = 8
    # y1 = 7 + 3 + 2  = 12
    # x2 = 5 + 2 + 11 = 18
    # y2 = 7 + 3 + 12 = 22
    assert result["coords"] == (8, 12, 18, 22)

    assert result["roi_coords"] == (1, 2, 11, 12)
    assert result["source_frame_number"] == 4
    assert result["roi"] == (5, 7, 30, 20)

    assert result["vehicle_class"] == "car"
    assert result["vehicle_confidence"] == 0.95
    assert result["vehicle_coords"] == (7, 10, 27, 25)

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
        vehicle_detector=FakeVehicleDetector(),
        plate_detector=FakePlateDetector(),
        ocr=FakeOCR(),
        min_detection_confidence=0.25,
    )

    assert len(results) == 1
    assert results[0]["plate_img"].shape == (10, 20, 3)
    assert results[0]["vehicle_img"].shape == (15, 20, 3)


def test_process_video_skips_when_no_vehicle_detected(monkeypatch):
    setup_pipeline_fakes(monkeypatch)

    class NoVehicleDetector:
        def vehicle_coordinates(self, frame, conf_threshold=0.35):
            return []

    results = pipeline.process_video(
        "fake.mp4",
        roi=(5, 7, 30, 20),
        save_to_db=False,
        vehicle_detector=NoVehicleDetector(),
        plate_detector=FakePlateDetector(),
        ocr=FakeOCR(),
        min_detection_confidence=0.25,
    )

    assert results == []


def test_process_video_skips_when_no_plate_detected(monkeypatch):
    setup_pipeline_fakes(monkeypatch)

    class NoPlateDetector:
        def license_coordinates(self, frame, min_confidence=0.0):
            return None

        def crop_into_plate(self, frame, x1, y1, x2, y2):
            raise AssertionError("crop_into_plate should not be called")

    results = pipeline.process_video(
        "fake.mp4",
        roi=(5, 7, 30, 20),
        save_to_db=False,
        vehicle_detector=FakeVehicleDetector(),
        plate_detector=NoPlateDetector(),
        ocr=FakeOCR(),
        min_detection_confidence=0.25,
    )

    assert results == []


def test_process_video_skips_when_ocr_fails(monkeypatch):
    setup_pipeline_fakes(monkeypatch)

    class NoOCR:
        def ocr_inference(self, plate_img):
            return None

    results = pipeline.process_video(
        "fake.mp4",
        roi=(5, 7, 30, 20),
        save_to_db=False,
        vehicle_detector=FakeVehicleDetector(),
        plate_detector=FakePlateDetector(),
        ocr=NoOCR(),
        min_detection_confidence=0.25,
    )

    assert results == []


def test_process_video_calls_progress_callback(monkeypatch):
    setup_pipeline_fakes(monkeypatch)

    callback_results = []

    results = pipeline.process_video(
        "fake.mp4",
        roi=(5, 7, 30, 20),
        save_to_db=False,
        vehicle_detector=FakeVehicleDetector(),
        plate_detector=FakePlateDetector(),
        ocr=FakeOCR(),
        min_detection_confidence=0.25,
        progress_callback=callback_results.append,
    )

    assert len(results) == 1
    assert callback_results == results
