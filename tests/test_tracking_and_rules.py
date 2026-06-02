from detection.vehicle import VehicleDetectionResult
from ingestion.zones import Zone
from rules.base import FrameContext
from rules.red_light import RedLightRule
from rules.restricted_zone import RestrictedZoneRule
from rules.traffic_signal import SignalState
from rules.zebra_crossing import ZebraCrossingRule
from tracking.centroid_tracker import CentroidTracker


def vehicle(bbox):
    return VehicleDetectionResult(bbox=bbox, class_name="car", confidence=0.9)


def test_centroid_tracker_keeps_stable_track_id():
    tracker = CentroidTracker(max_distance=50)

    first = tracker.update([vehicle((0, 0, 20, 20))], frame_number=1)
    second = tracker.update([vehicle((5, 0, 25, 20))], frame_number=2)

    assert first[0].track_id == second[0].track_id
    assert second[0].centroid_history == [(10, 10), (15, 10)]


def test_restricted_zone_rule_emits_once_per_track_zone_pair():
    tracker = CentroidTracker()
    tracks = tracker.update([vehicle((10, 10, 30, 30))], frame_number=1)
    context = FrameContext(
        frame_number=1,
        timestamp_seconds=0.1,
        frame=None,
        zones=[
            Zone(
                id="forbidden_1",
                type="forbidden_area",
                points=[(0, 0), (50, 0), (50, 50), (0, 50)],
            )
        ],
        vehicle_detections=[],
        tracks=tracks,
    )
    rule = RestrictedZoneRule()

    assert len(rule.evaluate(context)) == 1
    assert rule.evaluate(context) == []


def test_red_light_rule_requires_stop_line_crossing():
    tracker = CentroidTracker(max_distance=100)
    tracker.update([vehicle((40, -20, 60, 0))], frame_number=1)
    tracks = tracker.update([vehicle((40, 10, 60, 30))], frame_number=2)
    context = FrameContext(
        frame_number=2,
        timestamp_seconds=0.2,
        frame=None,
        zones=[Zone(id="stop_1", type="stop_line", points=[(0, 5), (100, 5)])],
        vehicle_detections=[],
        tracks=tracks,
        signal_state=SignalState("red", 0.9, 0.9, 0.0, 0.0, 2),
    )

    events = RedLightRule().evaluate(context)

    assert len(events) == 1
    assert events[0].violation_type == "red_light_violation"


def test_zebra_crossing_rule_requires_red_signal():
    tracker = CentroidTracker()
    tracks = tracker.update([vehicle((10, 10, 30, 30))], frame_number=1)
    context = FrameContext(
        frame_number=1,
        timestamp_seconds=0.1,
        frame=None,
        zones=[
            Zone(
                id="crosswalk_1",
                type="crosswalk",
                points=[(0, 0), (50, 0), (50, 50), (0, 50)],
            )
        ],
        vehicle_detections=[],
        tracks=tracks,
        signal_state=SignalState("green", 0.9, 0.0, 0.0, 0.9, 1),
    )

    assert ZebraCrossingRule().evaluate(context) == []
