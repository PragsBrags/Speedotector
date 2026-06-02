import json

from ingestion.zones import Zone
from ui_zones import (
    canvas_object_to_zone,
    validate_zone_setup,
    zone_config_json,
    zones_to_canvas_objects,
)


def test_canvas_rect_object_converts_to_zone_points():
    zone = canvas_object_to_zone(
        {
            "type": "rect",
            "left": 20,
            "top": 40,
            "width": 100,
            "height": 60,
            "scaleX": 1,
            "scaleY": 1,
        },
        zone_type="forbidden_area",
        zone_id="forbidden_area_1",
        label="Forbidden area",
        scale=2,
        frame_width=100,
        frame_height=80,
    )

    assert zone.points == [(10, 20), (60, 20), (60, 50), (10, 50)]


def test_zone_validation_matches_enabled_rules():
    validation = validate_zone_setup(
        zones=[Zone(id="stop_1", type="stop_line", points=[(0, 10), (100, 10)])],
        enabled_rules=["red_light_violation"],
    )

    assert not validation.valid
    assert validation.messages == ["Red-light detection needs a traffic light ROI."]


def test_zone_config_json_exports_run_settings():
    output = zone_config_json(
        "video.mp4",
        [
            Zone(
                id="crosswalk_1",
                type="crosswalk",
                points=[(0, 0), (10, 0), (10, 10), (0, 10)],
            )
        ],
        ["zebra_crossing_violation"],
        pre_event_frames=5,
        post_event_frames=7,
    )

    payload = json.loads(output)

    assert payload["video_path"] == "video.mp4"
    assert payload["zones"][0]["type"] == "crosswalk"
    assert payload["events"] == {"pre_event_frames": 5, "post_event_frames": 7}


def test_zones_to_canvas_objects_uses_line_for_stop_line():
    objects = zones_to_canvas_objects(
        [Zone(id="stop_1", type="stop_line", points=[(0, 5), (20, 5)])],
        scale=2,
    )

    assert objects[0]["type"] == "line"
    assert objects[0]["x2"] == 40
