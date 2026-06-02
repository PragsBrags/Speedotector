from ingestion.zones import (
    Zone,
    bbox_intersects_zone,
    centroid,
    line_crossed,
    point_in_polygon,
)


def test_zone_from_bbox_creates_polygon():
    zone = Zone.from_dict(
        {"id": "tl1", "type": "traffic_light", "bbox": [10, 20, 30, 40]}
    )

    assert zone.points == [(10, 20), (40, 20), (40, 60), (10, 60)]
    assert zone.bbox == (10, 20, 40, 60)


def test_geometry_helpers_cover_zone_logic():
    polygon = [(0, 0), (100, 0), (100, 100), (0, 100)]

    assert centroid((10, 20, 30, 40)) == (20, 30)
    assert point_in_polygon((50, 50), polygon)
    assert not point_in_polygon((150, 50), polygon)
    assert bbox_intersects_zone((90, 90, 120, 120), polygon)
    assert line_crossed((50, -10), (50, 10), [(0, 0), (100, 0)])
