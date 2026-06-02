import os

import pytest

sqlalchemy = pytest.importorskip("sqlalchemy")
orm = pytest.importorskip("sqlalchemy.orm")

os.environ["DATABASE_URL"] = "sqlite:///:memory:"


def test_save_zone_and_violation_records():
    from db.database import Base
    from db.repository import save_video, save_violation, save_zone
    from ingestion.zones import Zone
    from rules.base import ViolationEvent

    engine = sqlalchemy.create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = orm.sessionmaker(bind=engine)()

    try:
        video = save_video(session, "test_video.mp4")
        zone = save_zone(
            session,
            Zone(
                id="forbidden_1",
                type="forbidden_area",
                points=[(0, 0), (10, 0), (10, 10), (0, 10)],
            ),
            video_id=video.id,
        )
        violation = save_violation(
            session,
            video.id,
            ViolationEvent(
                violation_type="restricted_zone_violation",
                frame_number=12,
                timestamp_seconds=0.4,
                zone_id="forbidden_1",
                track_id=7,
                vehicle_bbox=(1, 2, 3, 4),
                signal_state=None,
                reason="test",
            ),
            plate_text="BA12PA3456",
            plate_bbox=(1, 2, 3, 4),
            detector_confidence=0.91,
            ocr_confidence=0.82,
            evidence_frame_path="outputs/run/evidence/frame.jpg",
            plate_crop_path="outputs/run/evidence/plate.jpg",
        )

        assert zone.zone_key == "forbidden_1"
        assert violation.violation_type == "restricted_zone_violation"
        assert violation.plate_text == "BA12PA3456"
        assert violation.detector_confidence == 0.91
    finally:
        session.close()
