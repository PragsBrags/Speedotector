from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from db.database import Base


class Video(Base):
    __tablename__ = "videos"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    file_name: Mapped[str] = mapped_column(String(255), nullable=False)
    file_path: Mapped[str] = mapped_column(String(500), nullable=False)
    date_processed: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )

    detections: Mapped[list["Detection"]] = relationship(
        "Detection",
        back_populates="video",
        cascade="all, delete-orphan",
    )


class Detection(Base):
    __tablename__ = "detections"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    video_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("videos.id"), nullable=False, index=True
    )
    plate_text: Mapped[str] = mapped_column(String(50), nullable=False)
    x1: Mapped[float] = mapped_column(Float, nullable=False)
    y1: Mapped[float] = mapped_column(Float, nullable=False)
    x2: Mapped[float] = mapped_column(Float, nullable=False)
    y2: Mapped[float] = mapped_column(Float, nullable=False)
    crop_width: Mapped[int] = mapped_column(Integer, nullable=False)
    crop_height: Mapped[int] = mapped_column(Integer, nullable=False)
    detector_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    ocr_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )

    video: Mapped[Video] = relationship("Video", back_populates="detections")


class ZoneRecord(Base):
    __tablename__ = "zones"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    video_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("videos.id"), nullable=True, index=True
    )
    zone_key: Mapped[str] = mapped_column(String(100), nullable=False)
    zone_type: Mapped[str] = mapped_column(String(50), nullable=False)
    label: Mapped[str | None] = mapped_column(String(255), nullable=True)
    points_json: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )


class Violation(Base):
    __tablename__ = "violations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    video_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("videos.id"), nullable=False, index=True
    )
    violation_type: Mapped[str] = mapped_column(String(100), nullable=False)
    frame_number: Mapped[int] = mapped_column(Integer, nullable=False)
    timestamp_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    zone_id: Mapped[str] = mapped_column(String(100), nullable=False)
    track_id: Mapped[int] = mapped_column(Integer, nullable=False)
    vehicle_x1: Mapped[float] = mapped_column(Float, nullable=False)
    vehicle_y1: Mapped[float] = mapped_column(Float, nullable=False)
    vehicle_x2: Mapped[float] = mapped_column(Float, nullable=False)
    vehicle_y2: Mapped[float] = mapped_column(Float, nullable=False)
    plate_text: Mapped[str | None] = mapped_column(String(50), nullable=True)
    plate_x1: Mapped[float | None] = mapped_column(Float, nullable=True)
    plate_y1: Mapped[float | None] = mapped_column(Float, nullable=True)
    plate_x2: Mapped[float | None] = mapped_column(Float, nullable=True)
    plate_y2: Mapped[float | None] = mapped_column(Float, nullable=True)
    detector_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    ocr_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    signal_state: Mapped[str | None] = mapped_column(String(20), nullable=True)
    signal_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    evidence_frame_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    plate_crop_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )
