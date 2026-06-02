"""Add research zone and violation tables.

Revision ID: 20260531_0004
Revises: 20260531_0003
Create Date: 2026-05-31

"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20260531_0004"
down_revision: str | None = "20260531_0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "zones",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("video_id", sa.Integer(), nullable=True),
        sa.Column("zone_key", sa.String(length=100), nullable=False),
        sa.Column("zone_type", sa.String(length=50), nullable=False),
        sa.Column("label", sa.String(length=255), nullable=True),
        sa.Column("points_json", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["video_id"], ["videos.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_zones_id"), "zones", ["id"], unique=False)
    op.create_index(op.f("ix_zones_video_id"), "zones", ["video_id"], unique=False)

    op.create_table(
        "violations",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("video_id", sa.Integer(), nullable=False),
        sa.Column("violation_type", sa.String(length=100), nullable=False),
        sa.Column("frame_number", sa.Integer(), nullable=False),
        sa.Column("timestamp_seconds", sa.Float(), nullable=True),
        sa.Column("zone_id", sa.String(length=100), nullable=False),
        sa.Column("track_id", sa.Integer(), nullable=False),
        sa.Column("vehicle_x1", sa.Float(), nullable=False),
        sa.Column("vehicle_y1", sa.Float(), nullable=False),
        sa.Column("vehicle_x2", sa.Float(), nullable=False),
        sa.Column("vehicle_y2", sa.Float(), nullable=False),
        sa.Column("plate_text", sa.String(length=50), nullable=True),
        sa.Column("plate_x1", sa.Float(), nullable=True),
        sa.Column("plate_y1", sa.Float(), nullable=True),
        sa.Column("plate_x2", sa.Float(), nullable=True),
        sa.Column("plate_y2", sa.Float(), nullable=True),
        sa.Column("detector_confidence", sa.Float(), nullable=True),
        sa.Column("ocr_confidence", sa.Float(), nullable=True),
        sa.Column("signal_state", sa.String(length=20), nullable=True),
        sa.Column("signal_confidence", sa.Float(), nullable=True),
        sa.Column("evidence_frame_path", sa.String(length=500), nullable=True),
        sa.Column("plate_crop_path", sa.String(length=500), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["video_id"], ["videos.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_violations_id"), "violations", ["id"], unique=False)
    op.create_index(
        op.f("ix_violations_video_id"), "violations", ["video_id"], unique=False
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_violations_video_id"), table_name="violations")
    op.drop_index(op.f("ix_violations_id"), table_name="violations")
    op.drop_table("violations")
    op.drop_index(op.f("ix_zones_video_id"), table_name="zones")
    op.drop_index(op.f("ix_zones_id"), table_name="zones")
    op.drop_table("zones")
