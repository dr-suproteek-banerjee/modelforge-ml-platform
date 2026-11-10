"""Create the ModelForge workspace tables."""

import sqlalchemy as sa
from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def created_at() -> sa.Column:
    return sa.Column("created_at", sa.DateTime(timezone=True), nullable=False)


def upgrade() -> None:
    op.create_table(
        "datasets",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("name", sa.String(120), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("rows", sa.Integer(), nullable=False),
        sa.Column("columns", sa.JSON(), nullable=False),
        sa.Column("sha256", sa.String(64), nullable=False),
        sa.Column("artifact_key", sa.String(120), nullable=False),
        created_at(),
        sa.UniqueConstraint("name", "version"),
    )
    op.create_table(
        "experiments",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("dataset_id", sa.String(36), sa.ForeignKey("datasets.id"), nullable=False),
        sa.Column("config", sa.JSON(), nullable=False),
        sa.Column("status", sa.String(20), nullable=False),
        sa.Column("job_id", sa.String(36)),
        sa.Column("progress", sa.Integer(), nullable=False),
        sa.Column("error", sa.String(400)),
        sa.Column("metrics", sa.JSON()),
        sa.Column("mlflow_run", sa.String(64)),
        created_at(),
    )
    op.create_index("ix_experiments_dataset_id", "experiments", ["dataset_id"])
    op.create_table(
        "models",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column(
            "experiment_id",
            sa.String(36),
            sa.ForeignKey("experiments.id"),
            nullable=False,
            unique=True,
        ),
        sa.Column("name", sa.String(120), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("artifact_key", sa.String(120), nullable=False),
        sa.Column("metadata", sa.JSON(), nullable=False),
        created_at(),
        sa.UniqueConstraint("name", "version"),
    )
    op.create_table(
        "activity",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("action", sa.String(40), nullable=False),
        sa.Column("resource_id", sa.String(36), nullable=False),
        sa.Column("details", sa.JSON(), nullable=False),
        created_at(),
    )


def downgrade() -> None:
    for name in ("activity", "models", "experiments", "datasets"):
        op.drop_table(name)
