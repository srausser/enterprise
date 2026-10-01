"""Separate conversation work generations from bookkeeping timestamps.

Revision ID: 166
Revises: 165
"""

from typing import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = '166'
down_revision: str | None = '165'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Historical conversations start at zero; only later work advances this value.
    op.add_column(
        'conversation_metadata',
        sa.Column(
            'lifecycle_generation',
            sa.Integer(),
            nullable=False,
            server_default='0',
        ),
    )


def downgrade() -> None:
    op.drop_column('conversation_metadata', 'lifecycle_generation')
