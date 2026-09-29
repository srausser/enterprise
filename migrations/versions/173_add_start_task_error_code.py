"""Persist classified sandbox start failures.

Revision ID: 173
Revises: 172
"""

from typing import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = '173'
down_revision: str | None = '172'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        'app_conversation_start_task',
        sa.Column('error_code', sa.String(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column('app_conversation_start_task', 'error_code')
