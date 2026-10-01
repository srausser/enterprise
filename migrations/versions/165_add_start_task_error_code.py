"""Persist classified app conversation start failures.

Revision ID: 165
Revises: 164
"""

from typing import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = '165'
down_revision: str | None = '164'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name != 'postgresql':
        raise RuntimeError(f'Unsupported database dialect: {bind.dialect.name}')
    columns = sa.inspect(bind).get_columns('app_conversation_start_task')
    if any(column['name'] == 'error_code' for column in columns):
        # The old private revision 158 already created this column. Leaving it
        # in place preserves both the schema and all classified failure data.
        return
    op.add_column(
        'app_conversation_start_task',
        sa.Column('error_code', sa.String(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column('app_conversation_start_task', 'error_code')
