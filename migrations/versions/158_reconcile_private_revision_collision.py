"""Reconcile databases stamped by the old private revision 158.

Revision ID: 158_private_schema_bridge
Revises: 158

The previous private derivative used revision ``158`` to add
``app_conversation_start_task.error_code``. Upstream later reused that revision
for the ``verified_models`` free/default schema. An old database therefore
looks current through upstream 158 to Alembic even though those upstream changes
are absent. This bridge is immediately after the colliding revision so it runs
before any later upstream migration can depend on the skipped schema.
"""

from typing import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = '158_private_schema_bridge'
down_revision: str | None = '158'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_DEFAULT_INDEX = 'uq_verified_model_default_per_provider'
_FREE_MODELS = ('deepseek-v4-flash',)
_DEFAULT_MODEL = 'deepseek-v4-flash'


def upgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name != 'postgresql':
        raise RuntimeError(f'Unsupported database dialect: {bind.dialect.name}')

    inspector = sa.inspect(bind)
    columns = {column['name'] for column in inspector.get_columns('verified_models')}
    indexes = {index['name'] for index in inspector.get_indexes('verified_models')}
    missing_columns = {'is_free', 'is_default'} - columns
    missing_index = _DEFAULT_INDEX not in indexes

    # A database reached from upstream 158 already has the complete change.
    # PostgreSQL applies that migration transactionally, so there is no
    # partially committed seed/backfill to repeat in this case.
    if not missing_columns and not missing_index:
        return

    if 'is_free' in missing_columns:
        op.add_column(
            'verified_models',
            sa.Column(
                'is_free',
                sa.Boolean(),
                nullable=False,
                server_default=sa.text('false'),
            ),
        )
    if 'is_default' in missing_columns:
        op.add_column(
            'verified_models',
            sa.Column(
                'is_default',
                sa.Boolean(),
                nullable=False,
                server_default=sa.text('false'),
            ),
        )
    if missing_index:
        op.create_index(
            _DEFAULT_INDEX,
            'verified_models',
            ['provider'],
            unique=True,
            postgresql_where=sa.text('is_default'),
        )

    for model_name in _FREE_MODELS:
        op.execute(
            sa.text(
                """
                INSERT INTO verified_models (
                    model_name, provider, is_enabled, is_free, is_default
                )
                VALUES (:model_name, 'openhands', true, true, false)
                ON CONFLICT (model_name, provider) DO UPDATE
                SET is_enabled = true,
                    is_free = true,
                    updated_at = CURRENT_TIMESTAMP
                """
            ).bindparams(model_name=model_name)
        )

    op.execute(
        sa.text(
            """
            INSERT INTO verified_models (
                model_name, provider, is_enabled, is_free, is_default
            )
            VALUES (:model_name, 'openhands', true, true, true)
            ON CONFLICT (model_name, provider) DO UPDATE
            SET is_enabled = true,
                is_free = true,
                is_default = true,
                updated_at = CURRENT_TIMESTAMP
            """
        ).bindparams(model_name=_DEFAULT_MODEL)
    )


def downgrade() -> None:
    # This bridge cannot distinguish whether upstream 158 or the historical
    # private 158 produced the incoming schema. Avoid destructive guesses.
    pass
