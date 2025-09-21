"""add trial tracking fields

Revision ID: trial_fields_001
Revises: b6bdaf33918a
Create Date: 2025-09-16 12:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import sqlite

# revision identifiers, used by Alembic.
revision = 'trial_fields_001'
down_revision = 'b6bdaf33918a'
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add trial tracking fields to user_subscriptions table"""
    # SQLite doesn't support ALTER COLUMN, so we use batch mode
    with op.batch_alter_table('user_subscriptions', schema=None) as batch_op:
        batch_op.add_column(sa.Column('is_trial', sa.Boolean(), nullable=False, default=False))
        batch_op.add_column(sa.Column('trial_start_date', sa.DateTime(timezone=True), nullable=True))
        batch_op.add_column(sa.Column('trial_end_date', sa.DateTime(timezone=True), nullable=True))


def downgrade() -> None:
    """Remove trial tracking fields from user_subscriptions table"""
    with op.batch_alter_table('user_subscriptions', schema=None) as batch_op:
        batch_op.drop_column('trial_end_date')
        batch_op.drop_column('trial_start_date')
        batch_op.drop_column('is_trial')
