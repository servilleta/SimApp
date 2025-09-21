"""Merge migration heads

Revision ID: 8755144999a0
Revises: stripe_subscription_features, trial_fields_001
Create Date: 2025-09-17 14:34:01.313332

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '8755144999a0'
down_revision: Union[str, None] = ('stripe_subscription_features', 'trial_fields_001')
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
