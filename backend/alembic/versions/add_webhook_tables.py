"""Add webhook tables

Revision ID: add_webhook_tables
Revises: add_stripe_subscription_features
Create Date: 2024-09-19 11:55:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = 'add_webhook_tables'
down_revision = 'stripe_subscription_features'
branch_labels = None
depends_on = None


def upgrade():
    # Create webhook_configurations table
    op.create_table('webhook_configurations',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('name', sa.String(), nullable=False),
        sa.Column('url', sa.String(), nullable=False),
        sa.Column('secret', sa.String(), nullable=True),
        sa.Column('events', sa.JSON(), nullable=False),
        sa.Column('enabled', sa.Boolean(), nullable=False),
        sa.Column('client_id', sa.String(), nullable=True),
        sa.Column('user_id', sa.Integer(), nullable=True),
        sa.Column('last_delivery_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('last_delivery_status', sa.String(), nullable=True),
        sa.Column('total_deliveries', sa.Integer(), nullable=True),
        sa.Column('failed_deliveries', sa.Integer(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_webhook_configurations_id'), 'webhook_configurations', ['id'], unique=False)

    # Create webhook_deliveries table
    op.create_table('webhook_deliveries',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('webhook_id', sa.Integer(), nullable=False),
        sa.Column('simulation_id', sa.String(), nullable=False),
        sa.Column('event_type', sa.String(), nullable=False),
        sa.Column('attempt', sa.Integer(), nullable=True),
        sa.Column('status', sa.String(), nullable=False),
        sa.Column('payload_data', sa.JSON(), nullable=False),
        sa.Column('response_status', sa.Integer(), nullable=True),
        sa.Column('response_body', sa.Text(), nullable=True),
        sa.Column('response_time_ms', sa.Integer(), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('delivered_at', sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(['webhook_id'], ['webhook_configurations.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_webhook_deliveries_id'), 'webhook_deliveries', ['id'], unique=False)
    op.create_index(op.f('ix_webhook_deliveries_simulation_id'), 'webhook_deliveries', ['simulation_id'], unique=False)


def downgrade():
    op.drop_index(op.f('ix_webhook_deliveries_simulation_id'), table_name='webhook_deliveries')
    op.drop_index(op.f('ix_webhook_deliveries_id'), table_name='webhook_deliveries')
    op.drop_table('webhook_deliveries')
    op.drop_index(op.f('ix_webhook_configurations_id'), table_name='webhook_configurations')
    op.drop_table('webhook_configurations')
