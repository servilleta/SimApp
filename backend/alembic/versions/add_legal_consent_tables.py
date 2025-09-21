"""Add legal consent tables

Revision ID: add_legal_consent
Revises: add_stripe_subscription_features
Create Date: 2025-01-19 12:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = 'add_legal_consent'
down_revision = 'add_stripe_subscription_features'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create legal_documents table
    op.create_table('legal_documents',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('document_type', sa.String(), nullable=False),
        sa.Column('version', sa.String(), nullable=False),
        sa.Column('title', sa.String(), nullable=False),
        sa.Column('content_path', sa.String(), nullable=False),
        sa.Column('content_hash', sa.String(), nullable=False),
        sa.Column('effective_date', sa.DateTime(timezone=True), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=False),
        sa.Column('requires_explicit_consent', sa.Boolean(), nullable=False),
        sa.Column('applies_to_existing_users', sa.Boolean(), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_legal_documents_id'), 'legal_documents', ['id'], unique=False)
    op.create_index(op.f('ix_legal_documents_document_type'), 'legal_documents', ['document_type'], unique=False)

    # Create user_legal_consents table
    op.create_table('user_legal_consents',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('document_id', sa.Integer(), nullable=False),
        sa.Column('consent_given', sa.Boolean(), nullable=False),
        sa.Column('consent_method', sa.String(), nullable=False),
        sa.Column('consent_timestamp', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('ip_address', sa.String(), nullable=True),
        sa.Column('user_agent', sa.Text(), nullable=True),
        sa.Column('session_id', sa.String(), nullable=True),
        sa.Column('consent_context', sa.JSON(), nullable=True),
        sa.Column('withdrawn_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('withdrawal_reason', sa.String(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(['document_id'], ['legal_documents.id'], ),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_user_legal_consents_id'), 'user_legal_consents', ['id'], unique=False)
    op.create_index(op.f('ix_user_legal_consents_user_id'), 'user_legal_consents', ['user_id'], unique=False)
    op.create_index(op.f('ix_user_legal_consents_document_id'), 'user_legal_consents', ['document_id'], unique=False)

    # Create consent_audit_logs table
    op.create_table('consent_audit_logs',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=True),
        sa.Column('action_type', sa.String(), nullable=False),
        sa.Column('document_type', sa.String(), nullable=True),
        sa.Column('document_version', sa.String(), nullable=True),
        sa.Column('details', sa.JSON(), nullable=True),
        sa.Column('ip_address', sa.String(), nullable=True),
        sa.Column('user_agent', sa.Text(), nullable=True),
        sa.Column('session_id', sa.String(), nullable=True),
        sa.Column('performed_by_admin', sa.Boolean(), nullable=False),
        sa.Column('admin_user_id', sa.Integer(), nullable=True),
        sa.Column('timestamp', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.ForeignKeyConstraint(['admin_user_id'], ['users.id'], ),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_consent_audit_logs_id'), 'consent_audit_logs', ['id'], unique=False)
    op.create_index(op.f('ix_consent_audit_logs_user_id'), 'consent_audit_logs', ['user_id'], unique=False)
    op.create_index(op.f('ix_consent_audit_logs_timestamp'), 'consent_audit_logs', ['timestamp'], unique=False)


def downgrade() -> None:
    op.drop_index(op.f('ix_consent_audit_logs_timestamp'), table_name='consent_audit_logs')
    op.drop_index(op.f('ix_consent_audit_logs_user_id'), table_name='consent_audit_logs')
    op.drop_index(op.f('ix_consent_audit_logs_id'), table_name='consent_audit_logs')
    op.drop_table('consent_audit_logs')
    
    op.drop_index(op.f('ix_user_legal_consents_document_id'), table_name='user_legal_consents')
    op.drop_index(op.f('ix_user_legal_consents_user_id'), table_name='user_legal_consents')
    op.drop_index(op.f('ix_user_legal_consents_id'), table_name='user_legal_consents')
    op.drop_table('user_legal_consents')
    
    op.drop_index(op.f('ix_legal_documents_document_type'), table_name='legal_documents')
    op.drop_index(op.f('ix_legal_documents_id'), table_name='legal_documents')
    op.drop_table('legal_documents')




