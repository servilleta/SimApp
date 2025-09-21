"""Add Stripe subscription features

Revision ID: stripe_subscription_features
Revises: 
Create Date: 2025-09-15 12:00:00.000000

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = 'stripe_subscription_features'
down_revision = None  # Update this to the previous revision if you have one
branch_labels = None
depends_on = None

def upgrade():
    """
    Upgrade to add new subscription features for Stripe integration
    """
    # Update user_subscriptions table with new columns
    op.add_column('user_subscriptions', sa.Column('max_iterations', sa.Integer(), nullable=True))
    op.add_column('user_subscriptions', sa.Column('max_formulas', sa.Integer(), nullable=True))
    op.add_column('user_subscriptions', sa.Column('projects_stored', sa.Integer(), nullable=True))
    op.add_column('user_subscriptions', sa.Column('gpu_priority', sa.String(), nullable=True))
    op.add_column('user_subscriptions', sa.Column('api_calls_per_month', sa.Integer(), nullable=True))
    op.add_column('user_subscriptions', sa.Column('stripe_webhook_endpoint_secret', sa.String(), nullable=True))
    
    # Update existing subscriptions to have proper tier values
    # This will ensure any existing subscriptions are properly categorized
    connection = op.get_bind()
    
    # Update any existing 'basic' tier to 'starter'
    connection.execute(sa.text("""
        UPDATE user_subscriptions 
        SET tier = 'starter' 
        WHERE tier = 'basic'
    """))
    
    # Update any existing 'pro' tier to 'professional'  
    connection.execute(sa.text("""
        UPDATE user_subscriptions 
        SET tier = 'professional' 
        WHERE tier = 'pro'
    """))
    
    # Add new status values to support Stripe subscription statuses
    # Note: This assumes you're using PostgreSQL. For SQLite, constraints work differently
    try:
        # Try to add constraint for PostgreSQL
        op.create_check_constraint(
            'valid_subscription_status',
            'user_subscriptions',
            sa.text("status IN ('active', 'cancelled', 'expired', 'suspended', 'past_due', 'unpaid')")
        )
    except:
        # SQLite doesn't support adding check constraints after table creation
        pass
    
    try:
        # Try to add constraint for tier validation
        op.create_check_constraint(
            'valid_subscription_tier',
            'user_subscriptions', 
            sa.text("tier IN ('free', 'starter', 'professional', 'enterprise', 'ultra')")
        )
    except:
        # SQLite doesn't support adding check constraints after table creation
        pass

def downgrade():
    """
    Downgrade to remove Stripe subscription features
    """
    # Remove the new columns
    op.drop_column('user_subscriptions', 'stripe_webhook_endpoint_secret')
    op.drop_column('user_subscriptions', 'api_calls_per_month')
    op.drop_column('user_subscriptions', 'gpu_priority')
    op.drop_column('user_subscriptions', 'projects_stored')
    op.drop_column('user_subscriptions', 'max_formulas')
    op.drop_column('user_subscriptions', 'max_iterations')
    
    # Revert tier names back to old format
    connection = op.get_bind()
    
    connection.execute(sa.text("""
        UPDATE user_subscriptions 
        SET tier = 'basic' 
        WHERE tier = 'starter'
    """))
    
    connection.execute(sa.text("""
        UPDATE user_subscriptions 
        SET tier = 'pro' 
        WHERE tier = 'professional'
    """))
    
    # Remove tier constraint if it exists
    try:
        op.drop_constraint('valid_subscription_tier', 'user_subscriptions', type_='check')
    except:
        pass
    
    # Remove status constraint if it exists
    try:
        op.drop_constraint('valid_subscription_status', 'user_subscriptions', type_='check')
    except:
        pass
