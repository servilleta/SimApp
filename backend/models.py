"""
Database models for the Monte Carlo Platform

This module contains all database models for persistent storage.
Migrates away from in-memory stores to proper database persistence.
"""

from sqlalchemy import Boolean, Column, Integer, String, Text, DateTime, ForeignKey, JSON, Float
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from database import Base


class User(Base):
    """
    User model for authentication and user management
    """
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    email = Column(String, unique=True, index=True, nullable=True)
    full_name = Column(String, index=True, nullable=True)
    hashed_password = Column(String, nullable=False)
    disabled = Column(Boolean, default=False)
    is_admin = Column(Boolean, default=False, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    
    # Auth0 integration
    auth0_user_id = Column(String, unique=True, index=True, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    webhooks = relationship("WebhookConfiguration", back_populates="user", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<User(username='{self.username}', email='{self.email}')>"


class SimulationResult(Base):
    """
    Database model for simulation results
    Replaces the in-memory SIMULATION_RESULTS_STORE
    """
    __tablename__ = "simulation_results"

    id = Column(Integer, primary_key=True, index=True)
    simulation_id = Column(String, unique=True, index=True, nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    
    # Simulation metadata
    status = Column(String, nullable=False, default="pending")  # pending, running, completed, failed, cancelled
    message = Column(Text, nullable=True)
    original_filename = Column(String, nullable=True)
    engine_type = Column(String, nullable=True)
    target_name = Column(String, nullable=True)
    
    # Simulation configuration
    file_id = Column(String, nullable=True)
    iterations_requested = Column(Integer, nullable=True)
    variables_config = Column(JSON, nullable=True)  # Monte Carlo input variables
    constants_config = Column(JSON, nullable=True)  # Constants
    target_cell = Column(String, nullable=True)
    
    # Results data (when completed)
    mean = Column(Float, nullable=True)
    median = Column(Float, nullable=True)
    std_dev = Column(Float, nullable=True)
    min_value = Column(Float, nullable=True)
    max_value = Column(Float, nullable=True)
    percentiles = Column(JSON, nullable=True)
    histogram = Column(JSON, nullable=True)
    iterations_run = Column(Integer, nullable=True)
    sensitivity_analysis = Column(JSON, nullable=True)
    errors = Column(JSON, nullable=True)
    
    # Multi-target simulation results
    multi_target_result = Column(JSON, nullable=True)  # For storing multiple target results
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    
    # Relationships
    user = relationship("User")

    def __repr__(self):
        return f"<SimulationResult(id='{self.simulation_id}', status='{self.status}', user_id={self.user_id})>"

    def to_simulation_response(self):
        """Convert database model to SimulationResponse schema"""
        from simulation.schemas import SimulationResponse, SimulationResult as SimulationResultSchema
        
        # Build results if completed
        results = None
        if self.status == "completed":
            # For single-target simulations with individual result fields
            if self.mean is not None:
                results = SimulationResultSchema(
                    mean=self.mean,
                    median=self.median,
                    std_dev=self.std_dev,
                    min_value=self.min_value,
                    max_value=self.max_value,
                    percentiles=self.percentiles or {},
                    histogram=self.histogram,
                    iterations_run=self.iterations_run or 0,
                    sensitivity_analysis=self.sensitivity_analysis,
                    errors=self.errors
                )
            # For multi-target simulations, create a basic results object if multi_target_result exists
            elif self.multi_target_result is not None:
                results = SimulationResultSchema(
                    mean=0.0,  # Dummy values for multi-target simulations
                    median=0.0,
                    std_dev=0.0,
                    min_value=0.0,
                    max_value=0.0,
                    percentiles={},
                    histogram=None,
                    iterations_run=self.iterations_run or 0,
                    sensitivity_analysis=self.sensitivity_analysis,
                    errors=self.errors or []
                )
        
        # Parse multi_target_result JSON string back to object if available
        multi_target_result_obj = None
        if self.multi_target_result:
            try:
                import json
                from simulation.schemas import MultiTargetSimulationResult
                
                if isinstance(self.multi_target_result, str):
                    multi_target_data = json.loads(self.multi_target_result)
                    multi_target_result_obj = MultiTargetSimulationResult(**multi_target_data)
                else:
                    # Already an object or dict
                    multi_target_result_obj = self.multi_target_result
            except Exception as e:
                # Log error but don't fail the whole response
                print(f"Warning: Failed to parse multi_target_result for {self.simulation_id}: {e}")
                multi_target_result_obj = None

        return SimulationResponse(
            simulation_id=self.simulation_id,
            status=self.status,
            message=self.message,
            results=results,
            created_at=self.created_at.isoformat() if self.created_at else None,
            updated_at=self.updated_at.isoformat() if self.updated_at else None,
            original_filename=self.original_filename,
            engine_type=self.engine_type,
            target_name=self.target_name,
            user=getattr(self.user, 'username', None) if self.user else None,
            file_id=self.file_id,
            variables_config=self.variables_config,
            target_cell=self.target_cell,
            iterations_requested=self.iterations_requested,
            multi_target_result=multi_target_result_obj
        )


class SecurityAuditLog(Base):
    """
    Database model for security audit logs
    Replaces in-memory audit logging in SecurityMiddlewareService
    """
    __tablename__ = "security_audit_logs"

    id = Column(Integer, primary_key=True, index=True)
    
    # Event information
    event_type = Column(String, nullable=False, index=True)  # blocked_ip, suspicious_headers, etc.
    client_ip = Column(String, nullable=False, index=True)
    user_agent = Column(Text, nullable=True)
    request_id = Column(String, nullable=True, index=True)
    
    # Request details
    method = Column(String, nullable=True)
    path = Column(String, nullable=True)
    query_params = Column(JSON, nullable=True)
    headers = Column(JSON, nullable=True)
    
    # Event details
    details = Column(JSON, nullable=True)
    severity = Column(String, nullable=False, default="info")  # info, warning, error, critical
    
    # User context (if available)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    user_email = Column(String, nullable=True)
    
    # Timestamp
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    
    # Relationships
    user = relationship("User")

    def __repr__(self):
        return f"<SecurityAuditLog(event_type='{self.event_type}', client_ip='{self.client_ip}', timestamp='{self.timestamp}')>"


class UserUsageMetrics(Base):
    """
    Database model for tracking user usage metrics
    Supports tiered limits and quota enforcement
    """
    __tablename__ = "user_usage_metrics"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    
    # Usage period
    period_start = Column(DateTime(timezone=True), nullable=False, index=True)
    period_end = Column(DateTime(timezone=True), nullable=False, index=True)
    period_type = Column(String, nullable=False, default="monthly")  # daily, weekly, monthly
    
    # Usage counters
    simulations_run = Column(Integer, nullable=False, default=0)
    total_iterations = Column(Integer, nullable=False, default=0)
    files_uploaded = Column(Integer, nullable=False, default=0)
    total_file_size_mb = Column(Float, nullable=False, default=0.0)
    api_calls = Column(Integer, nullable=False, default=0)
    
    # Feature usage
    gpu_simulations = Column(Integer, nullable=False, default=0)
    concurrent_simulations_peak = Column(Integer, nullable=False, default=0)
    engines_used = Column(JSON, nullable=True)  # Array of engine types used
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    user = relationship("User")

    def __repr__(self):
        return f"<UserUsageMetrics(user_id={self.user_id}, period='{self.period_start}', simulations={self.simulations_run})>"


class UserSession(Base):
    """
    Database model for tracking active user sessions
    Used for calculating real-time active user metrics
    """
    __tablename__ = "user_sessions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    
    # Session information
    session_id = Column(String, unique=True, index=True, nullable=False)
    ip_address = Column(String, nullable=True)
    user_agent = Column(String, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    last_activity = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    expires_at = Column(DateTime(timezone=True), nullable=True)
    
    # Status
    is_active = Column(Boolean, default=True, nullable=False)
    
    # Relationships
    user = relationship("User")

    def __repr__(self):
        return f"<UserSession(user_id={self.user_id}, session_id='{self.session_id}', last_activity='{self.last_activity}')>"


class APIKey(Base):
    """
    Database model for user API keys
    Provides secure, per-user API key management
    """
    __tablename__ = "api_keys"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    
    # Key information
    key_id = Column(String, unique=True, index=True, nullable=False)  # Public part (e.g., "ak_1234...")
    key_hash = Column(String, nullable=False)  # Hashed secret part
    name = Column(String, nullable=False)  # Human-readable name
    client_id = Column(String, nullable=False)  # Client identifier for API access
    
    # Status and limits
    is_active = Column(Boolean, default=True, nullable=False)
    subscription_tier = Column(String, nullable=False, default="starter")  # starter, professional, enterprise
    
    # Usage limits
    monthly_requests = Column(Integer, nullable=False, default=1000)
    max_iterations = Column(Integer, nullable=False, default=10000)
    max_file_size_mb = Column(Integer, nullable=False, default=10)
    
    # Usage tracking
    requests_used_this_month = Column(Integer, nullable=False, default=0)
    last_used_at = Column(DateTime(timezone=True), nullable=True)
    
    # Security
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    expires_at = Column(DateTime(timezone=True), nullable=True)  # Optional expiration
    last_rotated_at = Column(DateTime(timezone=True), nullable=True)
    
    # Relationships
    user = relationship("User")

    def __repr__(self):
        return f"<APIKey(key_id='{self.key_id}', user_id={self.user_id}, active={self.is_active})>"


class UserSubscription(Base):
    """
    Database model for user subscriptions and billing
    Replaces the placeholder BillingService
    """
    __tablename__ = "user_subscriptions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, unique=True, index=True)
    
    # Subscription details
    tier = Column(String, nullable=False, default="free")  # free, starter, professional, enterprise, ultra
    status = Column(String, nullable=False, default="active")  # active, cancelled, expired, suspended, past_due, unpaid
    
    # Stripe integration
    stripe_customer_id = Column(String, nullable=True, unique=True)
    stripe_subscription_id = Column(String, nullable=True, unique=True)
    stripe_price_id = Column(String, nullable=True)
    
    # Billing details
    current_period_start = Column(DateTime(timezone=True), nullable=True)
    current_period_end = Column(DateTime(timezone=True), nullable=True)
    cancel_at_period_end = Column(Boolean, nullable=False, default=False)
    
    # Trial tracking
    is_trial = Column(Boolean, nullable=False, default=False)
    trial_start_date = Column(DateTime(timezone=True), nullable=True)
    trial_end_date = Column(DateTime(timezone=True), nullable=True)
    
    # Limits (can override defaults based on custom plans)
    max_iterations = Column(Integer, nullable=True)  # Per simulation
    concurrent_simulations = Column(Integer, nullable=True)
    file_size_mb_limit = Column(Float, nullable=True)
    max_formulas = Column(Integer, nullable=True)
    projects_stored = Column(Integer, nullable=True)
    gpu_priority = Column(String, nullable=True)  # low, standard, high, premium, dedicated
    api_calls_per_month = Column(Integer, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    user = relationship("User")

    def __repr__(self):
        return f"<UserSubscription(user_id={self.user_id}, tier='{self.tier}', status='{self.status}')>"

    def get_limits(self):
        """Get the limits for this subscription tier based on the pricing matrix"""
        # Updated tier limits based on the pricing matrix
        DEFAULT_LIMITS = {
            "trial": {
                "price": 0,
                "simulations_per_month": 100,  # Professional level
                "max_iterations": 1000000,  # Professional level (1M)
                "concurrent_simulations": 10,  # Professional level
                "file_size_mb": 10,  # Professional level (10MB)
                "max_formulas": 50000,  # Professional level
                "projects_stored": 50,  # Professional level
                "gpu_priority": "high",  # Professional level (10x faster)
                "api_calls_per_month": 0,  # NO API ACCESS
                "result_retention_days": 365,  # Professional level
                "support_response_hours": 24,  # Professional level
                "engines": ["power", "arrow", "enhanced", "ultra"],  # Professional level
                "custom_integrations": False,  # No custom integrations in trial
                "trial_duration_days": 7
            },
            "free": {
                "price": 0,
                "max_iterations": 5000,
                "concurrent_simulations": 1,
                "file_size_mb": 10,
                "max_formulas": 1000,
                "projects_stored": 3,
                "gpu_priority": "low",
                "api_calls_per_month": 0,
                "result_retention_days": 30,
                "engines": ["power", "arrow"]
            },
            "starter": {
                "price": 19,
                "simulations_per_month": 50,
                "max_iterations": 50000,  # 50K iterations (updated to match StripeService)
                "concurrent_simulations": 3,
                "file_size_mb": 25,  # 25MB files (updated to match StripeService)
                "max_formulas": 10000,
                "projects_stored": 10,
                "gpu_priority": "standard",  # Standard GPU
                "api_calls_per_month": 0,
                "result_retention_days": 90,
                "support_response_hours": 48,  # 48 hour support
                "engines": ["power", "arrow", "enhanced"],
                "custom_integrations": False,
                "overage_rate_eur_per_1000_iterations": 1.0  # €1 per 1000 iterations overage
            },
            "professional": {
                "price": 49,
                "simulations_per_month": 100,
                "max_iterations": 500000,  # 500K iterations (updated to match StripeService)
                "concurrent_simulations": 10,
                "file_size_mb": 100,  # 100MB files (updated to match StripeService)
                "max_formulas": 50000,
                "projects_stored": 50,
                "gpu_priority": "high",  # 10x Faster GPU
                "api_calls_per_month": 1000,
                "result_retention_days": 365,
                "support_response_hours": 24,  # 24 hour support
                "engines": ["power", "arrow", "enhanced", "ultra"],
                "custom_integrations": False,
                "overage_rate_eur_per_1000_iterations": 1.0  # €1 per 1000 iterations overage
            },
            "enterprise": {
                "price": 149,
                "simulations_per_month": -1,  # Unlimited
                "max_iterations": 2000000,  # 2M iterations (updated to match StripeService)
                "concurrent_simulations": 25,  # 25 (updated to match StripeService)
                "file_size_mb": 500,  # 500MB (updated to match StripeService)
                "max_formulas": 500000,  # 500K (updated to match StripeService)
                "projects_stored": -1,  # Unlimited
                "gpu_priority": "premium",  # Premium GPU (updated to match StripeService)
                "api_calls_per_month": -1,  # Unlimited
                "result_retention_days": -1,  # Unlimited
                "support_response_hours": 4,  # 4 hour support
                "engines": ["power", "arrow", "enhanced", "ultra"],
                "custom_integrations": True,  # Custom integrations available
                "overage_rate_eur_per_1000_iterations": 1.0  # €1 per 1000 iterations overage
            },
            "ultra": {
                "price": 299,
                "max_iterations": -1,  # Unlimited
                "concurrent_simulations": -1,  # Unlimited
                "file_size_mb": -1,  # No limit
                "max_formulas": 1000000,
                "projects_stored": -1,  # Unlimited
                "gpu_priority": "dedicated",
                "api_calls_per_month": -1,  # Unlimited
                "result_retention_days": 365,
                "engines": ["power", "arrow", "enhanced", "ultra"]
            },
            "on_demand": {
                "price": 0,  # No monthly fee
                "max_iterations": 0,  # No included iterations
                "concurrent_simulations": 10,  # Same as professional
                "file_size_mb": 100,  # Same as professional
                "max_formulas": 50000,  # Same as professional
                "projects_stored": 50,  # Same as professional
                "gpu_priority": "high",  # Same as professional
                "api_calls_per_month": 1000,  # Same as professional
                "result_retention_days": 365,  # Same as professional
                "support_response_hours": 24,  # Same as professional
                "engines": ["power", "arrow", "enhanced", "ultra"],  # Same as professional
                "custom_integrations": False,  # Same as professional
                "overage_rate_eur_per_1000_iterations": 1.0,  # €1 per 1000 iterations
                "pay_per_use": True  # Flag to indicate pay-per-use model
            }
        }
        
        base_limits = DEFAULT_LIMITS.get(self.tier, DEFAULT_LIMITS["free"])
        
        # Override with custom limits if set
        if self.max_iterations is not None:
            base_limits["max_iterations"] = self.max_iterations
        if self.concurrent_simulations is not None:
            base_limits["concurrent_simulations"] = self.concurrent_simulations
        if self.file_size_mb_limit is not None:
            base_limits["file_size_mb"] = self.file_size_mb_limit
        if self.max_formulas is not None:
            base_limits["max_formulas"] = self.max_formulas
        if self.projects_stored is not None:
            base_limits["projects_stored"] = self.projects_stored
        if self.gpu_priority is not None:
            base_limits["gpu_priority"] = self.gpu_priority
        if self.api_calls_per_month is not None:
            base_limits["api_calls_per_month"] = self.api_calls_per_month
            
        return base_limits
    
    def is_trial_active(self) -> bool:
        """Check if trial is currently active"""
        if not self.is_trial or not self.trial_end_date:
            return False
        
        from datetime import datetime, timezone
        return datetime.now(timezone.utc) < self.trial_end_date
    
    def is_trial_expired(self) -> bool:
        """Check if trial has expired"""
        if not self.is_trial or not self.trial_end_date:
            return False
        
        from datetime import datetime, timezone
        return datetime.now(timezone.utc) >= self.trial_end_date
    
    def trial_days_remaining(self) -> int:
        """Get number of days remaining in trial"""
        if not self.is_trial_active():
            return 0
        
        from datetime import datetime, timezone
        remaining = self.trial_end_date - datetime.now(timezone.utc)
        return max(0, remaining.days)
    
    def start_trial(self, duration_days: int = 7):
        """Start a trial period"""
        from datetime import datetime, timezone, timedelta
        
        now = datetime.now(timezone.utc)
        self.is_trial = True
        self.trial_start_date = now
        self.trial_end_date = now + timedelta(days=duration_days)
        self.tier = "trial"
        self.status = "active"
    
    def expire_trial(self):
        """Expire trial and downgrade to free"""
        self.is_trial = False
        self.tier = "free"
        # Keep trial dates for historical tracking


class WebhookConfiguration(Base):
    """
    Webhook configuration model for storing webhook endpoints
    """
    __tablename__ = "webhook_configurations"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    url = Column(String, nullable=False)
    secret = Column(String, nullable=True)  # HMAC secret for signature verification
    events = Column(JSON, nullable=False)  # List of event types to subscribe to
    enabled = Column(Boolean, default=True, nullable=False)
    
    # Client/User association
    client_id = Column(String, nullable=True)  # For B2B API clients
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)  # For regular users
    
    # Status tracking
    last_delivery_at = Column(DateTime(timezone=True), nullable=True)
    last_delivery_status = Column(String, nullable=True)  # delivered, failed, abandoned
    total_deliveries = Column(Integer, default=0)
    failed_deliveries = Column(Integer, default=0)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    user = relationship("User", back_populates="webhooks")
    deliveries = relationship("WebhookDelivery", back_populates="webhook", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<WebhookConfiguration(name='{self.name}', url='{self.url}', enabled={self.enabled})>"


class WebhookDelivery(Base):
    """
    Webhook delivery attempt tracking model
    """
    __tablename__ = "webhook_deliveries"

    id = Column(Integer, primary_key=True, index=True)
    webhook_id = Column(Integer, ForeignKey("webhook_configurations.id"), nullable=False)
    simulation_id = Column(String, nullable=False, index=True)
    event_type = Column(String, nullable=False)
    
    # Delivery details
    attempt = Column(Integer, default=1)
    status = Column(String, nullable=False)  # pending, delivered, failed, retrying, abandoned
    payload_data = Column(JSON, nullable=False)
    
    # Response details
    response_status = Column(Integer, nullable=True)
    response_body = Column(Text, nullable=True)
    response_time_ms = Column(Integer, nullable=True)
    error_message = Column(Text, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    delivered_at = Column(DateTime(timezone=True), nullable=True)

    # Relationships
    webhook = relationship("WebhookConfiguration", back_populates="deliveries")

    def __repr__(self):
        return f"<WebhookDelivery(webhook_id={self.webhook_id}, event='{self.event_type}', status='{self.status}')>"