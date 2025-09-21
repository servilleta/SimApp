"""
Stripe Integration Service for Monte Carlo Platform

Handles Stripe customer creation, subscription management, and payment processing
for the five-tier pricing structure: Free, Starter, Professional, Enterprise, Ultra
"""

import logging
import stripe
from typing import Optional, Dict, Any, List
from sqlalchemy.orm import Session
from datetime import datetime, timezone

from config import settings
from models import User as UserModel, UserSubscription
from database import get_db

logger = logging.getLogger(__name__)

# Configure Stripe
stripe.api_key = settings.STRIPE_SECRET_KEY

class StripeService:
    """
    Service class for handling all Stripe operations
    """
    
    # Stripe Price IDs for each plan (created by setup_stripe_products.py)
    PRICE_IDS = {
        "free": None,  # Free tier doesn't have a Stripe price
        "basic": "price_1S7fWEGkZec0aS3MDjCHFnkH",  # Maps to Starter ($19/mo)
        "pro": "price_1S7fWFGkZec0aS3MSHiUzPoh",  # Maps to Professional ($49/mo)
        "enterprise": "price_1S7fWGGkZec0aS3M3lvFVmUa",  # Maps to Ultra ($299/mo)
        "on_demand": None,  # Pay-per-use plan (no subscription needed)
        # Legacy mappings for direct Stripe plan names
        "starter": "price_1S7fWEGkZec0aS3MDjCHFnkH",
        "professional": "price_1S7fWFGkZec0aS3MSHiUzPoh",
        "ultra": "price_1S7fWGGkZec0aS3M3lvFVmUa"
    }
    
    PLAN_METADATA = {
        "starter": {
            "name": "Starter Plan",
            "description": "Perfect for small teams getting started with Monte Carlo simulations",
            "features": ["50K max iterations", "3 concurrent simulations", "25MB file limit", "10K formulas", "10 projects", "Overage: €1/1000 iterations"]
        },
        "professional": {
            "name": "Professional Plan", 
            "description": "Advanced features for professional analysts and teams",
            "features": ["500K max iterations", "10 concurrent simulations", "100MB file limit", "50K formulas", "50 projects", "API access", "Overage: €1/1000 iterations"]
        },
        "enterprise": {
            "name": "Enterprise Plan",
            "description": "Enterprise-grade features for large organizations", 
            "features": ["2M max iterations", "25 concurrent simulations", "500MB file limit", "500K formulas", "Unlimited projects", "Unlimited API calls", "Overage: €1/1000 iterations"]
        },
        "ultra": {
            "name": "Ultra Plan",
            "description": "Unlimited power for the most demanding simulations",
            "features": ["Unlimited iterations", "Unlimited concurrent simulations", "No file size limit", "1M+ formulas", "Unlimited projects", "Dedicated GPU priority"]
        },
        "on_demand": {
            "name": "On Demand",
            "description": "Pay only for what you use - perfect for occasional simulations",
            "features": ["Pay per use", "€1 per 1000 iterations", "No monthly commitment", "Same features as Professional", "10MB file size limit", "API access"]
        }
    }

    @staticmethod
    def create_customer(user: UserModel) -> Optional[str]:
        """
        Create a Stripe customer for a user
        
        Args:
            user: User model instance
            
        Returns:
            Stripe customer ID or None if failed
        """
        try:
            customer = stripe.Customer.create(
                email=user.email,
                name=user.full_name or user.username,
                metadata={
                    "user_id": str(user.id),
                    "username": user.username,
                    "platform": "monte_carlo_platform"
                }
            )
            logger.info(f"Created Stripe customer {customer.id} for user {user.id}")
            return customer.id
            
        except stripe.error.StripeError as e:
            logger.error(f"Failed to create Stripe customer for user {user.id}: {str(e)}")
            return None
    
    @staticmethod
    def create_subscription(
        customer_id: str, 
        price_id: str, 
        metadata: Optional[Dict[str, str]] = None
    ) -> Optional[stripe.Subscription]:
        """
        Create a Stripe subscription
        
        Args:
            customer_id: Stripe customer ID
            price_id: Stripe price ID for the plan
            metadata: Optional metadata to attach to the subscription
            
        Returns:
            Stripe subscription object or None if failed
        """
        try:
            subscription = stripe.Subscription.create(
                customer=customer_id,
                items=[{"price": price_id}],
                payment_behavior="default_incomplete",
                payment_settings={"save_default_payment_method": "on_subscription"},
                expand=["latest_invoice.payment_intent"],
                metadata=metadata or {}
            )
            logger.info(f"Created subscription {subscription.id} for customer {customer_id}")
            return subscription
            
        except stripe.error.StripeError as e:
            logger.error(f"Failed to create subscription for customer {customer_id}: {str(e)}")
            return None
    
    @staticmethod
    def get_subscription(subscription_id: str) -> Optional[stripe.Subscription]:
        """
        Retrieve a Stripe subscription
        
        Args:
            subscription_id: Stripe subscription ID
            
        Returns:
            Stripe subscription object or None if not found
        """
        try:
            return stripe.Subscription.retrieve(subscription_id)
        except stripe.error.StripeError as e:
            logger.error(f"Failed to retrieve subscription {subscription_id}: {str(e)}")
            return None
    
    @staticmethod
    def cancel_subscription(
        subscription_id: str, 
        at_period_end: bool = True
    ) -> Optional[stripe.Subscription]:
        """
        Cancel a Stripe subscription
        
        Args:
            subscription_id: Stripe subscription ID
            at_period_end: Whether to cancel at period end or immediately
            
        Returns:
            Updated subscription object or None if failed
        """
        try:
            if at_period_end:
                subscription = stripe.Subscription.modify(
                    subscription_id,
                    cancel_at_period_end=True
                )
            else:
                subscription = stripe.Subscription.delete(subscription_id)
            
            logger.info(f"Cancelled subscription {subscription_id}, at_period_end={at_period_end}")
            return subscription
            
        except stripe.error.StripeError as e:
            logger.error(f"Failed to cancel subscription {subscription_id}: {str(e)}")
            return None
    
    @staticmethod
    def update_subscription(
        subscription_id: str,
        new_price_id: str,
        proration_behavior: str = "create_prorations"
    ) -> Optional[stripe.Subscription]:
        """
        Update a subscription to a new plan
        
        Args:
            subscription_id: Stripe subscription ID
            new_price_id: New Stripe price ID
            proration_behavior: How to handle proration
            
        Returns:
            Updated subscription object or None if failed
        """
        try:
            subscription = stripe.Subscription.retrieve(subscription_id)
            
            stripe.Subscription.modify(
                subscription_id,
                items=[{
                    "id": subscription["items"]["data"][0].id,
                    "price": new_price_id,
                }],
                proration_behavior=proration_behavior
            )
            
            updated_subscription = stripe.Subscription.retrieve(subscription_id)
            logger.info(f"Updated subscription {subscription_id} to price {new_price_id}")
            return updated_subscription
            
        except stripe.error.StripeError as e:
            logger.error(f"Failed to update subscription {subscription_id}: {str(e)}")
            return None
    
    @staticmethod
    def create_checkout_session(
        customer_id: str,
        price_id: str,
        success_url: str,
        cancel_url: str,
        user_id: int
    ) -> Optional[stripe.checkout.Session]:
        """
        Create a Stripe Checkout session for subscription
        
        Args:
            customer_id: Stripe customer ID
            price_id: Stripe price ID
            success_url: URL to redirect on success
            cancel_url: URL to redirect on cancel
            user_id: Internal user ID
            
        Returns:
            Checkout session or None if failed
        """
        try:
            session = stripe.checkout.Session.create(
                customer=customer_id,
                payment_method_types=["card"],
                line_items=[{
                    "price": price_id,
                    "quantity": 1,
                }],
                mode="subscription",
                success_url=success_url,
                cancel_url=cancel_url,
                metadata={
                    "user_id": str(user_id)
                }
            )
            logger.info(f"Created checkout session {session.id} for customer {customer_id}")
            return session
            
        except stripe.error.StripeError as e:
            logger.error(f"Failed to create checkout session for customer {customer_id}: {str(e)}")
            return None
    
    @staticmethod
    def create_billing_portal_session(
        customer_id: str,
        return_url: str
    ) -> Optional[stripe.billing_portal.Session]:
        """
        Create a Stripe billing portal session
        
        Args:
            customer_id: Stripe customer ID
            return_url: URL to return to from billing portal
            
        Returns:
            Billing portal session or None if failed
        """
        try:
            session = stripe.billing_portal.Session.create(
                customer=customer_id,
                return_url=return_url
            )
            logger.info(f"Created billing portal session for customer {customer_id}")
            return session
            
        except stripe.error.StripeError as e:
            logger.error(f"Failed to create billing portal session for customer {customer_id}: {str(e)}")
            return None
    
    @classmethod
    def sync_subscription_status(
        cls, 
        db: Session, 
        user_subscription: UserSubscription,
        stripe_subscription: stripe.Subscription
    ) -> UserSubscription:
        """
        Sync database subscription status with Stripe
        
        Args:
            db: Database session
            user_subscription: UserSubscription model instance
            stripe_subscription: Stripe subscription object
            
        Returns:
            Updated UserSubscription instance
        """
        # Map Stripe status to our internal status
        status_mapping = {
            "active": "active",
            "past_due": "past_due", 
            "unpaid": "unpaid",
            "canceled": "cancelled",
            "incomplete": "suspended",
            "incomplete_expired": "expired",
            "trialing": "active"
        }
        
        user_subscription.status = status_mapping.get(
            stripe_subscription.status, 
            "suspended"
        )
        
        # Update billing period
        if stripe_subscription.current_period_start:
            user_subscription.current_period_start = datetime.fromtimestamp(
                stripe_subscription.current_period_start, 
                tz=timezone.utc
            )
        
        if stripe_subscription.current_period_end:
            user_subscription.current_period_end = datetime.fromtimestamp(
                stripe_subscription.current_period_end,
                tz=timezone.utc
            )
        
        user_subscription.cancel_at_period_end = stripe_subscription.cancel_at_period_end
        
        db.commit()
        db.refresh(user_subscription)
        
        logger.info(f"Synced subscription status for user {user_subscription.user_id}: {user_subscription.status}")
        return user_subscription
    
    @classmethod
    def get_plan_limits(cls, tier: str) -> Dict[str, Any]:
        """
        Get the limits for a specific plan tier
        
        Args:
            tier: Plan tier name
            
        Returns:
            Dictionary containing plan limits and features
        """
        # This mirrors the get_limits method in UserSubscription model
        # but can be called without a database instance
        limits = {
            "free": {
                "price": 0,
                "max_iterations": 5000,
                "concurrent_simulations": 1,
                "file_size_mb": 10,
                "max_formulas": 1000,
                "projects_stored": 3,
                "gpu_priority": "low",
                "api_calls_per_month": 0,
                "overage_rate_eur_per_1000_iterations": None,  # No overage for free
            },
            "starter": {
                "price": 19,
                "max_iterations": 50000,
                "concurrent_simulations": 3,
                "file_size_mb": 25,
                "max_formulas": 10000,
                "projects_stored": 10,
                "gpu_priority": "standard",
                "api_calls_per_month": 0,
                "overage_rate_eur_per_1000_iterations": 1.0,  # €1 per 1000 iterations
            },
            "professional": {
                "price": 49,
                "max_iterations": 500000,
                "concurrent_simulations": 10,
                "file_size_mb": 100,
                "max_formulas": 50000,
                "projects_stored": 50,
                "gpu_priority": "high",
                "api_calls_per_month": 1000,
                "overage_rate_eur_per_1000_iterations": 1.0,  # €1 per 1000 iterations
            },
            "enterprise": {
                "price": 149,
                "max_iterations": 2000000,
                "concurrent_simulations": 25,
                "file_size_mb": 500,
                "max_formulas": 500000,
                "projects_stored": -1,  # Unlimited
                "gpu_priority": "premium", 
                "api_calls_per_month": -1,  # Unlimited
                "overage_rate_eur_per_1000_iterations": 1.0,  # €1 per 1000 iterations
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
                "overage_rate_eur_per_1000_iterations": None,  # Unlimited plan, no overage
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
                "overage_rate_eur_per_1000_iterations": 1.0,  # €1 per 1000 iterations (pay-per-use)
                "pay_per_use": True,  # Flag to indicate this is a pay-per-use plan
            }
        }
        
        return limits.get(tier, limits["free"])
    
    @classmethod
    def get_all_plans(cls) -> List[Dict[str, Any]]:
        """
        Get information about all available plans
        
        Returns:
            List of plan dictionaries with limits and metadata
        """
        # Define pricing for each tier (monthly in cents)
        tier_pricing = {
            "free": 0,
            "basic": 1900,  # $19.00 
            "pro": 4900,    # $49.00
            "enterprise": 14900,  # $149.00 (corrected from $299 to match the actual enterprise price)
            "on_demand": 0,  # No monthly fee, pay-per-use
            # Legacy Stripe tier names
            "starter": 1900,      # $19.00
            "professional": 4900, # $49.00
            "ultra": 29900        # $299.00
        }
        
        plans = []
        # Use frontend-friendly tier names including on_demand
        for tier in ["free", "basic", "pro", "enterprise", "on_demand"]:
            metadata = cls.PLAN_METADATA.get(tier, cls.PLAN_METADATA.get({
                "basic": "starter",
                "pro": "professional", 
                "enterprise": "ultra"
            }.get(tier), {}))
            
            plan_info = {
                "tier": tier,
                "name": metadata.get("name", tier.title() + " Plan"),
                "description": metadata.get("description", f"{tier.title()} tier for Monte Carlo simulations"),
                "price": tier_pricing.get(tier, 0),
                "limits": cls.get_plan_limits(tier),
                "features": metadata.get("features", []),
                "stripe_price_id": cls.PRICE_IDS.get(tier)
            }
            plans.append(plan_info)
        
        return plans
    
    @classmethod
    def calculate_overage_charge(cls, tier: str, iterations_used: int) -> Dict[str, Any]:
        """
        Calculate overage charges for a given tier and iteration usage
        
        Args:
            tier: Plan tier name
            iterations_used: Number of iterations used
            
        Returns:
            Dictionary with overage calculation details
        """
        limits = cls.get_plan_limits(tier)
        max_iterations = limits.get("max_iterations", 0)
        overage_rate = limits.get("overage_rate_eur_per_1000_iterations")
        
        # If unlimited plan or no overage rate, return no charge
        if max_iterations == -1 or overage_rate is None:
            return {
                "overage_iterations": 0,
                "overage_charge_eur": 0.0,
                "total_iterations": iterations_used,
                "included_iterations": max_iterations if max_iterations != -1 else iterations_used
            }
        
        # For on_demand plan, all iterations are overage
        if tier == "on_demand":
            overage_iterations = iterations_used
            included_iterations = 0
        else:
            # For subscription plans, calculate overage
            overage_iterations = max(0, iterations_used - max_iterations)
            included_iterations = min(iterations_used, max_iterations)
        
        # Calculate charge in euros (rate is per 1000 iterations)
        overage_charge_eur = (overage_iterations / 1000.0) * overage_rate
        
        return {
            "overage_iterations": overage_iterations,
            "overage_charge_eur": round(overage_charge_eur, 2),
            "total_iterations": iterations_used,
            "included_iterations": included_iterations,
            "rate_per_1000": overage_rate
        }
    
    @staticmethod
    def create_payment_intent_for_overage(
        customer_id: str,
        amount_eur: float,
        description: str,
        metadata: Optional[Dict[str, str]] = None
    ) -> Optional[stripe.PaymentIntent]:
        """
        Create a Stripe PaymentIntent for overage charges
        
        Args:
            customer_id: Stripe customer ID
            amount_eur: Amount in euros
            description: Payment description
            metadata: Optional metadata dictionary
            
        Returns:
            Stripe PaymentIntent object or None if failed
        """
        try:
            # Convert euros to cents for Stripe
            amount_cents = int(amount_eur * 100)
            
            payment_intent = stripe.PaymentIntent.create(
                amount=amount_cents,
                currency="eur",
                customer=customer_id,
                description=description,
                metadata=metadata or {},
                automatic_payment_methods={
                    "enabled": True
                }
            )
            
            logger.info(f"Created PaymentIntent {payment_intent.id} for €{amount_eur}")
            return payment_intent
            
        except stripe.error.StripeError as e:
            logger.error(f"Failed to create PaymentIntent: {str(e)}")
            return None
