"""
Billing and Subscription Management API

Provides endpoints for Stripe integration, subscription management,
and billing operations for the Monte Carlo Platform
"""

import logging
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Request
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field
from dateutil.relativedelta import relativedelta

from database import get_db
from auth.auth0_dependencies import get_current_active_auth0_user
from models import User as UserModel, UserSubscription
from services.stripe_service import StripeService
from config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/billing", tags=["billing"])

# Pydantic schemas for API requests and responses
class PlanInfo(BaseModel):
    tier: str
    name: str
    description: str
    price: int
    limits: dict
    features: List[str]
    stripe_price_id: Optional[str] = None

class SubscriptionInfo(BaseModel):
    tier: str
    status: str
    current_period_start: Optional[str] = None
    current_period_end: Optional[str] = None
    cancel_at_period_end: bool = False
    limits: dict

class CheckoutSessionRequest(BaseModel):
    plan: str = Field(..., description="Plan tier to subscribe to")
    success_url: str = Field(..., description="URL to redirect on successful payment")
    cancel_url: str = Field(..., description="URL to redirect on cancelled payment")

class CheckoutSessionResponse(BaseModel):
    checkout_url: str
    session_id: str

class BillingPortalResponse(BaseModel):
    portal_url: str

@router.get("/plans", response_model=List[PlanInfo])
async def get_available_plans():
    """
    Get all available subscription plans
    """
    try:
        # Use real Stripe service now that keys are configured
        plans = StripeService.get_all_plans()
        return [PlanInfo(**plan) for plan in plans]
    except Exception as e:
        logger.error(f"Failed to get plans: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve plans"
        )

@router.get("/subscription", response_model=SubscriptionInfo)
async def get_user_subscription(
    current_user: UserModel = Depends(get_current_active_auth0_user),
    db: Session = Depends(get_db)
):
    """
    Get current user's subscription information
    """
    try:
        # Get user subscription from database
        user_subscription = db.query(UserSubscription).filter(
            UserSubscription.user_id == current_user.id
        ).first()
        
        if not user_subscription:
            # Create default free subscription if none exists
            user_subscription = UserSubscription(
                user_id=current_user.id,
                tier="free",
                status="active"
            )
            db.add(user_subscription)
            db.commit()
            db.refresh(user_subscription)
        
        # If user has a Stripe subscription, sync the status
        if user_subscription.stripe_subscription_id:
            stripe_subscription = StripeService.get_subscription(
                user_subscription.stripe_subscription_id
            )
            if stripe_subscription:
                user_subscription = StripeService.sync_subscription_status(
                    db, user_subscription, stripe_subscription
                )
        
        return SubscriptionInfo(
            tier=user_subscription.tier,
            status=user_subscription.status,
            current_period_start=user_subscription.current_period_start.isoformat() if user_subscription.current_period_start else None,
            current_period_end=user_subscription.current_period_end.isoformat() if user_subscription.current_period_end else None,
            cancel_at_period_end=user_subscription.cancel_at_period_end,
            limits=user_subscription.get_limits()
        )
        
    except Exception as e:
        logger.error(f"Failed to get subscription for user {current_user.id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve subscription information"
        )

@router.post("/checkout", response_model=CheckoutSessionResponse)
@router.post("/create-checkout-session", response_model=CheckoutSessionResponse)  # Alias for frontend compatibility
async def create_checkout_session(
    request: CheckoutSessionRequest,
    current_user: UserModel = Depends(get_current_active_auth0_user),
    db: Session = Depends(get_db)
):
    """
    Create a Stripe checkout session for subscription
    """
    try:
        logger.info(f"Creating checkout session for user {current_user.id} with plan: {request.plan}")
        
        # Check Stripe configuration
        if not settings.STRIPE_SECRET_KEY or len(settings.STRIPE_SECRET_KEY) < 20:
            logger.error("Stripe API key not properly configured")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Payment system not configured. Please contact support."
            )
        
        # Validate plan
        logger.info(f"Available plans in StripeService.PRICE_IDS: {list(StripeService.PRICE_IDS.keys())}")
        logger.info(f"Requested plan: '{request.plan}' (type: {type(request.plan)})")
        
        if request.plan not in StripeService.PRICE_IDS:
            logger.error(f"Invalid plan requested: {request.plan}. Valid plans: {list(StripeService.PRICE_IDS.keys())}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid plan: {request.plan}. Valid plans: {list(StripeService.PRICE_IDS.keys())}"
            )
        
        if request.plan == "free":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot create checkout for free plan"
            )
        
        if request.plan == "on_demand":
            # For on_demand plan, just set up the customer and redirect to success
            # No subscription needed, users will be charged per usage
            return CheckoutSessionResponse(
                checkout_url=request.success_url + "?plan=on_demand",
                session_id="on_demand_setup"
            )
        
        price_id = StripeService.PRICE_IDS[request.plan]
        if not price_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Price ID not configured for plan: {request.plan}"
            )
        
        # Get or create user subscription record
        user_subscription = db.query(UserSubscription).filter(
            UserSubscription.user_id == current_user.id
        ).first()
        
        if not user_subscription:
            user_subscription = UserSubscription(
                user_id=current_user.id,
                tier="free",
                status="active"
            )
            db.add(user_subscription)
            db.commit()
            db.refresh(user_subscription)
        
        # Get or create Stripe customer
        customer_id = user_subscription.stripe_customer_id
        if not customer_id:
            customer_id = StripeService.create_customer(current_user)
            if not customer_id:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Failed to create Stripe customer"
                )
            user_subscription.stripe_customer_id = customer_id
            db.commit()
        
        # Create checkout session
        session = StripeService.create_checkout_session(
            customer_id=customer_id,
            price_id=price_id,
            success_url=request.success_url,
            cancel_url=request.cancel_url,
            user_id=current_user.id
        )
        
        if not session:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create checkout session"
            )
        
        return CheckoutSessionResponse(
            checkout_url=session.url,
            session_id=session.id
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create checkout session for user {current_user.id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create checkout session"
        )

@router.post("/portal", response_model=BillingPortalResponse)
async def create_billing_portal(
    current_user: UserModel = Depends(get_current_active_auth0_user),
    db: Session = Depends(get_db)
):
    """
    Create a Stripe billing portal session for subscription management
    """
    try:
        # Get user subscription
        user_subscription = db.query(UserSubscription).filter(
            UserSubscription.user_id == current_user.id
        ).first()
        
        if not user_subscription or not user_subscription.stripe_customer_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No active subscription found"
            )
        
        # Create billing portal session
        session = StripeService.create_billing_portal_session(
            customer_id=user_subscription.stripe_customer_id,
            return_url=f"https://simapp.ai/dashboard"  # Replace with your actual domain
        )
        
        if not session:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create billing portal session"
            )
        
        return BillingPortalResponse(portal_url=session.url)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create billing portal for user {current_user.id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create billing portal session"
        )

@router.post("/cancel")
async def cancel_subscription(
    current_user: UserModel = Depends(get_current_active_auth0_user),
    db: Session = Depends(get_db)
):
    """
    Cancel user's subscription (at period end)
    """
    try:
        # Get user subscription
        user_subscription = db.query(UserSubscription).filter(
            UserSubscription.user_id == current_user.id
        ).first()
        
        if not user_subscription or not user_subscription.stripe_subscription_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No active subscription found"
            )
        
        # Cancel subscription at period end
        stripe_subscription = StripeService.cancel_subscription(
            user_subscription.stripe_subscription_id,
            at_period_end=True
        )
        
        if not stripe_subscription:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to cancel subscription"
            )
        
        # Sync status with database
        StripeService.sync_subscription_status(db, user_subscription, stripe_subscription)
        
        return {"message": "Subscription will be cancelled at the end of the current period"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to cancel subscription for user {current_user.id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to cancel subscription"
        )

@router.post("/calculate-overage")
async def calculate_overage_cost(
    iterations_used: int,
    current_user: UserModel = Depends(get_current_active_auth0_user),
    db: Session = Depends(get_db)
):
    """
    Calculate overage cost for current user's plan based on iterations used
    """
    try:
        # Get user subscription
        user_subscription = db.query(UserSubscription).filter(
            UserSubscription.user_id == current_user.id
        ).first()
        
        if not user_subscription:
            user_subscription = UserSubscription(
                user_id=current_user.id,
                tier="free",
                status="active"
            )
        
        # Calculate overage
        overage_info = StripeService.calculate_overage_charge(
            user_subscription.tier, 
            iterations_used
        )
        
        return {
            "tier": user_subscription.tier,
            "overage_calculation": overage_info
        }
        
    except Exception as e:
        logger.error(f"Failed to calculate overage for user {current_user.id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to calculate overage cost"
        )

@router.post("/create-payment-intent")
async def create_payment_intent_for_usage(
    iterations_used: int,
    description: str = "Monte Carlo simulation usage",
    current_user: UserModel = Depends(get_current_active_auth0_user),
    db: Session = Depends(get_db)
):
    """
    Create a Stripe PaymentIntent for pay-per-use or overage charges
    """
    try:
        # Get user subscription
        user_subscription = db.query(UserSubscription).filter(
            UserSubscription.user_id == current_user.id
        ).first()
        
        if not user_subscription:
            user_subscription = UserSubscription(
                user_id=current_user.id,
                tier="free",
                status="active"
            )
            db.add(user_subscription)
            db.commit()
            db.refresh(user_subscription)
        
        # Get or create Stripe customer
        customer_id = user_subscription.stripe_customer_id
        if not customer_id:
            customer_id = StripeService.create_customer(current_user)
            if not customer_id:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Failed to create Stripe customer"
                )
            user_subscription.stripe_customer_id = customer_id
            db.commit()
        
        # Calculate charge
        overage_info = StripeService.calculate_overage_charge(
            user_subscription.tier, 
            iterations_used
        )
        
        if overage_info["overage_charge_eur"] <= 0:
            return {
                "message": "No payment required",
                "overage_calculation": overage_info
            }
        
        # Create PaymentIntent
        payment_intent = StripeService.create_payment_intent_for_overage(
            customer_id=customer_id,
            amount_eur=overage_info["overage_charge_eur"],
            description=f"{description} - {iterations_used} iterations",
            metadata={
                "user_id": str(current_user.id),
                "iterations_used": str(iterations_used),
                "tier": user_subscription.tier,
                "overage_iterations": str(overage_info["overage_iterations"])
            }
        )
        
        if not payment_intent:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create payment intent"
            )
        
        return {
            "payment_intent_id": payment_intent.id,
            "client_secret": payment_intent.client_secret,
            "amount_eur": overage_info["overage_charge_eur"],
            "overage_calculation": overage_info
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create payment intent for user {current_user.id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create payment intent"
        )

@router.get("/usage")
async def get_usage_info(
    current_user: UserModel = Depends(get_current_active_auth0_user),
    db: Session = Depends(get_db)
):
    """
    Get current usage information for the user
    """
    try:
        # Get user subscription to determine limits
        user_subscription = db.query(UserSubscription).filter(
            UserSubscription.user_id == current_user.id
        ).first()
        
        if not user_subscription:
            user_subscription = UserSubscription(
                user_id=current_user.id,
                tier="free",
                status="active"
            )
        
        limits = user_subscription.get_limits()
        
        # Get current usage from UserUsageMetrics
        from models import UserUsageMetrics
        from datetime import datetime, timezone
        from dateutil.relativedelta import relativedelta
        
        # Get current month's usage
        current_month_start = datetime.now(timezone.utc).replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        
        usage_metrics = db.query(UserUsageMetrics).filter(
            UserUsageMetrics.user_id == current_user.id,
            UserUsageMetrics.period_start >= current_month_start
        ).first()
        
        if not usage_metrics:
            # Create empty usage metrics
            usage_metrics = UserUsageMetrics(
                user_id=current_user.id,
                period_start=current_month_start,
                period_end=current_month_start + relativedelta(months=1),
                period_type="monthly"
            )
        
        return {
            "limits": limits,
            "current_usage": {
                "simulations_run": usage_metrics.simulations_run,
                "total_iterations": usage_metrics.total_iterations,
                "files_uploaded": usage_metrics.files_uploaded,
                "total_file_size_mb": usage_metrics.total_file_size_mb,
                "api_calls": usage_metrics.api_calls,
                "period_start": usage_metrics.period_start.isoformat() if usage_metrics.period_start else None,
                "period_end": usage_metrics.period_end.isoformat() if usage_metrics.period_end else None
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get usage info for user {current_user.id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve usage information"
        )


@router.post("/webhook")
async def stripe_webhook(
    request: Request,
    db: Session = Depends(get_db)
):
    """
    Handle Stripe webhook events for subscription updates
    """
    try:
        # Get the raw body and signature
        body = await request.body()
        sig_header = request.headers.get('stripe-signature')
        
        # For now, just log the webhook event (we'll add signature verification later)
        logger.info(f"Received Stripe webhook, signature: {sig_header}")
        
        # Parse the event
        try:
            import json
            event = json.loads(body)
        except json.JSONDecodeError:
            logger.error("Invalid JSON in webhook body")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid JSON"
            )
        
        # Handle different event types
        event_type = event.get('type')
        logger.info(f"Processing Stripe webhook event: {event_type}")
        
        if event_type == 'checkout.session.completed':
            # Handle successful checkout
            session = event['data']['object']
            customer_id = session.get('customer')
            subscription_id = session.get('subscription')
            
            if customer_id and subscription_id:
                # Find user by Stripe customer ID
                user_subscription = db.query(UserSubscription).filter(
                    UserSubscription.stripe_customer_id == customer_id
                ).first()
                
                if user_subscription:
                    # Get subscription details from Stripe
                    from services.stripe_service import StripeService
                    import stripe
                    stripe.api_key = settings.STRIPE_SECRET_KEY
                    
                    subscription = stripe.Subscription.retrieve(subscription_id)
                    price_id = subscription['items']['data'][0]['price']['id']
                    
                    # Map price ID back to tier
                    tier_mapping = {
                        "price_1S7fWEGkZec0aS3MDjCHFnkH": "basic",
                        "price_1S7fWFGkZec0aS3MSHiUzPoh": "pro", 
                        "price_1S7fWGGkZec0aS3M3lvFVmUa": "enterprise"
                    }
                    
                    new_tier = tier_mapping.get(price_id, "basic")
                    
                    # Update subscription in database
                    user_subscription.tier = new_tier
                    user_subscription.status = "active"
                    user_subscription.stripe_subscription_id = subscription_id
                    user_subscription.stripe_price_id = price_id
                    user_subscription.current_period_start = subscription['current_period_start']
                    user_subscription.current_period_end = subscription['current_period_end']
                    
                    db.commit()
                    
                    logger.info(f"Updated subscription for user {user_subscription.user_id} to {new_tier}")
                
        elif event_type == 'customer.subscription.updated':
            # Handle subscription updates (upgrades, downgrades, etc.)
            subscription = event['data']['object']
            customer_id = subscription.get('customer')
            subscription_id = subscription.get('id')
            
            if customer_id:
                user_subscription = db.query(UserSubscription).filter(
                    UserSubscription.stripe_customer_id == customer_id
                ).first()
                
                if user_subscription:
                    price_id = subscription['items']['data'][0]['price']['id']
                    tier_mapping = {
                        "price_1S7fWEGkZec0aS3MDjCHFnkH": "basic",
                        "price_1S7fWFGkZec0aS3MSHiUzPoh": "pro",
                        "price_1S7fWGGkZec0aS3M3lvFVmUa": "enterprise"
                    }
                    
                    new_tier = tier_mapping.get(price_id, user_subscription.tier)
                    
                    user_subscription.tier = new_tier
                    user_subscription.status = subscription.get('status', 'active')
                    user_subscription.stripe_subscription_id = subscription_id
                    user_subscription.stripe_price_id = price_id
                    user_subscription.current_period_start = subscription['current_period_start']
                    user_subscription.current_period_end = subscription['current_period_end']
                    
                    db.commit()
                    
                    logger.info(f"Updated subscription for user {user_subscription.user_id}: {new_tier} ({subscription.get('status')})")
        
        elif event_type == 'customer.subscription.deleted':
            # Handle subscription cancellation
            subscription = event['data']['object']
            customer_id = subscription.get('customer')
            
            if customer_id:
                user_subscription = db.query(UserSubscription).filter(
                    UserSubscription.stripe_customer_id == customer_id
                ).first()
                
                if user_subscription:
                    user_subscription.tier = "free"
                    user_subscription.status = "cancelled"
                    user_subscription.stripe_subscription_id = None
                    user_subscription.stripe_price_id = None
                    
                    db.commit()
                    
                    logger.info(f"Cancelled subscription for user {user_subscription.user_id}")
        
        return {"status": "success"}
        
    except Exception as e:
        logger.error(f"Failed to process webhook: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Webhook processing failed"
        )
