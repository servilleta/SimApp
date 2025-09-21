"""
Stripe Webhook Handler for Monte Carlo Platform

Handles Stripe webhook events for subscription lifecycle management,
payment processing, and automatic subscription status updates.
"""

import logging
import stripe
from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Request, status
from sqlalchemy.orm import Session
from datetime import datetime, timezone

from database import get_db
from models import User as UserModel, UserSubscription
from services.stripe_service import StripeService
from config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/webhooks", tags=["webhooks"])

@router.post("/stripe")
async def handle_stripe_webhook(request: Request):
    """
    Handle Stripe webhook events
    
    This endpoint receives webhook events from Stripe and processes them
    to keep our subscription status in sync with Stripe.
    """
    payload = await request.body()
    sig_header = request.headers.get("stripe-signature")
    
    if not sig_header:
        logger.error("Missing Stripe signature header")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Missing Stripe signature"
        )
    
    try:
        # Check if webhook secret is configured
        if not settings.STRIPE_WEBHOOK_SECRET:
            logger.error("STRIPE_WEBHOOK_SECRET is not configured")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Webhook secret not configured"
            )
        
        # Verify webhook signature
        event = stripe.Webhook.construct_event(
            payload, sig_header, settings.STRIPE_WEBHOOK_SECRET
        )
    except ValueError as e:
        logger.error(f"Invalid payload: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid payload"
        )
    except stripe.error.SignatureVerificationError as e:
        logger.error(f"Invalid signature - webhook secret may be incorrect: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid signature"
        )
    
    # Get database session
    db = next(get_db())
    
    try:
        # Handle the event
        event_type = event["type"]
        event_data = event["data"]["object"]
        
        logger.info(f"Processing Stripe webhook event: {event_type} for customer: {event_data.get('customer', 'unknown')}")
        
        try:
            if event_type == "customer.subscription.created":
                await handle_subscription_created(db, event_data)
                logger.info(f"Successfully processed subscription.created for customer: {event_data.get('customer')}")
            
            elif event_type == "customer.subscription.updated":
                await handle_subscription_updated(db, event_data)
                logger.info(f"Successfully processed subscription.updated for customer: {event_data.get('customer')}")
            
            elif event_type == "customer.subscription.deleted":
                await handle_subscription_deleted(db, event_data)
                logger.info(f"Successfully processed subscription.deleted for customer: {event_data.get('customer')}")
            
            elif event_type == "invoice.payment_succeeded":
                await handle_payment_succeeded(db, event_data)
                logger.info(f"Successfully processed payment.succeeded for customer: {event_data.get('customer')}")
            
            elif event_type == "invoice.payment_failed":
                await handle_payment_failed(db, event_data)
                logger.info(f"Successfully processed payment.failed for customer: {event_data.get('customer')}")
            
            elif event_type == "customer.subscription.trial_will_end":
                await handle_trial_will_end(db, event_data)
                logger.info(f"Successfully processed trial_will_end for customer: {event_data.get('customer')}")
            
            else:
                logger.info(f"Unhandled webhook event type: {event_type}")
            
            return {"status": "success", "event_type": event_type, "processed": True}
            
        except Exception as handler_error:
            logger.error(f"Error in webhook handler for {event_type}: {handler_error}")
            # Log the full event data for debugging (but not sensitive info)
            logger.error(f"Event data keys: {list(event_data.keys())}")
            if 'customer' in event_data:
                logger.error(f"Customer ID: {event_data['customer']}")
            if 'id' in event_data:
                logger.error(f"Subscription/Object ID: {event_data['id']}")
            raise handler_error
        
    except Exception as e:
        logger.error(f"Error processing webhook: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Webhook processing failed"
        )
    finally:
        db.close()

async def handle_subscription_created(db: Session, subscription_data: Dict[str, Any]):
    """
    Handle subscription.created webhook event
    """
    try:
        stripe_subscription_id = subscription_data["id"]
        stripe_customer_id = subscription_data["customer"]
        
        # Find user by Stripe customer ID
        user_subscription = db.query(UserSubscription).filter(
            UserSubscription.stripe_customer_id == stripe_customer_id
        ).first()
        
        if not user_subscription:
            logger.warning(f"No user subscription found for Stripe customer {stripe_customer_id}")
            return
        
        # Update subscription record
        user_subscription.stripe_subscription_id = stripe_subscription_id
        user_subscription.stripe_price_id = subscription_data["items"]["data"][0]["price"]["id"]
        
        # Determine tier from price ID
        tier = get_tier_from_price_id(subscription_data["items"]["data"][0]["price"]["id"])
        if tier:
            user_subscription.tier = tier
        
        # Sync status and billing info
        stripe_subscription = stripe.Subscription.retrieve(stripe_subscription_id)
        StripeService.sync_subscription_status(db, user_subscription, stripe_subscription)
        
        logger.info(f"Subscription created for user {user_subscription.user_id}: {stripe_subscription_id}")
        
    except Exception as e:
        logger.error(f"Failed to handle subscription created: {str(e)}")
        raise

async def handle_subscription_updated(db: Session, subscription_data: Dict[str, Any]):
    """
    Handle subscription.updated webhook event
    """
    try:
        stripe_subscription_id = subscription_data["id"]
        
        # Find user subscription
        user_subscription = db.query(UserSubscription).filter(
            UserSubscription.stripe_subscription_id == stripe_subscription_id
        ).first()
        
        if not user_subscription:
            logger.warning(f"No user subscription found for Stripe subscription {stripe_subscription_id}")
            return
        
        # Update tier if price changed
        current_price_id = subscription_data["items"]["data"][0]["price"]["id"]
        tier = get_tier_from_price_id(current_price_id)
        if tier and tier != user_subscription.tier:
            user_subscription.tier = tier
            user_subscription.stripe_price_id = current_price_id
            logger.info(f"User {user_subscription.user_id} plan changed to {tier}")
        
        # Sync status and billing info
        stripe_subscription = stripe.Subscription.retrieve(stripe_subscription_id)
        StripeService.sync_subscription_status(db, user_subscription, stripe_subscription)
        
        logger.info(f"Subscription updated for user {user_subscription.user_id}: {stripe_subscription_id}")
        
    except Exception as e:
        logger.error(f"Failed to handle subscription updated: {str(e)}")
        raise

async def handle_subscription_deleted(db: Session, subscription_data: Dict[str, Any]):
    """
    Handle subscription.deleted webhook event
    """
    try:
        stripe_subscription_id = subscription_data["id"]
        
        # Find user subscription
        user_subscription = db.query(UserSubscription).filter(
            UserSubscription.stripe_subscription_id == stripe_subscription_id
        ).first()
        
        if not user_subscription:
            logger.warning(f"No user subscription found for Stripe subscription {stripe_subscription_id}")
            return
        
        # Downgrade to free tier
        user_subscription.tier = "free"
        user_subscription.status = "cancelled"
        user_subscription.stripe_subscription_id = None
        user_subscription.stripe_price_id = None
        user_subscription.current_period_start = None
        user_subscription.current_period_end = None
        user_subscription.cancel_at_period_end = False
        
        db.commit()
        
        logger.info(f"Subscription cancelled for user {user_subscription.user_id}, downgraded to free tier")
        
    except Exception as e:
        logger.error(f"Failed to handle subscription deleted: {str(e)}")
        raise

async def handle_payment_succeeded(db: Session, invoice_data: Dict[str, Any]):
    """
    Handle invoice.payment_succeeded webhook event
    """
    try:
        stripe_subscription_id = invoice_data.get("subscription")
        
        if not stripe_subscription_id:
            logger.info("Payment succeeded for non-subscription invoice")
            return
        
        # Find user subscription
        user_subscription = db.query(UserSubscription).filter(
            UserSubscription.stripe_subscription_id == stripe_subscription_id
        ).first()
        
        if not user_subscription:
            logger.warning(f"No user subscription found for Stripe subscription {stripe_subscription_id}")
            return
        
        # Update subscription status to active if it was past_due or unpaid
        if user_subscription.status in ["past_due", "unpaid"]:
            user_subscription.status = "active"
            db.commit()
            logger.info(f"Payment succeeded for user {user_subscription.user_id}, status restored to active")
        
    except Exception as e:
        logger.error(f"Failed to handle payment succeeded: {str(e)}")
        raise

async def handle_payment_failed(db: Session, invoice_data: Dict[str, Any]):
    """
    Handle invoice.payment_failed webhook event
    """
    try:
        stripe_subscription_id = invoice_data.get("subscription")
        
        if not stripe_subscription_id:
            logger.info("Payment failed for non-subscription invoice")
            return
        
        # Find user subscription
        user_subscription = db.query(UserSubscription).filter(
            UserSubscription.stripe_subscription_id == stripe_subscription_id
        ).first()
        
        if not user_subscription:
            logger.warning(f"No user subscription found for Stripe subscription {stripe_subscription_id}")
            return
        
        # Update subscription status to past_due
        user_subscription.status = "past_due"
        db.commit()
        
        logger.warning(f"Payment failed for user {user_subscription.user_id}, status set to past_due")
        
        # TODO: Send notification email to user about failed payment
        
    except Exception as e:
        logger.error(f"Failed to handle payment failed: {str(e)}")
        raise

async def handle_trial_will_end(db: Session, subscription_data: Dict[str, Any]):
    """
    Handle customer.subscription.trial_will_end webhook event
    """
    try:
        stripe_subscription_id = subscription_data["id"]
        
        # Find user subscription
        user_subscription = db.query(UserSubscription).filter(
            UserSubscription.stripe_subscription_id == stripe_subscription_id
        ).first()
        
        if not user_subscription:
            logger.warning(f"No user subscription found for Stripe subscription {stripe_subscription_id}")
            return
        
        logger.info(f"Trial ending soon for user {user_subscription.user_id}")
        
        # TODO: Send notification email to user about trial ending
        
    except Exception as e:
        logger.error(f"Failed to handle trial will end: {str(e)}")
        raise

def get_tier_from_price_id(price_id: str) -> Optional[str]:
    """
    Get plan tier from Stripe price ID
    
    Args:
        price_id: Stripe price ID
        
    Returns:
        Plan tier name or None if not found
    """
    # Reverse lookup from StripeService.PRICE_IDS
    for tier, tier_price_id in StripeService.PRICE_IDS.items():
        if tier_price_id == price_id:
            return tier
    
    logger.warning(f"Unknown price ID: {price_id}")
    return None
