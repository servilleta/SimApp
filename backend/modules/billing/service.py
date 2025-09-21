"""
Billing service module - implements BillingServiceProtocol

Handles subscription management, payments, and billing operations using Stripe.
This service can be easily extracted to a microservice later.
"""

import logging
import stripe
import json
from typing import Dict, Optional, List, Any
from datetime import datetime, timezone, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import and_

from ..base import BaseService, BillingServiceProtocol
from database import get_db
from models import UserSubscription, UserUsageMetrics
from config import settings

logger = logging.getLogger(__name__)

# Configure Stripe
stripe.api_key = getattr(settings, 'STRIPE_SECRET_KEY', 'sk_test_...')  # Set in environment


class BillingService(BaseService, BillingServiceProtocol):
    """Stripe-based billing service implementation"""
    
    def __init__(self):
        super().__init__("billing")
        self.stripe_price_ids = {
            "basic": getattr(settings, 'STRIPE_BASIC_PRICE_ID', 'price_basic'),
            "pro": getattr(settings, 'STRIPE_PRO_PRICE_ID', 'price_pro'),
            "enterprise": getattr(settings, 'STRIPE_ENTERPRISE_PRICE_ID', 'price_enterprise')
        }
        self.webhook_secret = getattr(settings, 'STRIPE_WEBHOOK_SECRET', 'whsec_...')
        
    async def initialize(self) -> None:
        """Initialize the billing service"""
        await super().initialize()
        logger.info("Stripe billing service initialized")
        
        # Verify Stripe configuration
        try:
            stripe.Product.list(limit=1)
            logger.info("Stripe API connection verified")
        except Exception as e:
            logger.error(f"Stripe API connection failed: {e}")

    def _get_user_subscription(self, user_id: int, db: Session) -> UserSubscription:
        """Get or create user subscription record"""
        subscription = db.query(UserSubscription).filter(
            UserSubscription.user_id == user_id
        ).first()
        
        if not subscription:
            # Create default free subscription
            subscription = UserSubscription(
                user_id=user_id,
                tier="free",
                status="active"
            )
            db.add(subscription)
            db.commit()
            db.refresh(subscription)
            logger.info(f"Created free subscription for user {user_id}")
        
        return subscription

    async def create_customer(self, user_id: int, email: str, name: Optional[str] = None) -> Dict[str, Any]:
        """Create a Stripe customer"""
        db = next(get_db())
        try:
            subscription = self._get_user_subscription(user_id, db)
            
            # Check if customer already exists
            if subscription.stripe_customer_id:
                try:
                    customer = stripe.Customer.retrieve(subscription.stripe_customer_id)
                    return {
                        "success": True,
                        "customer_id": customer.id,
                        "message": "Customer already exists"
                    }
                except stripe.error.InvalidRequestError:
                    # Customer doesn't exist, create new one
                    pass
            
            # Create new Stripe customer
            customer_data = {
                "email": email,
                "metadata": {
                    "user_id": str(user_id),
                    "platform": "monte_carlo"
                }
            }
            
            if name:
                customer_data["name"] = name
            
            customer = stripe.Customer.create(**customer_data)
            
            # Update subscription record
            subscription.stripe_customer_id = customer.id
            db.commit()
            
            logger.info(f"Created Stripe customer {customer.id} for user {user_id}")
            
            return {
                "success": True,
                "customer_id": customer.id,
                "message": "Customer created successfully"
            }
            
        except Exception as e:
            db.rollback()
            logger.error(f"Error creating Stripe customer for user {user_id}: {e}")
            return {
                "success": False,
                "error": str(e)
            }
        finally:
            db.close()

    async def create_subscription(self, user_id: int, tier: str, payment_method_id: Optional[str] = None) -> Dict[str, Any]:
        """Create a subscription for a user"""
        if tier not in ["basic", "pro", "enterprise"]:
            return {
                "success": False,
                "error": "Invalid subscription tier"
            }
        
        db = next(get_db())
        try:
            subscription = self._get_user_subscription(user_id, db)
            
            # Ensure customer exists
            if not subscription.stripe_customer_id:
                return {
                    "success": False,
                    "error": "Customer not found. Please create customer first."
                }
            
            # Get price ID for tier
            price_id = self.stripe_price_ids.get(tier)
            if not price_id:
                return {
                    "success": False,
                    "error": f"Price not configured for tier: {tier}"
                }
            
            # Create subscription in Stripe
            stripe_subscription_data = {
                "customer": subscription.stripe_customer_id,
                "items": [{"price": price_id}],
                "payment_behavior": "default_incomplete",
                "payment_settings": {"save_default_payment_method": "on_subscription"},
                "expand": ["latest_invoice.payment_intent"],
                "metadata": {
                    "user_id": str(user_id),
                    "tier": tier
                }
            }
            
            if payment_method_id:
                stripe_subscription_data["default_payment_method"] = payment_method_id
            
            stripe_subscription = stripe.Subscription.create(**stripe_subscription_data)
            
            # Update local subscription record
            subscription.tier = tier
            subscription.status = "active" if stripe_subscription.status == "active" else "pending"
            subscription.stripe_subscription_id = stripe_subscription.id
            subscription.stripe_price_id = price_id
            subscription.current_period_start = datetime.fromtimestamp(
                stripe_subscription.current_period_start, tz=timezone.utc
            )
            subscription.current_period_end = datetime.fromtimestamp(
                stripe_subscription.current_period_end, tz=timezone.utc
            )
            
            db.commit()
            
            logger.info(f"Created subscription {stripe_subscription.id} for user {user_id} ({tier})")
            
            result = {
                "success": True,
                "subscription_id": stripe_subscription.id,
                "status": stripe_subscription.status,
                "tier": tier
            }
            
            # Handle payment intent if needed
            if stripe_subscription.latest_invoice.payment_intent:
                pi = stripe_subscription.latest_invoice.payment_intent
                if pi.status == "requires_action":
                    result["requires_action"] = True
                    result["payment_intent"] = {
                        "id": pi.id,
                        "client_secret": pi.client_secret
                    }
            
            return result
            
        except Exception as e:
            db.rollback()
            logger.error(f"Error creating subscription for user {user_id}: {e}")
            return {
                "success": False,
                "error": str(e)
            }
        finally:
            db.close()

    async def cancel_subscription(self, user_id: int, immediate: bool = False) -> Dict[str, Any]:
        """Cancel a user's subscription"""
        db = next(get_db())
        try:
            subscription = self._get_user_subscription(user_id, db)
            
            if not subscription.stripe_subscription_id:
                return {
                    "success": False,
                    "error": "No active subscription found"
                }
            
            # Cancel in Stripe
            if immediate:
                stripe_subscription = stripe.Subscription.delete(subscription.stripe_subscription_id)
                subscription.status = "cancelled"
            else:
                stripe_subscription = stripe.Subscription.modify(
                    subscription.stripe_subscription_id,
                    cancel_at_period_end=True
                )
                subscription.cancel_at_period_end = True
            
            db.commit()
            
            logger.info(f"Cancelled subscription {subscription.stripe_subscription_id} for user {user_id}")
            
            return {
                "success": True,
                "message": "Subscription cancelled" if immediate else "Subscription will cancel at period end",
                "cancelled_immediately": immediate
            }
            
        except Exception as e:
            db.rollback()
            logger.error(f"Error cancelling subscription for user {user_id}: {e}")
            return {
                "success": False,
                "error": str(e)
            }
        finally:
            db.close()

    async def get_subscription(self, user_id: int) -> Dict[str, Any]:
        """Get subscription details for a user"""
        db = next(get_db())
        try:
            subscription = self._get_user_subscription(user_id, db)
            
            result = {
                "user_id": user_id,
                "tier": subscription.tier,
                "status": subscription.status,
                "created_at": subscription.created_at.isoformat() if subscription.created_at else None,
                "current_period_start": subscription.current_period_start.isoformat() if subscription.current_period_start else None,
                "current_period_end": subscription.current_period_end.isoformat() if subscription.current_period_end else None,
                "cancel_at_period_end": subscription.cancel_at_period_end,
                "stripe_customer_id": subscription.stripe_customer_id,
                "stripe_subscription_id": subscription.stripe_subscription_id
            }
            
            # Get additional details from Stripe if available
            if subscription.stripe_subscription_id:
                try:
                    stripe_subscription = stripe.Subscription.retrieve(
                        subscription.stripe_subscription_id,
                        expand=["default_payment_method", "latest_invoice"]
                    )
                    
                    result["stripe_details"] = {
                        "status": stripe_subscription.status,
                        "current_period_start": stripe_subscription.current_period_start,
                        "current_period_end": stripe_subscription.current_period_end,
                        "cancel_at_period_end": stripe_subscription.cancel_at_period_end
                    }
                    
                    if stripe_subscription.default_payment_method:
                        pm = stripe_subscription.default_payment_method
                        result["payment_method"] = {
                            "type": pm.type,
                            "card": {
                                "brand": pm.card.brand,
                                "last4": pm.card.last4,
                                "exp_month": pm.card.exp_month,
                                "exp_year": pm.card.exp_year
                            } if pm.type == "card" else None
                        }
                    
                except Exception as e:
                    logger.warning(f"Could not fetch Stripe details for user {user_id}: {e}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting subscription for user {user_id}: {e}")
            return {
                "user_id": user_id,
                "tier": "free",
                "status": "active",
                "error": str(e)
            }
        finally:
            db.close()

    async def update_subscription(self, user_id: int, new_tier: str) -> Dict[str, Any]:
        """Update a user's subscription tier"""
        if new_tier not in ["basic", "pro", "enterprise"]:
            return {
                "success": False,
                "error": "Invalid subscription tier"
            }
        
        db = next(get_db())
        try:
            subscription = self._get_user_subscription(user_id, db)
            
            if not subscription.stripe_subscription_id:
                return {
                    "success": False,
                    "error": "No active subscription to update"
                }
            
            # Get new price ID
            new_price_id = self.stripe_price_ids.get(new_tier)
            if not new_price_id:
                return {
                    "success": False,
                    "error": f"Price not configured for tier: {new_tier}"
                }
            
            # Update subscription in Stripe
            stripe_subscription = stripe.Subscription.retrieve(subscription.stripe_subscription_id)
            
            stripe.Subscription.modify(
                subscription.stripe_subscription_id,
                items=[{
                    "id": stripe_subscription["items"]["data"][0].id,
                    "price": new_price_id,
                }],
                proration_behavior="create_prorations"
            )
            
            # Update local record
            subscription.tier = new_tier
            subscription.stripe_price_id = new_price_id
            db.commit()
            
            logger.info(f"Updated subscription for user {user_id} to {new_tier}")
            
            return {
                "success": True,
                "new_tier": new_tier,
                "message": "Subscription updated successfully"
            }
            
        except Exception as e:
            db.rollback()
            logger.error(f"Error updating subscription for user {user_id}: {e}")
            return {
                "success": False,
                "error": str(e)
            }
        finally:
            db.close()

    async def create_payment_intent(self, user_id: int, amount: int, currency: str = "usd") -> Dict[str, Any]:
        """Create a payment intent for one-time payments"""
        db = next(get_db())
        try:
            subscription = self._get_user_subscription(user_id, db)
            
            if not subscription.stripe_customer_id:
                return {
                    "success": False,
                    "error": "Customer not found"
                }
            
            payment_intent = stripe.PaymentIntent.create(
                amount=amount,  # Amount in cents
                currency=currency,
                customer=subscription.stripe_customer_id,
                metadata={
                    "user_id": str(user_id),
                    "type": "one_time_payment"
                }
            )
            
            return {
                "success": True,
                "payment_intent_id": payment_intent.id,
                "client_secret": payment_intent.client_secret,
                "amount": amount,
                "currency": currency
            }
            
        except Exception as e:
            logger.error(f"Error creating payment intent for user {user_id}: {e}")
            return {
                "success": False,
                "error": str(e)
            }
        finally:
            db.close()

    async def handle_webhook(self, payload: str, sig_header: str) -> Dict[str, Any]:
        """Handle Stripe webhook events"""
        try:
            event = stripe.Webhook.construct_event(
                payload, sig_header, self.webhook_secret
            )
        except ValueError:
            return {"success": False, "error": "Invalid payload"}
        except stripe.error.SignatureVerificationError:
            return {"success": False, "error": "Invalid signature"}
        
        # Handle the event
        if event["type"] == "customer.subscription.updated":
            await self._handle_subscription_updated(event["data"]["object"])
        elif event["type"] == "customer.subscription.deleted":
            await self._handle_subscription_deleted(event["data"]["object"])
        elif event["type"] == "invoice.payment_succeeded":
            await self._handle_payment_succeeded(event["data"]["object"])
        elif event["type"] == "invoice.payment_failed":
            await self._handle_payment_failed(event["data"]["object"])
        else:
            logger.info(f"Unhandled webhook event type: {event['type']}")
        
        return {"success": True}

    async def _handle_subscription_updated(self, stripe_subscription: Dict) -> None:
        """Handle subscription updated webhook"""
        user_id = int(stripe_subscription["metadata"].get("user_id", 0))
        if not user_id:
            return
        
        db = next(get_db())
        try:
            subscription = db.query(UserSubscription).filter(
                UserSubscription.stripe_subscription_id == stripe_subscription["id"]
            ).first()
            
            if subscription:
                subscription.status = stripe_subscription["status"]
                subscription.current_period_start = datetime.fromtimestamp(
                    stripe_subscription["current_period_start"], tz=timezone.utc
                )
                subscription.current_period_end = datetime.fromtimestamp(
                    stripe_subscription["current_period_end"], tz=timezone.utc
                )
                subscription.cancel_at_period_end = stripe_subscription.get("cancel_at_period_end", False)
                
                db.commit()
                logger.info(f"Updated subscription for user {user_id} via webhook")
        except Exception as e:
            db.rollback()
            logger.error(f"Error handling subscription updated webhook: {e}")
        finally:
            db.close()

    async def _handle_subscription_deleted(self, stripe_subscription: Dict) -> None:
        """Handle subscription deleted webhook"""
        db = next(get_db())
        try:
            subscription = db.query(UserSubscription).filter(
                UserSubscription.stripe_subscription_id == stripe_subscription["id"]
            ).first()
            
            if subscription:
                subscription.tier = "free"
                subscription.status = "cancelled"
                subscription.stripe_subscription_id = None
                subscription.stripe_price_id = None
                subscription.current_period_start = None
                subscription.current_period_end = None
                subscription.cancel_at_period_end = False
                
                db.commit()
                logger.info(f"Cancelled subscription for user {subscription.user_id} via webhook")
        except Exception as e:
            db.rollback()
            logger.error(f"Error handling subscription deleted webhook: {e}")
        finally:
            db.close()

    async def _handle_payment_succeeded(self, invoice: Dict) -> None:
        """Handle successful payment webhook"""
        subscription_id = invoice.get("subscription")
        if not subscription_id:
            return
        
        db = next(get_db())
        try:
            subscription = db.query(UserSubscription).filter(
                UserSubscription.stripe_subscription_id == subscription_id
            ).first()
            
            if subscription:
                # Update subscription status to active
                subscription.status = "active"
                db.commit()
                logger.info(f"Payment succeeded for user {subscription.user_id}")
        except Exception as e:
            db.rollback()
            logger.error(f"Error handling payment succeeded webhook: {e}")
        finally:
            db.close()

    async def _handle_payment_failed(self, invoice: Dict) -> None:
        """Handle failed payment webhook"""
        subscription_id = invoice.get("subscription")
        if not subscription_id:
            return
        
        db = next(get_db())
        try:
            subscription = db.query(UserSubscription).filter(
                UserSubscription.stripe_subscription_id == subscription_id
            ).first()
            
            if subscription:
                # Update subscription status
                subscription.status = "past_due"
                db.commit()
                logger.warning(f"Payment failed for user {subscription.user_id}")
        except Exception as e:
            db.rollback()
            logger.error(f"Error handling payment failed webhook: {e}")
        finally:
            db.close()

    async def get_usage_and_billing(self, user_id: int) -> Dict[str, Any]:
        """Get combined usage and billing information"""
        db = next(get_db())
        try:
            subscription = self._get_user_subscription(user_id, db)
            
            # Get current month usage
            now = datetime.now(timezone.utc)
            month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            
            usage_metrics = db.query(UserUsageMetrics).filter(
                and_(
                    UserUsageMetrics.user_id == user_id,
                    UserUsageMetrics.period_start == month_start,
                    UserUsageMetrics.period_type == "monthly"
                )
            ).first()
            
            usage_data = {
                "simulations_run": usage_metrics.simulations_run if usage_metrics else 0,
                "total_iterations": usage_metrics.total_iterations if usage_metrics else 0,
                "gpu_simulations": usage_metrics.gpu_simulations if usage_metrics else 0,
                "api_calls": usage_metrics.api_calls if usage_metrics else 0,
                "period_start": month_start.isoformat(),
                "period_end": (month_start + timedelta(days=32)).replace(day=1).isoformat()
            }
            
            return {
                "subscription": {
                    "tier": subscription.tier,
                    "status": subscription.status,
                    "current_period_start": subscription.current_period_start.isoformat() if subscription.current_period_start else None,
                    "current_period_end": subscription.current_period_end.isoformat() if subscription.current_period_end else None,
                    "cancel_at_period_end": subscription.cancel_at_period_end
                },
                "usage": usage_data,
                "limits": subscription.get_limits()
            }
            
        except Exception as e:
            logger.error(f"Error getting usage and billing for user {user_id}: {e}")
            return {"error": str(e)}
        finally:
            db.close()

    async def get_invoices(self, user_id: int, limit: int = 10) -> Dict[str, Any]:
        """Get user's billing history"""
        db = next(get_db())
        try:
            subscription = self._get_user_subscription(user_id, db)
            
            if not subscription.stripe_customer_id:
                return {
                    "invoices": [],
                    "message": "No billing history available"
                }
            
            invoices = stripe.Invoice.list(
                customer=subscription.stripe_customer_id,
                limit=limit
            )
            
            invoice_list = []
            for invoice in invoices.data:
                invoice_list.append({
                    "id": invoice.id,
                    "amount_paid": invoice.amount_paid,
                    "amount_due": invoice.amount_due,
                    "currency": invoice.currency,
                    "status": invoice.status,
                    "created": invoice.created,
                    "period_start": invoice.period_start,
                    "period_end": invoice.period_end,
                    "hosted_invoice_url": invoice.hosted_invoice_url,
                    "invoice_pdf": invoice.invoice_pdf
                })
            
            return {
                "invoices": invoice_list,
                "has_more": invoices.has_more
            }
            
        except Exception as e:
            logger.error(f"Error getting invoices for user {user_id}: {e}")
            return {
                "invoices": [],
                "error": str(e)
            }
        finally:
            db.close()

    async def cleanup_expired_subscriptions(self) -> Dict[str, int]:
        """Clean up expired subscriptions"""
        db = next(get_db())
        try:
            now = datetime.now(timezone.utc)
            
            # Find expired subscriptions
            expired_subscriptions = db.query(UserSubscription).filter(
                and_(
                    UserSubscription.current_period_end < now,
                    UserSubscription.status.in_(["active", "past_due"]),
                    UserSubscription.tier != "free"
                )
            ).all()
            
            cleanup_count = 0
            for subscription in expired_subscriptions:
                # Verify with Stripe
                if subscription.stripe_subscription_id:
                    try:
                        stripe_sub = stripe.Subscription.retrieve(subscription.stripe_subscription_id)
                        if stripe_sub.status in ["canceled", "incomplete_expired"]:
                            subscription.tier = "free"
                            subscription.status = "cancelled"
                            subscription.stripe_subscription_id = None
                            subscription.stripe_price_id = None
                            cleanup_count += 1
                    except Exception as e:
                        logger.warning(f"Could not verify Stripe subscription {subscription.stripe_subscription_id}: {e}")
            
            db.commit()
            logger.info(f"Cleaned up {cleanup_count} expired subscriptions")
            
            return {
                "cleaned_up": cleanup_count,
                "checked": len(expired_subscriptions)
            }
            
        except Exception as e:
            db.rollback()
            logger.error(f"Error cleaning up expired subscriptions: {e}")
            return {"error": str(e)}
        finally:
            db.close() 