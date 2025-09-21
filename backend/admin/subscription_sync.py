"""
Admin utilities for manual subscription synchronization.
This module provides tools to fix subscription sync issues when webhooks fail.
"""

import logging
import stripe
from typing import Dict, Any, List, Optional
from sqlalchemy.orm import Session
from datetime import datetime, timezone

from database import get_db
from models import User as UserModel, UserSubscription
from services.stripe_service import StripeService
from api.webhooks import get_tier_from_price_id

logger = logging.getLogger(__name__)

class SubscriptionSyncService:
    """Service for manually syncing subscription data from Stripe"""
    
    @staticmethod
    def sync_user_subscription(db: Session, user_email: str) -> Dict[str, Any]:
        """
        Manually sync a specific user's subscription from Stripe
        
        Args:
            db: Database session
            user_email: Email of the user to sync
            
        Returns:
            Dict with sync status and details
        """
        try:
            # Find the user
            user = db.query(UserModel).filter(UserModel.email == user_email).first()
            if not user:
                return {
                    "success": False,
                    "error": f"User not found: {user_email}"
                }
            
            # Get or create user subscription
            user_subscription = db.query(UserSubscription).filter(
                UserSubscription.user_id == user.id
            ).first()
            
            if not user_subscription:
                user_subscription = UserSubscription(
                    user_id=user.id,
                    tier="free",
                    status="active"
                )
                db.add(user_subscription)
                db.commit()
                db.refresh(user_subscription)
            
            # Check if user has Stripe customer ID
            if not user_subscription.stripe_customer_id:
                # Search for Stripe customers with this email
                customers = stripe.Customer.list(email=user_email, limit=5)
                
                if customers.data:
                    # Use the first customer found
                    customer = customers.data[0]
                    user_subscription.stripe_customer_id = customer.id
                    logger.info(f"Linked Stripe customer {customer.id} to user {user.id}")
                else:
                    return {
                        "success": True,
                        "message": f"No Stripe customer found for {user_email}",
                        "current_tier": user_subscription.tier,
                        "needs_upgrade": False
                    }
            
            # Get subscriptions for this customer
            subscriptions = stripe.Subscription.list(
                customer=user_subscription.stripe_customer_id,
                limit=10
            )
            
            sync_results = []
            subscription_updated = False
            
            for subscription in subscriptions.data:
                if subscription.status == "active":
                    # Get the price ID and map to tier
                    if subscription.items and subscription.items.data:
                        price_id = subscription.items.data[0].price.id
                        tier = get_tier_from_price_id(price_id)
                        
                        if tier and tier != user_subscription.tier:
                            old_tier = user_subscription.tier
                            
                            # Update subscription
                            user_subscription.tier = tier
                            user_subscription.stripe_subscription_id = subscription.id
                            user_subscription.stripe_price_id = price_id
                            user_subscription.status = subscription.status
                            
                            # Update billing period
                            if subscription.current_period_start:
                                user_subscription.current_period_start = datetime.fromtimestamp(
                                    subscription.current_period_start, tz=timezone.utc
                                )
                            if subscription.current_period_end:
                                user_subscription.current_period_end = datetime.fromtimestamp(
                                    subscription.current_period_end, tz=timezone.utc
                                )
                            
                            subscription_updated = True
                            
                            sync_results.append({
                                "subscription_id": subscription.id,
                                "old_tier": old_tier,
                                "new_tier": tier,
                                "price_id": price_id,
                                "status": subscription.status
                            })
                            
                            logger.info(f"Updated user {user.id} from {old_tier} to {tier}")
            
            if subscription_updated:
                db.commit()
                
                return {
                    "success": True,
                    "message": f"Subscription synced successfully for {user_email}",
                    "user_id": user.id,
                    "updates": sync_results,
                    "current_tier": user_subscription.tier,
                    "current_status": user_subscription.status
                }
            else:
                return {
                    "success": True,
                    "message": f"No subscription updates needed for {user_email}",
                    "current_tier": user_subscription.tier,
                    "current_status": user_subscription.status,
                    "stripe_customer_id": user_subscription.stripe_customer_id
                }
                
        except stripe.error.StripeError as e:
            logger.error(f"Stripe API error during sync for {user_email}: {e}")
            return {
                "success": False,
                "error": f"Stripe API error: {str(e)}"
            }
        except Exception as e:
            logger.error(f"Error syncing subscription for {user_email}: {e}")
            db.rollback()
            return {
                "success": False,
                "error": f"Database error: {str(e)}"
            }
    
    @staticmethod
    def sync_all_subscriptions(db: Session, limit: int = 100) -> Dict[str, Any]:
        """
        Sync all user subscriptions that have Stripe customer IDs
        
        Args:
            db: Database session
            limit: Maximum number of users to process
            
        Returns:
            Dict with sync results summary
        """
        try:
            # Get all users with Stripe customer IDs
            users_with_stripe = db.query(UserModel).join(UserSubscription).filter(
                UserSubscription.stripe_customer_id.isnot(None),
                UserSubscription.stripe_customer_id != ""
            ).limit(limit).all()
            
            results = {
                "success": True,
                "processed": 0,
                "updated": 0,
                "errors": 0,
                "user_results": []
            }
            
            for user in users_with_stripe:
                sync_result = SubscriptionSyncService.sync_user_subscription(db, user.email)
                results["processed"] += 1
                
                if sync_result["success"]:
                    if "updates" in sync_result and sync_result["updates"]:
                        results["updated"] += 1
                else:
                    results["errors"] += 1
                
                results["user_results"].append({
                    "email": user.email,
                    "result": sync_result
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Error during bulk subscription sync: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    @staticmethod
    def get_sync_status_report(db: Session) -> Dict[str, Any]:
        """
        Generate a report of subscription sync status
        
        Args:
            db: Database session
            
        Returns:
            Dict with sync status report
        """
        try:
            # Count users by subscription status
            total_users = db.query(UserModel).count()
            
            # Users with subscriptions
            users_with_subs = db.query(UserModel).join(UserSubscription).count()
            
            # Users with Stripe customer IDs
            users_with_stripe = db.query(UserModel).join(UserSubscription).filter(
                UserSubscription.stripe_customer_id.isnot(None),
                UserSubscription.stripe_customer_id != ""
            ).count()
            
            # Users with subscription IDs (properly synced)
            users_properly_synced = db.query(UserModel).join(UserSubscription).filter(
                UserSubscription.stripe_subscription_id.isnot(None),
                UserSubscription.stripe_subscription_id != ""
            ).count()
            
            # Users with Stripe customer but no subscription ID (sync issues)
            users_with_sync_issues = db.query(UserModel).join(UserSubscription).filter(
                UserSubscription.stripe_customer_id.isnot(None),
                UserSubscription.stripe_customer_id != "",
                UserSubscription.stripe_subscription_id.is_(None)
            ).count()
            
            # Tier distribution
            tier_counts = db.query(
                UserSubscription.tier,
                db.func.count(UserSubscription.tier)
            ).group_by(UserSubscription.tier).all()
            
            return {
                "success": True,
                "report": {
                    "total_users": total_users,
                    "users_with_subscriptions": users_with_subs,
                    "users_with_stripe_customer": users_with_stripe,
                    "users_properly_synced": users_properly_synced,
                    "users_with_sync_issues": users_with_sync_issues,
                    "sync_health_percentage": round(
                        (users_properly_synced / max(users_with_stripe, 1)) * 100, 2
                    ) if users_with_stripe > 0 else 100,
                    "tier_distribution": {tier: count for tier, count in tier_counts}
                },
                "issues_detected": users_with_sync_issues > 0,
                "recommendations": [
                    "Run manual sync for users with sync issues" if users_with_sync_issues > 0 else "All subscriptions are properly synced",
                    "Check Stripe webhook configuration" if users_with_sync_issues > 0 else "Webhook sync appears to be working",
                    f"Consider investigating {users_with_sync_issues} users with sync issues" if users_with_sync_issues > 0 else "No action needed"
                ]
            }
            
        except Exception as e:
            logger.error(f"Error generating sync status report: {e}")
            return {
                "success": False,
                "error": str(e)
            }
