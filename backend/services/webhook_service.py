"""
Webhook Delivery Service for Monte Carlo Platform

Handles webhook notifications with retry logic, signature verification,
and comprehensive delivery tracking.
"""

import hashlib
import hmac
import json
import logging
import time
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum

import aiohttp
import asyncio
from sqlalchemy.orm import Session
from pydantic import BaseModel

from database import get_db
from models import WebhookConfiguration, WebhookDelivery
from config import settings

logger = logging.getLogger(__name__)

class WebhookEventType(str, Enum):
    """Supported webhook event types"""
    SIMULATION_STARTED = "simulation.started"
    SIMULATION_PROGRESS = "simulation.progress" 
    SIMULATION_COMPLETED = "simulation.completed"
    SIMULATION_FAILED = "simulation.failed"
    SIMULATION_CANCELLED = "simulation.cancelled"

class WebhookDeliveryStatus(str, Enum):
    """Webhook delivery status"""
    PENDING = "pending"
    DELIVERED = "delivered"
    FAILED = "failed"
    RETRYING = "retrying"
    ABANDONED = "abandoned"

@dataclass
class WebhookPayload:
    """Webhook payload structure"""
    event: WebhookEventType
    timestamp: str
    data: Dict[str, Any]
    simulation_id: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "event": self.event.value,
            "timestamp": self.timestamp,
            "simulation_id": self.simulation_id,
            "data": self.data
        }

class WebhookRequest(BaseModel):
    """Webhook configuration request"""
    name: str
    url: str
    events: List[WebhookEventType]
    secret: Optional[str] = None
    enabled: bool = True

class WebhookService:
    """Webhook delivery service with retry logic and security"""
    
    def __init__(self):
        self.max_retries = 3
        self.retry_delays = [30, 300, 1800]  # 30s, 5m, 30m
        self.timeout = 30
        self.default_secret = settings.WEBHOOK_DEFAULT_SECRET or "default_webhook_secret"
        
    def generate_signature(self, payload: str, secret: str) -> str:
        """Generate HMAC-SHA256 signature for webhook verification"""
        signature = hmac.new(
            secret.encode('utf-8'),
            payload.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        return f"sha256={signature}"
    
    async def send_webhook(
        self, 
        webhook_config: WebhookConfiguration,
        payload: WebhookPayload,
        db: Session,
        attempt: int = 1
    ) -> bool:
        """
        Send webhook with signature verification and retry logic
        
        Returns:
            bool: True if delivery was successful, False otherwise
        """
        try:
            # Create delivery record
            delivery = WebhookDelivery(
                webhook_id=webhook_config.id,
                simulation_id=payload.simulation_id,
                event_type=payload.event.value,
                payload_data=payload.to_dict(),
                attempt=attempt,
                status=WebhookDeliveryStatus.PENDING.value,
                created_at=datetime.now(timezone.utc)
            )
            db.add(delivery)
            db.commit()
            
            # Prepare payload
            payload_json = json.dumps(payload.to_dict(), separators=(',', ':'))
            
            # Generate signature
            secret = webhook_config.secret or self.default_secret
            signature = self.generate_signature(payload_json, secret)
            
            # Prepare headers
            headers = {
                'Content-Type': 'application/json',
                'User-Agent': 'SimApp-Webhook/1.0',
                'X-SimApp-Event': payload.event.value,
                'X-SimApp-Signature': signature,
                'X-SimApp-Delivery': str(delivery.id),
                'X-SimApp-Timestamp': payload.timestamp
            }
            
            # Send webhook
            start_time = time.time()
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
                async with session.post(webhook_config.url, data=payload_json, headers=headers) as response:
                    response_time = (time.time() - start_time) * 1000  # ms
                    response_body = await response.text()
                    
                    # Update delivery record
                    delivery.response_status = response.status
                    delivery.response_body = response_body[:1000]  # Limit response body storage
                    delivery.response_time_ms = int(response_time)
                    delivery.delivered_at = datetime.now(timezone.utc)
                    
                    if 200 <= response.status < 300:
                        delivery.status = WebhookDeliveryStatus.DELIVERED.value
                        db.commit()
                        
                        logger.info(f"✅ Webhook delivered successfully to {webhook_config.url} "
                                  f"for {payload.event.value} (attempt {attempt}, {response_time:.0f}ms)")
                        return True
                    else:
                        delivery.status = WebhookDeliveryStatus.FAILED.value
                        delivery.error_message = f"HTTP {response.status}: {response_body[:200]}"
                        db.commit()
                        
                        logger.warning(f"❌ Webhook delivery failed to {webhook_config.url} "
                                     f"for {payload.event.value}: HTTP {response.status}")
                        return False
                        
        except asyncio.TimeoutError:
            delivery.status = WebhookDeliveryStatus.FAILED.value
            delivery.error_message = f"Timeout after {self.timeout}s"
            db.commit()
            
            logger.warning(f"⏰ Webhook delivery timeout to {webhook_config.url} "
                         f"for {payload.event.value} (attempt {attempt})")
            return False
            
        except Exception as e:
            delivery.status = WebhookDeliveryStatus.FAILED.value
            delivery.error_message = str(e)[:500]
            db.commit()
            
            logger.error(f"💥 Webhook delivery error to {webhook_config.url} "
                        f"for {payload.event.value} (attempt {attempt}): {e}")
            return False
    
    async def send_webhook_with_retry(
        self,
        webhook_config: WebhookConfiguration,
        payload: WebhookPayload,
        db: Session
    ):
        """Send webhook with automatic retry logic"""
        if not webhook_config.enabled:
            logger.debug(f"Skipping disabled webhook {webhook_config.name}")
            return
            
        # Check if webhook subscribes to this event
        if payload.event.value not in webhook_config.events:
            logger.debug(f"Webhook {webhook_config.name} not subscribed to {payload.event.value}")
            return
        
        # Attempt delivery with retries
        for attempt in range(1, self.max_retries + 1):
            success = await self.send_webhook(webhook_config, payload, db, attempt)
            
            if success:
                # Update webhook last delivery time
                webhook_config.last_delivery_at = datetime.now(timezone.utc)
                webhook_config.last_delivery_status = WebhookDeliveryStatus.DELIVERED.value
                db.commit()
                return
            
            # If not the last attempt, wait before retry
            if attempt < self.max_retries:
                delay = self.retry_delays[attempt - 1]
                logger.info(f"🔄 Retrying webhook delivery to {webhook_config.url} "
                          f"in {delay}s (attempt {attempt + 1}/{self.max_retries})")
                await asyncio.sleep(delay)
        
        # All attempts failed
        webhook_config.last_delivery_at = datetime.now(timezone.utc)
        webhook_config.last_delivery_status = WebhookDeliveryStatus.ABANDONED.value
        db.commit()
        
        logger.error(f"🚫 Webhook delivery abandoned to {webhook_config.url} "
                    f"for {payload.event.value} after {self.max_retries} attempts")
    
    async def notify_simulation_event(
        self,
        event_type: WebhookEventType,
        simulation_id: str,
        data: Dict[str, Any],
        client_id: Optional[str] = None
    ):
        """
        Send webhook notifications for simulation events
        
        Args:
            event_type: Type of simulation event
            simulation_id: ID of the simulation
            data: Event-specific data
            client_id: Client ID for B2B API filtering
        """
        try:
            db = next(get_db())
            
            # Create webhook payload
            payload = WebhookPayload(
                event=event_type,
                timestamp=datetime.now(timezone.utc).isoformat(),
                simulation_id=simulation_id,
                data=data
            )
            
            # Get all active webhooks for this client (or all for internal use)
            query = db.query(WebhookConfiguration).filter(
                WebhookConfiguration.enabled == True
            )
            
            if client_id:
                query = query.filter(WebhookConfiguration.client_id == client_id)
            
            webhooks = query.all()
            
            if not webhooks:
                logger.debug(f"No active webhooks found for {event_type.value}")
                return
            
            # Send webhooks concurrently
            tasks = [
                self.send_webhook_with_retry(webhook, payload, db)
                for webhook in webhooks
            ]
            
            await asyncio.gather(*tasks, return_exceptions=True)
            
        except Exception as e:
            logger.error(f"Failed to send webhook notifications for {event_type.value}: {e}")
        finally:
            db.close()
    
    async def test_webhook(
        self,
        webhook_config: WebhookConfiguration,
        db: Session
    ) -> Dict[str, Any]:
        """Test webhook delivery with a test payload"""
        test_payload = WebhookPayload(
            event=WebhookEventType.SIMULATION_STARTED,
            timestamp=datetime.now(timezone.utc).isoformat(),
            simulation_id="test_simulation_123",
            data={
                "message": "This is a test webhook delivery",
                "test": True,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )
        
        success = await self.send_webhook(webhook_config, test_payload, db)
        
        return {
            "success": success,
            "webhook_id": webhook_config.id,
            "webhook_name": webhook_config.name,
            "test_event": test_payload.event.value,
            "timestamp": test_payload.timestamp
        }

# Global webhook service instance
webhook_service = WebhookService()
