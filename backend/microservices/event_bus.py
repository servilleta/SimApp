"""
🚌 Enterprise Event Bus
Phase 2 Week 5: Event-Driven Communication

This implements a Redis-based event bus for microservices communication,
enabling loose coupling and scalable architecture.
"""

import asyncio
import json
import logging
import uuid
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum

try:
    import redis.asyncio as redis
except ImportError:
    redis = None
    logging.warning("Redis not available - using in-memory event bus")

logger = logging.getLogger(__name__)

class EventType(Enum):
    # User Events
    USER_CREATED = "user.created"
    USER_UPDATED = "user.updated"
    USER_DELETED = "user.deleted"
    
    # File Events
    FILE_UPLOADED = "file.uploaded"
    FILE_PARSED = "file.parsed"
    FILE_DELETED = "file.deleted"
    
    # Simulation Events
    SIMULATION_STARTED = "simulation.started"
    SIMULATION_PROGRESS = "simulation.progress"
    SIMULATION_COMPLETED = "simulation.completed"
    SIMULATION_FAILED = "simulation.failed"
    
    # Results Events
    RESULTS_GENERATED = "results.generated"
    RESULTS_EXPORTED = "results.exported"
    
    # Billing Events
    USAGE_RECORDED = "billing.usage_recorded"
    INVOICE_GENERATED = "billing.invoice_generated"
    
    # Notification Events
    NOTIFICATION_SENT = "notification.sent"
    EMAIL_SENT = "notification.email_sent"

@dataclass
class Event:
    """Standard event structure"""
    event_id: str
    event_type: EventType
    source_service: str
    user_id: int
    data: Dict[str, Any]
    timestamp: str
    correlation_id: Optional[str] = None
    
    @classmethod
    def create(cls, event_type: EventType, source_service: str, user_id: int, 
               data: Dict[str, Any], correlation_id: Optional[str] = None) -> 'Event':
        """Create a new event with auto-generated ID and timestamp"""
        return cls(
            event_id=str(uuid.uuid4()),
            event_type=event_type,
            source_service=source_service,
            user_id=user_id,
            data=data,
            timestamp=datetime.utcnow().isoformat(),
            correlation_id=correlation_id
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for serialization"""
        data = asdict(self)
        data['event_type'] = self.event_type.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Event':
        """Create event from dictionary"""
        data['event_type'] = EventType(data['event_type'])
        return cls(**data)

class EventHandler:
    """Base class for event handlers"""
    
    def __init__(self, service_name: str):
        self.service_name = service_name
        
    async def handle(self, event: Event) -> bool:
        """Handle an event. Return True if handled successfully."""
        raise NotImplementedError

class EnterpriseEventBus:
    """Enterprise Event Bus with Redis backend and fallback to in-memory"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        self.redis_url = redis_url
        self.redis_client: Optional[redis.Redis] = None
        self.subscribers: Dict[EventType, List[EventHandler]] = {}
        self.running = False
        
        # In-memory fallback
        self.memory_events: List[Event] = []
        self.use_memory_fallback = False
        
    async def initialize(self):
        """Initialize the event bus"""
        if redis:
            try:
                self.redis_client = redis.from_url(self.redis_url, decode_responses=True)
                await self.redis_client.ping()
                logger.info("✅ [EVENT_BUS] Connected to Redis")
            except Exception as e:
                logger.warning(f"⚠️ [EVENT_BUS] Redis unavailable, using memory fallback: {e}")
                self.use_memory_fallback = True
        else:
            logger.warning("⚠️ [EVENT_BUS] Redis package not installed, using memory fallback")
            self.use_memory_fallback = True
        
        self.running = True
        
        # Start event processing
        if not self.use_memory_fallback:
            asyncio.create_task(self._process_events_from_redis())
        
    async def shutdown(self):
        """Shutdown the event bus"""
        self.running = False
        if self.redis_client:
            await self.redis_client.close()
        logger.info("✅ [EVENT_BUS] Shutdown complete")
    
    async def publish(self, event: Event):
        """Publish an event"""
        event_data = json.dumps(event.to_dict())
        
        if self.use_memory_fallback:
            # In-memory implementation
            self.memory_events.append(event)
            await self._process_memory_events()
        else:
            # Redis implementation
            channel = f"events:{event.event_type.value}"
            await self.redis_client.publish(channel, event_data)
            
        logger.info(f"📤 [EVENT_BUS] Published {event.event_type.value} from {event.source_service}")
    
    def subscribe(self, event_type: EventType, handler: EventHandler):
        """Subscribe to an event type"""
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        
        self.subscribers[event_type].append(handler)
        logger.info(f"📝 [EVENT_BUS] {handler.service_name} subscribed to {event_type.value}")
    
    async def _process_events_from_redis(self):
        """Process events from Redis streams"""
        try:
            pubsub = self.redis_client.pubsub()
            
            # Subscribe to all event channels
            for event_type in EventType:
                channel = f"events:{event_type.value}"
                await pubsub.subscribe(channel)
            
            logger.info("✅ [EVENT_BUS] Subscribed to all event channels")
            
            while self.running:
                try:
                    message = await pubsub.get_message(timeout=1.0)
                    if message and message['type'] == 'message':
                        await self._handle_redis_message(message)
                except Exception as e:
                    logger.error(f"❌ [EVENT_BUS] Error processing Redis message: {e}")
                    await asyncio.sleep(1)
                    
        except Exception as e:
            logger.error(f"❌ [EVENT_BUS] Redis event processing failed: {e}")
    
    async def _handle_redis_message(self, message):
        """Handle a message from Redis"""
        try:
            event_data = json.loads(message['data'])
            event = Event.from_dict(event_data)
            await self._dispatch_event(event)
        except Exception as e:
            logger.error(f"❌ [EVENT_BUS] Error handling Redis message: {e}")
    
    async def _process_memory_events(self):
        """Process events from memory (fallback mode)"""
        events_to_process = self.memory_events.copy()
        self.memory_events.clear()
        
        for event in events_to_process:
            await self._dispatch_event(event)
    
    async def _dispatch_event(self, event: Event):
        """Dispatch event to all registered handlers"""
        handlers = self.subscribers.get(event.event_type, [])
        
        if not handlers:
            logger.debug(f"📭 [EVENT_BUS] No handlers for {event.event_type.value}")
            return
        
        for handler in handlers:
            try:
                success = await handler.handle(event)
                if success:
                    logger.info(f"✅ [EVENT_BUS] {handler.service_name} handled {event.event_type.value}")
                else:
                    logger.warning(f"⚠️ [EVENT_BUS] {handler.service_name} failed to handle {event.event_type.value}")
            except Exception as e:
                logger.error(f"❌ [EVENT_BUS] {handler.service_name} error handling {event.event_type.value}: {e}")

# ====================================================================
# CONVENIENCE FUNCTIONS FOR COMMON EVENTS
# ====================================================================

class SimulationEventHandler(EventHandler):
    """Example event handler for simulation events"""
    
    async def handle(self, event: Event) -> bool:
        """Handle simulation events"""
        try:
            if event.event_type == EventType.SIMULATION_STARTED:
                logger.info(f"🚀 [SIMULATION_EVENTS] Simulation {event.data.get('simulation_id')} started for user {event.user_id}")
                
            elif event.event_type == EventType.SIMULATION_COMPLETED:
                logger.info(f"✅ [SIMULATION_EVENTS] Simulation {event.data.get('simulation_id')} completed for user {event.user_id}")
                
                # Could trigger billing, notifications, etc.
                await self._trigger_billing_event(event)
                await self._trigger_notification_event(event)
                
            elif event.event_type == EventType.SIMULATION_FAILED:
                logger.error(f"❌ [SIMULATION_EVENTS] Simulation {event.data.get('simulation_id')} failed for user {event.user_id}")
                
            return True
            
        except Exception as e:
            logger.error(f"❌ [SIMULATION_EVENTS] Error handling event: {e}")
            return False
    
    async def _trigger_billing_event(self, simulation_event: Event):
        """Trigger billing event for completed simulation"""
        # This would integrate with the billing service
        logger.info(f"💰 [BILLING] Recording usage for simulation {simulation_event.data.get('simulation_id')}")
    
    async def _trigger_notification_event(self, simulation_event: Event):
        """Trigger notification for completed simulation"""
        # This would integrate with the notification service
        logger.info(f"📧 [NOTIFICATION] Notifying user {simulation_event.user_id} of completed simulation")

# Global event bus instance
enterprise_event_bus = EnterpriseEventBus()

# Convenience functions
async def publish_simulation_started(user_id: int, simulation_id: str, **kwargs):
    """Publish simulation started event"""
    event = Event.create(
        event_type=EventType.SIMULATION_STARTED,
        source_service="simulation-service",
        user_id=user_id,
        data={"simulation_id": simulation_id, **kwargs}
    )
    await enterprise_event_bus.publish(event)

async def publish_simulation_completed(user_id: int, simulation_id: str, results: Dict[str, Any], **kwargs):
    """Publish simulation completed event"""
    event = Event.create(
        event_type=EventType.SIMULATION_COMPLETED,
        source_service="simulation-service",
        user_id=user_id,
        data={"simulation_id": simulation_id, "results": results, **kwargs}
    )
    await enterprise_event_bus.publish(event)

async def publish_file_uploaded(user_id: int, file_id: str, filename: str, **kwargs):
    """Publish file uploaded event"""
    event = Event.create(
        event_type=EventType.FILE_UPLOADED,
        source_service="file-service",
        user_id=user_id,
        data={"file_id": file_id, "filename": filename, **kwargs}
    )
    await enterprise_event_bus.publish(event)
