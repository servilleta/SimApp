"""
🎯 ENTERPRISE SUPPORT SYSTEM - Phase 5 Week 19-20
Advanced support and ticketing system for Monte Carlo Enterprise Platform

This module implements the enterprise support features from enterprise.txt:
- Automated issue classification with AI
- SLA-based ticket management
- Tier-based support levels
- Proactive monitoring and escalation
- Knowledge base integration

Provides enterprise-grade support with guaranteed response times.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
from pathlib import Path

logger = logging.getLogger(__name__)


class TicketPriority(Enum):
    """Support ticket priority levels"""
    CRITICAL = "critical"
    HIGH = "high"  
    MEDIUM = "medium"
    LOW = "low"


class TicketStatus(Enum):
    """Support ticket status"""
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    WAITING_CUSTOMER = "waiting_customer"
    RESOLVED = "resolved"
    CLOSED = "closed"


class UserTier(Enum):
    """User support tier levels"""
    ENTERPRISE = "enterprise"
    PROFESSIONAL = "professional"
    STANDARD = "standard"


class IssueCategory(Enum):
    """Issue categories for classification"""
    ULTRA_ENGINE_PERFORMANCE = "ultra_engine_performance"
    SIMULATION_ERROR = "simulation_error"
    DATA_UPLOAD = "data_upload"
    AUTHENTICATION = "authentication"
    BILLING = "billing"
    INTEGRATION = "integration"
    GENERAL_QUESTION = "general_question"


@dataclass
class SupportTicket:
    """Support ticket data structure"""
    id: str
    user_id: int
    organization_id: int
    title: str
    description: str
    priority: TicketPriority
    category: IssueCategory
    status: TicketStatus
    user_tier: UserTier
    sla_hours: int
    assigned_to: Optional[str]
    created_at: datetime
    updated_at: datetime
    due_date: datetime
    resolution: Optional[str] = None
    satisfaction_rating: Optional[int] = None
    tags: List[str] = None


@dataclass
class SupportEngineer:
    """Support engineer information"""
    id: str
    name: str
    email: str
    specialties: List[IssueCategory]
    tier_access: List[UserTier]
    current_tickets: int
    max_tickets: int
    availability: bool = True


class AITechnicalAssistant:
    """
    AI-powered technical assistant for issue classification and resolution
    
    This simulates the AI assistant mentioned in enterprise.txt that automatically
    classifies issues and suggests resolutions.
    """
    
    def __init__(self):
        self.knowledge_base = self._load_knowledge_base()
        self.classification_rules = self._load_classification_rules()
    
    async def classify_issue(self, description: str) -> IssueCategory:
        """
        Automatically classify support issues using AI
        
        This analyzes the issue description and categorizes it for proper routing
        """
        try:
            description_lower = description.lower()
            
            # Ultra Engine performance issues
            if any(keyword in description_lower for keyword in 
                   ['progress bar', 'slow', 'performance', 'timeout', 'response time']):
                return IssueCategory.ULTRA_ENGINE_PERFORMANCE
            
            # Simulation errors
            if any(keyword in description_lower for keyword in 
                   ['simulation failed', 'error', 'crash', 'exception', 'monte carlo']):
                return IssueCategory.SIMULATION_ERROR
            
            # Data upload issues
            if any(keyword in description_lower for keyword in 
                   ['upload', 'excel', 'file', 'import', 'data']):
                return IssueCategory.DATA_UPLOAD
            
            # Authentication issues
            if any(keyword in description_lower for keyword in 
                   ['login', 'password', 'auth', 'sso', 'access denied']):
                return IssueCategory.AUTHENTICATION
            
            # Billing issues
            if any(keyword in description_lower for keyword in 
                   ['billing', 'payment', 'invoice', 'subscription', 'pricing']):
                return IssueCategory.BILLING
            
            # Integration issues
            if any(keyword in description_lower for keyword in 
                   ['api', 'integration', 'webhook', 'connector']):
                return IssueCategory.INTEGRATION
            
            # Default to general question
            return IssueCategory.GENERAL_QUESTION
            
        except Exception as e:
            logger.error(f"Error classifying issue: {e}")
            return IssueCategory.GENERAL_QUESTION
    
    async def suggest_resolution(self, ticket: SupportTicket) -> Optional[str]:
        """
        Suggest resolution based on issue category and knowledge base
        """
        try:
            suggestions = self.knowledge_base.get(ticket.category.value, [])
            
            if suggestions:
                # Return the most relevant suggestion
                return suggestions[0]
            
            return None
            
        except Exception as e:
            logger.error(f"Error suggesting resolution: {e}")
            return None
    
    def _load_knowledge_base(self) -> Dict[str, List[str]]:
        """Load knowledge base with common solutions"""
        return {
            "ultra_engine_performance": [
                "Check if GPU acceleration is enabled in system settings",
                "Verify network connectivity and latency to backend services",
                "Clear browser cache and restart the application",
                "Check system resources (CPU, memory) usage"
            ],
            "simulation_error": [
                "Verify Excel file format and data structure",
                "Check for invalid formulas or circular references",
                "Ensure data ranges are properly defined",
                "Try with a smaller dataset to isolate the issue"
            ],
            "data_upload": [
                "Verify file size is under 100MB limit",
                "Check Excel file format (.xlsx or .xls)",
                "Ensure no special characters in file name",
                "Try uploading from a different browser"
            ],
            "authentication": [
                "Clear browser cookies and cache",
                "Check SSO configuration with your IT administrator", 
                "Verify your organization's domain is properly configured",
                "Try logging in from an incognito/private browser window"
            ],
            "billing": [
                "Check your organization's billing settings",
                "Verify payment method is up to date",
                "Review usage reports for unexpected charges",
                "Contact billing team for invoice questions"
            ]
        }
    
    def _load_classification_rules(self) -> Dict[str, Any]:
        """Load classification rules"""
        return {
            "keywords": {
                "critical": ["down", "outage", "production", "urgent", "emergency"],
                "high": ["error", "failed", "broken", "not working"],
                "medium": ["slow", "issue", "problem", "question"],
                "low": ["enhancement", "feature request", "documentation"]
            }
        }


class EscalationManager:
    """
    Manages ticket escalation based on SLA breaches and priority
    """
    
    def __init__(self):
        self.escalation_rules = self._load_escalation_rules()
    
    async def check_sla_breach(self, ticket: SupportTicket) -> bool:
        """Check if ticket has breached SLA"""
        try:
            time_elapsed = (datetime.utcnow() - ticket.created_at).total_seconds() / 3600
            return time_elapsed > ticket.sla_hours
            
        except Exception as e:
            logger.error(f"Error checking SLA breach: {e}")
            return False
    
    async def escalate_ticket(self, ticket: SupportTicket) -> bool:
        """Escalate ticket to higher tier support"""
        try:
            # Escalation logic based on tier and priority
            if ticket.user_tier == UserTier.ENTERPRISE:
                # Enterprise tickets escalate to senior engineers
                await self._escalate_to_senior_support(ticket)
            elif ticket.priority == TicketPriority.CRITICAL:
                # Critical tickets always escalate
                await self._escalate_to_management(ticket)
            
            return True
            
        except Exception as e:
            logger.error(f"Error escalating ticket {ticket.id}: {e}")
            return False
    
    def _load_escalation_rules(self) -> Dict[str, Any]:
        """Load escalation rules"""
        return {
            "sla_breach_escalation": True,
            "critical_auto_escalate": True,
            "enterprise_priority": True
        }
    
    async def _escalate_to_senior_support(self, ticket: SupportTicket):
        """Escalate to senior support team"""
        logger.info(f"Escalating ticket {ticket.id} to senior support")
    
    async def _escalate_to_management(self, ticket: SupportTicket):
        """Escalate to management"""
        logger.info(f"Escalating ticket {ticket.id} to management")


class EnterpriseSupportService:
    """
    Main enterprise support service
    
    Implements the support system from enterprise.txt with:
    - SLA-based response times by user tier
    - Automatic issue classification
    - Intelligent engineer assignment
    - Proactive SLA monitoring
    """
    
    def __init__(self):
        self.tickets: Dict[str, SupportTicket] = {}
        self.engineers = self._initialize_engineers()
        self.ai_assistant = AITechnicalAssistant()
        self.escalation_manager = EscalationManager()
        
        # SLA matrix from enterprise.txt
        self.sla_matrix = {
            UserTier.ENTERPRISE: {
                TicketPriority.CRITICAL: 2,    # 2 hours
                TicketPriority.HIGH: 4,        # 4 hours  
                TicketPriority.MEDIUM: 8,      # 8 hours
                TicketPriority.LOW: 24         # 24 hours
            },
            UserTier.PROFESSIONAL: {
                TicketPriority.CRITICAL: 4,    # 4 hours
                TicketPriority.HIGH: 8,        # 8 hours
                TicketPriority.MEDIUM: 24,     # 24 hours
                TicketPriority.LOW: 72         # 72 hours
            },
            UserTier.STANDARD: {
                TicketPriority.CRITICAL: 8,    # 8 hours
                TicketPriority.HIGH: 24,       # 24 hours
                TicketPriority.MEDIUM: 72,     # 72 hours
                TicketPriority.LOW: 168        # 1 week
            }
        }
    
    async def create_support_ticket(
        self,
        user_id: int,
        organization_id: int,
        title: str,
        description: str,
        priority: str,
        user_tier: str
    ) -> SupportTicket:
        """
        Create a new support ticket with automatic classification and assignment
        
        This implements the ticket creation flow from enterprise.txt
        """
        try:
            # Generate ticket ID
            ticket_id = str(uuid.uuid4())
            
            # Convert string enums
            priority_enum = TicketPriority(priority.lower())
            user_tier_enum = UserTier(user_tier.lower())
            
            # AI classification
            issue_category = await self.ai_assistant.classify_issue(description)
            
            # Calculate SLA
            sla_hours = self.calculate_sla(user_tier_enum, priority_enum, issue_category)
            
            # Create ticket
            ticket = SupportTicket(
                id=ticket_id,
                user_id=user_id,
                organization_id=organization_id,
                title=title,
                description=description,
                priority=priority_enum,
                category=issue_category,
                status=TicketStatus.OPEN,
                user_tier=user_tier_enum,
                sla_hours=sla_hours,
                assigned_to=None,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
                due_date=datetime.utcnow() + timedelta(hours=sla_hours),
                tags=[]
            )
            
            # Auto-assign to best engineer
            engineer = await self.find_best_engineer(issue_category, user_tier_enum)
            if engineer:
                ticket.assigned_to = engineer.id
                engineer.current_tickets += 1
            
            # Store ticket
            self.tickets[ticket_id] = ticket
            
            # Schedule SLA monitoring
            await self.schedule_sla_monitoring(ticket_id, sla_hours)
            
            # Send notifications
            await self._send_ticket_notifications(ticket)
            
            logger.info(f"Created support ticket: {ticket_id} (SLA: {sla_hours}h)")
            return ticket
            
        except Exception as e:
            logger.error(f"Error creating support ticket: {e}")
            raise
    
    def calculate_sla(
        self, 
        user_tier: UserTier, 
        priority: TicketPriority, 
        category: IssueCategory
    ) -> int:
        """
        Calculate SLA based on user tier and priority
        
        Implements the SLA matrix from enterprise.txt
        """
        try:
            base_sla = self.sla_matrix.get(user_tier, {}).get(priority, 168)
            
            # Ultra Engine performance issues get priority treatment
            if category == IssueCategory.ULTRA_ENGINE_PERFORMANCE:
                base_sla = max(1, base_sla // 2)  # Halve SLA for performance issues
            
            return base_sla
            
        except Exception as e:
            logger.error(f"Error calculating SLA: {e}")
            return 168  # Default to 1 week
    
    async def find_best_engineer(
        self, 
        issue_category: IssueCategory, 
        user_tier: UserTier
    ) -> Optional[SupportEngineer]:
        """
        Find the best available engineer for the ticket
        
        Considers specialties, tier access, and current workload
        """
        try:
            # Filter engineers by tier access and specialties
            suitable_engineers = [
                eng for eng in self.engineers.values()
                if (user_tier in eng.tier_access and 
                    issue_category in eng.specialties and
                    eng.availability and
                    eng.current_tickets < eng.max_tickets)
            ]
            
            if not suitable_engineers:
                logger.warning(f"No suitable engineer found for {issue_category.value}")
                return None
            
            # Sort by current workload (ascending)
            suitable_engineers.sort(key=lambda x: x.current_tickets)
            
            return suitable_engineers[0]
            
        except Exception as e:
            logger.error(f"Error finding engineer: {e}")
            return None
    
    async def schedule_sla_monitoring(self, ticket_id: str, sla_hours: int):
        """
        Schedule proactive SLA monitoring for a ticket
        
        This monitors tickets and escalates before SLA breach
        """
        try:
            # Schedule monitoring task
            asyncio.create_task(self._monitor_ticket_sla(ticket_id, sla_hours))
            
        except Exception as e:
            logger.error(f"Error scheduling SLA monitoring: {e}")
    
    async def _monitor_ticket_sla(self, ticket_id: str, sla_hours: int):
        """Monitor individual ticket SLA"""
        try:
            # Wait until 80% of SLA time has passed
            warning_time = sla_hours * 0.8 * 3600  # Convert to seconds
            await asyncio.sleep(warning_time)
            
            # Check if ticket is still open
            ticket = self.tickets.get(ticket_id)
            if ticket and ticket.status not in [TicketStatus.RESOLVED, TicketStatus.CLOSED]:
                await self._send_sla_warning(ticket)
                
                # Wait for remaining 20% of SLA time
                remaining_time = sla_hours * 0.2 * 3600
                await asyncio.sleep(remaining_time)
                
                # Check for SLA breach
                if ticket.status not in [TicketStatus.RESOLVED, TicketStatus.CLOSED]:
                    await self.escalation_manager.escalate_ticket(ticket)
            
        except Exception as e:
            logger.error(f"Error monitoring SLA for ticket {ticket_id}: {e}")
    
    async def get_support_metrics(self) -> Dict[str, Any]:
        """
        Get support system metrics and KPIs
        """
        try:
            total_tickets = len(self.tickets)
            open_tickets = len([t for t in self.tickets.values() 
                               if t.status == TicketStatus.OPEN])
            
            # Calculate average resolution time
            resolved_tickets = [t for t in self.tickets.values() 
                               if t.status == TicketStatus.RESOLVED]
            
            avg_resolution_time = 0
            if resolved_tickets:
                total_time = sum(
                    (t.updated_at - t.created_at).total_seconds() / 3600
                    for t in resolved_tickets
                )
                avg_resolution_time = total_time / len(resolved_tickets)
            
            # Calculate SLA compliance
            sla_compliant = len([t for t in resolved_tickets 
                                if not await self.escalation_manager.check_sla_breach(t)])
            sla_compliance = (sla_compliant / len(resolved_tickets) * 100) if resolved_tickets else 100
            
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "total_tickets": total_tickets,
                "open_tickets": open_tickets,
                "resolved_tickets": len(resolved_tickets),
                "avg_resolution_time_hours": round(avg_resolution_time, 2),
                "sla_compliance_percent": round(sla_compliance, 2),
                "engineers_available": len([e for e in self.engineers.values() if e.availability]),
                "tickets_by_priority": {
                    priority.value: len([t for t in self.tickets.values() 
                                        if t.priority == priority])
                    for priority in TicketPriority
                },
                "tickets_by_tier": {
                    tier.value: len([t for t in self.tickets.values() 
                                    if t.user_tier == tier])
                    for tier in UserTier
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting support metrics: {e}")
            return {"status": "error", "message": str(e)}
    
    def _initialize_engineers(self) -> Dict[str, SupportEngineer]:
        """Initialize support engineering team"""
        return {
            "eng001": SupportEngineer(
                id="eng001",
                name="Sarah Chen",
                email="sarah.chen@montecarlo.com",
                specialties=[
                    IssueCategory.ULTRA_ENGINE_PERFORMANCE,
                    IssueCategory.SIMULATION_ERROR
                ],
                tier_access=[UserTier.ENTERPRISE, UserTier.PROFESSIONAL, UserTier.STANDARD],
                current_tickets=0,
                max_tickets=5
            ),
            "eng002": SupportEngineer(
                id="eng002", 
                name="Marcus Rodriguez",
                email="marcus.rodriguez@montecarlo.com",
                specialties=[
                    IssueCategory.INTEGRATION,
                    IssueCategory.AUTHENTICATION
                ],
                tier_access=[UserTier.ENTERPRISE, UserTier.PROFESSIONAL],
                current_tickets=0,
                max_tickets=8
            ),
            "eng003": SupportEngineer(
                id="eng003",
                name="Emily Watson", 
                email="emily.watson@montecarlo.com",
                specialties=[
                    IssueCategory.DATA_UPLOAD,
                    IssueCategory.GENERAL_QUESTION
                ],
                tier_access=[UserTier.PROFESSIONAL, UserTier.STANDARD],
                current_tickets=0,
                max_tickets=10
            )
        }
    
    async def _send_ticket_notifications(self, ticket: SupportTicket):
        """Send ticket creation notifications"""
        logger.info(f"Sending notifications for ticket {ticket.id}")
    
    async def _send_sla_warning(self, ticket: SupportTicket):
        """Send SLA warning notification"""
        logger.warning(f"SLA warning for ticket {ticket.id} - {ticket.sla_hours}h SLA")


# Global support service instance
support_service = EnterpriseSupportService()
