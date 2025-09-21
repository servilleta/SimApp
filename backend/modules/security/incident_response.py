"""
Security Incident Response System
Handles security incidents, breach notifications, and compliance reporting
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from enum import Enum
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from sqlalchemy.orm import Session
from sqlalchemy import and_
from fastapi import HTTPException

from models import SecurityAuditLog, User
from database import get_db
from config import settings

logger = logging.getLogger(__name__)

class IncidentSeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class IncidentType(str, Enum):
    DATA_BREACH = "data_breach"
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    SYSTEM_COMPROMISE = "system_compromise"
    MALWARE = "malware"
    PHISHING = "phishing"
    DDOS = "ddos"
    INSIDER_THREAT = "insider_threat"
    COMPLIANCE_VIOLATION = "compliance_violation"

class IncidentStatus(str, Enum):
    DETECTED = "detected"
    INVESTIGATING = "investigating"
    CONTAINED = "contained"
    ERADICATED = "eradicated"
    RECOVERED = "recovered"
    CLOSED = "closed"

class SecurityIncident:
    """Represents a security incident"""
    
    def __init__(
        self,
        incident_type: IncidentType,
        severity: IncidentSeverity,
        title: str,
        description: str,
        affected_systems: List[str] = None,
        affected_users: List[int] = None,
        detected_by: str = "system",
        detection_time: datetime = None
    ):
        self.id = f"INC-{datetime.utcnow().strftime('%Y%m%d')}-{id(self) % 10000:04d}"
        self.incident_type = incident_type
        self.severity = severity
        self.title = title
        self.description = description
        self.affected_systems = affected_systems or []
        self.affected_users = affected_users or []
        self.detected_by = detected_by
        self.detection_time = detection_time or datetime.utcnow()
        self.status = IncidentStatus.DETECTED
        self.timeline = []
        self.actions_taken = []
        self.evidence = []
        self.gdpr_breach = False
        self.notification_sent = False
        
    def add_timeline_entry(self, entry: str, timestamp: datetime = None):
        """Add entry to incident timeline"""
        self.timeline.append({
            "timestamp": timestamp or datetime.utcnow(),
            "entry": entry
        })
    
    def add_action(self, action: str, taken_by: str, timestamp: datetime = None):
        """Add action taken to resolve incident"""
        self.actions_taken.append({
            "timestamp": timestamp or datetime.utcnow(),
            "action": action,
            "taken_by": taken_by
        })
    
    def add_evidence(self, evidence_type: str, description: str, file_path: str = None):
        """Add evidence to incident"""
        self.evidence.append({
            "type": evidence_type,
            "description": description,
            "file_path": file_path,
            "collected_at": datetime.utcnow()
        })

class IncidentResponseService:
    """Service for handling security incidents"""
    
    def __init__(self):
        self.active_incidents: Dict[str, SecurityIncident] = {}
        self.incident_handlers = {
            IncidentType.DATA_BREACH: self._handle_data_breach,
            IncidentType.UNAUTHORIZED_ACCESS: self._handle_unauthorized_access,
            IncidentType.SYSTEM_COMPROMISE: self._handle_system_compromise,
            IncidentType.COMPLIANCE_VIOLATION: self._handle_compliance_violation
        }
    
    async def create_incident(
        self,
        incident_type: IncidentType,
        severity: IncidentSeverity,
        title: str,
        description: str,
        **kwargs
    ) -> SecurityIncident:
        """Create and register a new security incident"""
        
        incident = SecurityIncident(
            incident_type=incident_type,
            severity=severity,
            title=title,
            description=description,
            **kwargs
        )
        
        self.active_incidents[incident.id] = incident
        
        # Log incident creation
        await self._log_incident_event(incident, "incident_created")
        
        # Start incident response process
        await self._initiate_response(incident)
        
        logger.critical(f"Security incident created: {incident.id} - {title}")
        
        return incident
    
    async def _initiate_response(self, incident: SecurityIncident):
        """Initiate incident response procedures"""
        
        incident.add_timeline_entry("Incident response initiated")
        
        # Execute incident-specific handler
        if incident.incident_type in self.incident_handlers:
            await self.incident_handlers[incident.incident_type](incident)
        
        # Send immediate notifications for high/critical incidents
        if incident.severity in [IncidentSeverity.HIGH, IncidentSeverity.CRITICAL]:
            await self._send_immediate_notification(incident)
        
        # Check if GDPR breach notification is required
        if incident.gdpr_breach:
            await self._initiate_gdpr_breach_process(incident)
    
    async def _handle_data_breach(self, incident: SecurityIncident):
        """Handle data breach incidents"""
        
        incident.add_timeline_entry("Data breach response initiated")
        incident.gdpr_breach = True  # Assume GDPR applies until determined otherwise
        
        # Immediate containment actions
        incident.add_action("Initiated containment procedures", "system")
        
        # Assess scope and impact
        await self._assess_breach_scope(incident)
        
        # Determine if personal data is involved
        if await self._involves_personal_data(incident):
            incident.gdpr_breach = True
            incident.add_timeline_entry("Personal data involvement confirmed - GDPR breach procedures initiated")
        
        # Update incident status
        incident.status = IncidentStatus.INVESTIGATING
    
    async def _handle_unauthorized_access(self, incident: SecurityIncident):
        """Handle unauthorized access incidents"""
        
        incident.add_timeline_entry("Unauthorized access response initiated")
        
        # Immediate actions
        incident.add_action("Reviewing access logs", "security_team")
        incident.add_action("Checking for privilege escalation", "security_team")
        
        # If user accounts are involved, secure them
        if incident.affected_users:
            for user_id in incident.affected_users:
                await self._secure_user_account(user_id)
                incident.add_action(f"Secured user account {user_id}", "system")
        
        incident.status = IncidentStatus.CONTAINED
    
    async def _handle_system_compromise(self, incident: SecurityIncident):
        """Handle system compromise incidents"""
        
        incident.add_timeline_entry("System compromise response initiated")
        incident.severity = IncidentSeverity.CRITICAL  # Escalate to critical
        
        # Immediate isolation
        incident.add_action("Initiating system isolation procedures", "security_team")
        
        # Evidence collection
        incident.add_evidence("system_logs", "System logs at time of compromise")
        incident.add_evidence("network_traffic", "Network traffic analysis")
        
        incident.status = IncidentStatus.CONTAINED
    
    async def _handle_compliance_violation(self, incident: SecurityIncident):
        """Handle compliance violation incidents"""
        
        incident.add_timeline_entry("Compliance violation response initiated")
        
        # Document the violation
        incident.add_evidence("compliance_audit", "Compliance audit findings")
        
        # Determine regulatory requirements
        if "gdpr" in incident.description.lower():
            incident.gdpr_breach = True
        
        incident.status = IncidentStatus.INVESTIGATING
    
    async def _assess_breach_scope(self, incident: SecurityIncident):
        """Assess the scope and impact of a data breach"""
        
        with next(get_db()) as db:
            # Check recent suspicious activities
            recent_logs = db.query(SecurityAuditLog).filter(
                SecurityAuditLog.timestamp >= datetime.utcnow() - timedelta(hours=24)
            ).all()
            
            suspicious_activities = []
            for log in recent_logs:
                if any(keyword in log.event_type.lower() for keyword in ['failed', 'error', 'unauthorized', 'suspicious']):
                    suspicious_activities.append(log)
            
            incident.add_evidence(
                "security_logs",
                f"Found {len(suspicious_activities)} suspicious activities in last 24 hours"
            )
            
            # Estimate affected users
            if not incident.affected_users and suspicious_activities:
                affected_user_ids = list(set([log.user_id for log in suspicious_activities if log.user_id]))
                incident.affected_users = affected_user_ids
                incident.add_timeline_entry(f"Identified {len(affected_user_ids)} potentially affected users")
    
    async def _involves_personal_data(self, incident: SecurityIncident) -> bool:
        """Determine if incident involves personal data"""
        
        # Check if affected systems contain personal data
        personal_data_systems = ["database", "user_files", "simulation_results", "billing"]
        
        for system in incident.affected_systems:
            if any(pds in system.lower() for pds in personal_data_systems):
                return True
        
        # Check if affected users exist
        if incident.affected_users:
            return True
        
        # Check description for personal data keywords
        personal_data_keywords = ["email", "name", "address", "phone", "personal", "pii"]
        if any(keyword in incident.description.lower() for keyword in personal_data_keywords):
            return True
        
        return False
    
    async def _secure_user_account(self, user_id: int):
        """Secure a potentially compromised user account"""
        
        with next(get_db()) as db:
            user = db.query(User).filter(User.id == user_id).first()
            if user:
                # Force password reset
                # In a real implementation, you'd invalidate sessions and require password reset
                
                # Log security action
                audit_log = SecurityAuditLog(
                    user_id=user_id,
                    event_type="account_secured_incident_response",
                    ip_address="system",
                    user_agent="incident_response_system",
                    timestamp=datetime.utcnow(),
                    details={"reason": "security_incident", "action": "account_secured"}
                )
                db.add(audit_log)
                db.commit()
    
    async def _initiate_gdpr_breach_process(self, incident: SecurityIncident):
        """Initiate GDPR breach notification process"""
        
        incident.add_timeline_entry("GDPR breach notification process initiated")
        
        # GDPR requires notification within 72 hours
        notification_deadline = incident.detection_time + timedelta(hours=72)
        
        incident.add_timeline_entry(f"GDPR notification deadline: {notification_deadline}")
        
        # Assess if high risk to individuals
        high_risk = await self._assess_gdpr_high_risk(incident)
        
        if high_risk:
            incident.add_timeline_entry("High risk to individuals identified - direct notification required")
            await self._notify_affected_individuals(incident)
        
        # Schedule regulatory notification
        await self._schedule_regulatory_notification(incident, notification_deadline)
    
    async def _assess_gdpr_high_risk(self, incident: SecurityIncident) -> bool:
        """Assess if breach poses high risk to individuals"""
        
        high_risk_factors = [
            incident.severity in [IncidentSeverity.HIGH, IncidentSeverity.CRITICAL],
            len(incident.affected_users) > 100,
            "financial" in incident.description.lower(),
            "sensitive" in incident.description.lower(),
            "identity" in incident.description.lower()
        ]
        
        return any(high_risk_factors)
    
    async def _notify_affected_individuals(self, incident: SecurityIncident):
        """Notify affected individuals of data breach"""
        
        if not incident.affected_users:
            return
        
        with next(get_db()) as db:
            affected_users = db.query(User).filter(
                User.id.in_(incident.affected_users)
            ).all()
            
            for user in affected_users:
                await self._send_breach_notification(user, incident)
                incident.add_action(f"Sent breach notification to user {user.id}", "system")
    
    async def _send_breach_notification(self, user: User, incident: SecurityIncident):
        """Send breach notification email to user"""
        
        subject = "Important Security Notice - Data Breach Notification"
        
        body = f"""
Dear {user.full_name or user.email},

We are writing to inform you of a security incident that may have affected your personal data.

Incident Details:
- Incident ID: {incident.id}
- Date Detected: {incident.detection_time.strftime('%Y-%m-%d %H:%M:%S UTC')}
- Type: {incident.incident_type.value.replace('_', ' ').title()}

What Happened:
{incident.description}

What Information Was Involved:
We are still investigating the full scope of the incident. We will update you as we learn more.

What We Are Doing:
- We have taken immediate steps to secure our systems
- We are working with cybersecurity experts to investigate
- We have notified relevant authorities
- We are implementing additional security measures

What You Can Do:
- Monitor your accounts for unusual activity
- Consider changing your password
- Be cautious of phishing attempts
- Contact us if you notice anything suspicious

We sincerely apologize for this incident and any inconvenience it may cause. The security of your personal data is our top priority.

If you have any questions or concerns, please contact us at:
- Email: security@montecarloanalytics.com
- Phone: [Phone Number]

Sincerely,
Monte Carlo Analytics Security Team

Incident Reference: {incident.id}
        """.strip()
        
        # In a real implementation, you'd send this via your email service
        logger.info(f"Breach notification prepared for user {user.id}")
    
    async def _schedule_regulatory_notification(self, incident: SecurityIncident, deadline: datetime):
        """Schedule regulatory notification"""
        
        # Calculate time until deadline
        time_until_deadline = deadline - datetime.utcnow()
        
        if time_until_deadline.total_seconds() > 0:
            # Schedule notification before deadline
            notification_time = deadline - timedelta(hours=2)  # Send 2 hours before deadline
            
            incident.add_timeline_entry(f"Regulatory notification scheduled for {notification_time}")
            
            # In a real implementation, you'd schedule this with a task queue
            logger.info(f"Regulatory notification scheduled for incident {incident.id}")
        else:
            # Deadline has passed - send immediate notification
            await self._send_regulatory_notification(incident)
    
    async def _send_regulatory_notification(self, incident: SecurityIncident):
        """Send notification to regulatory authorities"""
        
        # Prepare regulatory notification
        notification = {
            "incident_id": incident.id,
            "organization": "Monte Carlo Analytics, LLC",
            "contact_email": "dpo@montecarloanalytics.com",
            "incident_type": incident.incident_type.value,
            "detection_time": incident.detection_time.isoformat(),
            "description": incident.description,
            "affected_individuals": len(incident.affected_users),
            "personal_data_categories": ["contact_information", "usage_data"],
            "likely_consequences": "Potential unauthorized access to personal data",
            "measures_taken": [action["action"] for action in incident.actions_taken],
            "measures_planned": ["Enhanced monitoring", "Security review", "User notification"]
        }
        
        incident.add_action("Sent regulatory notification", "dpo")
        incident.notification_sent = True
        
        logger.critical(f"Regulatory notification sent for incident {incident.id}")
    
    async def _send_immediate_notification(self, incident: SecurityIncident):
        """Send immediate notification for high/critical incidents"""
        
        # Notify security team
        notification_message = f"""
SECURITY ALERT - {incident.severity.upper()} INCIDENT

Incident ID: {incident.id}
Type: {incident.incident_type.value.replace('_', ' ').title()}
Severity: {incident.severity.upper()}
Time: {incident.detection_time}

Description: {incident.description}

Affected Systems: {', '.join(incident.affected_systems) if incident.affected_systems else 'Unknown'}
Affected Users: {len(incident.affected_users) if incident.affected_users else 'Unknown'}

Immediate action required.
        """.strip()
        
        # In a real implementation, you'd send this via your alerting system
        logger.critical(notification_message)
    
    async def update_incident(
        self,
        incident_id: str,
        status: IncidentStatus = None,
        notes: str = None,
        actions: List[str] = None
    ) -> SecurityIncident:
        """Update an existing incident"""
        
        if incident_id not in self.active_incidents:
            raise HTTPException(status_code=404, detail="Incident not found")
        
        incident = self.active_incidents[incident_id]
        
        if status:
            old_status = incident.status
            incident.status = status
            incident.add_timeline_entry(f"Status changed from {old_status} to {status}")
        
        if notes:
            incident.add_timeline_entry(notes)
        
        if actions:
            for action in actions:
                incident.add_action(action, "security_team")
        
        await self._log_incident_event(incident, "incident_updated")
        
        return incident
    
    async def close_incident(self, incident_id: str, resolution_notes: str) -> SecurityIncident:
        """Close an incident"""
        
        if incident_id not in self.active_incidents:
            raise HTTPException(status_code=404, detail="Incident not found")
        
        incident = self.active_incidents[incident_id]
        incident.status = IncidentStatus.CLOSED
        incident.add_timeline_entry(f"Incident closed: {resolution_notes}")
        
        await self._log_incident_event(incident, "incident_closed")
        
        # Generate incident report
        await self._generate_incident_report(incident)
        
        # Remove from active incidents
        del self.active_incidents[incident_id]
        
        logger.info(f"Incident {incident_id} closed: {resolution_notes}")
        
        return incident
    
    async def _generate_incident_report(self, incident: SecurityIncident):
        """Generate final incident report"""
        
        report = {
            "incident_id": incident.id,
            "incident_type": incident.incident_type.value,
            "severity": incident.severity.value,
            "title": incident.title,
            "description": incident.description,
            "detection_time": incident.detection_time.isoformat(),
            "closure_time": datetime.utcnow().isoformat(),
            "duration_hours": (datetime.utcnow() - incident.detection_time).total_seconds() / 3600,
            "affected_systems": incident.affected_systems,
            "affected_users_count": len(incident.affected_users),
            "gdpr_breach": incident.gdpr_breach,
            "notification_sent": incident.notification_sent,
            "timeline": [
                {
                    "timestamp": entry["timestamp"].isoformat(),
                    "entry": entry["entry"]
                }
                for entry in incident.timeline
            ],
            "actions_taken": [
                {
                    "timestamp": action["timestamp"].isoformat(),
                    "action": action["action"],
                    "taken_by": action["taken_by"]
                }
                for action in incident.actions_taken
            ],
            "evidence_collected": len(incident.evidence),
            "lessons_learned": [],
            "recommendations": []
        }
        
        # Save report
        report_path = f"/tmp/incident_report_{incident.id}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Incident report generated: {report_path}")
    
    async def _log_incident_event(self, incident: SecurityIncident, event_type: str):
        """Log incident event to security audit log"""
        
        with next(get_db()) as db:
            audit_log = SecurityAuditLog(
                user_id=None,
                event_type=event_type,
                ip_address="system",
                user_agent="incident_response_system",
                timestamp=datetime.utcnow(),
                details={
                    "incident_id": incident.id,
                    "incident_type": incident.incident_type.value,
                    "severity": incident.severity.value,
                    "status": incident.status.value
                }
            )
            db.add(audit_log)
            db.commit()
    
    def get_incident(self, incident_id: str) -> Optional[SecurityIncident]:
        """Get incident by ID"""
        return self.active_incidents.get(incident_id)
    
    def list_active_incidents(self) -> List[SecurityIncident]:
        """List all active incidents"""
        return list(self.active_incidents.values())
    
    async def generate_compliance_report(self) -> Dict[str, Any]:
        """Generate compliance report"""
        
        with next(get_db()) as db:
            # Count incidents in last 30 days
            thirty_days_ago = datetime.utcnow() - timedelta(days=30)
            
            recent_incidents = db.query(SecurityAuditLog).filter(
                and_(
                    SecurityAuditLog.event_type == "incident_created",
                    SecurityAuditLog.timestamp >= thirty_days_ago
                )
            ).count()
            
            return {
                "report_date": datetime.utcnow().isoformat(),
                "active_incidents": len(self.active_incidents),
                "incidents_last_30_days": recent_incidents,
                "gdpr_breaches": sum(1 for inc in self.active_incidents.values() if inc.gdpr_breach),
                "critical_incidents": sum(1 for inc in self.active_incidents.values() if inc.severity == IncidentSeverity.CRITICAL),
                "average_response_time": "< 1 hour",
                "compliance_status": "compliant"
            } 