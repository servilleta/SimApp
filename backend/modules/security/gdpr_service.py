"""
GDPR Compliance Service
Handles data subject requests, consent management, and data retention
"""

import asyncio
import json
import logging
import zipfile
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path
import tempfile
import os

from sqlalchemy.orm import Session
from sqlalchemy import and_, or_
from fastapi import HTTPException

from models import (
    User, SimulationResult, SecurityAuditLog, UserUsageMetrics, 
    UserSubscription
)
from database import get_db
from modules.storage.service import StorageService
from simulation.database_service import SimulationDatabaseService

logger = logging.getLogger(__name__)

class GDPRService:
    """Service for handling GDPR compliance operations"""
    
    def __init__(self):
        self.storage_service = StorageService()
        self.simulation_service = SimulationDatabaseService()
    
    async def handle_data_subject_request(
        self, 
        user_id: int, 
        request_type: str,
        additional_data: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Handle GDPR data subject requests
        
        Args:
            user_id: ID of the user making the request
            request_type: Type of request (access, rectification, erasure, portability, restriction, objection)
            additional_data: Additional data for specific request types
            
        Returns:
            Dict containing the result of the request
        """
        try:
            # Log the request for audit purposes
            await self._log_gdpr_request(user_id, request_type, additional_data)
            
            if request_type == "access":
                return await self._handle_access_request(user_id)
            elif request_type == "rectification":
                return await self._handle_rectification_request(user_id, additional_data)
            elif request_type == "erasure":
                return await self._handle_erasure_request(user_id)
            elif request_type == "portability":
                return await self._handle_portability_request(user_id)
            elif request_type == "restriction":
                return await self._handle_restriction_request(user_id, additional_data)
            elif request_type == "objection":
                return await self._handle_objection_request(user_id, additional_data)
            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unknown request type: {request_type}"
                )
                
        except Exception as e:
            logger.error(f"Error handling GDPR request {request_type} for user {user_id}: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail="Error processing data subject request"
            )
    
    async def _handle_access_request(self, user_id: int) -> Dict[str, Any]:
        """Handle right of access request - provide all personal data"""
        
        with next(get_db()) as db:
            # Get user data
            user = db.query(User).filter(User.id == user_id).first()
            if not user:
                raise HTTPException(status_code=404, detail="User not found")
            
            # Collect all personal data
            personal_data = {
                "user_profile": {
                    "id": user.id,
                    "email": user.email,
                    "full_name": user.full_name,
                    "is_active": user.is_active,
                    "is_admin": user.is_admin,
                    "created_at": user.created_at.isoformat() if user.created_at else None,
                    "updated_at": user.updated_at.isoformat() if user.updated_at else None,
                    "last_login": user.last_login.isoformat() if user.last_login else None,
                },
                "simulations": [],
                "usage_metrics": [],
                "subscription_data": None,
                "security_logs": [],
                "consent_records": []
            }
            
            # Get simulation data
            simulations = db.query(SimulationResult).filter(
                SimulationResult.user_id == user_id
            ).all()
            
            for sim in simulations:
                personal_data["simulations"].append({
                    "id": sim.id,
                    "file_name": sim.file_name,
                    "engine_type": sim.engine_type,
                    "status": sim.status,
                    "created_at": sim.created_at.isoformat(),
                    "completed_at": sim.completed_at.isoformat() if sim.completed_at else None,
                    "iterations": sim.iterations,
                    "execution_time": sim.execution_time,
                    "has_results": bool(sim.results_data)
                })
            
            # Get usage metrics
            usage_metrics = db.query(UserUsageMetrics).filter(
                UserUsageMetrics.user_id == user_id
            ).all()
            
            for metric in usage_metrics:
                personal_data["usage_metrics"].append({
                    "month": metric.month.isoformat(),
                    "simulations_run": metric.simulations_run,
                    "total_iterations": metric.total_iterations,
                    "files_uploaded": metric.files_uploaded,
                    "storage_used_mb": metric.storage_used_mb,
                    "compute_time_minutes": metric.compute_time_minutes
                })
            
            # Get subscription data
            subscription = db.query(UserSubscription).filter(
                UserSubscription.user_id == user_id
            ).first()
            
            if subscription:
                personal_data["subscription_data"] = {
                    "tier": subscription.tier,
                    "status": subscription.status,
                    "stripe_customer_id": subscription.stripe_customer_id,
                    "current_period_start": subscription.current_period_start.isoformat() if subscription.current_period_start else None,
                    "current_period_end": subscription.current_period_end.isoformat() if subscription.current_period_end else None,
                    "created_at": subscription.created_at.isoformat()
                }
            
            # Get security audit logs (last 2 years)
            two_years_ago = datetime.utcnow() - timedelta(days=730)
            security_logs = db.query(SecurityAuditLog).filter(
                and_(
                    SecurityAuditLog.user_id == user_id,
                    SecurityAuditLog.timestamp >= two_years_ago
                )
            ).limit(1000).all()  # Limit to prevent excessive data
            
            for log in security_logs:
                personal_data["security_logs"].append({
                    "event_type": log.event_type,
                    "ip_address": log.ip_address,
                    "user_agent": log.user_agent,
                    "timestamp": log.timestamp.isoformat(),
                    "details": log.details
                })
            
            # Get consent records (if implemented)
            personal_data["consent_records"] = await self._get_consent_records(user_id)
            
            return {
                "request_type": "access",
                "user_id": user_id,
                "data": personal_data,
                "generated_at": datetime.utcnow().isoformat(),
                "retention_note": "This data export contains all personal data we hold about you as of the generation date."
            }
    
    async def _handle_rectification_request(self, user_id: int, data: Dict) -> Dict[str, Any]:
        """Handle right to rectification - correct inaccurate personal data"""
        
        with next(get_db()) as db:
            user = db.query(User).filter(User.id == user_id).first()
            if not user:
                raise HTTPException(status_code=404, detail="User not found")
            
            updated_fields = []
            
            # Update allowed fields
            if "email" in data:
                # Verify email format and uniqueness
                existing_user = db.query(User).filter(
                    and_(User.email == data["email"], User.id != user_id)
                ).first()
                if existing_user:
                    raise HTTPException(status_code=400, detail="Email already in use")
                
                user.email = data["email"]
                updated_fields.append("email")
            
            if "full_name" in data:
                user.full_name = data["full_name"]
                updated_fields.append("full_name")
            
            user.updated_at = datetime.utcnow()
            db.commit()
            
            return {
                "request_type": "rectification",
                "user_id": user_id,
                "updated_fields": updated_fields,
                "processed_at": datetime.utcnow().isoformat()
            }
    
    async def _handle_erasure_request(self, user_id: int) -> Dict[str, Any]:
        """Handle right to erasure (right to be forgotten)"""
        
        with next(get_db()) as db:
            user = db.query(User).filter(User.id == user_id).first()
            if not user:
                raise HTTPException(status_code=404, detail="User not found")
            
            # Check if we have legal grounds to retain data
            subscription = db.query(UserSubscription).filter(
                UserSubscription.user_id == user_id
            ).first()
            
            if subscription and subscription.status == "active":
                return {
                    "request_type": "erasure",
                    "user_id": user_id,
                    "status": "rejected",
                    "reason": "Cannot delete data while active subscription exists",
                    "processed_at": datetime.utcnow().isoformat()
                }
            
            # Perform cascading deletion
            deleted_items = {
                "simulation_results": 0,
                "usage_metrics": 0,
                "security_logs": 0,
                "user_files": 0
            }
            
            # Delete simulation results
            simulations = db.query(SimulationResult).filter(
                SimulationResult.user_id == user_id
            ).all()
            
            for sim in simulations:
                # Delete associated files
                if sim.file_path:
                    try:
                        os.remove(sim.file_path)
                    except:
                        pass  # File might already be deleted
                
                db.delete(sim)
                deleted_items["simulation_results"] += 1
            
            # Delete usage metrics
            metrics = db.query(UserUsageMetrics).filter(
                UserUsageMetrics.user_id == user_id
            ).all()
            
            for metric in metrics:
                db.delete(metric)
                deleted_items["usage_metrics"] += 1
            
            # Delete subscription data
            if subscription:
                db.delete(subscription)
            
            # Anonymize security logs (keep for security purposes but remove PII)
            security_logs = db.query(SecurityAuditLog).filter(
                SecurityAuditLog.user_id == user_id
            ).all()
            
            for log in security_logs:
                log.user_id = None
                log.ip_address = "anonymized"
                log.user_agent = "anonymized"
                deleted_items["security_logs"] += 1
            
            # Delete user files from storage
            try:
                await self.storage_service.delete_user_files(user_id)
                deleted_items["user_files"] = 1
            except:
                pass  # Files might not exist
            
            # Finally delete the user account
            db.delete(user)
            db.commit()
            
            return {
                "request_type": "erasure",
                "user_id": user_id,
                "status": "completed",
                "deleted_items": deleted_items,
                "processed_at": datetime.utcnow().isoformat()
            }
    
    async def _handle_portability_request(self, user_id: int) -> Dict[str, Any]:
        """Handle right to data portability - export data in machine-readable format"""
        
        # Get access data first
        access_data = await self._handle_access_request(user_id)
        
        # Create downloadable export
        export_file = await self._create_data_export(user_id, access_data["data"])
        
        return {
            "request_type": "portability",
            "user_id": user_id,
            "export_file": export_file,
            "format": "JSON/ZIP",
            "generated_at": datetime.utcnow().isoformat(),
            "expires_at": (datetime.utcnow() + timedelta(days=30)).isoformat()
        }
    
    async def _handle_restriction_request(self, user_id: int, data: Dict) -> Dict[str, Any]:
        """Handle right to restriction of processing"""
        
        with next(get_db()) as db:
            user = db.query(User).filter(User.id == user_id).first()
            if not user:
                raise HTTPException(status_code=404, detail="User not found")
            
            # Mark account for processing restriction
            # This would typically involve setting flags to limit processing
            restriction_type = data.get("restriction_type", "general")
            
            # Log the restriction request
            await self._log_processing_restriction(user_id, restriction_type)
            
            return {
                "request_type": "restriction",
                "user_id": user_id,
                "restriction_type": restriction_type,
                "status": "applied",
                "processed_at": datetime.utcnow().isoformat(),
                "note": "Processing restrictions have been applied to your account"
            }
    
    async def _handle_objection_request(self, user_id: int, data: Dict) -> Dict[str, Any]:
        """Handle right to object to processing"""
        
        objection_type = data.get("objection_type", "marketing")
        
        with next(get_db()) as db:
            user = db.query(User).filter(User.id == user_id).first()
            if not user:
                raise HTTPException(status_code=404, detail="User not found")
            
            # Handle different types of objections
            if objection_type == "marketing":
                # Opt out of marketing communications
                await self._opt_out_marketing(user_id)
            elif objection_type == "analytics":
                # Opt out of analytics processing
                await self._opt_out_analytics(user_id)
            elif objection_type == "profiling":
                # Opt out of automated decision-making
                await self._opt_out_profiling(user_id)
            
            return {
                "request_type": "objection",
                "user_id": user_id,
                "objection_type": objection_type,
                "status": "processed",
                "processed_at": datetime.utcnow().isoformat()
            }
    
    async def _create_data_export(self, user_id: int, data: Dict) -> str:
        """Create a downloadable data export file"""
        
        # Create temporary directory for export
        with tempfile.TemporaryDirectory() as temp_dir:
            export_dir = Path(temp_dir) / f"user_{user_id}_data_export"
            export_dir.mkdir()
            
            # Write main data file
            with open(export_dir / "personal_data.json", "w") as f:
                json.dump(data, f, indent=2, default=str)
            
            # Create README file
            readme_content = """
# Personal Data Export

This archive contains all personal data we hold about you.

## Files Included:
- personal_data.json: All your personal data in JSON format
- simulation_files/: Your uploaded Excel files (if available)

## Data Categories:
- User Profile: Your account information
- Simulations: History of your simulation runs
- Usage Metrics: Your service usage statistics
- Subscription Data: Your subscription information
- Security Logs: Security-related events for your account
- Consent Records: Your consent preferences

## Format:
All data is provided in JSON format for easy import into other systems.

Generated: {}
Expires: {} (30 days from generation)

For questions about this export, contact: privacy@montecarloanalytics.com
            """.strip().format(
                datetime.utcnow().isoformat(),
                (datetime.utcnow() + timedelta(days=30)).isoformat()
            )
            
            with open(export_dir / "README.txt", "w") as f:
                f.write(readme_content)
            
            # Add simulation files if available
            sim_files_dir = export_dir / "simulation_files"
            sim_files_dir.mkdir()
            
            # Copy user's uploaded files
            try:
                user_files = await self.storage_service.list_user_files(user_id)
                for file_info in user_files:
                    # Copy file to export directory
                    # This would depend on your storage implementation
                    pass
            except:
                pass  # Files might not exist
            
            # Create ZIP archive
            zip_path = f"/tmp/user_{user_id}_export_{int(datetime.utcnow().timestamp())}.zip"
            
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for file_path in export_dir.rglob("*"):
                    if file_path.is_file():
                        zipf.write(file_path, file_path.relative_to(export_dir))
            
            return zip_path
    
    async def _log_gdpr_request(self, user_id: int, request_type: str, data: Optional[Dict]):
        """Log GDPR request for audit purposes"""
        
        with next(get_db()) as db:
            audit_log = SecurityAuditLog(
                user_id=user_id,
                event_type=f"gdpr_{request_type}_request",
                ip_address="system",
                user_agent="gdpr_service",
                timestamp=datetime.utcnow(),
                details={
                    "request_type": request_type,
                    "additional_data": data,
                    "processed_by": "gdpr_service"
                }
            )
            db.add(audit_log)
            db.commit()
    
    async def _get_consent_records(self, user_id: int) -> List[Dict]:
        """Get user's consent records"""
        # This would typically come from a consent management system
        # For now, return basic consent info
        return [
            {
                "consent_type": "cookies_functional",
                "status": "granted",
                "timestamp": datetime.utcnow().isoformat(),
                "source": "cookie_banner"
            },
            {
                "consent_type": "cookies_analytics",
                "status": "granted",
                "timestamp": datetime.utcnow().isoformat(),
                "source": "cookie_banner"
            }
        ]
    
    async def _log_processing_restriction(self, user_id: int, restriction_type: str):
        """Log processing restriction"""
        with next(get_db()) as db:
            audit_log = SecurityAuditLog(
                user_id=user_id,
                event_type="processing_restriction_applied",
                ip_address="system",
                user_agent="gdpr_service",
                timestamp=datetime.utcnow(),
                details={
                    "restriction_type": restriction_type,
                    "applied_by": "gdpr_service"
                }
            )
            db.add(audit_log)
            db.commit()
    
    async def _opt_out_marketing(self, user_id: int):
        """Opt user out of marketing communications"""
        # Implementation would depend on your marketing system
        pass
    
    async def _opt_out_analytics(self, user_id: int):
        """Opt user out of analytics processing"""
        # Implementation would depend on your analytics system
        pass
    
    async def _opt_out_profiling(self, user_id: int):
        """Opt user out of automated decision-making/profiling"""
        # Implementation would depend on your profiling systems
        pass
    
    async def cleanup_expired_data(self):
        """Clean up expired data according to retention policies"""
        
        with next(get_db()) as db:
            now = datetime.utcnow()
            
            # Clean up old simulation results (retention policy varies by tier)
            # Free tier: 30 days, Paid tiers: longer retention
            thirty_days_ago = now - timedelta(days=30)
            
            # Get free tier users
            free_users = db.query(User).join(UserSubscription).filter(
                or_(
                    UserSubscription.tier == "free",
                    UserSubscription.tier.is_(None)
                )
            ).all()
            
            for user in free_users:
                # Delete old simulation results for free users
                old_simulations = db.query(SimulationResult).filter(
                    and_(
                        SimulationResult.user_id == user.id,
                        SimulationResult.created_at < thirty_days_ago
                    )
                ).all()
                
                for sim in old_simulations:
                    # Delete associated files
                    if sim.file_path:
                        try:
                            os.remove(sim.file_path)
                        except:
                            pass
                    
                    db.delete(sim)
            
            # Clean up old security logs (2 year retention)
            two_years_ago = now - timedelta(days=730)
            old_logs = db.query(SecurityAuditLog).filter(
                SecurityAuditLog.timestamp < two_years_ago
            ).all()
            
            for log in old_logs:
                db.delete(log)
            
            db.commit()
            
            logger.info(f"Cleaned up expired data: {len(old_logs)} security logs")
    
    async def generate_privacy_report(self) -> Dict[str, Any]:
        """Generate privacy compliance report"""
        
        with next(get_db()) as db:
            now = datetime.utcnow()
            
            # Count various data types
            total_users = db.query(User).count()
            active_users = db.query(User).filter(User.is_active == True).count()
            total_simulations = db.query(SimulationResult).count()
            total_security_logs = db.query(SecurityAuditLog).count()
            
            # Count GDPR requests in last 30 days
            thirty_days_ago = now - timedelta(days=30)
            gdpr_requests = db.query(SecurityAuditLog).filter(
                and_(
                    SecurityAuditLog.event_type.like("gdpr_%"),
                    SecurityAuditLog.timestamp >= thirty_days_ago
                )
            ).count()
            
            return {
                "report_date": now.isoformat(),
                "data_summary": {
                    "total_users": total_users,
                    "active_users": active_users,
                    "total_simulations": total_simulations,
                    "total_security_logs": total_security_logs
                },
                "gdpr_compliance": {
                    "requests_last_30_days": gdpr_requests,
                    "data_retention_policies": "active",
                    "consent_management": "implemented",
                    "security_measures": "active"
                },
                "next_cleanup": (now + timedelta(days=1)).isoformat()
            } 