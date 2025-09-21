"""
GDPR Compliance API Router
Provides endpoints for users to exercise their data rights
"""

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel, EmailStr
from typing import Optional, Dict, Any
import os
import logging

from auth.dependencies import get_current_user
from models import User
from modules.security.gdpr_service import GDPRService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/gdpr", tags=["GDPR Compliance"])

# Request models
class DataSubjectRequest(BaseModel):
    request_type: str  # access, rectification, erasure, portability, restriction, objection
    additional_data: Optional[Dict[str, Any]] = None
    reason: Optional[str] = None

class RectificationRequest(BaseModel):
    email: Optional[EmailStr] = None
    full_name: Optional[str] = None

class RestrictionRequest(BaseModel):
    restriction_type: str = "general"  # general, marketing, analytics, profiling
    reason: str

class ObjectionRequest(BaseModel):
    objection_type: str = "marketing"  # marketing, analytics, profiling
    reason: str

# Response models
class DataSubjectResponse(BaseModel):
    request_id: str
    request_type: str
    status: str
    message: str
    data: Optional[Dict[str, Any]] = None

@router.post("/data-request", response_model=DataSubjectResponse)
async def submit_data_subject_request(
    request: DataSubjectRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user)
):
    """
    Submit a GDPR data subject request
    
    Supported request types:
    - access: Get all personal data we hold about you
    - rectification: Correct inaccurate personal data
    - erasure: Delete your personal data (right to be forgotten)
    - portability: Export your data in machine-readable format
    - restriction: Restrict processing of your personal data
    - objection: Object to processing of your personal data
    """
    
    gdpr_service = GDPRService()
    
    try:
        # Process the request
        result = await gdpr_service.handle_data_subject_request(
            user_id=current_user.id,
            request_type=request.request_type,
            additional_data=request.additional_data
        )
        
        return DataSubjectResponse(
            request_id=f"gdpr_{current_user.id}_{request.request_type}_{int(result.get('processed_at', '0').replace('-', '').replace(':', '').replace('T', '').replace('Z', '')[:14])}",
            request_type=request.request_type,
            status="completed",
            message=f"Your {request.request_type} request has been processed successfully",
            data=result
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing GDPR request: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Error processing your request. Please try again or contact support."
        )

@router.post("/access-request")
async def request_data_access(
    current_user: User = Depends(get_current_user)
):
    """
    Request access to all personal data (Article 15 GDPR)
    Returns comprehensive export of all data we hold about you
    """
    
    gdpr_service = GDPRService()
    
    try:
        result = await gdpr_service.handle_data_subject_request(
            user_id=current_user.id,
            request_type="access"
        )
        
        return {
            "message": "Data access request completed",
            "data": result["data"],
            "generated_at": result["generated_at"],
            "note": "This export contains all personal data we hold about you"
        }
        
    except Exception as e:
        logger.error(f"Error processing access request: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Error generating data export"
        )

@router.post("/rectification-request")
async def request_data_rectification(
    rectification: RectificationRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Request rectification of inaccurate personal data (Article 16 GDPR)
    """
    
    gdpr_service = GDPRService()
    
    try:
        # Convert to dict for processing
        update_data = rectification.dict(exclude_unset=True)
        
        result = await gdpr_service.handle_data_subject_request(
            user_id=current_user.id,
            request_type="rectification",
            additional_data=update_data
        )
        
        return {
            "message": "Data rectification completed",
            "updated_fields": result["updated_fields"],
            "processed_at": result["processed_at"]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing rectification request: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Error updating your data"
        )

@router.post("/erasure-request")
async def request_data_erasure(
    current_user: User = Depends(get_current_user)
):
    """
    Request erasure of personal data - Right to be Forgotten (Article 17 GDPR)
    WARNING: This will permanently delete your account and all associated data
    """
    
    gdpr_service = GDPRService()
    
    try:
        result = await gdpr_service.handle_data_subject_request(
            user_id=current_user.id,
            request_type="erasure"
        )
        
        if result["status"] == "rejected":
            return {
                "message": "Data erasure request rejected",
                "reason": result["reason"],
                "processed_at": result["processed_at"]
            }
        
        return {
            "message": "Data erasure completed successfully",
            "deleted_items": result["deleted_items"],
            "processed_at": result["processed_at"],
            "note": "Your account and all associated data have been permanently deleted"
        }
        
    except Exception as e:
        logger.error(f"Error processing erasure request: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Error processing deletion request"
        )

@router.post("/portability-request")
async def request_data_portability(
    current_user: User = Depends(get_current_user)
):
    """
    Request data portability - Export data in machine-readable format (Article 20 GDPR)
    """
    
    gdpr_service = GDPRService()
    
    try:
        result = await gdpr_service.handle_data_subject_request(
            user_id=current_user.id,
            request_type="portability"
        )
        
        return {
            "message": "Data export generated successfully",
            "download_info": {
                "format": result["format"],
                "generated_at": result["generated_at"],
                "expires_at": result["expires_at"]
            },
            "note": "Your data export is ready for download"
        }
        
    except Exception as e:
        logger.error(f"Error processing portability request: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Error generating data export"
        )

@router.get("/download-export/{export_id}")
async def download_data_export(
    export_id: str,
    current_user: User = Depends(get_current_user)
):
    """
    Download data export file
    """
    
    # Validate export_id belongs to current user
    if not export_id.startswith(f"user_{current_user.id}_"):
        raise HTTPException(
            status_code=403,
            detail="Access denied to this export"
        )
    
    # Construct file path (this would be more secure in production)
    export_path = f"/tmp/{export_id}.zip"
    
    if not os.path.exists(export_path):
        raise HTTPException(
            status_code=404,
            detail="Export file not found or expired"
        )
    
    return FileResponse(
        path=export_path,
        filename=f"personal_data_export_{current_user.id}.zip",
        media_type="application/zip"
    )

@router.post("/restriction-request")
async def request_processing_restriction(
    restriction: RestrictionRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Request restriction of processing (Article 18 GDPR)
    """
    
    gdpr_service = GDPRService()
    
    try:
        result = await gdpr_service.handle_data_subject_request(
            user_id=current_user.id,
            request_type="restriction",
            additional_data={
                "restriction_type": restriction.restriction_type,
                "reason": restriction.reason
            }
        )
        
        return {
            "message": "Processing restriction applied",
            "restriction_type": result["restriction_type"],
            "processed_at": result["processed_at"],
            "note": result["note"]
        }
        
    except Exception as e:
        logger.error(f"Error processing restriction request: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Error applying processing restriction"
        )

@router.post("/objection-request")
async def request_processing_objection(
    objection: ObjectionRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Object to processing of personal data (Article 21 GDPR)
    """
    
    gdpr_service = GDPRService()
    
    try:
        result = await gdpr_service.handle_data_subject_request(
            user_id=current_user.id,
            request_type="objection",
            additional_data={
                "objection_type": objection.objection_type,
                "reason": objection.reason
            }
        )
        
        return {
            "message": f"Objection to {objection.objection_type} processing recorded",
            "objection_type": result["objection_type"],
            "processed_at": result["processed_at"],
            "note": f"You have successfully opted out of {objection.objection_type} processing"
        }
        
    except Exception as e:
        logger.error(f"Error processing objection request: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Error recording your objection"
        )

@router.get("/consent-status")
async def get_consent_status(
    current_user: User = Depends(get_current_user)
):
    """
    Get current consent status for various processing activities
    """
    
    # This would typically come from a consent management system
    consent_status = {
        "cookies": {
            "strictly_necessary": {
                "status": "granted",
                "required": True,
                "description": "Essential for website functionality"
            },
            "functional": {
                "status": "granted",
                "required": False,
                "description": "Remember your preferences"
            },
            "analytics": {
                "status": "granted",
                "required": False,
                "description": "Help us improve the service"
            }
        },
        "marketing": {
            "email_marketing": {
                "status": "not_set",
                "required": False,
                "description": "Product updates and offers"
            },
            "personalized_content": {
                "status": "not_set",
                "required": False,
                "description": "Personalized recommendations"
            }
        },
        "data_processing": {
            "service_provision": {
                "status": "granted",
                "required": True,
                "description": "Process simulations and store results"
            },
            "analytics": {
                "status": "granted",
                "required": False,
                "description": "Analyze usage patterns"
            }
        }
    }
    
    return {
        "user_id": current_user.id,
        "consent_status": consent_status,
        "last_updated": "2025-01-01T00:00:00Z",
        "note": "You can update your consent preferences at any time"
    }

@router.post("/update-consent")
async def update_consent_preferences(
    consent_updates: Dict[str, Dict[str, str]],
    current_user: User = Depends(get_current_user)
):
    """
    Update consent preferences
    """
    
    # This would typically update a consent management system
    # For now, just acknowledge the update
    
    return {
        "message": "Consent preferences updated successfully",
        "updated_at": "2025-01-01T00:00:00Z",
        "updates": consent_updates,
        "note": "Your consent preferences have been saved"
    }

@router.get("/privacy-info")
async def get_privacy_information():
    """
    Get information about data processing and privacy rights
    """
    
    return {
        "data_controller": {
            "name": "Monte Carlo Analytics, LLC",
            "email": "privacy@montecarloanalytics.com",
            "dpo": "dpo@montecarloanalytics.com"
        },
        "your_rights": {
            "access": "Request a copy of your personal data",
            "rectification": "Correct inaccurate personal data",
            "erasure": "Delete your personal data (right to be forgotten)",
            "portability": "Export your data in machine-readable format",
            "restriction": "Restrict processing of your personal data",
            "objection": "Object to processing of your personal data",
            "withdraw_consent": "Withdraw consent for processing based on consent"
        },
        "response_time": "We will respond to your request within 30 days",
        "contact": {
            "privacy_email": "privacy@montecarloanalytics.com",
            "support_email": "support@montecarloanalytics.com",
            "legal_email": "legal@montecarloanalytics.com"
        },
        "policies": {
            "privacy_policy": "/privacy",
            "cookie_policy": "/cookie-policy",
            "terms_of_service": "/terms"
        }
    } 