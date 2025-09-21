"""
Legal Consent Management Service

Handles legal document management, user consent tracking, and GDPR compliance
for Terms of Service, Privacy Policy, and other legal documents.
"""

import logging
import hashlib
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path

from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc
from fastapi import HTTPException, status

from database import get_db
from models import User
from models_legal_consent import LegalDocument, UserLegalConsent, ConsentAuditLog

logger = logging.getLogger(__name__)


class LegalConsentService:
    """
    Service for managing legal document consent and compliance
    """
    
    def __init__(self):
        self.legal_documents_path = Path("legal")
    
    async def get_required_consents_for_user(self, user_id: int) -> List[Dict[str, Any]]:
        """
        Get all legal documents that user needs to consent to
        Returns documents that are active and haven't been consented to by the user
        """
        with next(get_db()) as db:
            # Get all active legal documents
            active_docs = db.query(LegalDocument).filter(
                LegalDocument.is_active == True,
                LegalDocument.requires_explicit_consent == True
            ).all()
            
            required_consents = []
            
            for doc in active_docs:
                # Check if user has already consented to this version
                existing_consent = db.query(UserLegalConsent).filter(
                    and_(
                        UserLegalConsent.user_id == user_id,
                        UserLegalConsent.document_id == doc.id,
                        UserLegalConsent.consent_given == True,
                        UserLegalConsent.withdrawn_at.is_(None)
                    )
                ).first()
                
                if not existing_consent:
                    required_consents.append({
                        "document_id": doc.id,
                        "document_type": doc.document_type,
                        "version": doc.version,
                        "title": doc.title,
                        "content_path": doc.content_path,
                        "effective_date": doc.effective_date.isoformat(),
                        "requires_consent": True
                    })
            
            return required_consents
    
    async def record_user_consent(
        self,
        user_id: int,
        document_consents: Dict[str, bool],
        consent_method: str = "registration",
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        session_id: Optional[str] = None,
        consent_context: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Record user consent for multiple legal documents
        
        Args:
            user_id: User ID
            document_consents: Dict mapping document_type to consent boolean
            consent_method: How consent was obtained
            ip_address: User's IP address
            user_agent: User's browser/client info
            session_id: Session identifier
            consent_context: Additional context (e.g., registration flow, plan)
        """
        with next(get_db()) as db:
            try:
                consent_records = []
                
                for document_type, consent_given in document_consents.items():
                    # Get the current active version of the document
                    document = db.query(LegalDocument).filter(
                        and_(
                            LegalDocument.document_type == document_type,
                            LegalDocument.is_active == True
                        )
                    ).order_by(desc(LegalDocument.effective_date)).first()
                    
                    if not document:
                        logger.warning(f"No active document found for type: {document_type}")
                        continue
                    
                    # Create consent record
                    consent_record = UserLegalConsent(
                        user_id=user_id,
                        document_id=document.id,
                        consent_given=consent_given,
                        consent_method=consent_method,
                        ip_address=ip_address,
                        user_agent=user_agent,
                        session_id=session_id,
                        consent_context=consent_context
                    )
                    
                    db.add(consent_record)
                    consent_records.append(consent_record)
                    
                    # Create audit log entry
                    audit_log = ConsentAuditLog(
                        user_id=user_id,
                        action_type="consent_given" if consent_given else "consent_denied",
                        document_type=document_type,
                        document_version=document.version,
                        details={
                            "consent_method": consent_method,
                            "consent_context": consent_context
                        },
                        ip_address=ip_address,
                        user_agent=user_agent,
                        session_id=session_id
                    )
                    
                    db.add(audit_log)
                
                db.commit()
                
                logger.info(f"Recorded consent for user {user_id}: {document_consents}")
                
                return {
                    "status": "success",
                    "message": "Consent preferences recorded successfully",
                    "consents_recorded": len(consent_records),
                    "timestamp": datetime.utcnow().isoformat()
                }
                
            except Exception as e:
                db.rollback()
                logger.error(f"Error recording consent for user {user_id}: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Failed to record consent preferences"
                )
    
    async def get_user_consent_history(self, user_id: int) -> List[Dict[str, Any]]:
        """
        Get complete consent history for a user
        """
        with next(get_db()) as db:
            consents = db.query(UserLegalConsent).filter(
                UserLegalConsent.user_id == user_id
            ).order_by(desc(UserLegalConsent.consent_timestamp)).all()
            
            consent_history = []
            for consent in consents:
                consent_history.append({
                    "id": consent.id,
                    "document_type": consent.document.document_type,
                    "document_version": consent.document.version,
                    "document_title": consent.document.title,
                    "consent_given": consent.consent_given,
                    "consent_method": consent.consent_method,
                    "consent_timestamp": consent.consent_timestamp.isoformat(),
                    "withdrawn_at": consent.withdrawn_at.isoformat() if consent.withdrawn_at else None,
                    "ip_address": consent.ip_address,
                    "consent_context": consent.consent_context
                })
            
            return consent_history
    
    async def withdraw_consent(
        self,
        user_id: int,
        document_type: str,
        withdrawal_reason: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Withdraw user consent for a specific document type
        """
        with next(get_db()) as db:
            try:
                # Find the current consent record
                consent_record = db.query(UserLegalConsent).join(LegalDocument).filter(
                    and_(
                        UserLegalConsent.user_id == user_id,
                        LegalDocument.document_type == document_type,
                        UserLegalConsent.consent_given == True,
                        UserLegalConsent.withdrawn_at.is_(None)
                    )
                ).first()
                
                if not consent_record:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail="No active consent found for this document type"
                    )
                
                # Mark as withdrawn
                consent_record.withdrawn_at = datetime.utcnow()
                consent_record.withdrawal_reason = withdrawal_reason
                
                # Create audit log
                audit_log = ConsentAuditLog(
                    user_id=user_id,
                    action_type="consent_withdrawn",
                    document_type=document_type,
                    document_version=consent_record.document.version,
                    details={
                        "withdrawal_reason": withdrawal_reason,
                        "original_consent_timestamp": consent_record.consent_timestamp.isoformat()
                    }
                )
                
                db.add(audit_log)
                db.commit()
                
                logger.info(f"Consent withdrawn for user {user_id}, document: {document_type}")
                
                return {
                    "status": "success",
                    "message": "Consent withdrawn successfully",
                    "document_type": document_type,
                    "withdrawn_at": consent_record.withdrawn_at.isoformat()
                }
                
            except HTTPException:
                raise
            except Exception as e:
                db.rollback()
                logger.error(f"Error withdrawing consent for user {user_id}: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Failed to withdraw consent"
                )
    
    async def initialize_legal_documents(self) -> Dict[str, Any]:
        """
        Initialize legal documents in the database from the legal/ directory
        Should be called during application startup
        """
        with next(get_db()) as db:
            try:
                documents_initialized = []
                
                # Define the legal documents to initialize
                legal_docs = [
                    {
                        "document_type": "terms_of_service",
                        "version": "2025.1",
                        "title": "Terms of Service",
                        "content_path": "legal/TERMS_OF_SERVICE.md",
                        "requires_explicit_consent": True
                    },
                    {
                        "document_type": "privacy_policy",
                        "version": "2025.1",
                        "title": "Privacy Policy",
                        "content_path": "legal/PRIVACY_POLICY.md",
                        "requires_explicit_consent": True
                    },
                    {
                        "document_type": "cookie_policy",
                        "version": "2025.1",
                        "title": "Cookie Policy",
                        "content_path": "legal/COOKIE_POLICY.md",
                        "requires_explicit_consent": False  # Handled by cookie banner
                    },
                    {
                        "document_type": "acceptable_use_policy",
                        "version": "2025.1",
                        "title": "Acceptable Use Policy",
                        "content_path": "legal/ACCEPTABLE_USE_POLICY.md",
                        "requires_explicit_consent": True
                    }
                ]
                
                for doc_config in legal_docs:
                    # Check if this document version already exists
                    existing_doc = db.query(LegalDocument).filter(
                        and_(
                            LegalDocument.document_type == doc_config["document_type"],
                            LegalDocument.version == doc_config["version"]
                        )
                    ).first()
                    
                    if existing_doc:
                        logger.info(f"Legal document already exists: {doc_config['document_type']} v{doc_config['version']}")
                        continue
                    
                    # Calculate content hash for integrity
                    content_path = Path(doc_config["content_path"])
                    if content_path.exists():
                        content_hash = hashlib.sha256(content_path.read_bytes()).hexdigest()
                    else:
                        logger.warning(f"Legal document file not found: {content_path}")
                        content_hash = "unknown"
                    
                    # Create document record
                    legal_doc = LegalDocument(
                        document_type=doc_config["document_type"],
                        version=doc_config["version"],
                        title=doc_config["title"],
                        content_path=doc_config["content_path"],
                        content_hash=content_hash,
                        effective_date=datetime.utcnow(),
                        requires_explicit_consent=doc_config["requires_explicit_consent"]
                    )
                    
                    db.add(legal_doc)
                    documents_initialized.append(doc_config["document_type"])
                
                db.commit()
                
                logger.info(f"Initialized {len(documents_initialized)} legal documents")
                
                return {
                    "status": "success",
                    "documents_initialized": documents_initialized,
                    "total_count": len(documents_initialized)
                }
                
            except Exception as e:
                db.rollback()
                logger.error(f"Error initializing legal documents: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Failed to initialize legal documents"
                )
    
    async def get_document_content(self, document_type: str, version: Optional[str] = None) -> Dict[str, Any]:
        """
        Get the content of a legal document
        """
        with next(get_db()) as db:
            query = db.query(LegalDocument).filter(
                LegalDocument.document_type == document_type
            )
            
            if version:
                query = query.filter(LegalDocument.version == version)
            else:
                query = query.filter(LegalDocument.is_active == True)
            
            document = query.order_by(desc(LegalDocument.effective_date)).first()
            
            if not document:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Document not found: {document_type}"
                )
            
            # Read content from file
            content_path = Path(document.content_path)
            if content_path.exists():
                content = content_path.read_text(encoding='utf-8')
            else:
                logger.error(f"Legal document file not found: {content_path}")
                content = f"# {document.title}\n\nDocument content not available."
            
            return {
                "document_id": document.id,
                "document_type": document.document_type,
                "version": document.version,
                "title": document.title,
                "content": content,
                "effective_date": document.effective_date.isoformat(),
                "requires_explicit_consent": document.requires_explicit_consent
            }




