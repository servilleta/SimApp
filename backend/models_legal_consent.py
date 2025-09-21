"""
Legal Consent Database Models

Models for tracking user acceptance of legal documents (Terms of Service, Privacy Policy, etc.)
to ensure GDPR compliance and legal protection.
"""

from sqlalchemy import Boolean, Column, Integer, String, Text, DateTime, ForeignKey, JSON
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from database import Base


class LegalDocument(Base):
    """
    Model for storing legal documents and their versions
    """
    __tablename__ = "legal_documents"

    id = Column(Integer, primary_key=True, index=True)
    
    # Document identification
    document_type = Column(String, nullable=False, index=True)  # 'terms_of_service', 'privacy_policy', 'cookie_policy', etc.
    version = Column(String, nullable=False)  # e.g., "1.0", "2024.1", "2025-01-01"
    title = Column(String, nullable=False)
    
    # Document content
    content_path = Column(String, nullable=False)  # Path to markdown file
    content_hash = Column(String, nullable=False)  # SHA-256 hash for integrity verification
    
    # Metadata
    effective_date = Column(DateTime(timezone=True), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    is_active = Column(Boolean, default=True, nullable=False)
    
    # Legal requirements
    requires_explicit_consent = Column(Boolean, default=True, nullable=False)
    applies_to_existing_users = Column(Boolean, default=False, nullable=False)
    
    # Relationships
    consents = relationship("UserLegalConsent", back_populates="document")

    def __repr__(self):
        return f"<LegalDocument(type='{self.document_type}', version='{self.version}')>"


class UserLegalConsent(Base):
    """
    Model for tracking user acceptance of legal documents
    Critical for GDPR compliance and legal protection
    """
    __tablename__ = "user_legal_consents"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    document_id = Column(Integer, ForeignKey("legal_documents.id"), nullable=False, index=True)
    
    # Consent details
    consent_given = Column(Boolean, nullable=False)
    consent_method = Column(String, nullable=False)  # 'registration', 'update_prompt', 'api', 'admin'
    consent_timestamp = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    
    # Technical details for audit
    ip_address = Column(String, nullable=True)
    user_agent = Column(Text, nullable=True)
    session_id = Column(String, nullable=True)
    
    # Consent context
    consent_context = Column(JSON, nullable=True)  # Additional context: registration flow, plan selection, etc.
    
    # Withdrawal tracking
    withdrawn_at = Column(DateTime(timezone=True), nullable=True)
    withdrawal_reason = Column(String, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    user = relationship("User")
    document = relationship("LegalDocument", back_populates="consents")

    def __repr__(self):
        return f"<UserLegalConsent(user_id={self.user_id}, document_id={self.document_id}, consent={self.consent_given})>"


class ConsentAuditLog(Base):
    """
    Audit log for all consent-related activities
    Provides comprehensive tracking for legal compliance
    """
    __tablename__ = "consent_audit_logs"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True, index=True)
    
    # Action details
    action_type = Column(String, nullable=False)  # 'consent_given', 'consent_withdrawn', 'document_updated', 'consent_expired'
    document_type = Column(String, nullable=True)
    document_version = Column(String, nullable=True)
    
    # Context
    details = Column(JSON, nullable=True)
    ip_address = Column(String, nullable=True)
    user_agent = Column(Text, nullable=True)
    session_id = Column(String, nullable=True)
    
    # Administrative
    performed_by_admin = Column(Boolean, default=False, nullable=False)
    admin_user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    
    # Timestamp
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    
    # Relationships
    user = relationship("User", foreign_keys=[user_id])
    admin_user = relationship("User", foreign_keys=[admin_user_id])

    def __repr__(self):
        return f"<ConsentAuditLog(action='{self.action_type}', user_id={self.user_id}, timestamp='{self.timestamp}')>"




