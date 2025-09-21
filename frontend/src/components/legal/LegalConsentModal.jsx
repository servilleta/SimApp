import React, { useState, useEffect } from 'react';
import { useAuth0 } from '@auth0/auth0-react';

const LegalConsentModal = ({ 
  isOpen, 
  onClose, 
  onConsentGiven, 
  requiredDocuments = [],
  context = "registration" 
}) => {
  const [consents, setConsents] = useState({});
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [expandedDocument, setExpandedDocument] = useState(null);
  const { getAccessTokenSilently } = useAuth0();

  useEffect(() => {
    if (isOpen && requiredDocuments.length > 0) {
      // Initialize consent state - all required documents start as false
      const initialConsents = {};
      requiredDocuments.forEach(doc => {
        initialConsents[doc.document_type] = false;
      });
      setConsents(initialConsents);
    }
  }, [isOpen, requiredDocuments]);

  const handleConsentChange = (documentType, value) => {
    setConsents(prev => ({
      ...prev,
      [documentType]: value
    }));
  };

  const areAllRequiredConsentsGiven = () => {
    return requiredDocuments.every(doc => 
      !doc.requires_consent || consents[doc.document_type] === true
    );
  };

  const handleSubmit = async () => {
    if (!areAllRequiredConsentsGiven()) {
      setError('Please accept all required legal documents to continue.');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const token = await getAccessTokenSilently();
      const apiBaseUrl = import.meta.env.VITE_API_URL || 'http://localhost:9090/api';

      const response = await fetch(`${apiBaseUrl}/legal/record-consent`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify({
          document_consents: consents,
          consent_context: {
            context_type: context,
            timestamp: new Date().toISOString(),
            user_agent: navigator.userAgent
          }
        })
      });

      if (response.ok) {
        const result = await response.json();
        console.log('✅ Legal consent recorded:', result);
        onConsentGiven && onConsentGiven(consents);
        onClose();
      } else {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to record consent');
      }

    } catch (err) {
      console.error('❌ Error recording legal consent:', err);
      setError('Failed to record your consent. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const openDocumentPreview = async (documentType) => {
    try {
      const apiBaseUrl = import.meta.env.VITE_API_URL || 'http://localhost:9090/api';
      window.open(`${apiBaseUrl}/legal/document/${documentType}`, '_blank');
    } catch (err) {
      console.error('Error opening document:', err);
    }
  };

  if (!isOpen) return null;

  const styles = {
    overlay: {
      position: 'fixed',
      top: 0,
      left: 0,
      right: 0,
      bottom: 0,
      backgroundColor: 'rgba(0, 0, 0, 0.7)',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      zIndex: 10000,
      padding: '20px'
    },
    modal: {
      backgroundColor: '#ffffff',
      borderRadius: '12px',
      boxShadow: '0 20px 40px rgba(0, 0, 0, 0.3)',
      maxWidth: '600px',
      width: '100%',
      maxHeight: '80vh',
      overflow: 'hidden',
      display: 'flex',
      flexDirection: 'column'
    },
    header: {
      padding: '24px',
      borderBottom: '1px solid #e0e0e0',
      backgroundColor: '#f8f9fa'
    },
    title: {
      margin: 0,
      fontSize: '24px',
      fontWeight: '600',
      color: '#1a1a1a'
    },
    subtitle: {
      margin: '8px 0 0 0',
      fontSize: '14px',
      color: '#666',
      lineHeight: '1.4'
    },
    content: {
      padding: '24px',
      flex: 1,
      overflow: 'auto'
    },
    documentItem: {
      marginBottom: '20px',
      padding: '16px',
      border: '1px solid #e0e0e0',
      borderRadius: '8px',
      backgroundColor: '#fafafa'
    },
    documentHeader: {
      display: 'flex',
      alignItems: 'flex-start',
      gap: '12px',
      marginBottom: '12px'
    },
    checkbox: {
      marginTop: '2px',
      transform: 'scale(1.2)'
    },
    documentTitle: {
      flex: 1,
      margin: 0,
      fontSize: '16px',
      fontWeight: '500',
      color: '#1a1a1a'
    },
    documentActions: {
      display: 'flex',
      gap: '12px',
      marginTop: '8px'
    },
    linkButton: {
      background: 'none',
      border: 'none',
      color: '#4f46e5',
      textDecoration: 'underline',
      cursor: 'pointer',
      fontSize: '14px',
      padding: 0
    },
    footer: {
      padding: '24px',
      borderTop: '1px solid #e0e0e0',
      backgroundColor: '#f8f9fa'
    },
    error: {
      marginBottom: '16px',
      padding: '12px',
      backgroundColor: '#fee',
      border: '1px solid #fcc',
      borderRadius: '6px',
      color: '#c33',
      fontSize: '14px'
    },
    buttonGroup: {
      display: 'flex',
      gap: '12px',
      justifyContent: 'flex-end'
    },
    button: {
      padding: '12px 24px',
      borderRadius: '6px',
      fontSize: '14px',
      fontWeight: '500',
      cursor: 'pointer',
      border: 'none',
      transition: 'background-color 0.2s'
    },
    cancelButton: {
      backgroundColor: '#6b7280',
      color: 'white'
    },
    submitButton: {
      backgroundColor: '#4f46e5',
      color: 'white',
      opacity: areAllRequiredConsentsGiven() && !loading ? 1 : 0.5
    },
    requiredLabel: {
      color: '#dc2626',
      fontSize: '12px',
      fontWeight: '500',
      marginLeft: '4px'
    }
  };

  return (
    <div style={styles.overlay} onClick={(e) => e.target === e.currentTarget && onClose()}>
      <div style={styles.modal}>
        <div style={styles.header}>
          <h2 style={styles.title}>Legal Agreements</h2>
          <p style={styles.subtitle}>
            Please review and accept the following legal documents to continue using our Monte Carlo simulation platform.
          </p>
        </div>

        <div style={styles.content}>
          {requiredDocuments.map((doc) => (
            <div key={doc.document_type} style={styles.documentItem}>
              <div style={styles.documentHeader}>
                <input
                  type="checkbox"
                  id={`consent-${doc.document_type}`}
                  checked={consents[doc.document_type] || false}
                  onChange={(e) => handleConsentChange(doc.document_type, e.target.checked)}
                  style={styles.checkbox}
                />
                <div style={{ flex: 1 }}>
                  <label htmlFor={`consent-${doc.document_type}`} style={styles.documentTitle}>
                    I agree to the {doc.title}
                    {doc.requires_consent && <span style={styles.requiredLabel}>*</span>}
                  </label>
                  <div style={styles.documentActions}>
                    <button
                      type="button"
                      onClick={() => openDocumentPreview(doc.document_type)}
                      style={styles.linkButton}
                    >
                      Read Full Document
                    </button>
                    <span style={{ fontSize: '12px', color: '#666' }}>
                      Version {doc.version} • Effective {new Date(doc.effective_date).toLocaleDateString()}
                    </span>
                  </div>
                </div>
              </div>
            </div>
          ))}

          <div style={{ fontSize: '12px', color: '#666', marginTop: '16px', lineHeight: '1.4' }}>
            <p>
              <strong>Your Rights:</strong> You can withdraw your consent at any time through your account settings. 
              For questions about data processing, contact us at privacy@montecarloanalytics.com.
            </p>
            <p>
              <strong>Required for Service:</strong> Acceptance of Terms of Service and Privacy Policy is required 
              to use our simulation platform. Cookie preferences can be managed separately.
            </p>
          </div>
        </div>

        <div style={styles.footer}>
          {error && <div style={styles.error}>{error}</div>}
          
          <div style={styles.buttonGroup}>
            <button
              type="button"
              onClick={onClose}
              style={{ ...styles.button, ...styles.cancelButton }}
              disabled={loading}
            >
              Cancel
            </button>
            <button
              type="button"
              onClick={handleSubmit}
              disabled={!areAllRequiredConsentsGiven() || loading}
              style={{ ...styles.button, ...styles.submitButton }}
            >
              {loading ? 'Processing...' : 'Accept & Continue'}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default LegalConsentModal;




