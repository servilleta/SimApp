import React, { useState, useEffect } from 'react';
import { useLegalConsent } from '../../hooks/useLegalConsent';

const LegalDocumentsSection = () => {
  const [consentHistory, setConsentHistory] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const { getConsentHistory, withdrawConsent } = useLegalConsent();

  useEffect(() => {
    loadConsentHistory();
  }, []);

  const loadConsentHistory = async () => {
    setLoading(true);
    setError(null);
    
    try {
      const history = await getConsentHistory();
      setConsentHistory(history);
    } catch (err) {
      setError('Failed to load consent history');
    } finally {
      setLoading(false);
    }
  };

  const handleWithdrawConsent = async (documentType) => {
    if (!window.confirm(`Are you sure you want to withdraw consent for ${documentType}? This may affect your access to certain features.`)) {
      return;
    }

    try {
      await withdrawConsent(documentType, 'User withdrawal via account settings');
      await loadConsentHistory(); // Refresh the list
      alert('Consent withdrawn successfully');
    } catch (err) {
      alert('Failed to withdraw consent. Please try again.');
    }
  };

  const openDocument = (documentType) => {
    const apiBaseUrl = import.meta.env.VITE_API_URL || 'http://localhost:9090/api';
    window.open(`${apiBaseUrl}/legal/document/${documentType}`, '_blank');
  };

  const downloadDocument = async (documentType, version) => {
    try {
      const apiBaseUrl = import.meta.env.VITE_API_URL || 'http://localhost:9090/api';
      const response = await fetch(`${apiBaseUrl}/legal/document/${documentType}?version=${version}`);
      
      if (response.ok) {
        const data = await response.json();
        
        // Create a downloadable file
        const blob = new Blob([data.content], { type: 'text/markdown' });
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `${data.title.replace(/\s+/g, '_')}_v${data.version}.md`;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
      }
    } catch (err) {
      console.error('Error downloading document:', err);
      alert('Failed to download document');
    }
  };

  const getStatusColor = (consent) => {
    if (consent.withdrawn_at) return '#dc2626'; // Red for withdrawn
    if (consent.consent_given) return '#059669'; // Green for active
    return '#d97706'; // Orange for denied
  };

  const getStatusText = (consent) => {
    if (consent.withdrawn_at) return 'Withdrawn';
    if (consent.consent_given) return 'Active';
    return 'Denied';
  };

  const styles = {
    container: {
      padding: '24px',
      backgroundColor: '#fff',
      borderRadius: '8px',
      boxShadow: '0 1px 3px rgba(0, 0, 0, 0.1)'
    },
    header: {
      marginBottom: '24px'
    },
    title: {
      fontSize: '20px',
      fontWeight: '600',
      color: '#1f2937',
      margin: '0 0 8px 0'
    },
    subtitle: {
      fontSize: '14px',
      color: '#6b7280',
      lineHeight: '1.5'
    },
    loading: {
      textAlign: 'center',
      padding: '40px',
      color: '#6b7280'
    },
    error: {
      padding: '16px',
      backgroundColor: '#fee',
      border: '1px solid #fcc',
      borderRadius: '6px',
      color: '#c33',
      marginBottom: '16px'
    },
    consentItem: {
      padding: '16px',
      border: '1px solid #e5e7eb',
      borderRadius: '8px',
      marginBottom: '16px'
    },
    consentHeader: {
      display: 'flex',
      justifyContent: 'space-between',
      alignItems: 'flex-start',
      marginBottom: '12px'
    },
    documentTitle: {
      fontSize: '16px',
      fontWeight: '500',
      color: '#1f2937',
      margin: '0 0 4px 0'
    },
    documentVersion: {
      fontSize: '12px',
      color: '#6b7280'
    },
    status: {
      padding: '4px 8px',
      borderRadius: '4px',
      fontSize: '12px',
      fontWeight: '500',
      color: 'white'
    },
    details: {
      fontSize: '14px',
      color: '#6b7280',
      marginBottom: '12px'
    },
    actions: {
      display: 'flex',
      gap: '12px',
      flexWrap: 'wrap'
    },
    button: {
      padding: '6px 12px',
      borderRadius: '4px',
      fontSize: '12px',
      fontWeight: '500',
      border: 'none',
      cursor: 'pointer',
      textDecoration: 'none',
      display: 'inline-block'
    },
    primaryButton: {
      backgroundColor: '#4f46e5',
      color: 'white'
    },
    secondaryButton: {
      backgroundColor: '#6b7280',
      color: 'white'
    },
    dangerButton: {
      backgroundColor: '#dc2626',
      color: 'white'
    },
    emptyState: {
      textAlign: 'center',
      padding: '40px',
      color: '#6b7280'
    }
  };

  if (loading) {
    return (
      <div style={styles.container}>
        <div style={styles.loading}>Loading legal documents...</div>
      </div>
    );
  }

  return (
    <div style={styles.container}>
      <div style={styles.header}>
        <h3 style={styles.title}>Legal Documents & Consent</h3>
        <p style={styles.subtitle}>
          View and manage your consent for legal documents. You can download copies of documents 
          you've accepted and withdraw consent where applicable.
        </p>
      </div>

      {error && <div style={styles.error}>{error}</div>}

      {consentHistory.length === 0 ? (
        <div style={styles.emptyState}>
          <p>No legal document consents recorded.</p>
        </div>
      ) : (
        <div>
          {consentHistory.map((consent) => (
            <div key={consent.id} style={styles.consentItem}>
              <div style={styles.consentHeader}>
                <div>
                  <h4 style={styles.documentTitle}>{consent.document_title}</h4>
                  <p style={styles.documentVersion}>Version {consent.document_version}</p>
                </div>
                <span 
                  style={{
                    ...styles.status,
                    backgroundColor: getStatusColor(consent)
                  }}
                >
                  {getStatusText(consent)}
                </span>
              </div>

              <div style={styles.details}>
                <p>
                  <strong>Consent Given:</strong> {new Date(consent.consent_timestamp).toLocaleString()}
                </p>
                <p>
                  <strong>Method:</strong> {consent.consent_method}
                </p>
                {consent.withdrawn_at && (
                  <p>
                    <strong>Withdrawn:</strong> {new Date(consent.withdrawn_at).toLocaleString()}
                  </p>
                )}
              </div>

              <div style={styles.actions}>
                <button
                  onClick={() => openDocument(consent.document_type)}
                  style={{ ...styles.button, ...styles.primaryButton }}
                >
                  View Document
                </button>

                <button
                  onClick={() => downloadDocument(consent.document_type, consent.document_version)}
                  style={{ ...styles.button, ...styles.secondaryButton }}
                >
                  Download Copy
                </button>

                {consent.consent_given && !consent.withdrawn_at && (
                  <button
                    onClick={() => handleWithdrawConsent(consent.document_type)}
                    style={{ ...styles.button, ...styles.dangerButton }}
                  >
                    Withdraw Consent
                  </button>
                )}
              </div>
            </div>
          ))}
        </div>
      )}

      <div style={{ marginTop: '24px', padding: '16px', backgroundColor: '#f9fafb', borderRadius: '6px' }}>
        <h4 style={{ margin: '0 0 8px 0', fontSize: '14px', fontWeight: '500' }}>Your Privacy Rights</h4>
        <p style={{ margin: 0, fontSize: '12px', color: '#6b7280', lineHeight: '1.4' }}>
          Under GDPR, you have the right to access, rectify, erase, restrict processing, and port your personal data. 
          You can also object to processing and withdraw consent at any time. Contact us at privacy@montecarloanalytics.com 
          for questions about your data or to exercise these rights.
        </p>
      </div>
    </div>
  );
};

export default LegalDocumentsSection;




