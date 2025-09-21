import { useState, useEffect } from 'react';
import { useAuth0 } from '@auth0/auth0-react';

/**
 * Custom hook for managing legal consent requirements
 * Checks if user needs to accept legal documents and provides consent management
 */
export const useLegalConsent = () => {
  const [requiredConsents, setRequiredConsents] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [needsConsent, setNeedsConsent] = useState(false);
  const { getAccessTokenSilently, isAuthenticated } = useAuth0();

  const apiBaseUrl = import.meta.env.VITE_API_URL || 'http://localhost:9090/api';

  /**
   * Check if the current user has any outstanding legal consent requirements
   */
  const checkConsentRequirements = async () => {
    if (!isAuthenticated) {
      setRequiredConsents([]);
      setNeedsConsent(false);
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const token = await getAccessTokenSilently();
      
      const response = await fetch(`${apiBaseUrl}/legal/required-consents`, {
        headers: {
          'Authorization': `Bearer ${token}`
        }
      });

      if (response.ok) {
        const data = await response.json();
        setRequiredConsents(data.required_consents || []);
        setNeedsConsent(data.required_consents && data.required_consents.length > 0);
      } else {
        throw new Error('Failed to check consent requirements');
      }

    } catch (err) {
      console.error('Error checking consent requirements:', err);
      setError('Failed to check legal consent requirements');
      setRequiredConsents([]);
      setNeedsConsent(false);
    } finally {
      setLoading(false);
    }
  };

  /**
   * Record user consent for legal documents
   */
  const recordConsent = async (documentConsents, context = 'user_action') => {
    setLoading(true);
    setError(null);

    try {
      const token = await getAccessTokenSilently();
      
      const response = await fetch(`${apiBaseUrl}/legal/record-consent`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify({
          document_consents: documentConsents,
          consent_context: {
            context_type: context,
            timestamp: new Date().toISOString(),
            user_agent: navigator.userAgent
          }
        })
      });

      if (response.ok) {
        const result = await response.json();
        
        // Refresh consent requirements after recording
        await checkConsentRequirements();
        
        return result;
      } else {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to record consent');
      }

    } catch (err) {
      console.error('Error recording consent:', err);
      setError('Failed to record legal consent');
      throw err;
    } finally {
      setLoading(false);
    }
  };

  /**
   * Get user's consent history
   */
  const getConsentHistory = async () => {
    if (!isAuthenticated) return [];

    try {
      const token = await getAccessTokenSilently();
      
      const response = await fetch(`${apiBaseUrl}/legal/consent-history`, {
        headers: {
          'Authorization': `Bearer ${token}`
        }
      });

      if (response.ok) {
        const data = await response.json();
        return data.consent_history || [];
      } else {
        throw new Error('Failed to fetch consent history');
      }

    } catch (err) {
      console.error('Error fetching consent history:', err);
      return [];
    }
  };

  /**
   * Withdraw consent for a specific document type
   */
  const withdrawConsent = async (documentType, reason = null) => {
    setLoading(true);
    setError(null);

    try {
      const token = await getAccessTokenSilently();
      
      const response = await fetch(`${apiBaseUrl}/legal/withdraw-consent`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify({
          document_type: documentType,
          withdrawal_reason: reason
        })
      });

      if (response.ok) {
        const result = await response.json();
        
        // Refresh consent requirements after withdrawal
        await checkConsentRequirements();
        
        return result;
      } else {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to withdraw consent');
      }

    } catch (err) {
      console.error('Error withdrawing consent:', err);
      setError('Failed to withdraw consent');
      throw err;
    } finally {
      setLoading(false);
    }
  };

  // Check consent requirements when authentication state changes
  useEffect(() => {
    if (isAuthenticated) {
      checkConsentRequirements();
    }
  }, [isAuthenticated]);

  return {
    requiredConsents,
    needsConsent,
    loading,
    error,
    checkConsentRequirements,
    recordConsent,
    getConsentHistory,
    withdrawConsent
  };
};




