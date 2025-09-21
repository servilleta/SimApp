import React, { useState, useEffect } from 'react';
import axios from 'axios';

const TrialStatus = ({ onTrialExpired }) => {
  const [trialStatus, setTrialStatus] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:9090/api';

  const getToken = () => {
    return localStorage.getItem('authToken') || '';
  };

  const fetchTrialStatus = async () => {
    try {
      setLoading(true);
      const response = await axios.get(`${API_BASE_URL}/trial/status`, {
        headers: { Authorization: `Bearer ${getToken()}` },
      });

      if (response.data.success) {
        setTrialStatus(response.data.trial_status);
        
        // Notify parent if trial expired
        if (response.data.trial_status.trial_expired && onTrialExpired) {
          onTrialExpired();
        }
      }
    } catch (err) {
      console.error('Error fetching trial status:', err);
      setError(err.response?.data?.detail || 'Failed to fetch trial status');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchTrialStatus();
    
    // Refresh trial status every 5 minutes
    const interval = setInterval(fetchTrialStatus, 5 * 60 * 1000);
    
    return () => clearInterval(interval);
  }, []);

  if (loading) {
    return (
      <div style={styles.container}>
        <div style={styles.loadingText}>Loading trial status...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div style={styles.errorContainer}>
        <span style={styles.errorText}>⚠️ {error}</span>
      </div>
    );
  }

  if (!trialStatus || !trialStatus.has_subscription) {
    return null;
  }

  // Don't show anything for paid plans
  if (!trialStatus.is_trial && trialStatus.current_tier !== 'free') {
    return null;
  }

  // Trial expired
  if (trialStatus.trial_expired) {
    return (
      <div style={styles.expiredContainer}>
        <div style={styles.expiredHeader}>
          <span style={styles.expiredIcon}>⏰</span>
          <span style={styles.expiredTitle}>Trial Expired</span>
        </div>
        <p style={styles.expiredText}>
          Your 7-day trial has ended. You're now on the free plan with limited features.
        </p>
        <button style={styles.upgradeButton} onClick={() => window.location.href = '/account'}>
          Upgrade Now
        </button>
      </div>
    );
  }

  // Active trial
  if (trialStatus.is_trial && trialStatus.trial_active) {
    const daysRemaining = trialStatus.days_remaining;
    const isUrgent = daysRemaining <= 2;

    return (
      <div style={{
        ...styles.activeContainer,
        ...(isUrgent ? styles.urgentContainer : {})
      }}>
        <div style={styles.activeHeader}>
          <span style={styles.activeIcon}>🎉</span>
          <span style={styles.activeTitle}>
            Trial Active: {daysRemaining} day{daysRemaining !== 1 ? 's' : ''} remaining
          </span>
        </div>
        <p style={styles.activeText}>
          You have access to all professional features! 
          {isUrgent && ' Upgrade now to keep unlimited access.'}
        </p>
        <div style={styles.featureList}>
          <div style={styles.feature}>✓ 100 simulations per month</div>
          <div style={styles.feature}>✓ Up to 1M iterations</div>
          <div style={styles.feature}>✓ 10 concurrent simulations</div>
          <div style={styles.feature}>✓ 10MB file uploads</div>
          <div style={styles.feature}>✓ GPU acceleration (10x faster)</div>
          <div style={styles.feature}>✓ All simulation engines</div>
        </div>
        <button style={styles.upgradeButton} onClick={() => window.location.href = '/account'}>
          {isUrgent ? 'Upgrade Now' : 'View Plans'}
        </button>
      </div>
    );
  }

  // Free plan (no trial)
  if (trialStatus.current_tier === 'free' && !trialStatus.is_trial) {
    return (
      <div style={styles.freeContainer}>
        <div style={styles.freeHeader}>
          <span style={styles.freeIcon}>🆓</span>
          <span style={styles.freeTitle}>Free Plan</span>
        </div>
        <p style={styles.freeText}>
          Limited features available. Upgrade for unlimited access.
        </p>
        <button style={styles.upgradeButton} onClick={() => window.location.href = '/account'}>
          View Plans
        </button>
      </div>
    );
  }

  return null;
};

const styles = {
  container: {
    padding: '1rem',
    margin: '1rem 0',
    borderRadius: '8px',
    backgroundColor: 'var(--color-warm-white)',
    border: '1px solid var(--color-border-light)',
  },
  
  loadingText: {
    color: 'var(--color-medium-grey)',
    textAlign: 'center',
    fontSize: '0.875rem',
  },
  
  errorContainer: {
    padding: '1rem',
    margin: '1rem 0',
    borderRadius: '8px',
    backgroundColor: 'var(--color-error-bg)',
    border: '1px solid var(--color-error-border)',
  },
  
  errorText: {
    color: 'var(--color-error)',
    fontSize: '0.875rem',
  },
  
  expiredContainer: {
    padding: '1.5rem',
    margin: '1rem 0',
    borderRadius: '8px',
    backgroundColor: '#fff3cd',
    border: '1px solid #ffeaa7',
  },
  
  expiredHeader: {
    display: 'flex',
    alignItems: 'center',
    gap: '0.5rem',
    marginBottom: '0.5rem',
  },
  
  expiredIcon: {
    fontSize: '1.2rem',
  },
  
  expiredTitle: {
    fontWeight: '600',
    color: '#856404',
    fontSize: '1rem',
  },
  
  expiredText: {
    color: '#856404',
    fontSize: '0.875rem',
    margin: '0.5rem 0 1rem 0',
    lineHeight: '1.4',
  },
  
  activeContainer: {
    padding: '1.5rem',
    margin: '1rem 0',
    borderRadius: '8px',
    backgroundColor: '#d1ecf1',
    border: '1px solid var(--color-braun-orange)',
  },
  
  urgentContainer: {
    backgroundColor: '#f8d7da',
    border: '1px solid #f5c6cb',
  },
  
  activeHeader: {
    display: 'flex',
    alignItems: 'center',
    gap: '0.5rem',
    marginBottom: '0.5rem',
  },
  
  activeIcon: {
    fontSize: '1.2rem',
  },
  
  activeTitle: {
    fontWeight: '600',
    color: 'var(--color-charcoal)',
    fontSize: '1rem',
  },
  
  activeText: {
    color: 'var(--color-charcoal)',
    fontSize: '0.875rem',
    margin: '0.5rem 0 1rem 0',
    lineHeight: '1.4',
  },
  
  featureList: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
    gap: '0.5rem',
    margin: '1rem 0',
  },
  
  feature: {
    fontSize: '0.8rem',
    color: 'var(--color-charcoal)',
    display: 'flex',
    alignItems: 'center',
  },
  
  freeContainer: {
    padding: '1.5rem',
    margin: '1rem 0',
    borderRadius: '8px',
    backgroundColor: 'var(--color-warm-white)',
    border: '1px solid var(--color-border-light)',
  },
  
  freeHeader: {
    display: 'flex',
    alignItems: 'center',
    gap: '0.5rem',
    marginBottom: '0.5rem',
  },
  
  freeIcon: {
    fontSize: '1.2rem',
  },
  
  freeTitle: {
    fontWeight: '600',
    color: 'var(--color-charcoal)',
    fontSize: '1rem',
  },
  
  freeText: {
    color: 'var(--color-medium-grey)',
    fontSize: '0.875rem',
    margin: '0.5rem 0 1rem 0',
    lineHeight: '1.4',
  },
  
  upgradeButton: {
    background: 'var(--color-braun-orange)',
    color: 'white',
    border: 'none',
    padding: '0.75rem 1.5rem',
    borderRadius: '6px',
    fontSize: '0.875rem',
    fontWeight: '500',
    cursor: 'pointer',
    transition: 'all var(--transition-base)',
  },
};

export default TrialStatus;
