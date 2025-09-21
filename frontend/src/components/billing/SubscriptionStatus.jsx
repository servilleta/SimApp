import React, { useState, useEffect } from 'react';
import { useAuth0 } from '@auth0/auth0-react';
import BillingService from '../../services/billingService';

const SubscriptionStatus = ({ showUpgradeButton = true, compact = false }) => {
  const { isAuthenticated } = useAuth0();
  const [subscription, setSubscription] = useState(null);
  const [usage, setUsage] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (isAuthenticated) {
      loadSubscriptionData();
    }
  }, [isAuthenticated]);

  const loadSubscriptionData = async () => {
    try {
      setLoading(true);
      const [subData, usageData] = await Promise.all([
        BillingService.getCurrentSubscription(),
        BillingService.getUsageInfo()
      ]);
      setSubscription(subData);
      setUsage(usageData);
      setError(null);
    } catch (err) {
      console.error('Failed to load subscription data:', err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleUpgrade = () => {
    // Navigate to pricing page or open plan selector
    window.location.href = '/pricing';
  };

  const handleManageBilling = async () => {
    try {
      await BillingService.openBillingPortal();
    } catch (err) {
      console.error('Failed to open billing portal:', err);
      alert('Failed to open billing portal. Please try again.');
    }
  };

  const styles = {
    container: {
      background: 'var(--color-white)',
      border: '1px solid var(--color-border-light)',
      borderRadius: compact ? '8px' : '12px',
      padding: compact ? '16px' : '24px',
      boxShadow: 'var(--shadow-sm)'
    },
    header: {
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'space-between',
      marginBottom: compact ? '12px' : '16px'
    },
    title: {
      fontSize: compact ? '16px' : '18px',
      fontWeight: '600',
      color: 'var(--color-charcoal)',
      margin: 0
    },
    planBadge: {
      padding: '4px 12px',
      borderRadius: '16px',
      fontSize: '12px',
      fontWeight: '600',
      textTransform: 'uppercase',
      letterSpacing: '0.5px'
    },
    freeBadge: {
      background: 'var(--color-light-grey)',
      color: 'var(--color-dark-grey)'
    },
    paidBadge: {
      background: 'var(--color-braun-orange)',
      color: 'var(--color-white)'
    },
    usageGrid: {
      display: 'grid',
      gridTemplateColumns: compact ? '1fr' : 'repeat(auto-fit, minmax(200px, 1fr))',
      gap: compact ? '8px' : '16px',
      marginBottom: compact ? '12px' : '16px'
    },
    usageItem: {
      display: 'flex',
      justifyContent: 'space-between',
      alignItems: 'center',
      padding: compact ? '8px' : '12px',
      background: 'var(--color-warm-white)',
      borderRadius: '6px',
      fontSize: compact ? '13px' : '14px'
    },
    usageLabel: {
      color: 'var(--color-text-secondary)',
      fontWeight: '500'
    },
    usageValue: {
      color: 'var(--color-charcoal)',
      fontWeight: '600'
    },
    progressBar: {
      width: '100%',
      height: '4px',
      background: 'var(--color-light-grey)',
      borderRadius: '2px',
      overflow: 'hidden',
      marginTop: '4px'
    },
    progressFill: {
      height: '100%',
      background: 'var(--color-braun-orange)',
      transition: 'width 0.3s ease'
    },
    warningProgress: {
      background: 'var(--color-warning)'
    },
    dangerProgress: {
      background: 'var(--color-error)'
    },
    buttonContainer: {
      display: 'flex',
      gap: '12px',
      marginTop: compact ? '12px' : '16px'
    },
    upgradeButton: {
      background: 'var(--color-braun-orange)',
      color: 'var(--color-white)',
      border: 'none',
      padding: compact ? '8px 16px' : '10px 20px',
      borderRadius: '6px',
      fontSize: compact ? '13px' : '14px',
      fontWeight: '600',
      cursor: 'pointer',
      transition: 'all var(--transition-base)'
    },
    manageButton: {
      background: 'var(--color-white)',
      color: 'var(--color-dark-grey)',
      border: '1px solid var(--color-border-light)',
      padding: compact ? '8px 16px' : '10px 20px',
      borderRadius: '6px',
      fontSize: compact ? '13px' : '14px',
      fontWeight: '500',
      cursor: 'pointer',
      transition: 'all var(--transition-base)'
    },
    loading: {
      textAlign: 'center',
      padding: '20px',
      color: 'var(--color-medium-grey)'
    },
    error: {
      textAlign: 'center',
      padding: '20px',
      color: 'var(--color-error)',
      background: 'var(--color-error-bg)',
      borderRadius: '6px',
      fontSize: '14px'
    }
  };

  const formatUsagePercentage = (current, limit) => {
    if (limit === -1) return 0; // Unlimited
    if (limit === 0) return 100; // No access
    return Math.min((current / limit) * 100, 100);
  };

  const getProgressBarStyle = (percentage) => {
    let fillStyle = { ...styles.progressFill, width: `${percentage}%` };
    
    if (percentage >= 90) {
      fillStyle = { ...fillStyle, ...styles.dangerProgress };
    } else if (percentage >= 80) {
      fillStyle = { ...fillStyle, ...styles.warningProgress };
    }
    
    return fillStyle;
  };

  if (!isAuthenticated) {
    return null;
  }

  if (loading) {
    return (
      <div style={styles.container}>
        <div style={styles.loading}>Loading subscription details...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div style={styles.container}>
        <div style={styles.error}>
          Failed to load subscription data: {error}
        </div>
      </div>
    );
  }

  const isFreePlan = !subscription || subscription?.tier === 'free' || subscription?.tier === 'trial';
  const limits = BillingService.formatLimits(usage?.limits || {});
  
  // Map tier names to display names
  const tierDisplayNames = {
    'free': 'Free',
    'trial': '7-Day Trial',
    'starter': 'Starter',
    'professional': 'Professional',
    'pro': 'Professional',
    'enterprise': 'Enterprise'
  };

  const displayTier = tierDisplayNames[subscription?.tier?.toLowerCase()] || subscription?.tier || 'Free';

  return (
    <div style={styles.container}>
      <div style={styles.header}>
        <h3 style={styles.title}>
          {compact ? 'Plan' : 'Current Plan'}
        </h3>
        <div 
          style={{
            ...styles.planBadge,
            ...(isFreePlan ? styles.freeBadge : styles.paidBadge)
          }}
        >
          {displayTier}
        </div>
      </div>

      {!compact && usage && (
        <div style={styles.usageGrid}>
          <div style={styles.usageItem}>
            <span style={styles.usageLabel}>Monthly Iterations</span>
            <div>
              <div style={styles.usageValue}>
                {usage.current_usage.total_iterations.toLocaleString()} / {limits.maxIterations}
              </div>
              <div style={styles.progressBar}>
                <div style={getProgressBarStyle(
                  formatUsagePercentage(usage.current_usage.total_iterations, usage.limits.max_iterations)
                )}></div>
              </div>
            </div>
          </div>

          <div style={styles.usageItem}>
            <span style={styles.usageLabel}>Concurrent Sims</span>
            <div style={styles.usageValue}>
              {usage.current_usage.concurrent_simulations || 0} / {limits.concurrentSimulations}
            </div>
          </div>

          <div style={styles.usageItem}>
            <span style={styles.usageLabel}>API Calls</span>
            <div>
              <div style={styles.usageValue}>
                {usage.current_usage.api_calls.toLocaleString()} / {limits.apiCallsPerMonth}
              </div>
              {usage.limits.api_calls_per_month > 0 && (
                <div style={styles.progressBar}>
                  <div style={getProgressBarStyle(
                    formatUsagePercentage(usage.current_usage.api_calls, usage.limits.api_calls_per_month)
                  )}></div>
                </div>
              )}
            </div>
          </div>

          <div style={styles.usageItem}>
            <span style={styles.usageLabel}>File Size Limit</span>
            <div style={styles.usageValue}>{limits.fileSizeLimit}</div>
          </div>
        </div>
      )}

      {showUpgradeButton && (
        <div style={styles.buttonContainer}>
          {isFreePlan ? (
            <button
              onClick={handleUpgrade}
              style={styles.upgradeButton}
              onMouseEnter={(e) => {
                e.target.style.background = 'var(--color-braun-orange-dark)';
                e.target.style.transform = 'translateY(-1px)';
              }}
              onMouseLeave={(e) => {
                e.target.style.background = 'var(--color-braun-orange)';
                e.target.style.transform = 'translateY(0)';
              }}
            >
              {compact ? 'Upgrade' : 'Upgrade Plan'}
            </button>
          ) : (
            <button
              onClick={handleManageBilling}
              style={styles.manageButton}
              onMouseEnter={(e) => {
                e.target.style.background = 'var(--color-warm-white)';
                e.target.style.borderColor = 'var(--color-medium-grey)';
              }}
              onMouseLeave={(e) => {
                e.target.style.background = 'var(--color-white)';
                e.target.style.borderColor = 'var(--color-border-light)';
              }}
            >
              {compact ? 'Manage' : 'Manage Billing'}
            </button>
          )}
        </div>
      )}
    </div>
  );
};

export default SubscriptionStatus;
