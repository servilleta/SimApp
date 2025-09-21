import React, { useState } from 'react';

const PricingTable = ({ onPlanSelect, selectedPlan = 'professional' }) => {
  const [hoveredPlan, setHoveredPlan] = useState(null);

  const plans = [
    {
      tier: 'starter',
      name: 'Starter',
      price: 19,
      description: 'Perfect for individuals and small teams',
      popular: false
    },
    {
      tier: 'professional',
      name: 'Professional',
      price: 49,
      description: 'Best for growing businesses',
      popular: true
    },
    {
      tier: 'enterprise',
      name: 'Enterprise',
      price: 149,
      description: 'For large organizations',
      popular: false
    },
    {
      tier: 'on_demand',
      name: 'On Demand',
      price: 0,
      priceLabel: 'Pay per use',
      description: 'Pay only for what you use',
      popular: false,
      isPAYG: true
    }
  ];

  const features = [
    { label: 'Max Iterations', starter: '50K', professional: '500K', enterprise: '2M', on_demand: '€1/1000' },
    { label: 'Concurrent Sims', starter: '3', professional: '10', enterprise: '25', on_demand: '10' },
    { label: 'File Size Limit', starter: '25MB', professional: '100MB', enterprise: '500MB', on_demand: '100MB' },
    { label: 'GPU Priority', starter: 'Standard', professional: 'High', enterprise: 'Premium', on_demand: 'High' },
    { label: 'Support Response', starter: '48 hours', professional: '24 hours', enterprise: '4 hours', on_demand: '24 hours' },
    { label: 'API Access', starter: '❌', professional: '✅', enterprise: '✅', on_demand: '✅' },
    { label: 'Overage Rate', starter: '€1/1000', professional: '€1/1000', enterprise: '€1/1000', on_demand: 'N/A' }
  ];

  const handlePlanClick = (plan) => {
    if (onPlanSelect) {
      onPlanSelect(plan);
    }
  };

  const styles = {
    container: {
      width: '100%',
      margin: '0 auto',
      padding: '0 20px',
      overflowX: 'auto', // Allow horizontal scrolling if needed
      overflowY: 'visible'
    },
    tableContainer: {
      background: 'var(--color-white)',
      borderRadius: '16px',
      boxShadow: 'var(--shadow-sm)',
      overflow: 'visible', // Changed from 'hidden' to show full table
      border: '1px solid var(--color-border-light)',
      minWidth: '1000px' // Ensure minimum width to prevent cutting (increased for 4 plans)
    },
    table: {
      width: '100%',
      borderCollapse: 'collapse',
      fontSize: '14px',
      tableLayout: 'fixed', // Fixed layout for consistent column widths
      minWidth: '1000px' // Ensure table doesn't get compressed (increased for 4 plans)
    },
    headerRow: {
      background: 'var(--color-warm-white)'
    },
    headerCell: {
      padding: '32px 16px 24px 16px', // Increased top padding to accommodate badge
      textAlign: 'center',
      borderRight: '1px solid var(--color-border-light)',
      width: '20%', // Equal width for 5 columns (features + 4 plans)
      minWidth: '160px',
      position: 'relative' // Required for absolute positioned badge
    },
    planHeader: {
      position: 'relative'
    },
    popularBadge: {
      position: 'absolute',
      top: '8px', // Changed from -8px to 8px to show inside the cell
      left: '50%',
      transform: 'translateX(-50%)',
      background: 'var(--color-braun-orange)',
      color: 'var(--color-white)',
      padding: '4px 12px',
      borderRadius: '12px',
      fontSize: '10px',
      fontWeight: '600',
      textTransform: 'uppercase',
      letterSpacing: '0.5px',
      zIndex: 1,
      whiteSpace: 'nowrap' // Prevent text wrapping
    },
    planName: {
      fontSize: '18px',
      fontWeight: '700',
      color: 'var(--color-charcoal)',
      marginBottom: '4px',
      marginTop: '20px' // Add top margin to push content below the badge
    },
    planPrice: {
      fontSize: '24px',
      fontWeight: '700',
      color: 'var(--color-braun-orange)',
      marginBottom: '4px'
    },
    planDescription: {
      fontSize: '12px',
      color: 'var(--color-text-secondary)',
      marginBottom: '12px'
    },
    selectButton: {
      width: '100%',
      padding: '8px 16px',
      borderRadius: '6px',
      fontSize: '12px',
      fontWeight: '600',
      cursor: 'pointer',
      transition: 'all var(--transition-base)',
      border: 'none'
    },
    primaryButton: {
      background: 'var(--color-braun-orange)',
      color: 'var(--color-white)'
    },
    secondaryButton: {
      background: 'var(--color-white)',
      color: 'var(--color-dark-grey)',
      border: '1px solid var(--color-border-light)'
    },
    selectedButton: {
      background: 'var(--color-success)',
      color: 'var(--color-white)'
    },
    featureRow: {
      borderBottom: '1px solid var(--color-border-light)'
    },
    featureRowEven: {
      background: 'var(--color-warm-white)'
    },
    featureLabel: {
      padding: '16px',
      fontWeight: '600',
      color: 'var(--color-charcoal)',
      borderRight: '1px solid var(--color-border-light)',
      background: 'var(--color-white)',
      textAlign: 'left',
      width: '20%', // Equal width for feature column
      minWidth: '160px'
    },
    featureCell: {
      padding: '16px',
      textAlign: 'center',
      color: 'var(--color-text-secondary)',
      borderRight: '1px solid var(--color-border-light)',
      fontWeight: '500',
      width: '20%', // Equal width for each plan column
      minWidth: '160px'
    },
    highlightedColumn: {
      background: 'rgba(255, 107, 53, 0.05)',
      borderLeft: '2px solid var(--color-braun-orange)',
      borderRight: '2px solid var(--color-braun-orange)'
    },
    mobileContainer: {
      display: 'none'
    },
    mobileCard: {
      background: 'var(--color-white)',
      border: '1px solid var(--color-border-light)',
      borderRadius: '12px',
      padding: '20px',
      marginBottom: '16px',
      boxShadow: 'var(--shadow-sm)'
    },
    mobileCardHeader: {
      textAlign: 'center',
      marginBottom: '16px',
      paddingBottom: '16px',
      paddingTop: '20px', // Add top padding for badge space
      borderBottom: '1px solid var(--color-border-light)',
      position: 'relative'
    },
    mobilePlanName: {
      fontSize: '18px',
      fontWeight: '700',
      color: 'var(--color-charcoal)',
      marginBottom: '4px'
    },
    mobilePlanPrice: {
      fontSize: '24px',
      fontWeight: '700',
      color: 'var(--color-braun-orange)',
      marginBottom: '8px'
    },
    mobileFeatureList: {
      listStyle: 'none',
      padding: 0,
      margin: '0 0 16px 0'
    },
    mobileFeatureItem: {
      display: 'flex',
      justifyContent: 'space-between',
      padding: '8px 0',
      borderBottom: '1px solid var(--color-border-light)',
      fontSize: '14px'
    },
    mobileFeatureLabel: {
      color: 'var(--color-text-secondary)',
      fontWeight: '500'
    },
    mobileFeatureValue: {
      color: 'var(--color-charcoal)',
      fontWeight: '600'
    }
  };

  const getButtonStyle = (plan) => {
    if (selectedPlan === plan.tier) {
      return { ...styles.selectButton, ...styles.selectedButton };
    }
    if (plan.popular) {
      return { ...styles.selectButton, ...styles.primaryButton };
    }
    return { ...styles.selectButton, ...styles.secondaryButton };
  };

  const getButtonText = (plan) => {
    if (selectedPlan === plan.tier) {
      return '✓ Selected';
    }
    return `Start ${plan.name}`;
  };

  const isColumnHighlighted = (planTier) => {
    return selectedPlan === planTier || hoveredPlan === planTier;
  };

  return (
    <div style={styles.container}>
      <div className="pricing-table-container" style={styles.tableContainer}>
        <table style={styles.table}>
          {/* Header Row */}
          <thead>
            <tr style={styles.headerRow}>
              <th style={styles.featureLabel}>Features</th>
              {plans.map((plan) => (
                <th
                  key={plan.tier}
                  style={{
                    ...styles.headerCell,
                    ...styles.planHeader,
                    ...(isColumnHighlighted(plan.tier) ? styles.highlightedColumn : {})
                  }}
                  onMouseEnter={() => setHoveredPlan(plan.tier)}
                  onMouseLeave={() => setHoveredPlan(null)}
                >
                  {plan.popular && <div style={styles.popularBadge}>Most Popular</div>}
                  <div style={styles.planName}>{plan.name}</div>
                  <div style={styles.planPrice}>
                    {plan.isPAYG ? (
                      plan.priceLabel || 'Pay per use'
                    ) : (
                      <>
                        {plan.price === 0 ? 'Free' : `$${plan.price}`}
                        {plan.price > 0 && <span style={{ fontSize: '12px', fontWeight: '400' }}>/mo</span>}
                      </>
                    )}
                  </div>
                  <div style={styles.planDescription}>{plan.description}</div>
                  <button
                    style={getButtonStyle(plan)}
                    onClick={() => handlePlanClick(plan)}
                    onMouseEnter={(e) => {
                      if (selectedPlan !== plan.tier) {
                        e.target.style.transform = 'translateY(-1px)';
                        e.target.style.boxShadow = 'var(--shadow-sm)';
                      }
                    }}
                    onMouseLeave={(e) => {
                      if (selectedPlan !== plan.tier) {
                        e.target.style.transform = 'translateY(0)';
                        e.target.style.boxShadow = 'none';
                      }
                    }}
                  >
                    {getButtonText(plan)}
                  </button>
                </th>
              ))}
            </tr>
          </thead>
          
          {/* Feature Rows */}
          <tbody>
            {features.map((feature, index) => (
              <tr
                key={feature.label}
                style={{
                  ...styles.featureRow,
                  ...(index % 2 === 1 ? styles.featureRowEven : {})
                }}
              >
                <td style={styles.featureLabel}>{feature.label}</td>
                {plans.map((plan) => (
                  <td
                    key={`${feature.label}-${plan.tier}`}
                    style={{
                      ...styles.featureCell,
                      ...(index % 2 === 1 ? styles.featureRowEven : {}),
                      ...(isColumnHighlighted(plan.tier) ? styles.highlightedColumn : {})
                    }}
                    onMouseEnter={() => setHoveredPlan(plan.tier)}
                    onMouseLeave={() => setHoveredPlan(null)}
                  >
                    {feature[plan.tier]}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Mobile fallback - show card layout */}
      <div style={styles.mobileContainer}>
        <style>{`
          @media (max-width: 1200px) {
            .pricing-table-container {
              overflow-x: auto;
              -webkit-overflow-scrolling: touch;
            }
          }
          
          @media (max-width: 768px) {
            .pricing-table-container {
              display: none !important;
            }
            .pricing-mobile-container {
              display: block !important;
            }
          }
        `}</style>
        <div className="pricing-mobile-container" style={{ display: 'none' }}>
          {plans.map((plan) => (
            <div key={plan.tier} style={styles.mobileCard}>
              <div style={styles.mobileCardHeader}>
                {plan.popular && (
                  <div style={{
                    ...styles.popularBadge,
                    top: '8px', // Position within the card header
                    position: 'absolute'
                  }}>
                    Most Popular
                  </div>
                )}
                <div style={styles.mobilePlanName}>{plan.name}</div>
                <div style={styles.mobilePlanPrice}>
                  {plan.isPAYG ? (
                    plan.priceLabel || 'Pay per use'
                  ) : (
                    <>
                      {plan.price === 0 ? 'Free' : `$${plan.price}`}
                      {plan.price > 0 && <span style={{ fontSize: '14px', fontWeight: '400' }}>/mo</span>}
                    </>
                  )}
                </div>
                <p style={{ 
                  fontSize: '14px', 
                  color: 'var(--color-text-secondary)',
                  margin: 0 
                }}>
                  {plan.description}
                </p>
              </div>
              
              <ul style={styles.mobileFeatureList}>
                {features.map((feature) => (
                  <li key={feature.label} style={styles.mobileFeatureItem}>
                    <span style={styles.mobileFeatureLabel}>{feature.label}</span>
                    <span style={styles.mobileFeatureValue}>{feature[plan.tier]}</span>
                  </li>
                ))}
              </ul>
              
              <button
                style={getButtonStyle(plan)}
                onClick={() => handlePlanClick(plan)}
                onMouseEnter={(e) => {
                  if (selectedPlan !== plan.tier) {
                    e.target.style.transform = 'translateY(-1px)';
                    e.target.style.boxShadow = 'var(--shadow-sm)';
                  }
                }}
                onMouseLeave={(e) => {
                  if (selectedPlan !== plan.tier) {
                    e.target.style.transform = 'translateY(0)';
                    e.target.style.boxShadow = 'none';
                  }
                }}
              >
                {getButtonText(plan)}
              </button>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default PricingTable;
