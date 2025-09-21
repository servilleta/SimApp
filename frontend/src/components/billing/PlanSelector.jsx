import React, { useState, useEffect } from 'react';
import { useAuth0 } from '@auth0/auth0-react';

const PlanSelector = ({ onPlanSelect, selectedPlan, showFree = true }) => {
  const { isAuthenticated } = useAuth0();
  const [plans, setPlans] = useState([]);
  const [loading, setLoading] = useState(true);

  // Define plans with exact pricing matrix
  const planData = [
    {
      tier: 'free',
      name: 'Free',
      price: 0,
      description: 'Perfect for getting started with Monte Carlo simulations',
      features: [
        '5K max iterations per simulation',
        '1 concurrent simulation',
        '10MB file size limit',
        '1K formulas maximum',
        '3 projects stored',
        'Low GPU priority',
        'Community support'
      ],
      highlighted: false,
      cta: 'Start Free'
    },
    {
      tier: 'starter',
      name: 'Starter',
      price: 19,
      description: 'Perfect for small teams getting started',
      features: [
        '50K max iterations per simulation',
        '3 concurrent simulations',
        '25MB file size limit',
        '10K formulas maximum',
        '10 projects stored',
        'Standard GPU priority',
        'Email support',
        'Overage: €1/1000 iterations'
      ],
      highlighted: false,
      cta: 'Start Starter Plan'
    },
    {
      tier: 'professional',
      name: 'Professional',
      price: 49,
      description: 'Advanced features for professional analysts',
      features: [
        '500K max iterations per simulation',
        '10 concurrent simulations',
        '100MB file size limit',
        '50K formulas maximum',
        '50 projects stored',
        'High GPU priority',
        '1,000 API calls per month',
        'Priority email support',
        'Overage: €1/1000 iterations'
      ],
      highlighted: true,
      cta: 'Start Professional Plan'
    },
    {
      tier: 'enterprise',
      name: 'Enterprise',
      price: 149,
      description: 'Enterprise-grade features for large organizations',
      features: [
        '2M max iterations per simulation',
        '25 concurrent simulations',
        '500MB file size limit',
        '500K formulas maximum',
        'Unlimited projects stored',
        'Premium GPU priority',
        'Unlimited API calls',
        '24/7 priority support',
        'Overage: €1/1000 iterations'
      ],
      highlighted: false,
      cta: 'Start Enterprise Plan'
    },
    {
      tier: 'on_demand',
      name: 'On Demand',
      price: 0,
      priceLabel: 'Pay per use',
      description: 'Pay only for what you use - perfect for occasional simulations',
      features: [
        '€1 per 1000 iterations',
        'No monthly commitment',
        'Same features as Professional',
        '100MB file size limit',
        'High GPU priority',
        '1,000 API calls per month',
        'Priority email support'
      ],
      highlighted: false,
      cta: 'Start On Demand Plan',
      isPAYG: true
    }
  ];

  useEffect(() => {
    setPlans(showFree ? planData : planData.filter(p => p.tier !== 'free'));
    setLoading(false);
  }, [showFree]);

  const styles = {
    container: {
      maxWidth: '1200px',
      margin: '0 auto',
      padding: '0 20px'
    },
    header: {
      textAlign: 'center',
      marginBottom: '48px'
    },
    title: {
      fontSize: '36px',
      fontWeight: '700',
      color: 'var(--color-charcoal)',
      marginBottom: '16px',
      lineHeight: '1.2'
    },
    subtitle: {
      fontSize: '18px',
      color: 'var(--color-text-secondary)',
      maxWidth: '600px',
      margin: '0 auto',
      lineHeight: '1.6'
    },
    plansGrid: {
      display: 'grid',
      gridTemplateColumns: showFree 
        ? 'repeat(auto-fit, minmax(280px, 1fr))' 
        : 'repeat(auto-fit, minmax(320px, 1fr))',
      gap: '24px',
      marginTop: '40px'
    },
    planCard: {
      background: 'var(--color-white)',
      border: '1px solid var(--color-border-light)',
      borderRadius: '16px',
      padding: '32px 24px',
      position: 'relative',
      transition: 'all 0.3s ease',
      cursor: 'pointer',
      boxShadow: 'var(--shadow-sm)'
    },
    selectedPlan: {
      borderColor: 'var(--color-braun-orange)',
      boxShadow: '0 8px 25px rgba(255, 107, 53, 0.15)',
      transform: 'translateY(-4px)'
    },
    popularBadge: {
      position: 'absolute',
      top: '-12px',
      left: '50%',
      transform: 'translateX(-50%)',
      background: 'var(--color-braun-orange)',
      color: 'var(--color-white)',
      padding: '6px 16px',
      borderRadius: '12px',
      fontSize: '12px',
      fontWeight: '600',
      textTransform: 'uppercase',
      letterSpacing: '0.5px'
    },
    planName: {
      fontSize: '24px',
      fontWeight: '700',
      color: 'var(--color-charcoal)',
      marginBottom: '8px'
    },
    planPrice: {
      display: 'flex',
      alignItems: 'baseline',
      marginBottom: '16px'
    },
    priceAmount: {
      fontSize: '48px',
      fontWeight: '700',
      color: 'var(--color-charcoal)',
      lineHeight: '1'
    },
    pricePrefix: {
      fontSize: '24px',
      fontWeight: '500',
      color: 'var(--color-dark-grey)',
      marginRight: '4px'
    },
    priceSuffix: {
      fontSize: '16px',
      color: 'var(--color-medium-grey)',
      marginLeft: '8px',
      alignSelf: 'flex-end',
      marginBottom: '8px'
    },
    planDescription: {
      fontSize: '16px',
      color: 'var(--color-text-secondary)',
      marginBottom: '24px',
      lineHeight: '1.5'
    },
    featuresList: {
      marginBottom: '32px'
    },
    featureItem: {
      display: 'flex',
      alignItems: 'center',
      gap: '12px',
      marginBottom: '10px',
      fontSize: '14px',
      color: 'var(--color-charcoal)'
    },
    featureIcon: {
      width: '16px',
      height: '16px',
      borderRadius: '50%',
      background: 'var(--color-success)',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      fontSize: '10px',
      color: 'white',
      flexShrink: 0
    },
    selectButton: {
      width: '100%',
      padding: '14px 24px',
      borderRadius: '8px',
      fontSize: '16px',
      fontWeight: '600',
      textAlign: 'center',
      transition: 'all 0.3s ease',
      border: 'none',
      cursor: 'pointer',
      textDecoration: 'none',
      display: 'block'
    },
    primaryButton: {
      background: 'var(--color-braun-orange)',
      color: 'var(--color-white)'
    },
    secondaryButton: {
      background: 'var(--color-white)',
      color: 'var(--color-charcoal)',
      border: '1px solid var(--color-border-light)'
    },
    selectedButton: {
      background: 'var(--color-success)',
      color: 'var(--color-white)'
    },
    loading: {
      textAlign: 'center',
      padding: '48px',
      color: 'var(--color-medium-grey)'
    }
  };

  const handlePlanSelect = (plan) => {
    if (onPlanSelect) {
      onPlanSelect(plan);
    }
  };

  const getButtonStyle = (plan) => {
    if (selectedPlan === plan.tier) {
      return { ...styles.selectButton, ...styles.selectedButton };
    }
    if (plan.highlighted) {
      return { ...styles.selectButton, ...styles.primaryButton };
    }
    return { ...styles.selectButton, ...styles.secondaryButton };
  };

  const getButtonText = (plan) => {
    if (selectedPlan === plan.tier) {
      return '✓ Selected';
    }
    return plan.cta;
  };

  if (loading) {
    return (
      <div style={styles.loading}>
        <div>Loading plans...</div>
      </div>
    );
  }

  return (
    <div style={styles.container}>
      <div style={styles.header}>
        <h2 style={styles.title}>Choose Your Plan</h2>
        <p style={styles.subtitle}>
          Select the perfect plan for your Monte Carlo simulation needs. 
          You can upgrade or downgrade at any time.
        </p>
      </div>

      <div style={styles.plansGrid}>
        {plans.map((plan) => (
          <div
            key={plan.tier}
            style={{
              ...styles.planCard,
              ...(selectedPlan === plan.tier ? styles.selectedPlan : {}),
              ...(plan.highlighted && selectedPlan !== plan.tier ? {
                borderColor: 'rgba(255, 107, 53, 0.3)',
                transform: 'translateY(-2px)'
              } : {})
            }}
            onClick={() => handlePlanSelect(plan)}
            onMouseEnter={(e) => {
              if (selectedPlan !== plan.tier) {
                e.currentTarget.style.transform = 'translateY(-4px)';
                e.currentTarget.style.boxShadow = 'var(--shadow-md)';
              }
            }}
            onMouseLeave={(e) => {
              if (selectedPlan !== plan.tier) {
                const baseTransform = plan.highlighted ? 'translateY(-2px)' : 'translateY(0)';
                e.currentTarget.style.transform = baseTransform;
                e.currentTarget.style.boxShadow = plan.highlighted ? 'var(--shadow-sm)' : 'var(--shadow-sm)';
              }
            }}
          >
            {plan.highlighted && <div style={styles.popularBadge}>Most Popular</div>}
            
            <h3 style={styles.planName}>{plan.name}</h3>
            
            <div style={styles.planPrice}>
              {plan.isPAYG ? (
                <span style={styles.priceAmount}>
                  {plan.priceLabel || 'Pay per use'}
                </span>
              ) : (
                <>
                  {plan.price > 0 && <span style={styles.pricePrefix}>$</span>}
                  <span style={styles.priceAmount}>
                    {plan.price === 0 ? 'Free' : plan.price}
                  </span>
                  {plan.price > 0 && <span style={styles.priceSuffix}>per month</span>}
                </>
              )}
            </div>
            
            <p style={styles.planDescription}>{plan.description}</p>

            <div style={styles.featuresList}>
              {plan.features.map((feature, idx) => (
                <div key={idx} style={styles.featureItem}>
                  <div style={styles.featureIcon}>✓</div>
                  {feature}
                </div>
              ))}
            </div>

            <button
              style={getButtonStyle(plan)}
              onClick={(e) => {
                e.stopPropagation();
                handlePlanSelect(plan);
              }}
              onMouseEnter={(e) => {
                if (selectedPlan !== plan.tier) {
                  e.target.style.transform = 'translateY(-1px)';
                  e.target.style.boxShadow = 'var(--shadow-md)';
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
  );
};

export default PlanSelector;
