import React, { useState, useEffect } from 'react';
import { useAuth0 } from '@auth0/auth0-react';
import { Link, useNavigate } from 'react-router-dom';
import PlanSelector from '../components/billing/PlanSelector';

const RegisterPage = () => {
  const { loginWithRedirect, isLoading, isAuthenticated } = useAuth0();
  const navigate = useNavigate();
  const [step, setStep] = useState('welcome'); // welcome, plans, signup
  const [selectedPlan, setSelectedPlan] = useState('professional');

  useEffect(() => {
    if (isAuthenticated) {
      navigate('/dashboard');
    }
  }, [isAuthenticated, navigate]);

  const handleGetStarted = () => {
    setStep('plans');
  };

  const handlePlanSelect = (plan) => {
    setSelectedPlan(plan.tier);
  };

  const handleContinueWithPlan = () => {
    // Store selected plan in localStorage for after auth
    localStorage.setItem('selectedPlan', selectedPlan);
    
    loginWithRedirect({
      authorizationParams: {
        screen_hint: 'signup'
      }
    });
  };

  const handleSignInInstead = () => {
    loginWithRedirect({
      authorizationParams: {
        screen_hint: 'login'
      }
    });
  };

  const styles = {
    container: {
      minHeight: '100vh',
      background: 'var(--color-warm-white)',
      fontFamily: 'system-ui, -apple-system, sans-serif',
      position: 'relative'
    },
    header: {
      background: 'var(--color-white)',
      borderBottom: '1px solid var(--color-border-light)',
      padding: '16px 0',
      position: 'sticky',
      top: 0,
      zIndex: 10
    },
    headerContent: {
      maxWidth: '1200px',
      margin: '0 auto',
      padding: '0 20px',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'space-between'
    },
    logo: {
      fontSize: '24px',
      fontWeight: 'bold',
      color: 'var(--color-charcoal)',
      textDecoration: 'none'
    },
    backButton: {
      background: 'var(--color-white)',
      border: '1px solid var(--color-border-light)',
      borderRadius: '8px',
      padding: '8px 16px',
      color: 'var(--color-dark-grey)',
      textDecoration: 'none',
      fontSize: '14px',
      fontWeight: '500',
      transition: 'all var(--transition-base)',
      display: 'flex',
      alignItems: 'center',
      gap: '8px'
    },
    content: {
      padding: '60px 20px',
      maxWidth: step === 'plans' ? '1200px' : '600px',
      margin: '0 auto',
      textAlign: 'center'
    },
    welcomeCard: {
      background: 'var(--color-white)',
      border: '1px solid var(--color-border-light)',
      borderRadius: '16px',
      padding: '48px 40px',
      boxShadow: 'var(--shadow-sm)',
      textAlign: 'center'
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
      marginBottom: '32px',
      lineHeight: '1.6'
    },
    features: {
      display: 'grid',
      gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
      gap: '24px',
      margin: '40px 0',
      textAlign: 'left'
    },
    featureItem: {
      display: 'flex',
      alignItems: 'flex-start',
      gap: '12px'
    },
    featureIcon: {
      width: '24px',
      height: '24px',
      borderRadius: '50%',
      background: 'var(--color-braun-orange)',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      fontSize: '12px',
      color: 'white',
      flexShrink: 0,
      marginTop: '2px'
    },
    featureContent: {
      flex: 1
    },
    featureTitle: {
      fontSize: '16px',
      fontWeight: '600',
      color: 'var(--color-charcoal)',
      marginBottom: '4px'
    },
    featureDescription: {
      fontSize: '14px',
      color: 'var(--color-text-secondary)',
      lineHeight: '1.4'
    },
    primaryButton: {
      background: 'var(--color-braun-orange)',
      color: 'var(--color-white)',
      border: 'none',
      padding: '16px 32px',
      borderRadius: '8px',
      fontSize: '16px',
      fontWeight: '600',
      cursor: 'pointer',
      transition: 'all var(--transition-base)',
      textDecoration: 'none',
      display: 'inline-block',
      marginBottom: '16px'
    },
    secondaryButton: {
      background: 'var(--color-white)',
      color: 'var(--color-charcoal)',
      border: '1px solid var(--color-border-light)',
      padding: '12px 24px',
      borderRadius: '8px',
      fontSize: '14px',
      fontWeight: '500',
      cursor: 'pointer',
      transition: 'all var(--transition-base)',
      textDecoration: 'none',
      display: 'inline-block',
      marginTop: '8px'
    },
    planSection: {
      background: 'var(--color-white)',
      borderRadius: '16px',
      padding: '40px',
      boxShadow: 'var(--shadow-sm)',
      margin: '40px 0'
    },
    stepIndicator: {
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      gap: '32px',
      marginBottom: '40px'
    },
    stepItem: {
      display: 'flex',
      alignItems: 'center',
      gap: '8px',
      fontSize: '14px',
      color: 'var(--color-medium-grey)'
    },
    stepNumber: {
      width: '24px',
      height: '24px',
      borderRadius: '50%',
      background: 'var(--color-light-grey)',
      color: 'var(--color-medium-grey)',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      fontSize: '12px',
      fontWeight: '600'
    },
    stepNumberActive: {
      background: 'var(--color-braun-orange)',
      color: 'var(--color-white)'
    },
    stepNumberCompleted: {
      background: 'var(--color-success)',
      color: 'var(--color-white)'
    },
    continueButton: {
      background: 'var(--color-braun-orange)',
      color: 'var(--color-white)',
      border: 'none',
      padding: '16px 32px',
      borderRadius: '8px',
      fontSize: '16px',
      fontWeight: '600',
      cursor: 'pointer',
      transition: 'all var(--transition-base)',
      marginTop: '32px'
    },
    footer: {
      fontSize: '14px',
      color: 'var(--color-medium-grey)',
      marginTop: '24px'
    },
    link: {
      color: 'var(--color-braun-orange)',
      textDecoration: 'none',
      fontWeight: '500'
    },
    spinner: {
      width: '20px',
      height: '20px',
      border: '2px solid rgba(255, 255, 255, 0.3)',
      borderTop: '2px solid white',
      borderRadius: '50%',
      animation: 'spin 1s linear infinite',
      display: 'inline-block',
      marginRight: '8px'
    }
  };

  const renderWelcomeStep = () => (
    <div style={styles.welcomeCard}>
      <h1 style={styles.title}>Start Your Monte Carlo Journey</h1>
      <p style={styles.subtitle}>
        Join thousands of professionals using SimApp for data-driven decision making.
        From startups to Fortune 500 companies, we power the most demanding simulations.
      </p>

      <div style={styles.features}>
        <div style={styles.featureItem}>
          <div style={styles.featureIcon}>🚀</div>
          <div style={styles.featureContent}>
            <div style={styles.featureTitle}>GPU-Accelerated</div>
            <div style={styles.featureDescription}>
              Up to 1000x faster than traditional CPU-based simulations
            </div>
          </div>
        </div>
        <div style={styles.featureItem}>
          <div style={styles.featureIcon}>📊</div>
          <div style={styles.featureContent}>
            <div style={styles.featureTitle}>Any Excel File</div>
            <div style={styles.featureDescription}>
              Works with your existing models without modification
            </div>
          </div>
        </div>
        <div style={styles.featureItem}>
          <div style={styles.featureIcon}>⚡</div>
          <div style={styles.featureContent}>
            <div style={styles.featureTitle}>Real-time Results</div>
            <div style={styles.featureDescription}>
              Live progress tracking and instant result visualization
            </div>
          </div>
        </div>
        <div style={styles.featureItem}>
          <div style={styles.featureIcon}>🔒</div>
          <div style={styles.featureContent}>
            <div style={styles.featureTitle}>Enterprise Security</div>
            <div style={styles.featureDescription}>
              Bank-grade security with Auth0 integration
            </div>
          </div>
        </div>
      </div>

      <button
        onClick={handleGetStarted}
        style={styles.primaryButton}
        onMouseEnter={(e) => {
          e.target.style.background = 'var(--color-braun-orange-dark)';
          e.target.style.transform = 'translateY(-2px)';
          e.target.style.boxShadow = 'var(--shadow-md)';
        }}
        onMouseLeave={(e) => {
          e.target.style.background = 'var(--color-braun-orange)';
          e.target.style.transform = 'translateY(0)';
          e.target.style.boxShadow = 'none';
        }}
      >
        Get Started - It's Free
      </button>

      <button
        onClick={handleSignInInstead}
        style={styles.secondaryButton}
        onMouseEnter={(e) => {
          e.target.style.background = 'var(--color-warm-white)';
          e.target.style.borderColor = 'var(--color-medium-grey)';
        }}
        onMouseLeave={(e) => {
          e.target.style.background = 'var(--color-white)';
          e.target.style.borderColor = 'var(--color-border-light)';
        }}
      >
        Already have an account? Sign In
      </button>
    </div>
  );

  const renderPlanStep = () => (
    <div>
      <div style={styles.planSection}>
        <PlanSelector
          onPlanSelect={handlePlanSelect}
          selectedPlan={selectedPlan}
          showFree={true}
        />
        
        <button
          onClick={handleContinueWithPlan}
          disabled={isLoading}
          style={{
            ...styles.continueButton,
            opacity: isLoading ? 0.7 : 1,
            cursor: isLoading ? 'not-allowed' : 'pointer'
          }}
          onMouseEnter={(e) => {
            if (!isLoading) {
              e.target.style.background = 'var(--color-braun-orange-dark)';
              e.target.style.transform = 'translateY(-2px)';
              e.target.style.boxShadow = 'var(--shadow-md)';
            }
          }}
          onMouseLeave={(e) => {
            if (!isLoading) {
              e.target.style.background = 'var(--color-braun-orange)';
              e.target.style.transform = 'translateY(0)';
              e.target.style.boxShadow = 'none';
            }
          }}
        >
          {isLoading && <div style={styles.spinner}></div>}
          Continue with {selectedPlan.charAt(0).toUpperCase() + selectedPlan.slice(1)} Plan
        </button>

        <div style={styles.footer}>
          <p>
            ✅ Start with any plan and upgrade/downgrade anytime<br/>
            ✅ All plans include secure Auth0 authentication<br/>
            ✅ Free plan available with no credit card required
          </p>
        </div>
      </div>
    </div>
  );

  return (
    <div style={styles.container}>
      <div style={styles.header}>
        <div style={styles.headerContent}>
          <Link to="/" style={styles.logo}>
            SimApp
          </Link>
          
          <Link 
            to="/" 
            style={styles.backButton}
            onMouseEnter={(e) => {
              e.target.style.background = 'var(--color-warm-white)';
              e.target.style.borderColor = 'var(--color-medium-grey)';
            }}
            onMouseLeave={(e) => {
              e.target.style.background = 'var(--color-white)';
              e.target.style.borderColor = 'var(--color-border-light)';
            }}
          >
            ← Back to Home
          </Link>
        </div>
      </div>

      <div style={styles.content}>
        {step === 'plans' && (
          <div style={styles.stepIndicator}>
            <div style={styles.stepItem}>
              <div style={{...styles.stepNumber, ...styles.stepNumberCompleted}}>✓</div>
              <span style={{color: 'var(--color-success)'}}>Welcome</span>
            </div>
            <div style={{width: '32px', height: '1px', background: 'var(--color-border-light)'}}></div>
            <div style={styles.stepItem}>
              <div style={{...styles.stepNumber, ...styles.stepNumberActive}}>2</div>
              <span style={{color: 'var(--color-braun-orange)'}}>Choose Plan</span>
            </div>
            <div style={{width: '32px', height: '1px', background: 'var(--color-border-light)'}}></div>
            <div style={styles.stepItem}>
              <div style={styles.stepNumber}>3</div>
              <span>Create Account</span>
            </div>
          </div>
        )}

        {step === 'welcome' && renderWelcomeStep()}
        {step === 'plans' && renderPlanStep()}
      </div>

      <style>{`
        @keyframes spin {
          0% { transform: rotate(0deg); }
          100% { transform: rotate(360deg); }
        }
      `}</style>
    </div>
  );
};

export default RegisterPage; 