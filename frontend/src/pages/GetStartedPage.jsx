import React, { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { useAuth0 } from '@auth0/auth0-react';
import PlanSelector from '../components/billing/PlanSelector';
import PricingTable from '../components/billing/PricingTable';

const GetStartedPage = () => {
  const navigate = useNavigate();
  const { loginWithRedirect, isAuthenticated } = useAuth0();
  const [showPlans, setShowPlans] = useState(true); // Show plans by default
  const [selectedPlan, setSelectedPlan] = useState('professional');

  React.useEffect(() => {
    if (isAuthenticated) {
      navigate('/dashboard');
    }
  }, [isAuthenticated, navigate]);

  const handleStartFree = () => {
    setSelectedPlan('starter');
    handleSignUp();
  };

  const handleViewPlans = () => {
    setShowPlans(true);
    // Scroll to plans section
    setTimeout(() => {
      const plansSection = document.getElementById('plans-section');
      if (plansSection) {
        plansSection.scrollIntoView({ behavior: 'smooth' });
      }
    }, 100);
  };

  const handlePlanSelect = (plan) => {
    setSelectedPlan(plan.tier);
  };

  const handleSignUp = () => {
    // Store selected plan for after auth
    localStorage.setItem('selectedPlan', selectedPlan);
    
    loginWithRedirect({
      authorizationParams: {
        screen_hint: 'signup'
      }
    });
  };

  const handleSignIn = () => {
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
    headerButtons: {
      display: 'flex',
      alignItems: 'center',
      gap: '12px'
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
    signInButton: {
      background: 'transparent',
      border: 'none',
      color: 'var(--color-dark-grey)',
      padding: '8px 16px',
      borderRadius: '6px',
      fontSize: '14px',
      fontWeight: '500',
      cursor: 'pointer',
      transition: 'all var(--transition-base)'
    },
    heroSection: {
      background: 'linear-gradient(135deg, var(--color-white) 0%, var(--color-warm-white) 100%)',
      padding: '80px 20px 60px',
      textAlign: 'center'
    },
    heroContent: {
      maxWidth: '800px',
      margin: '0 auto'
    },
    badge: {
      display: 'inline-block',
      background: 'var(--color-braun-orange)',
      color: 'var(--color-white)',
      padding: '8px 20px',
      borderRadius: '20px',
      fontSize: '14px',
      fontWeight: '600',
      textTransform: 'uppercase',
      letterSpacing: '0.5px',
      marginBottom: '24px'
    },
    heroTitle: {
      fontSize: '48px',
      fontWeight: '700',
      color: 'var(--color-charcoal)',
      marginBottom: '24px',
      lineHeight: '1.1'
    },
    heroSubtitle: {
      fontSize: '20px',
      color: 'var(--color-text-secondary)',
      marginBottom: '40px',
      lineHeight: '1.6'
    },
    ctaContainer: {
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      gap: '16px',
      maxWidth: '400px',
      margin: '0 auto'
    },
    primaryCTA: {
      background: 'var(--color-braun-orange)',
      color: 'var(--color-white)',
      border: 'none',
      padding: '16px 32px',
      borderRadius: '8px',
      fontSize: '18px',
      fontWeight: '600',
      cursor: 'pointer',
      transition: 'all var(--transition-base)',
      width: '100%',
      textAlign: 'center'
    },
    secondaryCTA: {
      background: 'var(--color-white)',
      color: 'var(--color-dark-grey)',
      border: '1px solid var(--color-border-light)',
      padding: '14px 28px',
      borderRadius: '8px',
      fontSize: '16px',
      fontWeight: '500',
      cursor: 'pointer',
      transition: 'all var(--transition-base)',
      width: '100%',
      textAlign: 'center'
    },
    trustIndicators: {
      display: 'flex',
      justifyContent: 'center',
      alignItems: 'center',
      gap: '32px',
      marginTop: '40px',
      fontSize: '14px',
      color: 'var(--color-medium-grey)'
    },
    trustItem: {
      display: 'flex',
      alignItems: 'center',
      gap: '8px'
    },
    featuresSection: {
      padding: '80px 20px',
      background: 'var(--color-warm-white)'
    },
    sectionContent: {
      maxWidth: '1200px',
      margin: '0 auto'
    },
    sectionTitle: {
      fontSize: '36px',
      fontWeight: '700',
      color: 'var(--color-charcoal)',
      marginBottom: '16px',
      textAlign: 'center'
    },
    sectionSubtitle: {
      fontSize: '18px',
      color: 'var(--color-text-secondary)',
      marginBottom: '60px',
      textAlign: 'center',
      maxWidth: '600px',
      margin: '0 auto 60px'
    },
    featuresGrid: {
      display: 'grid',
      gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))',
      gap: '32px'
    },
    featureCard: {
      background: 'var(--color-white)',
      border: '1px solid var(--color-border-light)',
      borderRadius: '12px',
      padding: '32px 24px',
      textAlign: 'center',
      transition: 'all var(--transition-base)',
      boxShadow: 'var(--shadow-sm)'
    },
    featureIcon: {
      width: '64px',
      height: '64px',
      borderRadius: '50%',
      background: 'linear-gradient(135deg, var(--color-braun-orange), var(--color-braun-orange-dark))',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      fontSize: '24px',
      color: 'white',
      margin: '0 auto 20px'
    },
    featureTitle: {
      fontSize: '20px',
      fontWeight: '600',
      color: 'var(--color-charcoal)',
      marginBottom: '12px'
    },
    featureDescription: {
      fontSize: '16px',
      color: 'var(--color-text-secondary)',
      lineHeight: '1.6'
    },
    plansSection: {
      padding: '80px 20px',
      background: 'var(--color-white)'
    },
    plansSectionTitle: {
      fontSize: '36px',
      fontWeight: '700',
      color: 'var(--color-charcoal)',
      marginBottom: '16px',
      textAlign: 'center'
    },
    plansSectionSubtitle: {
      fontSize: '18px',
      color: 'var(--color-text-secondary)',
      marginBottom: '40px',
      textAlign: 'center',
      maxWidth: '600px',
      margin: '0 auto 40px'
    },
    statsSection: {
      padding: '60px 20px',
      background: 'var(--color-charcoal)',
      color: 'var(--color-white)',
      textAlign: 'center'
    },
    statsGrid: {
      display: 'grid',
      gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
      gap: '40px',
      maxWidth: '800px',
      margin: '0 auto'
    },
    statItem: {
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center'
    },
    statNumber: {
      fontSize: '32px',
      fontWeight: '700',
      color: 'var(--color-braun-orange)',
      marginBottom: '8px'
    },
    statLabel: {
      fontSize: '16px',
      color: 'var(--color-light-grey)',
      fontWeight: '500'
    },
    finalCTASection: {
      padding: '80px 20px',
      background: 'var(--color-white)',
      textAlign: 'center'
    },
    finalCTATitle: {
      fontSize: '32px',
      fontWeight: '700',
      color: 'var(--color-charcoal)',
      marginBottom: '16px'
    },
    finalCTASubtitle: {
      fontSize: '18px',
      color: 'var(--color-text-secondary)',
      marginBottom: '32px',
      maxWidth: '500px',
      margin: '0 auto 32px'
    },
    finalCTAButtons: {
      display: 'flex',
      justifyContent: 'center',
      gap: '16px',
      flexWrap: 'wrap'
    }
  };

  const features = [
    {
      icon: '🚀',
      title: 'Ultra-Fast Simulations',
      description: 'GPU-accelerated Monte Carlo simulations up to 1000x faster than traditional CPU-based solutions.'
    },
    {
      icon: '📊',
      title: 'Any Excel File',
      description: 'Works with your existing Excel models without modification. Upload and simulate in seconds.'
    },
    {
      icon: '⚡',
      title: 'Real-time Results',
      description: 'Live progress tracking with instant visualization of results as your simulation runs.'
    },
    {
      icon: '🔒',
      title: 'Enterprise Security',
      description: 'Bank-grade security with Auth0 integration and comprehensive audit trails for compliance.'
    },
    {
      icon: '📈',
      title: 'Advanced Analytics',
      description: 'Sensitivity analysis, tornado charts, and statistical insights to drive better decisions.'
    },
    {
      icon: '🌐',
      title: 'Cloud-Native',
      description: 'Fully cloud-based platform with automatic scaling and global availability.'
    }
  ];

  return (
    <div style={styles.container}>
      <div style={styles.header}>
        <div style={styles.headerContent}>
          <Link to="/" style={styles.logo}>
            SimApp
          </Link>
          
          <div style={styles.headerButtons}>
            <button
              onClick={handleSignIn}
              style={styles.signInButton}
              onMouseEnter={(e) => {
                e.target.style.background = 'var(--color-warm-white)';
              }}
              onMouseLeave={(e) => {
                e.target.style.background = 'transparent';
              }}
            >
              Sign In
            </button>
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
      </div>

      {/* Hero Section */}
      <section style={styles.heroSection}>
        <div style={styles.heroContent}>
          <div style={styles.badge}>Now Available</div>
          <h1 style={styles.heroTitle}>Transform Your Excel Models into Powerful Simulations</h1>
          <p style={styles.heroSubtitle}>
            The world's fastest Monte Carlo simulation platform. Upload your Excel file, 
            run millions of iterations in seconds, and make data-driven decisions with confidence.
          </p>
          
          <div style={styles.ctaContainer}>
            <button
              onClick={handleStartFree}
              style={styles.primaryCTA}
              onMouseEnter={(e) => {
                e.target.style.background = 'var(--color-braun-orange-dark)';
                e.target.style.transform = 'translateY(-2px)';
                e.target.style.boxShadow = 'var(--shadow-lg)';
              }}
              onMouseLeave={(e) => {
                e.target.style.background = 'var(--color-braun-orange)';
                e.target.style.transform = 'translateY(0)';
                e.target.style.boxShadow = 'none';
              }}
            >
              Start 7-Day Free Trial
            </button>
            
            <button
              onClick={handleViewPlans}
              style={styles.secondaryCTA}
              onMouseEnter={(e) => {
                e.target.style.background = 'var(--color-warm-white)';
                e.target.style.borderColor = 'var(--color-medium-grey)';
              }}
              onMouseLeave={(e) => {
                e.target.style.background = 'var(--color-white)';
                e.target.style.borderColor = 'var(--color-border-light)';
              }}
            >
              View All Plans & Pricing
            </button>
          </div>

          <div style={styles.trustIndicators}>
            <div style={styles.trustItem}>
              <span>✅</span>
              <span>7-Day Free Trial</span>
            </div>
            <div style={styles.trustItem}>
              <span>🔒</span>
              <span>Secure Auth0 Login</span>
            </div>
            <div style={styles.trustItem}>
              <span>⚡</span>
              <span>GPU-Accelerated</span>
            </div>
          </div>
        </div>
      </section>

      {/* Plans Section (Always Visible) */}
      <section 
        id="plans-section"
        style={styles.plansSection}
      >
        <div style={styles.sectionContent}>
          <h2 style={styles.plansSectionTitle}>Choose Your Plan</h2>
          <p style={styles.plansSectionSubtitle}>
            Select the perfect plan for your Monte Carlo simulation needs. 
            You can upgrade or downgrade at any time.
          </p>
          
          <PricingTable
            onPlanSelect={handlePlanSelect}
            selectedPlan={selectedPlan}
          />
          
          <div style={{ textAlign: 'center', marginTop: '40px' }}>
            <button
              onClick={handleSignUp}
              style={{
                ...styles.primaryCTA,
                fontSize: '16px',
                maxWidth: '300px'
              }}
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
              Get Started with {selectedPlan.charAt(0).toUpperCase() + selectedPlan.slice(1)} Plan
            </button>
          </div>
        </div>
      </section>

      {/* Features Section - Moved below plans */}
      <section style={styles.featuresSection}>
        <div style={styles.sectionContent}>
          <h2 style={styles.sectionTitle}>Why Choose SimApp?</h2>
          <p style={styles.sectionSubtitle}>
            Built for professionals who need fast, reliable Monte Carlo simulations without the complexity.
          </p>
          
          <div style={styles.featuresGrid}>
            {features.map((feature, index) => (
              <div
                key={index}
                style={styles.featureCard}
                onMouseEnter={(e) => {
                  e.currentTarget.style.transform = 'translateY(-4px)';
                  e.currentTarget.style.boxShadow = 'var(--shadow-md)';
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.transform = 'translateY(0)';
                  e.currentTarget.style.boxShadow = 'var(--shadow-sm)';
                }}
              >
                <div style={styles.featureIcon}>{feature.icon}</div>
                <h3 style={styles.featureTitle}>{feature.title}</h3>
                <p style={styles.featureDescription}>{feature.description}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Stats Section */}
      <section style={styles.statsSection}>
        <div style={styles.sectionContent}>
          <div style={styles.statsGrid}>
            <div style={styles.statItem}>
              <div style={styles.statNumber}>1000x</div>
              <div style={styles.statLabel}>Faster Than CPU</div>
            </div>
            <div style={styles.statItem}>
              <div style={styles.statNumber}>10M+</div>
              <div style={styles.statLabel}>Iterations Per Second</div>
            </div>
            <div style={styles.statItem}>
              <div style={styles.statNumber}>99.9%</div>
              <div style={styles.statLabel}>Uptime SLA</div>
            </div>
            <div style={styles.statItem}>
              <div style={styles.statNumber}>Enterprise</div>
              <div style={styles.statLabel}>Security Grade</div>
            </div>
          </div>
        </div>
      </section>

      {/* Final CTA Section */}
      <section style={styles.finalCTASection}>
        <div style={styles.sectionContent}>
          <h2 style={styles.finalCTATitle}>Ready to Get Started?</h2>
          <p style={styles.finalCTASubtitle}>
            Join thousands of professionals already using SimApp to make better decisions with Monte Carlo simulations.
          </p>
          
          <div style={styles.finalCTAButtons}>
            <button
              onClick={handleStartFree}
              style={{
                ...styles.primaryCTA,
                width: 'auto',
                minWidth: '250px'
              }}
              onMouseEnter={(e) => {
                e.target.style.background = 'var(--color-braun-orange-dark)';
                e.target.style.transform = 'translateY(-2px)';
                e.target.style.boxShadow = 'var(--shadow-lg)';
              }}
              onMouseLeave={(e) => {
                e.target.style.background = 'var(--color-braun-orange)';
                e.target.style.transform = 'translateY(0)';
                e.target.style.boxShadow = 'none';
              }}
            >
              Start 7-Day Free Trial
            </button>
            
            <Link
              to="/contact"
              style={{
                ...styles.secondaryCTA,
                width: 'auto',
                minWidth: '200px',
                textDecoration: 'none',
                display: 'inline-flex',
                alignItems: 'center',
                justifyContent: 'center'
              }}
              onMouseEnter={(e) => {
                e.target.style.background = 'var(--color-warm-white)';
                e.target.style.borderColor = 'var(--color-medium-grey)';
              }}
              onMouseLeave={(e) => {
                e.target.style.background = 'var(--color-white)';
                e.target.style.borderColor = 'var(--color-border-light)';
              }}
            >
              Contact Sales
            </Link>
          </div>
        </div>
      </section>
    </div>
  );
};

export default GetStartedPage;
