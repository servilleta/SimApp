import React, { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import Footer from '../components/layout/Footer';
import BillingService from '../services/billingService';

const PricingPage = () => {
  const navigate = useNavigate();
  const [isAnnual, setIsAnnual] = useState(false);
  const [loadingPlan, setLoadingPlan] = useState(null);

  const handlePlanSelection = async (plan) => {
    // Handle Enterprise plan differently - contact sales
    if (plan.name === 'Enterprise') {
      // You could redirect to a contact form or open a modal
      window.location.href = 'mailto:sales@simapp.com?subject=Enterprise Plan Inquiry';
      return;
    }

    // Enterprise plan now has Stripe checkout too

    try {
      setLoadingPlan(plan.name);
      
      // Create Stripe checkout session
      const checkoutData = await BillingService.createCheckoutSession(
        plan.name.toLowerCase(), // Convert to lowercase for backend
        `${window.location.origin}/my-dashboard?payment=success`,
        `${window.location.origin}/pricing?payment=cancelled`
      );

      // Redirect to Stripe checkout
      if (checkoutData.checkout_url) {
        window.location.href = checkoutData.checkout_url;
      } else {
        throw new Error('No checkout URL received from server');
      }
    } catch (error) {
      console.error('Error starting checkout:', error);
      alert(`Error starting checkout: ${error.message}`);
      setLoadingPlan(null);
    }
  };

  const plans = [
    {
      name: 'Starter',
      price: { monthly: 19, annual: 190 },
      description: 'Perfect for individuals and small teams getting started',
      useCase: 'Ideal for consultants, analysts, and small finance teams',
      realWorldExample: 'Run project risk assessments with up to 50K scenarios',
      features: [
        '50 simulations per month',
        'Up to 100,000 iterations per simulation',
        '1MB file size limit',
        'Standard GPU acceleration',
        'All Excel functions supported',
        'Email support (48hr response)',
        'Basic visualizations & reports',
        'Export to PDF/Excel',
        'Community forum access'
      ],
      highlights: [
        'Perfect for budget planning',
        'Risk analysis for projects under $1M',
        'Investment portfolio optimization'
      ],
      cta: 'Start 7-Day Free Trial',
      popular: false,
      badge: null,
      onboarding: 'Instant access • No setup required • Cancel anytime'
    },
    {
      name: 'Professional',
      price: { monthly: 49, annual: 490 },
      description: 'Best for growing businesses and advanced analysis',
      useCase: 'Perfect for finance teams, consultancies, and growing companies',
      realWorldExample: 'Model complex scenarios with millions of iterations',
      features: [
        '100 simulations per month',
        'Up to 1,000,000 iterations per simulation',
        '10MB file size limit',
        'GPU-accelerated processing (10x faster)',
        'Priority support (24hr response)',
        'Advanced sensitivity analysis',
        'Custom visualizations & dashboards',
        'API access for integrations',
        'Team collaboration tools',
        'Batch processing capabilities',
        'Advanced export options'
      ],
      highlights: [
        'Enterprise financial planning',
        'Complex risk modeling',
        'Portfolio optimization for $1M+ assets',
        'Multi-scenario planning'
      ],
      cta: 'Start 7-Day Free Trial',
      popular: true,
      badge: 'Most Popular',
      onboarding: 'Setup in 5 minutes • Dedicated onboarding • Priority support'
    },
    {
      name: 'Enterprise',
      price: { monthly: 149, annual: 1490 },
      description: 'For large organizations with mission-critical needs',
      useCase: 'Built for Fortune 500 companies and financial institutions',
      realWorldExample: 'Enterprise-grade modeling with unlimited scale',
      features: [
        'Unlimited simulations per month',
        'Unlimited iterations & scale',
        'Unlimited file size',
        'Dedicated GPU clusters',
        'Premium support (4hr SLA)',
        'Custom integrations & APIs',
        'SSO/SAML authentication',
        'Advanced security & compliance',
        'On-premise deployment options',
        'Custom training & onboarding',
        'Dedicated account manager',
        'Custom SLAs available',
        'Priority feature requests'
      ],
      highlights: [
        'Bank stress testing',
        'Insurance risk modeling',
        'Government budget planning',
        'Multi-billion dollar portfolios'
      ],
      cta: 'Contact Sales',
      popular: false,
      badge: null,
      onboarding: 'Custom setup • Dedicated team • Enterprise onboarding'
    }
  ];

  const styles = {
    container: {
      minHeight: '100vh',
      background: 'var(--color-warm-white)',
      fontFamily: 'system-ui, -apple-system, sans-serif',
      position: 'relative',
      overflow: 'hidden'
    },
    backgroundPattern: {
      display: 'none'
    },
    header: {
      position: 'relative',
      zIndex: 1,
      padding: '40px 20px',
      textAlign: 'center'
    },
    backButton: {
      position: 'absolute',
      top: '24px',
      left: '24px',
      background: 'var(--color-white)',
      border: '1px solid var(--color-border-light)',
      borderRadius: '12px',
      padding: '12px 20px',
      color: 'var(--color-dark-grey)',
      textDecoration: 'none',
      fontSize: '14px',
      fontWeight: '500',
      transition: 'all var(--transition-base)',
      display: 'flex',
      alignItems: 'center',
      gap: '8px'
    },
    title: {
      fontSize: '48px',
      fontWeight: 'bold',
      color: 'var(--color-charcoal)',
      marginBottom: '16px',
      background: 'none',
      WebkitBackgroundClip: 'unset',
      WebkitTextFillColor: 'unset',
      backgroundClip: 'unset'
    },
    subtitle: {
      fontSize: '20px',
      color: 'var(--color-text-secondary)',
      maxWidth: '800px',
      margin: '0 auto 40px',
      lineHeight: 1.6
    },
    billingToggle: {
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      gap: '16px',
      marginBottom: '60px'
    },
    toggleLabel: {
      color: 'var(--color-text-secondary)',
      fontSize: '16px',
      fontWeight: '500'
    },
    toggleSwitch: {
      position: 'relative',
      width: '60px',
      height: '30px',
      background: isAnnual ? 'var(--color-braun-orange)' : 'var(--color-light-grey)',
      borderRadius: '15px',
      cursor: 'pointer',
      transition: 'all var(--transition-base)'
    },
    toggleThumb: {
      position: 'absolute',
      top: '3px',
      left: isAnnual ? '33px' : '3px',
      width: '24px',
      height: '24px',
      background: 'var(--color-white)',
      borderRadius: '50%',
      transition: 'all var(--transition-base)'
    },
    savingsBadge: {
      background: 'var(--color-success)',
      color: 'var(--color-white)',
      padding: '4px 12px',
      borderRadius: '12px',
      fontSize: '12px',
      fontWeight: '600'
    },
    content: {
      position: 'relative',
      zIndex: 1,
      padding: '0 20px 80px',
      maxWidth: '1400px',
      margin: '0 auto'
    },
    plansGrid: {
      display: 'grid',
      gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))',
      gap: '32px',
      marginTop: '40px'
    },
    planCard: {
      background: 'var(--color-white)',
      border: '1px solid var(--color-border-light)',
      borderRadius: '24px',
      padding: '40px 32px',
      position: 'relative',
      transition: 'all 0.3s ease',
      boxShadow: 'var(--shadow-sm)'
    },
    popularBadge: {
      position: 'absolute',
      top: '-12px',
      left: '50%',
      transform: 'translateX(-50%)',
      background: 'var(--color-braun-orange)',
      color: 'var(--color-white)',
      padding: '6px 20px',
      borderRadius: '12px',
      fontSize: '12px',
      fontWeight: '700',
      textTransform: 'uppercase',
      letterSpacing: '0.5px'
    },
    planName: {
      fontSize: '24px',
      fontWeight: 'bold',
      color: 'var(--color-charcoal)',
      marginBottom: '8px'
    },
    planPrice: {
      fontSize: '48px',
      fontWeight: 'bold',
      color: 'var(--color-charcoal)',
      marginBottom: '8px',
      display: 'flex',
      alignItems: 'baseline',
      gap: '8px'
    },
    pricePrefix: {
      fontSize: '24px',
      color: 'var(--color-dark-grey)'
    },
    priceSuffix: {
      fontSize: '16px',
      color: 'var(--color-medium-grey)'
    },
    planDescription: {
      fontSize: '16px',
      color: 'var(--color-text-secondary)',
      marginBottom: '32px',
      lineHeight: 1.5
    },
    featuresList: {
      marginBottom: '32px'
    },
    featureItem: {
      display: 'flex',
      alignItems: 'center',
      gap: '12px',
      marginBottom: '12px',
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
      color: 'white'
    },
    limitationItem: {
      display: 'flex',
      alignItems: 'center',
      gap: '12px',
      marginBottom: '12px',
      fontSize: '14px',
      color: 'var(--color-medium-grey)'
    },
    limitationIcon: {
      width: '16px',
      height: '16px',
      borderRadius: '50%',
      background: 'rgba(239, 68, 68, 0.2)',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      fontSize: '10px',
      color: '#f87171'
    },
    ctaButton: {
      width: '100%',
      padding: '16px 24px',
      borderRadius: '12px',
      fontSize: '16px',
      fontWeight: '600',
      textDecoration: 'none',
      textAlign: 'center',
      transition: 'all 0.3s ease',
      border: 'none',
      cursor: 'pointer',
      display: 'block'
    },
    useCaseSection: {
      marginBottom: '20px',
      padding: '16px',
      background: 'var(--color-warm-white)',
      borderRadius: '12px',
      border: '1px solid var(--color-border-light)'
    },
    useCase: {
      fontSize: '14px',
      color: 'var(--color-text-secondary)',
      marginBottom: '8px',
      fontWeight: '500'
    },
    realWorldExample: {
      fontSize: '13px',
      color: 'var(--color-braun-orange)',
      fontStyle: 'italic'
    },
    highlightsSection: {
      marginBottom: '24px'
    },
    highlightsTitle: {
      fontSize: '14px',
      fontWeight: '600',
      color: 'var(--color-charcoal)',
      marginBottom: '12px',
      margin: '0 0 12px 0'
    },
    highlightItem: {
      display: 'flex',
      alignItems: 'center',
      gap: '8px',
      marginBottom: '8px',
      fontSize: '13px',
      color: 'var(--color-text-secondary)',
      fontWeight: '500'
    },
    highlightIcon: {
      fontSize: '12px'
    },
    featuresTitle: {
      fontSize: '14px',
      fontWeight: '600',
      color: 'var(--color-charcoal)',
      marginBottom: '12px',
      margin: '0 0 12px 0'
    },
    ctaSection: {
      marginTop: '24px'
    },
    onboarding: {
      fontSize: '12px',
      color: 'var(--color-text-secondary)',
      textAlign: 'center',
      marginTop: '8px',
      lineHeight: 1.4
    },
    socialProofSection: {
      marginBottom: '60px',
      textAlign: 'center'
    },
    socialProofStats: {
      display: 'flex',
      justifyContent: 'center',
      gap: '40px',
      marginBottom: '40px',
      flexWrap: 'wrap'
    },
    statItem: {
      minWidth: '120px'
    },
    statNumber: {
      fontSize: '32px',
      fontWeight: 'bold',
      color: 'var(--color-braun-orange)',
      marginBottom: '4px'
    },
    statLabel: {
      fontSize: '14px',
      color: 'var(--color-text-secondary)',
      fontWeight: '500'
    },
    testimonialSection: {
      display: 'grid',
      gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))',
      gap: '24px',
      marginBottom: '32px',
      maxWidth: '800px',
      margin: '0 auto 32px'
    },
    testimonial: {
      background: 'var(--color-white)',
      padding: '24px',
      borderRadius: '16px',
      border: '1px solid var(--color-border-light)',
      boxShadow: 'var(--shadow-sm)'
    },
    testimonialQuote: {
      fontSize: '16px',
      color: 'var(--color-charcoal)',
      marginBottom: '12px',
      lineHeight: 1.5,
      fontStyle: 'italic'
    },
    testimonialAuthor: {
      fontSize: '14px',
      color: 'var(--color-text-secondary)'
    },
    urgencyBanner: {
      background: 'linear-gradient(135deg, var(--color-braun-orange), #ff7849)',
      color: 'white',
      padding: '16px 24px',
      borderRadius: '12px',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      gap: '12px',
      maxWidth: '600px',
      margin: '0 auto',
      boxShadow: 'var(--shadow-md)'
    },
    urgencyIcon: {
      fontSize: '20px'
    },
    urgencyText: {
      fontSize: '14px',
      textAlign: 'center'
    }
  };

  return (
    <div style={styles.container}>
      <div style={styles.backgroundPattern}></div>
      
      <Link 
        to="/" 
        style={styles.backButton}
        onMouseOver={(e) => {
          e.target.style.background = 'var(--color-border-light)';
          e.target.style.transform = 'translateY(-1px)';
        }}
        onMouseOut={(e) => {
          e.target.style.background = 'var(--color-white)';
          e.target.style.transform = 'translateY(0)';
        }}
      >
        ← Back to Home
      </Link>

      <div style={styles.header}>
        <h1 style={styles.title}>Choose Your Success Level</h1>
        <p style={styles.subtitle}>
          From startup risk analysis to Fortune 500 financial modeling - we have the perfect plan to transform your Excel models into powerful simulations. <strong>Start your free trial today</strong> and see results in minutes.
        </p>
        
        <div style={styles.billingToggle}>
          <span style={{...styles.toggleLabel, opacity: isAnnual ? 0.7 : 1}}>Monthly</span>
          <div 
            style={styles.toggleSwitch}
            onClick={() => setIsAnnual(!isAnnual)}
          >
            <div style={styles.toggleThumb}></div>
          </div>
          <span style={{...styles.toggleLabel, opacity: isAnnual ? 1 : 0.7}}>Annual</span>
          {isAnnual && <span style={styles.savingsBadge}>Save 17%</span>}
        </div>
      </div>

      <div style={styles.content}>
        {/* Social Proof Section */}
        <div style={styles.socialProofSection}>
          <div style={styles.socialProofStats}>
            <div style={styles.statItem}>
              <div style={styles.statNumber}>10,000+</div>
              <div style={styles.statLabel}>Simulations Run Daily</div>
            </div>
            <div style={styles.statItem}>
              <div style={styles.statNumber}>500+</div>
              <div style={styles.statLabel}>Companies Trust Us</div>
            </div>
            <div style={styles.statItem}>
              <div style={styles.statNumber}>$2B+</div>
              <div style={styles.statLabel}>Decisions Modeled</div>
            </div>
          </div>
          
          <div style={styles.testimonialSection}>
            <div style={styles.testimonial}>
              <div style={styles.testimonialQuote}>
                "SimApp reduced our financial modeling time from weeks to hours. The GPU acceleration is incredible."
              </div>
              <div style={styles.testimonialAuthor}>
                <strong>Sarah Chen</strong>, CFO at TechCorp
              </div>
            </div>
            <div style={styles.testimonial}>
              <div style={styles.testimonialQuote}>
                "Finally, a Monte Carlo platform that actually works with our complex Excel models."
              </div>
              <div style={styles.testimonialAuthor}>
                <strong>Michael Rodriguez</strong>, Risk Analyst at GlobalBank
              </div>
            </div>
          </div>
          
          <div style={styles.urgencyBanner}>
            <span style={styles.urgencyIcon}>🔥</span>
            <span style={styles.urgencyText}>
              <strong>Limited Time:</strong> Start your free trial today and get priority setup support
            </span>
          </div>
        </div>

        <div style={styles.plansGrid}>
          {plans.map((plan, index) => (
            <div
              key={index}
              style={{
                ...styles.planCard,
                borderColor: plan.popular ? 'var(--color-braun-orange)' : 'var(--color-border-light)',
                borderWidth: plan.popular ? '2px' : '1px',
                transform: plan.popular ? 'scale(1.02)' : 'scale(1)',
                boxShadow: plan.popular ? 'var(--shadow-lg)' : 'var(--shadow-sm)'
              }}
              onMouseOver={(e) => {
                if (!plan.popular) {
                  e.currentTarget.style.transform = 'translateY(-4px)';
                  e.currentTarget.style.boxShadow = 'var(--shadow-lg)';
                  e.currentTarget.style.borderColor = 'var(--color-braun-orange)';
                }
              }}
              onMouseOut={(e) => {
                if (!plan.popular) {
                  e.currentTarget.style.transform = 'translateY(0)';
                  e.currentTarget.style.boxShadow = 'var(--shadow-sm)';
                  e.currentTarget.style.borderColor = 'var(--color-border-light)';
                }
              }}
            >
              {plan.badge && <div style={styles.popularBadge}>{plan.badge}</div>}
              
              <h3 style={styles.planName}>{plan.name}</h3>
              <div style={styles.planPrice}>
                <span style={styles.pricePrefix}>$</span>
                <span>
                  {isAnnual ? plan.price.annual : plan.price.monthly}
                  {isAnnual && plan.price.annual !== 0 ? '/year' : '/month'}
                </span>
              </div>
              <p style={styles.planDescription}>{plan.description}</p>
              
              <div style={styles.useCaseSection}>
                <div style={styles.useCase}>{plan.useCase}</div>
                <div style={styles.realWorldExample}>💡 {plan.realWorldExample}</div>
              </div>

              <div style={styles.highlightsSection}>
                <h4 style={styles.highlightsTitle}>Perfect for:</h4>
                {plan.highlights.map((highlight, idx) => (
                  <div key={idx} style={styles.highlightItem}>
                    <div style={styles.highlightIcon}>⭐</div>
                    {highlight}
                  </div>
                ))}
              </div>

              <div style={styles.featuresList}>
                <h4 style={styles.featuresTitle}>What's included:</h4>
                {plan.features.map((feature, idx) => (
                  <div key={idx} style={styles.featureItem}>
                    <div style={styles.featureIcon}>✓</div>
                    {feature}
                  </div>
                ))}
              </div>

              <div style={styles.ctaSection}>
                <button
                  onClick={() => handlePlanSelection(plan)}
                  disabled={loadingPlan === plan.name}
                  style={{
                    ...styles.ctaButton,
                    backgroundColor: plan.popular 
                      ? 'var(--color-braun-orange)'
                      : 'var(--color-white)',
                    color: plan.popular ? 'white' : 'var(--color-charcoal)',
                    border: plan.popular 
                      ? 'none' 
                      : '2px solid var(--color-border-light)',
                    boxShadow: plan.popular 
                      ? 'var(--shadow-md)' 
                      : 'var(--shadow-sm)',
                    opacity: loadingPlan === plan.name ? 0.7 : 1
                  }}
                  onMouseOver={(e) => {
                    if (loadingPlan === plan.name) return;
                    if (plan.popular) {
                      e.target.style.backgroundColor = 'var(--color-braun-orange-dark)';
                      e.target.style.transform = 'translateY(-2px)';
                      e.target.style.boxShadow = 'var(--shadow-lg)';
                    } else {
                      e.target.style.backgroundColor = 'var(--color-warm-white)';
                      e.target.style.borderColor = 'var(--color-braun-orange)';
                      e.target.style.transform = 'translateY(-2px)';
                      e.target.style.boxShadow = 'var(--shadow-md)';
                    }
                  }}
                  onMouseOut={(e) => {
                    if (loadingPlan === plan.name) return;
                    if (plan.popular) {
                      e.target.style.backgroundColor = 'var(--color-braun-orange)';
                      e.target.style.transform = 'translateY(0)';
                      e.target.style.boxShadow = 'var(--shadow-md)';
                    } else {
                      e.target.style.backgroundColor = 'var(--color-white)';
                      e.target.style.borderColor = 'var(--color-border-light)';
                      e.target.style.transform = 'translateY(0)';
                      e.target.style.boxShadow = 'var(--shadow-sm)';
                    }
                  }}
                >
                  {loadingPlan === plan.name ? 'Processing...' : plan.cta}
                </button>
                <div style={styles.onboarding}>{plan.onboarding}</div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Footer */}
      <footer style={{
        backgroundColor: 'var(--color-charcoal)',
        color: 'var(--color-white)',
        padding: '48px 0',
        marginTop: '80px'
      }}>
        <div style={{
          maxWidth: '1280px',
          margin: '0 auto',
          padding: '0 24px'
        }}>
          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))',
            gap: '32px',
            marginBottom: '32px'
          }}>
            <div>
              <div style={{ fontSize: '24px', fontWeight: 'bold', marginBottom: '16px' }}>SimApp</div>
              <p style={{ color: 'var(--color-light-grey)', fontSize: '14px' }}>
                Enterprise-grade Monte Carlo simulation platform for data-driven decision making.
              </p>
            </div>
            <div>
              <h4 style={{ fontSize: '16px', fontWeight: '600', marginBottom: '16px' }}>Product</h4>
              <button onClick={() => navigate('/features')} style={{ background: 'none', border: 'none', color: 'var(--color-light-grey)', cursor: 'pointer', padding: 0, marginBottom: '8px', display: 'block' }}>Features</button>
              <button onClick={() => navigate('/pricing')} style={{ background: 'none', border: 'none', color: 'var(--color-braun-orange)', cursor: 'pointer', padding: 0, marginBottom: '8px', display: 'block' }}>Pricing</button>
              <a href="#api" style={{ color: 'var(--color-light-grey)', textDecoration: 'none', display: 'block', marginBottom: '8px' }}>API</a>
              <a href="#docs" style={{ color: 'var(--color-light-grey)', textDecoration: 'none', display: 'block', marginBottom: '8px' }}>Documentation</a>
            </div>
            <div>
              <h4 style={{ fontSize: '16px', fontWeight: '600', marginBottom: '16px' }}>Company</h4>
              <button onClick={() => navigate('/about')} style={{ background: 'none', border: 'none', color: 'var(--color-light-grey)', cursor: 'pointer', padding: 0, marginBottom: '8px', display: 'block' }}>About</button>
              <a href="#blog" style={{ color: 'var(--color-light-grey)', textDecoration: 'none', display: 'block', marginBottom: '8px' }}>Blog</a>
              <a href="#careers" style={{ color: 'var(--color-light-grey)', textDecoration: 'none', display: 'block', marginBottom: '8px' }}>Careers</a>
              <button onClick={() => navigate('/contact')} style={{ background: 'none', border: 'none', color: 'var(--color-light-grey)', cursor: 'pointer', padding: 0, marginBottom: '8px', display: 'block' }}>Contact</button>
            </div>
            <div>
              <h4 style={{ fontSize: '16px', fontWeight: '600', marginBottom: '16px' }}>Legal</h4>
              <button onClick={() => navigate('/privacy')} style={{ background: 'none', border: 'none', color: 'var(--color-light-grey)', cursor: 'pointer', padding: 0, marginBottom: '8px', display: 'block' }}>Privacy Policy</button>
              <button onClick={() => navigate('/terms')} style={{ background: 'none', border: 'none', color: 'var(--color-light-grey)', cursor: 'pointer', padding: 0, marginBottom: '8px', display: 'block' }}>Terms of Service</button>
              <button onClick={() => navigate('/cookie-policy')} style={{ background: 'none', border: 'none', color: 'var(--color-light-grey)', cursor: 'pointer', padding: 0, marginBottom: '8px', display: 'block' }}>Cookie Policy</button>
            </div>
          </div>
          <div style={{ 
            borderTop: '1px solid var(--color-medium-grey)', 
            paddingTop: '32px', 
            textAlign: 'center', 
            color: 'var(--color-light-grey)',
            fontSize: '14px' 
          }}>
            <p>&copy; 2025 SimApp. All rights reserved.</p>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default PricingPage; 