import React from 'react';
import { Link, useNavigate } from 'react-router-dom';

const FeaturesPage = () => {
  const navigate = useNavigate();

  const features = [
    {
      icon: '🚀',
      title: 'GPU-Accelerated Computing',
      description: 'Harness the power of CUDA-enabled GPUs for lightning-fast Monte Carlo simulations. Experience up to 10x performance improvements over traditional CPU-based calculations.',
      highlights: ['CUDA Optimization', 'Memory Pooling', 'Batch Processing', 'Real-time Results']
    },
    {
      icon: '🎯',
      title: 'Multi-Engine Architecture',
      description: 'Choose from 5 specialized simulation engines (GPU, Power, Enhanced, Arrow, BIG) each optimized for different use cases and performance requirements.',
      highlights: ['Auto-optimization', 'Fallback Support', 'Performance Monitoring']
    },
    {
      icon: '📊',
      title: 'Advanced Analytics',
      description: 'Deep statistical analysis with sensitivity analysis, correlation matrices, variable impact assessment, and comprehensive uncertainty quantification.',
      highlights: ['Sensitivity Analysis', 'Correlation Matrices', 'Impact Assessment', 'Risk Quantification']
    },
    {
      icon: '📈',
      title: 'Excel Integration',
      description: 'Seamlessly work with your existing Excel models. Support for complex formulas, named ranges, circular references, and structured tables.',
      highlights: ['Formula Evaluation', 'Named Ranges', 'Circular References', 'Table Support']
    },
    {
      icon: '🔒',
      title: 'Enterprise Security',
      description: 'Bank-grade security with GDPR compliance, role-based access control, audit logging, and comprehensive data protection.',
      highlights: ['GDPR Compliant', 'Role-based Access', 'Audit Logging', 'Data Encryption']
    },
    {
      icon: '☁️',
      title: 'Cloud-Native Platform',
      description: 'Built for scale with Docker containerization, Kubernetes orchestration, auto-scaling, and global deployment capabilities.',
      highlights: ['Auto-scaling', 'Global CDN', 'Load Balancing', '99.9% Uptime']
    },
    {
      icon: '🔧',
      title: 'Developer-Friendly API',
      description: 'RESTful API with comprehensive documentation, SDKs for popular languages, webhooks, and real-time WebSocket connections.',
      highlights: ['REST API', 'Multiple SDKs', 'Webhooks', 'WebSocket Support']
    },
    {
      icon: '📱',
      title: 'Modern Interface',
      description: 'Intuitive, responsive web interface with real-time progress tracking, interactive visualizations, and mobile-optimized design.',
      highlights: ['Responsive Design', 'Real-time Updates', 'Interactive Charts', 'Mobile Support']
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
      margin: '0 auto',
      lineHeight: 1.6
    },
    content: {
      position: 'relative',
      zIndex: 1,
      padding: '0 20px 80px',
      maxWidth: '1200px',
      margin: '0 auto'
    },
    featuresGrid: {
      display: 'grid',
      gridTemplateColumns: 'repeat(auto-fit, minmax(350px, 1fr))',
      gap: '32px',
      marginTop: '60px'
    },
    featureCard: {
      background: 'var(--color-white)',
      border: '1px solid var(--color-border-light)',
      borderRadius: '24px',
      padding: '32px',
      transition: 'all var(--transition-base)',
      boxShadow: 'var(--shadow-sm)'
    },
    featureCardHover: {
      transform: 'translateY(-8px)',
      boxShadow: 'var(--shadow-md)',
      borderColor: 'var(--color-braun-orange)'
    },
    featureIcon: {
      fontSize: '48px',
      marginBottom: '20px',
      display: 'block',
      color: 'var(--color-braun-orange)'
    },
    featureTitle: {
      fontSize: '24px',
      fontWeight: 'bold',
      color: 'var(--color-charcoal)',
      marginBottom: '16px'
    },
    featureDescription: {
      fontSize: '16px',
      color: 'var(--color-text-secondary)',
      lineHeight: 1.6,
      marginBottom: '24px'
    },
    highlightsList: {
      display: 'grid',
      gridTemplateColumns: 'repeat(2, 1fr)',
      gap: '8px'
    },
    highlight: {
      fontSize: '14px',
      color: 'var(--color-braun-orange)',
      display: 'flex',
      alignItems: 'center',
      gap: '8px'
    },
    highlightDot: {
      width: '6px',
      height: '6px',
      borderRadius: '50%',
      background: 'var(--color-braun-orange)'
    },
    ctaSection: {
      textAlign: 'center',
      marginTop: '80px',
      padding: '60px 32px',
      background: 'var(--color-warm-white)',
      borderRadius: '24px',
      border: '1px solid var(--color-border-light)'
    },
    ctaTitle: {
      fontSize: '32px',
      fontWeight: 'bold',
      color: 'var(--color-charcoal)',
      marginBottom: '16px'
    },
    ctaDescription: {
      fontSize: '18px',
      color: 'var(--color-text-secondary)',
      marginBottom: '32px'
    },
    ctaButtons: {
      display: 'flex',
      gap: '16px',
      justifyContent: 'center',
      flexWrap: 'wrap'
    },
    primaryButton: {
      background: 'var(--color-braun-orange)',
      color: 'var(--color-white)',
      border: 'none',
      borderRadius: '8px',
      padding: '16px 32px',
      fontWeight: '600',
      fontSize: '18px',
      cursor: 'pointer',
      transition: 'all var(--transition-base)'
    },
    secondaryButton: {
      background: 'var(--color-white)',
      color: 'var(--color-dark-grey)',
      border: '1px solid var(--color-dark-grey)',
      borderRadius: '8px',
      padding: '16px 32px',
      fontWeight: '600',
      fontSize: '18px',
      cursor: 'pointer',
      transition: 'all var(--transition-base)'
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
        <h1 style={styles.title}>Platform Features</h1>
        <p style={styles.subtitle}>
          Discover the powerful capabilities that make SimApp the leading Monte Carlo simulation platform for enterprises worldwide.
        </p>
      </div>

      <div style={styles.content}>
        <div style={styles.featuresGrid}>
          {features.map((feature, index) => (
            <div
              key={index}
              style={styles.featureCard}
              onMouseOver={(e) => {
                Object.assign(e.currentTarget.style, styles.featureCardHover);
              }}
              onMouseOut={(e) => {
                e.currentTarget.style.transform = 'translateY(0)';
                e.currentTarget.style.boxShadow = 'none';
                e.currentTarget.style.borderColor = 'rgba(255, 255, 255, 0.1)';
              }}
            >
              <span style={styles.featureIcon}>{feature.icon}</span>
              <h3 style={styles.featureTitle}>{feature.title}</h3>
              <p style={styles.featureDescription}>{feature.description}</p>
              <div style={styles.highlightsList}>
                {feature.highlights.map((highlight, idx) => (
                  <div key={idx} style={styles.highlight}>
                    <div style={styles.highlightDot}></div>
                    {highlight}
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>

        <div style={styles.ctaSection}>
          <h2 style={styles.ctaTitle}>Ready to Experience These Features?</h2>
          <p style={styles.ctaDescription}>
            Start your free trial today and see how SimApp can transform your decision-making process.
          </p>
          <div style={styles.ctaButtons}>
            <Link 
              to="/register" 
              style={styles.primaryButton}
              onMouseOver={(e) => {
                e.target.style.transform = 'translateY(-2px)';
                e.target.style.boxShadow = '0 10px 25px rgba(59, 130, 246, 0.3)';
              }}
              onMouseOut={(e) => {
                e.target.style.transform = 'translateY(0)';
                e.target.style.boxShadow = 'none';
              }}
            >
              Start Free Trial
            </Link>
            <Link 
              to="/pricing" 
              style={styles.secondaryButton}
              onMouseOver={(e) => {
                e.target.style.background = 'rgba(255, 255, 255, 0.15)';
                e.target.style.transform = 'translateY(-2px)';
              }}
              onMouseOut={(e) => {
                e.target.style.background = 'rgba(255, 255, 255, 0.1)';
                e.target.style.transform = 'translateY(0)';
              }}
            >
              View Pricing
            </Link>
          </div>
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
              <button onClick={() => navigate('/features')} style={{ background: 'none', border: 'none', color: 'var(--color-braun-orange)', cursor: 'pointer', padding: 0, marginBottom: '8px', display: 'block' }}>Features</button>
              <button onClick={() => navigate('/pricing')} style={{ background: 'none', border: 'none', color: 'var(--color-light-grey)', cursor: 'pointer', padding: 0, marginBottom: '8px', display: 'block' }}>Pricing</button>
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

export default FeaturesPage; 