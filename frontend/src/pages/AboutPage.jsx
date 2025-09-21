import React from 'react';
import { Link, useNavigate } from 'react-router-dom';

const AboutPage = () => {
  const navigate = useNavigate();

  const teamMembers = [
    {
      name: 'Dr. Sarah Chen',
      role: 'CEO & Co-Founder',
      bio: 'Former quantitative analyst at Goldman Sachs with 15+ years in financial modeling and risk assessment.',
      image: '👩‍💼'
    },
    {
      name: 'Marcus Rodriguez',
      role: 'CTO & Co-Founder',
      bio: 'Ex-Google engineer specializing in high-performance computing and GPU acceleration technologies.',
      image: '👨‍💻'
    },
    {
      name: 'Dr. Emily Watson',
      role: 'Head of Research',
      bio: 'PhD in Statistics from MIT, published researcher in Monte Carlo methods and uncertainty quantification.',
      image: '👩‍🔬'
    },
    {
      name: 'David Kim',
      role: 'VP of Engineering',
      bio: 'Former Microsoft principal engineer with expertise in cloud architecture and distributed systems.',
      image: '👨‍🔧'
    }
  ];

  const milestones = [
    {
      year: '2020',
      title: 'Company Founded',
      description: 'Started with a vision to democratize Monte Carlo simulation for businesses worldwide.'
    },
    {
      year: '2021',
      title: 'First Product Launch',
      description: 'Released our MVP with basic Excel integration and CPU-based simulation engines.'
    },
    {
      year: '2022',
      title: 'GPU Acceleration',
      description: 'Introduced CUDA-powered GPU acceleration, achieving 10x performance improvements.'
    },
    {
      year: '2023',
      title: 'Enterprise Features',
      description: 'Added advanced security, compliance features, and enterprise-grade infrastructure.'
    },
    {
      year: '2024',
      title: 'Global Expansion',
      description: 'Serving 10,000+ users across 50+ countries with 99.9% uptime SLA.'
    }
  ];

  const values = [
    {
      icon: '🎯',
      title: 'Accuracy First',
      description: 'We prioritize mathematical precision and statistical rigor in every simulation.'
    },
    {
      icon: '🚀',
      title: 'Performance Driven',
      description: 'Continuous optimization to deliver the fastest, most efficient simulations possible.'
    },
    {
      icon: '🔒',
      title: 'Security Focused',
      description: 'Enterprise-grade security and compliance built into every aspect of our platform.'
    },
    {
      icon: '🤝',
      title: 'Customer Success',
      description: 'Your success is our success. We provide the tools and support you need to excel.'
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
    section: {
      marginTop: '80px'
    },
    sectionTitle: {
      fontSize: '32px',
      fontWeight: 'bold',
      color: 'var(--color-charcoal)',
      textAlign: 'center',
      marginBottom: '40px'
    },
    storySection: {
      background: 'var(--color-white)',
      border: '1px solid var(--color-border-light)',
      borderRadius: '24px',
      padding: '48px',
      marginTop: '60px',
      boxShadow: 'var(--shadow-sm)'
    },
    storyText: {
      fontSize: '18px',
      color: 'var(--color-text-secondary)',
      lineHeight: 1.8,
      marginBottom: '24px'
    },
    teamGrid: {
      display: 'grid',
      gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))',
      gap: '32px',
      marginTop: '40px'
    },
    teamCard: {
      background: 'var(--color-white)',
      border: '1px solid var(--color-border-light)',
      borderRadius: '24px',
      padding: '32px',
      textAlign: 'center',
      transition: 'all var(--transition-base)',
      boxShadow: 'var(--shadow-sm)'
    },
    teamImage: {
      fontSize: '64px',
      marginBottom: '20px',
      display: 'block',
      color: 'var(--color-braun-orange)'
    },
    teamName: {
      fontSize: '20px',
      fontWeight: 'bold',
      color: 'var(--color-charcoal)',
      marginBottom: '8px'
    },
    teamRole: {
      fontSize: '16px',
      color: 'var(--color-braun-orange)',
      marginBottom: '16px'
    },
    teamBio: {
      fontSize: '14px',
      color: 'var(--color-text-secondary)',
      lineHeight: 1.6
    },
    timelineContainer: {
      position: 'relative',
      paddingLeft: '40px'
    },
    timelineLine: {
      position: 'absolute',
      left: '20px',
      top: '0',
      bottom: '0',
      width: '2px',
      background: 'var(--color-braun-orange)'
    },
    milestoneItem: {
      position: 'relative',
      marginBottom: '40px',
      background: 'var(--color-white)',
      border: '1px solid var(--color-border-light)',
      borderRadius: '16px',
      padding: '24px',
      marginLeft: '20px'
    },
    milestoneDot: {
      position: 'absolute',
      left: '-30px',
      top: '24px',
      width: '12px',
      height: '12px',
      borderRadius: '50%',
      background: 'var(--color-braun-orange)',
      border: '3px solid var(--color-warm-white)'
    },
    milestoneYear: {
      fontSize: '18px',
      fontWeight: 'bold',
      color: 'var(--color-braun-orange)',
      marginBottom: '8px'
    },
    milestoneTitle: {
      fontSize: '20px',
      fontWeight: 'bold',
      color: 'var(--color-charcoal)',
      marginBottom: '12px'
    },
    milestoneDescription: {
      fontSize: '16px',
      color: 'var(--color-text-secondary)',
      lineHeight: 1.6
    },
    valuesGrid: {
      display: 'grid',
      gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))',
      gap: '32px',
      marginTop: '40px'
    },
    valueCard: {
      background: 'var(--color-white)',
      border: '1px solid var(--color-border-light)',
      borderRadius: '20px',
      padding: '32px',
      textAlign: 'center',
      transition: 'all var(--transition-base)',
      boxShadow: 'var(--shadow-sm)'
    },
    valueIcon: {
      fontSize: '48px',
      marginBottom: '20px',
      display: 'block',
      color: 'var(--color-braun-orange)'
    },
    valueTitle: {
      fontSize: '20px',
      fontWeight: 'bold',
      color: 'var(--color-charcoal)',
      marginBottom: '16px'
    },
    valueDescription: {
      fontSize: '16px',
      color: 'var(--color-text-secondary)',
      lineHeight: 1.6
    },
    ctaSection: {
      textAlign: 'center',
      marginTop: '80px',
      padding: '60px 32px',
      background: 'var(--color-white)',
      borderRadius: '24px',
      border: '1px solid var(--color-border-light)',
      boxShadow: 'var(--shadow-md)'
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
    ctaButton: {
      background: 'var(--color-braun-orange)',
      color: 'var(--color-white)',
      padding: '16px 32px',
      borderRadius: '12px',
      textDecoration: 'none',
      fontSize: '16px',
      fontWeight: '600',
      transition: 'all var(--transition-base)',
      display: 'inline-block'
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
        <h1 style={styles.title}>About SimApp</h1>
        <p style={styles.subtitle}>
          We're on a mission to democratize Monte Carlo simulation and make advanced analytics accessible to every business decision-maker.
        </p>
      </div>

      <div style={styles.content}>
        <div style={styles.storySection}>
          <p style={styles.storyText}>
            Founded in 2020 by a team of quantitative analysts and software engineers, SimApp was born from the frustration of seeing powerful Monte Carlo simulation techniques locked away in expensive, complex software packages that only large corporations could afford.
          </p>
          <p style={styles.storyText}>
            We believed that every business, regardless of size, should have access to the same advanced analytical tools used by Fortune 500 companies. Our vision was to create a platform that combines the mathematical rigor of academic research with the ease-of-use that modern business users expect.
          </p>
          <p style={styles.storyText}>
            Today, SimApp serves thousands of users across 50+ countries, from startup founders making critical business decisions to enterprise risk managers running complex financial models. We've processed over 10 million simulations and helped our users quantify billions of dollars in potential outcomes.
          </p>
        </div>

        <div style={styles.section}>
          <h2 style={styles.sectionTitle}>Our Team</h2>
          <div style={styles.teamGrid}>
            {teamMembers.map((member, index) => (
              <div
                key={index}
                style={styles.teamCard}
                onMouseOver={(e) => {
                  e.currentTarget.style.transform = 'translateY(-8px)';
                  e.currentTarget.style.boxShadow = '0 25px 50px -12px rgba(255, 165, 0, 0.3)';
                }}
                onMouseOut={(e) => {
                  e.currentTarget.style.transform = 'translateY(0)';
                  e.currentTarget.style.boxShadow = 'none';
                }}
              >
                <span style={styles.teamImage}>{member.image}</span>
                <h3 style={styles.teamName}>{member.name}</h3>
                <p style={styles.teamRole}>{member.role}</p>
                <p style={styles.teamBio}>{member.bio}</p>
              </div>
            ))}
          </div>
        </div>

        <div style={styles.section}>
          <h2 style={styles.sectionTitle}>Our Journey</h2>
          <div style={styles.timelineContainer}>
            <div style={styles.timelineLine}></div>
            {milestones.map((milestone, index) => (
              <div key={index} style={styles.milestoneItem}>
                <div style={styles.milestoneDot}></div>
                <div style={styles.milestoneYear}>{milestone.year}</div>
                <h3 style={styles.milestoneTitle}>{milestone.title}</h3>
                <p style={styles.milestoneDescription}>{milestone.description}</p>
              </div>
            ))}
          </div>
        </div>

        <div style={styles.section}>
          <h2 style={styles.sectionTitle}>Our Values</h2>
          <div style={styles.valuesGrid}>
            {values.map((value, index) => (
              <div
                key={index}
                style={styles.valueCard}
                onMouseOver={(e) => {
                  e.currentTarget.style.transform = 'translateY(-4px)';
                  e.currentTarget.style.borderColor = 'rgba(255, 165, 0, 0.3)';
                }}
                onMouseOut={(e) => {
                  e.currentTarget.style.transform = 'translateY(0)';
                  e.currentTarget.style.borderColor = 'var(--color-border-light)';
                }}
              >
                <span style={styles.valueIcon}>{value.icon}</span>
                <h3 style={styles.valueTitle}>{value.title}</h3>
                <p style={styles.valueDescription}>{value.description}</p>
              </div>
            ))}
          </div>
        </div>

        <div style={styles.ctaSection}>
          <h2 style={styles.ctaTitle}>Join Us on Our Mission</h2>
          <p style={styles.ctaDescription}>
            Ready to experience the future of Monte Carlo simulation? Start your free trial today.
          </p>
          <Link 
            to="/register" 
            style={styles.ctaButton}
            onMouseOver={(e) => {
              e.target.style.transform = 'translateY(-2px)';
              e.target.style.boxShadow = '0 10px 25px rgba(255, 165, 0, 0.3)';
            }}
            onMouseOut={(e) => {
              e.target.style.transform = 'translateY(0)';
              e.target.style.boxShadow = 'none';
            }}
          >
            Start Free Trial
          </Link>
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
              <button onClick={() => navigate('/pricing')} style={{ background: 'none', border: 'none', color: 'var(--color-light-grey)', cursor: 'pointer', padding: 0, marginBottom: '8px', display: 'block' }}>Pricing</button>
              <a href="#api" style={{ color: 'var(--color-light-grey)', textDecoration: 'none', display: 'block', marginBottom: '8px' }}>API</a>
              <a href="#docs" style={{ color: 'var(--color-light-grey)', textDecoration: 'none', display: 'block', marginBottom: '8px' }}>Documentation</a>
            </div>
            <div>
              <h4 style={{ fontSize: '16px', fontWeight: '600', marginBottom: '16px' }}>Company</h4>
              <button onClick={() => navigate('/about')} style={{ background: 'none', border: 'none', color: 'var(--color-braun-orange)', cursor: 'pointer', padding: 0, marginBottom: '8px', display: 'block' }}>About</button>
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

export default AboutPage; 