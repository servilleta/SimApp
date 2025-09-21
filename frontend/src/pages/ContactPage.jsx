import React, { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import Footer from '../components/layout/Footer';

const ContactPage = () => {
  const navigate = useNavigate();
  const [formData, setFormData] = useState({
    name: '',
    email: '',
    company: '',
    subject: '',
    message: ''
  });

  const handleChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value
    });
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    alert('Thank you for your message! We\'ll get back to you within 24 hours.');
    setFormData({ name: '', email: '', company: '', subject: '', message: '' });
  };

  const contactInfo = [
    {
      icon: '📧',
      title: 'Email',
      details: ['hello@simapp.com', 'support@simapp.com'],
      description: 'Send us an email anytime'
    },
    {
      icon: '📞',
      title: 'Phone',
      details: ['+1 (555) 123-4567'],
      description: 'Call us during business hours'
    },
    {
      icon: '📍',
      title: 'Office',
      details: ['123 Innovation Drive', 'San Francisco, CA 94105'],
      description: 'Visit our headquarters'
    },
    {
      icon: '🕒',
      title: 'Hours',
      details: ['Mon-Fri: 9AM-6PM PST'],
      description: 'We\'re here to help'
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
    mainGrid: {
      display: 'grid',
      gridTemplateColumns: 'repeat(auto-fit, minmax(500px, 1fr))',
      gap: '40px',
      marginTop: '60px'
    },
    formSection: {
      background: 'var(--color-white)',
      border: '1px solid var(--color-border-light)',
      borderRadius: '24px',
      padding: '40px',
      boxShadow: 'var(--shadow-sm)'
    },
    formTitle: {
      fontSize: '24px',
      fontWeight: 'bold',
      color: 'var(--color-charcoal)',
      marginBottom: '24px'
    },
    form: {
      display: 'flex',
      flexDirection: 'column',
      gap: '20px'
    },
    inputGroup: {
      display: 'flex',
      flexDirection: 'column',
      gap: '8px'
    },
    label: {
      fontSize: '14px',
      fontWeight: '500',
      color: 'var(--color-dark-grey)',
      marginLeft: '4px'
    },
    input: {
      width: '100%',
      padding: '16px 20px',
      background: 'var(--color-white)',
      border: '1px solid var(--color-border-light)',
      borderRadius: '12px',
      fontSize: '16px',
      color: 'var(--color-charcoal)',
      boxSizing: 'border-box',
      transition: 'all var(--transition-base)',
      outline: 'none'
    },
    textarea: {
      minHeight: '120px',
      resize: 'vertical'
    },
    submitButton: {
      background: 'var(--color-braun-orange)',
      color: 'var(--color-white)',
      padding: '16px 24px',
      borderRadius: '12px',
      fontSize: '16px',
      fontWeight: '600',
      border: 'none',
      cursor: 'pointer',
      transition: 'all var(--transition-base)',
      marginTop: '8px'
    },
    contactInfoSection: {
      display: 'flex',
      flexDirection: 'column',
      gap: '24px'
    },
    infoCard: {
      background: 'var(--color-white)',
      border: '1px solid var(--color-border-light)',
      borderRadius: '20px',
      padding: '24px',
      transition: 'all var(--transition-base)',
      boxShadow: 'var(--shadow-sm)'
    },
    infoIcon: {
      fontSize: '32px',
      marginBottom: '16px',
      display: 'block',
      color: 'var(--color-braun-orange)'
    },
    infoTitle: {
      fontSize: '18px',
      fontWeight: 'bold',
      color: 'var(--color-charcoal)',
      marginBottom: '8px'
    },
    infoDetails: {
      marginBottom: '12px'
    },
    infoDetail: {
      fontSize: '16px',
      color: 'var(--color-text-secondary)',
      marginBottom: '4px'
    },
    infoDescription: {
      fontSize: '14px',
      color: 'var(--color-text-secondary)'
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
        <h1 style={styles.title}>Contact Us</h1>
        <p style={styles.subtitle}>
          Have questions about SimApp? We'd love to hear from you. Send us a message and we'll respond as soon as possible.
        </p>
      </div>

      <div style={styles.content}>
        <div style={styles.mainGrid}>
          <div style={styles.formSection}>
            <h2 style={styles.formTitle}>Send us a message</h2>
            <form onSubmit={handleSubmit} style={styles.form}>
              <div style={styles.inputGroup}>
                <label style={styles.label}>Full Name *</label>
                <input
                  type="text"
                  name="name"
                  value={formData.name}
                  onChange={handleChange}
                  placeholder="Enter your full name"
                  style={styles.input}
                  required
                />
              </div>

              <div style={styles.inputGroup}>
                <label style={styles.label}>Email Address *</label>
                <input
                  type="email"
                  name="email"
                  value={formData.email}
                  onChange={handleChange}
                  placeholder="Enter your email address"
                  style={styles.input}
                  required
                />
              </div>

              <div style={styles.inputGroup}>
                <label style={styles.label}>Company</label>
                <input
                  type="text"
                  name="company"
                  value={formData.company}
                  onChange={handleChange}
                  placeholder="Enter your company name"
                  style={styles.input}
                />
              </div>

              <div style={styles.inputGroup}>
                <label style={styles.label}>Subject *</label>
                <input
                  type="text"
                  name="subject"
                  value={formData.subject}
                  onChange={handleChange}
                  placeholder="What is this about?"
                  style={styles.input}
                  required
                />
              </div>

              <div style={styles.inputGroup}>
                <label style={styles.label}>Message *</label>
                <textarea
                  name="message"
                  value={formData.message}
                  onChange={handleChange}
                  placeholder="Tell us more about your inquiry..."
                  style={{...styles.input, ...styles.textarea}}
                  required
                />
              </div>

              <button
                type="submit"
                style={styles.submitButton}
                onMouseOver={(e) => {
                  e.target.style.transform = 'translateY(-2px)';
                  e.target.style.boxShadow = '0 10px 25px rgba(255, 165, 0, 0.3)';
                }}
                onMouseOut={(e) => {
                  e.target.style.transform = 'translateY(0)';
                  e.target.style.boxShadow = 'none';
                }}
              >
                Send Message
              </button>
            </form>
          </div>

          <div style={styles.contactInfoSection}>
            {contactInfo.map((info, index) => (
              <div
                key={index}
                style={styles.infoCard}
                onMouseOver={(e) => {
                  e.currentTarget.style.transform = 'translateY(-4px)';
                  e.currentTarget.style.borderColor = 'var(--color-border-light)';
                }}
                onMouseOut={(e) => {
                  e.currentTarget.style.transform = 'translateY(0)';
                  e.currentTarget.style.borderColor = 'var(--color-white)';
                }}
              >
                <span style={styles.infoIcon}>{info.icon}</span>
                <h3 style={styles.infoTitle}>{info.title}</h3>
                <div style={styles.infoDetails}>
                  {info.details.map((detail, idx) => (
                    <div key={idx} style={styles.infoDetail}>{detail}</div>
                  ))}
                </div>
                <p style={styles.infoDescription}>{info.description}</p>
              </div>
            ))}
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
              <button onClick={() => navigate('/features')} style={{ background: 'none', border: 'none', color: 'var(--color-light-grey)', cursor: 'pointer', padding: 0, marginBottom: '8px', display: 'block' }}>Features</button>
              <button onClick={() => navigate('/pricing')} style={{ background: 'none', border: 'none', color: 'var(--color-light-grey)', cursor: 'pointer', padding: 0, marginBottom: '8px', display: 'block' }}>Pricing</button>
              <a href="#api" style={{ color: 'var(--color-light-grey)', textDecoration: 'none', display: 'block', marginBottom: '8px' }}>API</a>
              <a href="#docs" style={{ color: 'var(--color-light-grey)', textDecoration: 'none', display: 'block', marginBottom: '8px' }}>Documentation</a>
            </div>
            <div>
              <h4 style={{ fontSize: '16px', fontWeight: '600', marginBottom: '16px' }}>Company</h4>
              <button onClick={() => navigate('/about')} style={{ background: 'none', border: 'none', color: 'var(--color-light-grey)', cursor: 'pointer', padding: 0, marginBottom: '8px', display: 'block' }}>About</button>
              <a href="#blog" style={{ color: 'var(--color-light-grey)', textDecoration: 'none', display: 'block', marginBottom: '8px' }}>Blog</a>
              <a href="#careers" style={{ color: 'var(--color-light-grey)', textDecoration: 'none', display: 'block', marginBottom: '8px' }}>Careers</a>
              <button onClick={() => navigate('/contact')} style={{ background: 'none', border: 'none', color: 'var(--color-braun-orange)', cursor: 'pointer', padding: 0, marginBottom: '8px', display: 'block' }}>Contact</button>
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

export default ContactPage; 