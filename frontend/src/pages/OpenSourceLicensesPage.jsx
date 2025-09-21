import React, { useEffect, useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import ReactMarkdown from 'react-markdown';

function useMarkdown(url) {
  const [content, setContent] = useState('Loading...');
  useEffect(() => {
    fetch(url)
      .then(r => r.text())
      .then(setContent)
      .catch(() => setContent('Failed to load open source licenses information.'));
  }, [url]);
  return content;
}

function OpenSourceLicensesPage() {
  const navigate = useNavigate();
  const licensesText = useMarkdown('/legal/OPEN_SOURCE_LICENSES.md');

  const styles = {
    container: {
      minHeight: '100vh',
      background: 'var(--color-warm-white)',
      fontFamily: 'system-ui, -apple-system, sans-serif',
      position: 'relative'
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
      marginBottom: '16px'
    },
    subtitle: {
      fontSize: '18px',
      color: 'var(--color-medium-grey)',
      marginBottom: '32px'
    },
    content: {
      maxWidth: '1000px',
      margin: '0 auto',
      padding: '0 20px 80px',
      background: 'var(--color-white)',
      borderRadius: '24px',
      boxShadow: 'var(--shadow-sm)',
      border: '1px solid var(--color-border-light)'
    },
    markdownContent: {
      padding: '40px',
      color: 'var(--color-charcoal)',
      lineHeight: '1.6'
    },
    highlightBox: {
      background: 'linear-gradient(135deg, #e8f8f5 0%, #f0fff0 100%)',
      border: '2px solid #28a745',
      borderRadius: '16px',
      padding: '24px',
      margin: '32px 0',
      textAlign: 'center'
    },
    highlightTitle: {
      fontSize: '24px',
      fontWeight: 'bold',
      color: '#28a745',
      marginBottom: '12px'
    },
    highlightText: {
      fontSize: '16px',
      color: 'var(--color-charcoal)',
      marginBottom: '16px'
    },
    benefitsList: {
      display: 'grid',
      gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))',
      gap: '16px',
      margin: '24px 0'
    },
    benefitItem: {
      display: 'flex',
      alignItems: 'center',
      gap: '8px',
      padding: '12px',
      background: 'rgba(40, 167, 69, 0.1)',
      borderRadius: '8px'
    }
  };

  return (
    <div style={styles.container}>
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
        <h1 style={styles.title}>Open Source Licenses</h1>
        <p style={styles.subtitle}>
          Comprehensive attribution and license compliance information for all third-party components
        </p>
      </div>

      <div style={styles.content}>
        {/* Executive Summary */}
        <div style={styles.highlightBox}>
          <h2 style={styles.highlightTitle}>✅ Commercial-Friendly Platform</h2>
          <p style={styles.highlightText}>
            This platform is built entirely on open-source technologies with business-friendly licenses. 
            <strong> No licensing fees required for any component.</strong>
          </p>
          
          <div style={styles.benefitsList}>
            <div style={styles.benefitItem}>
              <span style={{color: '#28a745', fontSize: '18px'}}>🟢</span>
              <span>$0 licensing costs</span>
            </div>
            <div style={styles.benefitItem}>
              <span style={{color: '#28a745', fontSize: '18px'}}>🟢</span>
              <span>Full commercial rights</span>
            </div>
            <div style={styles.benefitItem}>
              <span style={{color: '#28a745', fontSize: '18px'}}>🟢</span>
              <span>No copyleft restrictions</span>
            </div>
            <div style={styles.benefitItem}>
              <span style={{color: '#28a745', fontSize: '18px'}}>🟢</span>
              <span>International compliance</span>
            </div>
          </div>
        </div>

        <div style={styles.markdownContent}>
          <ReactMarkdown
            components={{
              // Style tables properly
              table: ({node, ...props}) => (
                <table style={{
                  width: '100%',
                  borderCollapse: 'collapse',
                  margin: '16px 0',
                  fontSize: '14px'
                }} {...props} />
              ),
              th: ({node, ...props}) => (
                <th style={{
                  background: 'var(--color-warm-white)',
                  padding: '12px',
                  border: '1px solid var(--color-border-light)',
                  textAlign: 'left',
                  fontWeight: '600'
                }} {...props} />
              ),
              td: ({node, ...props}) => (
                <td style={{
                  padding: '12px',
                  border: '1px solid var(--color-border-light)',
                  verticalAlign: 'top'
                }} {...props} />
              ),
              // Style headings
              h1: ({node, ...props}) => (
                <h1 style={{
                  fontSize: '32px',
                  fontWeight: 'bold',
                  color: 'var(--color-charcoal)',
                  marginTop: '32px',
                  marginBottom: '16px'
                }} {...props} />
              ),
              h2: ({node, ...props}) => (
                <h2 style={{
                  fontSize: '24px',
                  fontWeight: '600',
                  color: 'var(--color-charcoal)',
                  marginTop: '24px',
                  marginBottom: '12px'
                }} {...props} />
              ),
              h3: ({node, ...props}) => (
                <h3 style={{
                  fontSize: '20px',
                  fontWeight: '600',
                  color: 'var(--color-charcoal)',
                  marginTop: '20px',
                  marginBottom: '10px'
                }} {...props} />
              ),
              // Style code blocks
              code: ({node, inline, ...props}) => 
                inline ? (
                  <code style={{
                    background: 'var(--color-warm-white)',
                    padding: '2px 6px',
                    borderRadius: '4px',
                    fontSize: '13px',
                    fontFamily: 'Monaco, Consolas, monospace'
                  }} {...props} />
                ) : (
                  <code style={{
                    display: 'block',
                    background: 'var(--color-warm-white)',
                    padding: '16px',
                    borderRadius: '8px',
                    fontSize: '13px',
                    fontFamily: 'Monaco, Consolas, monospace',
                    overflow: 'auto'
                  }} {...props} />
                ),
              // Style links
              a: ({node, ...props}) => (
                <a style={{
                  color: 'var(--color-braun-orange)',
                  textDecoration: 'none'
                }} {...props} />
              )
            }}
          >
            {licensesText}
          </ReactMarkdown>
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
            </div>
            <div>
              <h4 style={{ fontSize: '16px', fontWeight: '600', marginBottom: '16px' }}>Company</h4>
              <button onClick={() => navigate('/about')} style={{ background: 'none', border: 'none', color: 'var(--color-light-grey)', cursor: 'pointer', padding: 0, marginBottom: '8px', display: 'block' }}>About</button>
              <button onClick={() => navigate('/contact')} style={{ background: 'none', border: 'none', color: 'var(--color-light-grey)', cursor: 'pointer', padding: 0, marginBottom: '8px', display: 'block' }}>Contact</button>
            </div>
            <div>
              <h4 style={{ fontSize: '16px', fontWeight: '600', marginBottom: '16px' }}>Legal</h4>
              <button onClick={() => navigate('/privacy')} style={{ background: 'none', border: 'none', color: 'var(--color-light-grey)', cursor: 'pointer', padding: 0, marginBottom: '8px', display: 'block' }}>Privacy Policy</button>
              <button onClick={() => navigate('/terms')} style={{ background: 'none', border: 'none', color: 'var(--color-light-grey)', cursor: 'pointer', padding: 0, marginBottom: '8px', display: 'block' }}>Terms of Service</button>
              <button onClick={() => navigate('/cookie-policy')} style={{ background: 'none', border: 'none', color: 'var(--color-light-grey)', cursor: 'pointer', padding: 0, marginBottom: '8px', display: 'block' }}>Cookie Policy</button>
              <button onClick={() => navigate('/open-source-licenses')} style={{ background: 'none', border: 'none', color: 'var(--color-braun-orange)', cursor: 'pointer', padding: 0, marginBottom: '8px', display: 'block' }}>Open Source Licenses</button>
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
}

export default OpenSourceLicensesPage;




