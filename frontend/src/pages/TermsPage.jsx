import React, { useEffect, useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import ReactMarkdown from 'react-markdown';

function useMarkdown(url) {
  const [content, setContent] = useState('Loading...');
  useEffect(() => {
    fetch(url)
      .then(r => r.text())
      .then(setContent)
      .catch(() => setContent('Failed to load.'));
  }, [url]);
  return content;
}

function TermsPage() {
  const navigate = useNavigate();
  const termsText = useMarkdown('/legal/TERMS_OF_SERVICE.md');

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
    content: {
      maxWidth: '800px',
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
        <h1 style={styles.title}>Terms of Service</h1>
      </div>

      <div style={styles.content}>
        <div style={styles.markdownContent}>
          <ReactMarkdown>{termsText}</ReactMarkdown>
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
              <button onClick={() => navigate('/terms')} style={{ background: 'none', border: 'none', color: 'var(--color-braun-orange)', cursor: 'pointer', padding: 0, marginBottom: '8px', display: 'block' }}>Terms of Service</button>
              <button onClick={() => navigate('/cookie-policy')} style={{ background: 'none', border: 'none', color: 'var(--color-light-grey)', cursor: 'pointer', padding: 0, marginBottom: '8px', display: 'block' }}>Cookie Policy</button>
              <button onClick={() => navigate('/acceptable-use')} style={{ background: 'none', border: 'none', color: 'var(--color-light-grey)', cursor: 'pointer', padding: 0, marginBottom: '8px', display: 'block' }}>Acceptable Use</button>
              <button onClick={() => navigate('/open-source-licenses')} style={{ background: 'none', border: 'none', color: 'var(--color-light-grey)', cursor: 'pointer', padding: 0, marginBottom: '8px', display: 'block' }}>Open Source Licenses</button>
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

export default TermsPage; 