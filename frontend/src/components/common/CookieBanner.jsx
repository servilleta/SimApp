import React, { useState, useEffect } from 'react';

const COOKIE_CONSENT_KEY = 'cookieConsentGiven_v1';

const bannerStyle = {
  position: 'fixed',
  bottom: 0,
  left: 0,
  right: 0,
  background: 'rgba(0, 0, 0, 0.85)',
  color: '#fff',
  padding: '12px 20px',
  display: 'flex',
  flexWrap: 'wrap',
  justifyContent: 'space-between',
  alignItems: 'center',
  zIndex: 10000,
};

const buttonStyle = {
  background: '#4caf50',
  border: 'none',
  color: '#fff',
  padding: '8px 16px',
  cursor: 'pointer',
  borderRadius: '4px',
  fontWeight: 'bold',
};

/**
 * Simple GDPR-compliant cookie consent banner.
 * Stores consent flag in localStorage and hides afterwards.
 */
function CookieBanner() {
  const [visible, setVisible] = useState(false);

  useEffect(() => {
    const consentGiven = localStorage.getItem(COOKIE_CONSENT_KEY);
    if (!consentGiven) {
      setVisible(true);
    }
  }, []);

  const handleAccept = () => {
    localStorage.setItem(COOKIE_CONSENT_KEY, 'true');
    setVisible(false);
  };

  if (!visible) return null;

  return (
    <div style={bannerStyle} role="dialog" aria-live="polite">
      <span>
        We use cookies to improve your experience. By using this site, you agree to our{' '}
        <a href="/privacy" style={{ color: '#90caf9' }}>Privacy Policy</a> and{' '}
        <a href="/cookie-policy" style={{ color: '#90caf9' }}>Cookie Policy</a>.
      </span>
      <button type="button" onClick={handleAccept} style={buttonStyle}>
        Accept
      </button>
    </div>
  );
}

export default CookieBanner; 