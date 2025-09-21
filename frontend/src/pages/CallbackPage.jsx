import React, { useEffect } from 'react';
import { useAuth0 } from '@auth0/auth0-react';
import { useNavigate } from 'react-router-dom';

export default function CallbackPage() {
  const { isLoading, error, isAuthenticated } = useAuth0();
  const navigate = useNavigate();

  useEffect(() => {
    if (!isLoading) {
      if (isAuthenticated) {
        // Add delay to ensure Auth0 state is fully processed
        console.log('✅ Authentication successful, redirecting to dashboard');
        const timer = setTimeout(() => {
          navigate('/my-dashboard', { replace: true });
        }, 500);
        return () => clearTimeout(timer);
      } else if (error) {
        // Redirect to login page on error
        console.error('Auth0 authentication error:', error);
        const timer = setTimeout(() => {
          navigate('/login', { replace: true });
        }, 1000);
        return () => clearTimeout(timer);
      }
    }
  }, [isLoading, isAuthenticated, error, navigate]);

  if (isLoading) {
    return (
      <div style={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        minHeight: '100vh',
        backgroundColor: 'var(--color-warm-white)'
      }}>
        <div style={{
          padding: '2rem',
          textAlign: 'center',
          backgroundColor: 'var(--color-white)',
          borderRadius: '8px',
          boxShadow: '0 2px 10px rgba(0,0,0,0.07)'
        }}>
          <div style={{
            width: '40px',
            height: '40px',
            border: '4px solid var(--color-border-light)',
            borderTop: '4px solid var(--color-braun-orange)',
            borderRadius: '50%',
            animation: 'spin 1s linear infinite',
            margin: '0 auto 1rem'
          }}></div>
          <h2 style={{ margin: '0 0 0.5rem', color: 'var(--color-charcoal)' }}>Completing Sign In...</h2>
          <p style={{ margin: 0, color: 'var(--color-text-secondary)' }}>Please wait while we finish setting up your account.</p>
        </div>
        <style>{`
          @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
          }
        `}</style>
      </div>
    );
  }

  if (error) {
    return (
      <div style={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        minHeight: '100vh',
        backgroundColor: 'var(--color-warm-white)'
      }}>
        <div style={{
          padding: '2rem',
          textAlign: 'center',
          backgroundColor: 'var(--color-white)',
          borderRadius: '8px',
          boxShadow: '0 2px 10px rgba(0,0,0,0.07)',
          maxWidth: '400px'
        }}>
          <div style={{
            width: '48px',
            height: '48px',
            backgroundColor: 'var(--color-braun-orange)',
            borderRadius: '50%',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            margin: '0 auto 1rem'
          }}>
            <span style={{ color: 'white', fontSize: '24px' }}>⚠️</span>
          </div>
          <h2 style={{ margin: '0 0 0.5rem', color: 'var(--color-charcoal)' }}>Authentication Error</h2>
          <p style={{ margin: '0 0 1rem', color: 'var(--color-text-secondary)' }}>
            There was a problem signing you in. Please try again.
          </p>
          <button
            onClick={() => navigate('/login')}
            style={{
              padding: '0.75rem 1.5rem',
              backgroundColor: 'var(--color-braun-orange)',
              color: 'white',
              border: 'none',
              borderRadius: '6px',
              cursor: 'pointer',
              fontSize: '14px',
              fontWeight: '500'
            }}
          >
            Back to Login
          </button>
        </div>
      </div>
    );
  }

  return null;
} 