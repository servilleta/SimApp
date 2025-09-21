import React, { useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth0 } from '@auth0/auth0-react';

const LoginPage = () => {
  const { loginWithRedirect, isAuthenticated, isLoading } = useAuth0();
  const navigate = useNavigate();

  // Redirect to dashboard if already authenticated (with enhanced redirect loop protection)
  useEffect(() => {
    if (isAuthenticated && !isLoading) {
      console.log('🔄 User already authenticated, redirecting to dashboard');
      console.log('🔍 Auth0 Debug - isAuthenticated:', isAuthenticated, 'isLoading:', isLoading);
      
      // Enhanced protection against rapid redirects
      const hasRecentRedirect = sessionStorage.getItem('loginRedirectTime');
      const redirectCount = sessionStorage.getItem('loginRedirectCount') || '0';
      const now = Date.now();
      
      // Check for too many recent redirects
      if (hasRecentRedirect && (now - parseInt(hasRecentRedirect)) < 5000) {
        const count = parseInt(redirectCount) + 1;
        if (count > 3) {
          console.warn('⚠️ Too many redirects detected, staying on login page');
          sessionStorage.setItem('loginRedirectCount', '0');
          return;
        }
        sessionStorage.setItem('loginRedirectCount', count.toString());
      } else {
        // Reset count if enough time has passed
        sessionStorage.setItem('loginRedirectCount', '1');
      }
      
      sessionStorage.setItem('loginRedirectTime', now.toString());
      
      // Use delayed redirect to ensure Auth0 state is stable
      const timer = setTimeout(() => {
        console.log('🔄 Executing redirect to dashboard');
        navigate('/my-dashboard', { replace: true });
      }, 200);
      
      return () => clearTimeout(timer);
    }
  }, [isAuthenticated, isLoading, navigate]);

  // Only redirect to Auth0 login if not authenticated and not loading
  useEffect(() => {
    if (isLoading) {
      console.log('🔄 Auth0 is loading, waiting...');
      return;
    }

    // Add a small delay to prevent immediate redirect loops
    const timer = setTimeout(() => {
      if (!isAuthenticated && !isLoading) {
        console.log('🔐 Starting Auth0 login redirect');
        loginWithRedirect({
          authorizationParams: {
            redirect_uri: `${window.location.origin}/callback`
          }
        });
      }
    }, 500); // 500ms delay to prevent redirect loops

    return () => clearTimeout(timer);
  }, [loginWithRedirect, isAuthenticated, isLoading]);

  // Show different content based on authentication status
  const getLoadingMessage = () => {
    if (isAuthenticated) {
      return "Redirecting to dashboard...";
    } else if (isLoading) {
      return "Checking authentication...";
    } else {
      return "Redirecting to secure login...";
    }
  };

  return (
    <div style={{
      minHeight: '100vh',
      background: 'var(--color-warm-white)',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      fontFamily: 'system-ui, -apple-system, sans-serif'
    }}>
      <div style={{
        color: 'var(--color-charcoal)',
        textAlign: 'center'
      }}>
        <div style={{
          fontSize: '32px',
          fontWeight: 'bold',
          marginBottom: '16px',
          color: 'var(--color-braun-orange)'
        }}>
          SimApp
        </div>
        <div style={{
          fontSize: '16px',
          color: 'var(--color-text-secondary)'
        }}>
          {getLoadingMessage()}
        </div>
        <div style={{
          width: '32px',
          height: '32px',
          border: '3px solid var(--color-border-light)',
          borderTop: '3px solid var(--color-braun-orange)',
          borderRadius: '50%',
          animation: 'spin 1s linear infinite',
          margin: '24px auto'
        }}></div>
        
        {/* Show manual redirect button if stuck */}
        {isAuthenticated && (
          <button 
            onClick={() => navigate('/my-dashboard', { replace: true })}
            style={{
              marginTop: '20px',
              padding: '10px 20px',
              backgroundColor: 'var(--color-braun-orange)',
              color: 'white',
              border: 'none',
              borderRadius: '6px',
              cursor: 'pointer',
              fontSize: '14px'
            }}
          >
            Go to Dashboard
          </button>
        )}
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

export default LoginPage; 