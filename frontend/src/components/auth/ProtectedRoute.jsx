import React from 'react';
import { useSelector } from 'react-redux';
import { useAuth0 } from '@auth0/auth0-react';
import { Navigate, useLocation } from 'react-router-dom';

const ProtectedRoute = ({ children }) => {
  const { isAuthenticated: auth0IsAuthenticated, isLoading: auth0Loading } = useAuth0();
  const reduxIsAuthenticated = useSelector(state => state.auth.isAuthenticated);
  const reduxAuthLoading = useSelector(state => state.auth.isLoading);
  const location = useLocation();
  
  // Use Auth0 as the primary source of truth, with Redux as fallback
  const isAuthenticated = auth0IsAuthenticated || reduxIsAuthenticated;
  const isLoading = auth0Loading || reduxAuthLoading;

  if (isLoading) {
    // Show a loading spinner while auth state is being determined
    return (
      <div style={{ 
        display: 'flex', 
        justifyContent: 'center', 
        alignItems: 'center', 
        height: '100vh',
        flexDirection: 'column',
        gap: '16px'
      }}>
        <div style={{
          width: '32px',
          height: '32px',
          border: '3px solid #f3f3f3',
          borderTop: '3px solid #667eea',
          borderRadius: '50%',
          animation: 'spin 1s linear infinite'
        }}></div>
        <p style={{ color: '#666', fontSize: '14px' }}>Authenticating...</p>
        <style>{`
          @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
          }
        `}</style>
      </div>
    );
  }

  if (!isAuthenticated) {
    // Redirect to login page, saving the current location
    return <Navigate to="/login" state={{ from: location }} replace />;
  }

  return children; // Render the protected content
};

export default ProtectedRoute; 