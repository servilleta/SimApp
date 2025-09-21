import React from 'react';
import { useSelector } from 'react-redux';
import { Navigate } from 'react-router-dom';
import '../../styles/colors.css';
import '../../styles/WebhookBraun.css';

/**
 * AdminRoute component - Protects routes that should only be accessible to admin users
 * 
 * @param {Object} props - Component props
 * @param {React.ReactNode} props.children - Child components to render if user is admin
 * @param {string} props.redirectTo - Path to redirect to if user is not admin (default: '/dashboard')
 * @returns {React.ReactNode} Either the children or a redirect
 */
const AdminRoute = ({ children, redirectTo = '/dashboard' }) => {
  const { isAuthenticated, user } = useSelector((state) => state.auth);

  // If not authenticated, redirect to dashboard (or login will handle it)
  if (!isAuthenticated) {
    return <Navigate to={redirectTo} replace />;
  }

  // If authenticated but user data not loaded yet, show loading
  if (!user) {
    return (
      <div className="page-container">
        <div style={{ 
          display: 'flex', 
          flexDirection: 'column', 
          alignItems: 'center', 
          justifyContent: 'center', 
          padding: '4rem',
          color: 'var(--color-text-secondary)' 
        }}>
          <div className="spinner" style={{ marginBottom: '1rem' }}></div>
          <p>Loading user information...</p>
        </div>
      </div>
    );
  }

  // If user is not admin, show access denied
  if (!user.is_admin) {
    return (
      <div className="page-container">
        <div className="page-header">
          <h1 className="page-title">Access Denied</h1>
          <p className="page-subtitle">You need administrator privileges to access this page</p>
        </div>
        
        <div className="card-braun" style={{ textAlign: 'center', padding: '3rem 2rem' }}>
          <div style={{ 
            width: '64px', 
            height: '64px', 
            borderRadius: '50%', 
            backgroundColor: 'var(--color-error-bg)', 
            color: 'var(--color-error)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            margin: '0 auto 1.5rem',
            fontSize: '1.5rem',
            fontWeight: 'bold'
          }}>
            !
          </div>
          <h3 style={{ color: 'var(--color-error)', marginBottom: '1rem' }}>
            Administrator Access Required
          </h3>
          <p style={{ color: 'var(--color-text-secondary)', marginBottom: '1rem' }}>
            This area contains sensitive administrative tools and is only accessible to system administrators.
            If you believe you should have access to this page, please contact your system administrator.
          </p>
          <div style={{ 
            background: 'var(--color-warm-white)', 
            border: '1px solid var(--color-border-light)', 
            borderRadius: '4px', 
            padding: '1rem', 
            marginBottom: '2rem',
            fontSize: '0.9rem',
            color: 'var(--color-text-tertiary)'
          }}>
            <strong>Current User:</strong> {user.full_name || user.email}<br/>
            <strong>Admin Status:</strong> {user.is_admin ? 'Yes' : 'No'}
          </div>
          <button 
            className="btn-braun-primary"
            onClick={() => window.history.back()}
          >
            Go Back
          </button>
        </div>
      </div>
    );
  }

  // User is admin, render the protected content
  return children;
};

export default AdminRoute;
