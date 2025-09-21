import React from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { useSelector, useDispatch } from 'react-redux';
import { logoutUser } from '../../store/authSlice'; // Correct path to authSlice

const Header = () => {
  const dispatch = useDispatch();
  const navigate = useNavigate();
  const { isAuthenticated, user } = useSelector((state) => state.auth);

  const handleLogout = () => {
    // Clear Redux state
    dispatch(logoutUser());
    // Clear localStorage
    localStorage.removeItem('authToken');
    // Trigger Auth0 logout
    window.dispatchEvent(new CustomEvent('auth0-logout'));
  };

  const headerStyle = {
    background: 'linear-gradient(145deg, #f0f0f0, #e6e6e6)',
    boxShadow: '20px 20px 60px #d1d1d1, -20px -20px 60px #ffffff',
    padding: '1rem 2rem',
    color: '#333',
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    borderBottom: '1px solid rgba(0,0,0,0.1)',
  };

  const navStyle = {
    display: 'flex',
    gap: '1rem',
  };

  const linkStyle = {
    color: '#333',
    textDecoration: 'none',
    padding: '0.75rem 1.25rem',
    borderRadius: '15px',
    background: 'linear-gradient(145deg, #f0f0f0, #e6e6e6)',
    boxShadow: '8px 8px 16px #d1d1d1, -8px -8px 16px #ffffff',
    transition: 'all 0.3s ease',
    fontWeight: '500',
    '&:hover': {
      boxShadow: 'inset 8px 8px 16px #d1d1d1, inset -8px -8px 16px #ffffff',
    }
  };

  const adminLinkStyle = {
    ...linkStyle,
    background: 'linear-gradient(145deg, #ff6b6b, #ee5a5a)',
    color: 'white',
    fontWeight: 'bold',
    boxShadow: '8px 8px 16px #d1d1d1, -8px -8px 16px #ffffff',
  };

  const logoStyle = {
    fontSize: '1.8rem',
    fontWeight: 'bold',
    background: 'linear-gradient(145deg, #667eea, #764ba2)',
    WebkitBackgroundClip: 'text',
    WebkitTextFillColor: 'transparent',
    backgroundClip: 'text',
  };

  const authActionsStyle = {
    display: 'flex',
    gap: '1rem',
    alignItems: 'center',
  };

  const buttonStyle = {
    ...linkStyle,
    background: 'linear-gradient(145deg, #e6e6e6, #f0f0f0)',
    border: 'none',
    cursor: 'pointer',
    color: '#333',
  };

  return (
    <header style={headerStyle}>
      <div style={logoStyle}>
        <Link to="/" style={{ color: 'inherit', textDecoration: 'none' }}>SimApp</Link>
      </div>
      <nav style={navStyle}>
        {isAuthenticated && (
          <>
            <Link to="/" style={linkStyle}>Home</Link>
            <Link to="/simulate" style={linkStyle}>Simulate</Link>
            {(user?.is_admin || (user && (user.username === 'matias redard' || user.email === 'mredard@gmail.com'))) && (
              <Link to="/admin/users" style={adminLinkStyle}>
                👑 Admin
              </Link>
            )}
          </>
        )}
      </nav>
      <div style={authActionsStyle}>
        {isAuthenticated && user ? (
          <>
            <span style={{ color: '#666', fontWeight: '500' }}>Welcome, {user.full_name || user.email}</span>
            <button 
              onClick={handleLogout} 
              style={buttonStyle}
            >
              Logout
            </button>
          </>
        ) : (
          <>
            <Link to="/login" style={linkStyle}>Login</Link>
            <Link to="/register" style={linkStyle}>Register</Link>
          </>
        )}
      </div>
    </header>
  );
};

export default Header;
