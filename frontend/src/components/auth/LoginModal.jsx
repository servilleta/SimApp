import React, { useState } from 'react';
import { login } from '../../services/authService';
import './LoginModal.css';

const LoginModal = ({ isOpen, onClose, onLoginSuccess }) => {
  const [formData, setFormData] = useState({
    username: '',
    password: '',
  });
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setIsLoading(true);
    setError('');

    try {
      await login(formData.username, formData.password);
      onLoginSuccess();
      onClose();
      setFormData({ username: '', password: '' });
    } catch (err) {
      // Handle error that might be an object with message property
      const errorMessage = typeof err === 'string' 
        ? err 
        : err?.message || 'Login failed';
      setError(errorMessage);
    } finally {
      setIsLoading(false);
    }
  };

  const handleClose = () => {
    setFormData({ username: '', password: '' });
    setError('');
    onClose();
  };

  if (!isOpen) return null;

  return (
    <div className="login-modal-overlay">
      <div className="login-modal">
        <div className="login-modal-header">
          <h3>Login Required</h3>
          <button className="close-btn" onClick={handleClose}>×</button>
        </div>
        
        <div className="login-modal-content">
          <p className="login-message">Please log in to access the simulation features.</p>
          
          <form onSubmit={handleSubmit} className="login-form">
            <div className="form-group">
              <label htmlFor="username">Username</label>
              <input
                type="text"
                id="username"
                name="username"
                value={formData.username}
                onChange={handleInputChange}
                required
                placeholder="Enter your username"
                disabled={isLoading}
              />
            </div>
            
            <div className="form-group">
              <label htmlFor="password">Password</label>
              <input
                type="password"
                id="password"
                name="password"
                value={formData.password}
                onChange={handleInputChange}
                required
                placeholder="Enter your password"
                disabled={isLoading}
              />
            </div>
            
            {error && (
              <div className="error-message">
                {error}
              </div>
            )}
            
            <div className="form-actions">
              <button 
                type="button" 
                className="cancel-btn"
                onClick={handleClose}
                disabled={isLoading}
              >
                Cancel
              </button>
              <button 
                type="submit" 
                className="login-btn"
                disabled={isLoading}
              >
                {isLoading ? 'Logging in...' : 'Login'}
              </button>
            </div>
          </form>
          
          <div className="login-help">
            <p><strong>Demo Credentials:</strong></p>
            <p>Username: <code>matias redard</code></p>
            <p>Password: <code>password</code></p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default LoginModal; 