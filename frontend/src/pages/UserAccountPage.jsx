import React, { useState, useEffect } from 'react';
import { useSelector, useDispatch } from 'react-redux';
import { useNavigate } from 'react-router-dom';
import { logoutUser } from '../store/authSlice';
import userAccountService from '../services/userAccountService';
import BillingService from '../services/billingService';
import TrialStatus from '../components/trial/TrialStatus';

const UserAccountPage = () => {
  const dispatch = useDispatch();
  const navigate = useNavigate();
  const { user } = useSelector((state) => state.auth);
  const [activeTab, setActiveTab] = useState('profile');
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState(null);
  const [error, setError] = useState(null);

  // Profile form state
  const [profileForm, setProfileForm] = useState({
    full_name: '',
    email: ''
  });

  // Password form state
  const [passwordForm, setPasswordForm] = useState({
    current_password: '',
    new_password: '',
    confirm_password: ''
  });

  // Privacy settings state
  const [privacySettings, setPrivacySettings] = useState({
    email_notifications: true,
    marketing_emails: false,
    data_sharing: false
  });

  // Current plan state
  const [currentPlan, setCurrentPlan] = useState({
    name: 'Free Trial',
    tier: 'free',
    simulations_per_month: 100,
    max_iterations: 1000,
    max_file_size: '10MB',
    gpu_acceleration: false,
    priority_support: false,
    advanced_analytics: false,
    price: null
  });

  // Available plans for upgrade comparison
  const [availablePlans, setAvailablePlans] = useState([
    {
      name: 'Starter',
      tier: 'starter',
      price: { monthly: 19, annual: 190 },
      description: 'Perfect for individuals and small teams',
      features: [
        '50 simulations per month',
        'Up to 100,000 iterations per simulation',
        '1MB file size limit',
        'Standard GPU acceleration',
        'Email support (48hr response)'
      ],
      limits: {
        simulations_per_month: 50,
        max_iterations: 100000,
        max_file_size: '1MB',
        gpu_acceleration: true,
        gpu_priority: 'standard'
      }
    },
    {
      name: 'Professional',
      tier: 'professional',
      price: { monthly: 49, annual: 490 },
      description: 'Best for growing businesses and advanced analysis',
      features: [
        '100 simulations per month',
        'Up to 1,000,000 iterations per simulation',
        '10MB file size limit',
        'GPU-accelerated processing (10x faster)',
        'Priority support (24hr response)'
      ],
      limits: {
        simulations_per_month: 100,
        max_iterations: 1000000,
        max_file_size: '10MB',
        gpu_acceleration: true,
        gpu_priority: 'high'
      },
      popular: true
    },
    {
      name: 'Enterprise',
      tier: 'enterprise',
      price: { monthly: 149, annual: 1490 },
      description: 'For large organizations with mission-critical needs',
      features: [
        'Unlimited simulations per month',
        'Unlimited iterations & scale',
        'Unlimited file size',
        'Dedicated GPU clusters',
        'Premium support (4hr SLA)',
        'Custom integrations & APIs'
      ],
      limits: {
        simulations_per_month: -1,
        max_iterations: -1,
        max_file_size: 'Unlimited',
        gpu_acceleration: true,
        gpu_priority: 'dedicated'
      }
    }
  ]);

  // Load user data and settings on component mount
  useEffect(() => {
    const loadUserData = async () => {
      if (user) {
        setProfileForm({
          full_name: user.full_name || '',
          email: user.email || ''
        });

        // Load privacy settings
        try {
          const privacy = await userAccountService.getPrivacySettings();
          setPrivacySettings(privacy);
        } catch (error) {
          console.error('Failed to load privacy settings:', error);
        }

        // Load current plan info via Billing Service
        try {
          const [subscription, usage] = await Promise.all([
            BillingService.getCurrentSubscription(),
            BillingService.getUsageInfo()
          ]);
          
          // Determine plan name based on tier and trial status
          let planName = 'Free Plan';
          if (subscription.tier === 'trial') {
            planName = '7-Day Trial (Professional Features)';
          } else if (subscription.tier === 'professional') {
            planName = 'Professional';
          } else if (subscription.tier === 'starter') {
            planName = 'Starter';
          } else if (subscription.tier === 'enterprise') {
            planName = 'Enterprise';
          }
          
          setCurrentPlan({
            name: planName,
            tier: subscription.tier || 'free',
            simulations_per_month: usage.limits.max_iterations || 100,
            max_iterations: usage.limits.max_iterations || 1000,
            max_file_size: usage.limits.file_size_mb ? `${usage.limits.file_size_mb}MB` : '10MB',
            gpu_acceleration: usage.limits.gpu_priority || false,
            priority_support: subscription.tier !== 'free',
            advanced_analytics: subscription.tier === 'professional' || subscription.tier === 'enterprise' || subscription.tier === 'trial',
            price: subscription.price || null,
            usage: usage.current_usage,
            limits: usage.limits
          });
        } catch (error) {
          console.error('Failed to load plan info:', error);
          // Keep default free plan state on error
        }
      }
    };

    loadUserData();
  }, [user]);

  const showMessage = (msg, isError = false) => {
    if (isError) {
      setError(msg);
      setMessage(null);
    } else {
      setMessage(msg);
      setError(null);
    }
    
    setTimeout(() => {
      setMessage(null);
      setError(null);
    }, 5000);
  };

  const handleProfileUpdate = async (e) => {
    e.preventDefault();
    setLoading(true);

    try {
      const updatedUser = await userAccountService.updateUserProfile(profileForm);
      
      // Update Redux store with new user data
      dispatch({
        type: 'auth/loginSuccess',
        payload: {
          user: updatedUser,
          token: localStorage.getItem('authToken')
        }
      });
      
      showMessage('Profile updated successfully!');
    } catch (error) {
      showMessage(error.message, true);
    } finally {
      setLoading(false);
    }
  };

  const handlePasswordChange = async (e) => {
    e.preventDefault();
    
    if (passwordForm.new_password !== passwordForm.confirm_password) {
      showMessage('New passwords do not match', true);
      return;
    }

    // Validate password strength
    const validation = userAccountService.validatePassword(passwordForm.new_password);
    if (!validation.isValid) {
      showMessage(validation.errors.join(', '), true);
      return;
    }

    setLoading(true);

    try {
      await userAccountService.changePassword(passwordForm);
      
      setPasswordForm({
        current_password: '',
        new_password: '',
        confirm_password: ''
      });
      
      showMessage('Password changed successfully!');
    } catch (error) {
      showMessage(error.message, true);
    } finally {
      setLoading(false);
    }
  };

  const handleCloseAllSessions = async () => {
    if (!window.confirm('This will log you out from all devices. Continue?')) {
      return;
    }

    setLoading(true);
    try {
      await userAccountService.revokeAllSessions();
      
      // Logout current session
      dispatch(logoutUser());
      localStorage.removeItem('authToken');
      window.dispatchEvent(new CustomEvent('auth0-logout'));
      showMessage('All sessions closed successfully!');
      navigate('/login');
    } catch (error) {
      showMessage(error.message, true);
    } finally {
      setLoading(false);
    }
  };

  const handleDeleteAccount = async () => {
    const confirmText = 'DELETE';
    const userInput = window.prompt(
      `This action cannot be undone. All your data will be permanently deleted.\n\nType "${confirmText}" to confirm:`
    );

    if (userInput !== confirmText) {
      showMessage('Account deletion cancelled', false);
      return;
    }

    setLoading(true);
    try {
      const result = await userAccountService.deleteUserAccount();
      showMessage(result.message, false);
      
      // After account deletion request, logout the user
      setTimeout(() => {
        dispatch(logoutUser());
        localStorage.removeItem('authToken');
        window.dispatchEvent(new CustomEvent('auth0-logout'));
        navigate('/login');
      }, 3000);
    } catch (error) {
      showMessage(error.message, true);
    } finally {
      setLoading(false);
    }
  };

  const handleUpgradePlan = (planTier = null) => {
    if (planTier) {
      // Direct upgrade to specific plan
      handleSubscribeToPlan(planTier);
    } else {
      // Redirect to pricing page for plan selection
      navigate('/pricing');
    }
  };

  const handleSubscribeToPlan = async (planTier) => {
    try {
      setLoading(true);
      await BillingService.subscribeToPlan(planTier);
    } catch (error) {
      showMessage(error.message, true);
    } finally {
      setLoading(false);
    }
  };

  const handleManageBilling = async () => {
    try {
      setLoading(true);
      await BillingService.openBillingPortal();
    } catch (error) {
      showMessage(error.message, true);
    } finally {
      setLoading(false);
    }
  };

  const handleCancelSubscription = async () => {
    if (!window.confirm('Are you sure you want to cancel your subscription? You will lose access to premium features at the end of your billing period.')) {
      return;
    }

    try {
      setLoading(true);
      const result = await BillingService.cancelSubscription();
      showMessage(result.message || 'Subscription cancelled successfully');
      
      // Refresh plan data
      setTimeout(() => {
        window.location.reload();
      }, 2000);
    } catch (error) {
      showMessage(error.message, true);
    } finally {
      setLoading(false);
    }
  };

  const handleSavePrivacySettings = async () => {
    setLoading(true);
    try {
      await userAccountService.updatePrivacySettings(privacySettings);
      showMessage('Privacy settings saved successfully!');
    } catch (error) {
      showMessage(error.message, true);
    } finally {
      setLoading(false);
    }
  };

  const pageStyle = {
    maxWidth: '1200px',
    margin: '0 auto',
    padding: '2rem 3rem',
    background: 'var(--color-white)',
    minHeight: '100vh',
    '@media (max-width: 1024px)': {
      maxWidth: '900px',
      padding: '2rem'
    },
    '@media (max-width: 768px)': {
      maxWidth: '100%',
      padding: '1rem'
    }
  };

  const cardStyle = {
    background: 'var(--color-white)',
    borderRadius: '12px',
    padding: '2.5rem 3rem',
    boxShadow: 'var(--shadow-sm)',
    marginBottom: '2rem',
    border: '1px solid var(--color-border-light)',
    '@media (max-width: 768px)': {
      padding: '1.5rem'
    }
  };

  const tabContainerStyle = {
    display: 'flex',
    gap: '0.5rem',
    marginBottom: '2rem',
    borderBottom: '1px solid var(--color-border-light)',
    paddingBottom: '1rem'
  };

  const tabStyle = (isActive) => ({
    padding: '0.75rem 1.5rem',
    borderRadius: '6px',
    background: isActive 
      ? 'var(--color-braun-orange)' 
      : 'var(--color-warm-white)',
    color: isActive ? 'var(--color-white)' : 'var(--color-text-secondary)',
    border: isActive ? 'none' : '1px solid var(--color-border-light)',
    cursor: 'pointer',
    fontWeight: '500',
    fontSize: '0.875rem',
    transition: 'all var(--transition-base)',
    boxShadow: 'none'
  });

  const formGroupStyle = {
    marginBottom: '1.5rem'
  };

  const labelStyle = {
    display: 'block',
    marginBottom: '0.5rem',
    fontWeight: '500',
    color: 'var(--color-text-primary)',
    fontSize: '0.875rem'
  };

  const inputStyle = {
    width: '100%',
    maxWidth: '400px',
    padding: '0.75rem 1rem',
    borderRadius: '6px',
    border: '1px solid var(--color-border-light)',
    background: 'var(--color-white)',
    fontSize: '0.875rem',
    color: 'var(--color-text-primary)',
    outline: 'none',
    transition: 'all var(--transition-base)',
    ':focus': {
      borderColor: 'var(--color-braun-orange)',
      boxShadow: 'var(--shadow-focus)'
    }
  };

  const buttonStyle = {
    padding: '0.75rem 1.5rem',
    borderRadius: '6px',
    background: 'var(--color-braun-orange)',
    color: 'var(--color-white)',
    border: 'none',
    cursor: 'pointer',
    fontWeight: '500',
    fontSize: '0.875rem',
    transition: 'all var(--transition-base)',
    marginRight: '1rem',
    ':hover': {
      background: 'var(--color-braun-orange-dark)',
      transform: 'translateY(-1px)',
      boxShadow: 'var(--shadow-md)'
    }
  };

  const dangerButtonStyle = {
    ...buttonStyle,
    background: 'var(--color-error)',
    ':hover': {
      background: '#c62828',
      transform: 'translateY(-1px)',
      boxShadow: 'var(--shadow-md)'
    }
  };

  const messageStyle = {
    padding: '1rem',
    borderRadius: '6px',
    marginBottom: '1rem',
    fontSize: '0.875rem',
    fontWeight: '500',
    border: '1px solid'
  };

  const successMessageStyle = {
    ...messageStyle,
    background: 'var(--color-success-bg)',
    color: 'var(--color-success)',
    borderColor: 'var(--color-success-border)'
  };

  const errorMessageStyle = {
    ...messageStyle,
    background: 'var(--color-error-bg)',
    color: 'var(--color-error)',
    borderColor: 'var(--color-error-border)'
  };

  const planCardStyle = {
    background: 'var(--color-warm-white)',
    borderRadius: '6px',
    padding: '1.5rem',
    marginBottom: '1rem',
    border: '1px solid var(--color-border-light)',
    boxShadow: 'var(--shadow-xs)'
  };

  const featureListStyle = {
    listStyle: 'none',
    padding: 0,
    margin: '1rem 0'
  };

  const featureItemStyle = {
    padding: '0.5rem 0',
    display: 'flex',
    alignItems: 'center',
    gap: '0.5rem',
    fontSize: '0.875rem',
    color: 'var(--color-text-secondary)'
  };

  return (
    <>
      <style>{`
        .plans-grid {
          display: grid;
          grid-template-columns: repeat(3, 1fr);
          gap: 1.5rem;
          max-width: 100%;
          min-height: auto;
        }
        
        @media (max-width: 1024px) {
          .plans-grid {
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
          }
        }
        
        @media (max-width: 768px) {
          .plans-grid {
            grid-template-columns: 1fr;
          }
        }
      `}</style>
      
      <div style={pageStyle}>
        <h1 style={{ 
          color: 'var(--color-text-primary)', 
          textAlign: 'center', 
          marginBottom: '2rem', 
          fontSize: '2rem', 
          fontWeight: '600',
          letterSpacing: '-0.025em'
        }}>
        Account Settings
      </h1>

      <TrialStatus onTrialExpired={() => setActiveTab('plan')} />

      {message && <div style={successMessageStyle}>{message}</div>}
      {error && <div style={errorMessageStyle}>{error}</div>}

      <div style={cardStyle}>
        {/* Tab Navigation */}
        <div style={tabContainerStyle}>
          <button 
            style={tabStyle(activeTab === 'profile')}
            onClick={() => setActiveTab('profile')}
          >
            👤 Profile
          </button>
          <button 
            style={tabStyle(activeTab === 'password')}
            onClick={() => setActiveTab('password')}
          >
            🔒 Password
          </button>
          <button 
            style={tabStyle(activeTab === 'account')}
            onClick={() => setActiveTab('account')}
          >
            ⚙️ Account
          </button>
          <button 
            style={tabStyle(activeTab === 'privacy')}
            onClick={() => setActiveTab('privacy')}
          >
            🛡️ Privacy
          </button>
          <button 
            style={tabStyle(activeTab === 'plan')}
            onClick={() => setActiveTab('plan')}
          >
            💎 Plan
          </button>
        </div>

        {/* Profile Tab */}
        {activeTab === 'profile' && (
          <div>
            <h2 style={{ marginBottom: '1.5rem', color: 'var(--color-text-primary)', fontSize: '1.25rem', fontWeight: '600' }}>Profile Information</h2>
            <form onSubmit={handleProfileUpdate}>
              <div style={formGroupStyle}>
                <label style={labelStyle}>Full Name</label>
                <input
                  type="text"
                  value={profileForm.full_name}
                  onChange={(e) => setProfileForm({...profileForm, full_name: e.target.value})}
                  style={inputStyle}
                  placeholder="Enter your full name"
                  onFocus={(e) => {
                    e.target.style.borderColor = 'var(--color-braun-orange)';
                    e.target.style.boxShadow = 'var(--shadow-focus)';
                  }}
                  onBlur={(e) => {
                    e.target.style.borderColor = 'var(--color-border-light)';
                    e.target.style.boxShadow = 'none';
                  }}
                />
              </div>
              <div style={formGroupStyle}>
                <label style={labelStyle}>Email Address</label>
                <input
                  type="email"
                  value={profileForm.email}
                  onChange={(e) => setProfileForm({...profileForm, email: e.target.value})}
                  style={inputStyle}
                  placeholder="Enter your email address"
                />
              </div>
              <button 
                type="submit" 
                style={buttonStyle} 
                disabled={loading}
                onMouseEnter={(e) => {
                  if (!loading) {
                    e.target.style.background = 'var(--color-braun-orange-dark)';
                    e.target.style.transform = 'translateY(-1px)';
                    e.target.style.boxShadow = 'var(--shadow-md)';
                  }
                }}
                onMouseLeave={(e) => {
                  if (!loading) {
                    e.target.style.background = 'var(--color-braun-orange)';
                    e.target.style.transform = 'translateY(0)';
                    e.target.style.boxShadow = 'none';
                  }
                }}
              >
                {loading ? 'Updating...' : 'Update Profile'}
              </button>
            </form>
          </div>
        )}

        {/* Password Tab */}
        {activeTab === 'password' && (
          <div>
            <h2 style={{ marginBottom: '1.5rem', color: 'var(--color-text-primary)', fontSize: '1.25rem', fontWeight: '600' }}>Change Password</h2>
            <form onSubmit={handlePasswordChange}>
              <div style={formGroupStyle}>
                <label style={labelStyle}>Current Password</label>
                <input
                  type="password"
                  value={passwordForm.current_password}
                  onChange={(e) => setPasswordForm({...passwordForm, current_password: e.target.value})}
                  style={inputStyle}
                  placeholder="Enter current password"
                />
              </div>
              <div style={formGroupStyle}>
                <label style={labelStyle}>New Password</label>
                <input
                  type="password"
                  value={passwordForm.new_password}
                  onChange={(e) => setPasswordForm({...passwordForm, new_password: e.target.value})}
                  style={inputStyle}
                  placeholder="Enter new password (min 8 characters)"
                />
              </div>
              <div style={formGroupStyle}>
                <label style={labelStyle}>Confirm New Password</label>
                <input
                  type="password"
                  value={passwordForm.confirm_password}
                  onChange={(e) => setPasswordForm({...passwordForm, confirm_password: e.target.value})}
                  style={inputStyle}
                  placeholder="Confirm new password"
                />
              </div>
              <button type="submit" style={buttonStyle} disabled={loading}>
                {loading ? 'Changing...' : 'Change Password'}
              </button>
            </form>
          </div>
        )}

        {/* Account Tab */}
        {activeTab === 'account' && (
          <div>
            <h2 style={{ marginBottom: '1.5rem', color: 'var(--color-text-primary)', fontSize: '1.25rem', fontWeight: '600' }}>Account Management</h2>
            
            <div style={formGroupStyle}>
              <h3 style={{ marginBottom: '1rem', color: 'var(--color-text-primary)', fontSize: '1rem', fontWeight: '500' }}>Session Management</h3>
              <p style={{ marginBottom: '1rem', color: 'var(--color-text-secondary)', fontSize: '0.875rem' }}>
                Close all active sessions on all devices. You will need to log in again.
              </p>
              <button onClick={handleCloseAllSessions} style={buttonStyle} disabled={loading}>
                {loading ? 'Closing...' : 'Close All Sessions'}
              </button>
            </div>

            <div style={{...formGroupStyle, marginTop: '3rem', padding: '2rem', borderRadius: '6px', background: 'var(--color-error-bg)', border: '1px solid var(--color-error-border)'}}>
              <h3 style={{ marginBottom: '1rem', color: 'var(--color-error)', fontSize: '1rem', fontWeight: '500' }}>Danger Zone</h3>
              <p style={{ marginBottom: '1rem', color: 'var(--color-error)', fontSize: '0.875rem' }}>
                <strong>Delete Account:</strong> This action cannot be undone. All your data, simulations, and files will be permanently deleted.
              </p>
              <button onClick={handleDeleteAccount} style={dangerButtonStyle} disabled={loading}>
                {loading ? 'Processing...' : 'Delete Account'}
              </button>
            </div>
          </div>
        )}

        {/* Privacy Tab */}
        {activeTab === 'privacy' && (
          <div>
            <h2 style={{ marginBottom: '1.5rem', color: 'var(--color-text-primary)', fontSize: '1.25rem', fontWeight: '600' }}>Privacy Settings</h2>
            
            <div style={formGroupStyle}>
              <h3 style={{ marginBottom: '1rem', color: 'var(--color-text-primary)', fontSize: '1rem', fontWeight: '500' }}>Email Preferences</h3>
              
              <div style={{ marginBottom: '1rem' }}>
                <label style={{ display: 'flex', alignItems: 'center', cursor: 'pointer' }}>
                  <input
                    type="checkbox"
                    checked={privacySettings.email_notifications}
                    onChange={(e) => setPrivacySettings({...privacySettings, email_notifications: e.target.checked})}
                    style={{ marginRight: '0.5rem' }}
                  />
                  <span>Email notifications for simulation results</span>
                </label>
              </div>
              
              <div style={{ marginBottom: '1rem' }}>
                <label style={{ display: 'flex', alignItems: 'center', cursor: 'pointer' }}>
                  <input
                    type="checkbox"
                    checked={privacySettings.marketing_emails}
                    onChange={(e) => setPrivacySettings({...privacySettings, marketing_emails: e.target.checked})}
                    style={{ marginRight: '0.5rem' }}
                  />
                  <span>Marketing emails and product updates</span>
                </label>
              </div>
            </div>

            <div style={formGroupStyle}>
              <h3 style={{ marginBottom: '1rem', color: 'var(--color-text-primary)', fontSize: '1rem', fontWeight: '500' }}>Data Sharing</h3>
              
              <div style={{ marginBottom: '1rem' }}>
                <label style={{ display: 'flex', alignItems: 'center', cursor: 'pointer' }}>
                  <input
                    type="checkbox"
                    checked={privacySettings.data_sharing}
                    onChange={(e) => setPrivacySettings({...privacySettings, data_sharing: e.target.checked})}
                    style={{ marginRight: '0.5rem' }}
                  />
                  <span>Share anonymized usage data to improve the platform</span>
                </label>
              </div>
            </div>

            <button onClick={handleSavePrivacySettings} style={buttonStyle} disabled={loading}>
              {loading ? 'Saving...' : 'Save Privacy Settings'}
            </button>
          </div>
        )}

        {/* Plan Tab */}
        {activeTab === 'plan' && (
          <div>
            <h2 style={{ marginBottom: '1.5rem', color: 'var(--color-text-primary)', fontSize: '1.25rem', fontWeight: '600' }}>Subscription Plan</h2>
            
            {/* Current Plan Card */}
            <div style={planCardStyle}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
                <h3 style={{ color: 'var(--color-text-primary)', fontSize: '1.125rem', fontWeight: '600', display: 'flex', alignItems: 'center', gap: '0.5rem', margin: 0 }}>
                  <span>📦</span> {currentPlan.name}
                </h3>
                <div style={{
                  padding: '4px 12px',
                  borderRadius: '16px',
                  fontSize: '12px',
                  fontWeight: '600',
                  textTransform: 'uppercase',
                  letterSpacing: '0.5px',
                  background: currentPlan.tier === 'free' ? 'var(--color-light-grey)' : 'var(--color-braun-orange)',
                  color: currentPlan.tier === 'free' ? 'var(--color-dark-grey)' : 'var(--color-white)'
                }}>
                  {currentPlan.tier}
                </div>
              </div>
              
              <ul style={featureListStyle}>
                <li style={featureItemStyle}>
                  <span>🔄</span> {currentPlan.simulations_per_month === -1 ? 'Unlimited' : currentPlan.simulations_per_month.toLocaleString()} simulations per month
                </li>
                <li style={featureItemStyle}>
                  <span>⚡</span> {currentPlan.max_iterations === -1 ? 'Unlimited' : currentPlan.max_iterations.toLocaleString()} iterations per simulation
                </li>
                <li style={featureItemStyle}>
                  <span>📁</span> Max file size: {currentPlan.max_file_size}
                </li>
                <li style={featureItemStyle}>
                  <span>{currentPlan.gpu_acceleration ? '✅' : '❌'}</span> 
                  GPU acceleration
                </li>
                <li style={featureItemStyle}>
                  <span>{currentPlan.priority_support ? '✅' : '❌'}</span> 
                  Priority support
                </li>
                <li style={featureItemStyle}>
                  <span>{currentPlan.advanced_analytics ? '✅' : '❌'}</span> 
                  Advanced analytics
                </li>
                {currentPlan.price && (
                  <li style={featureItemStyle}>
                    <span>💰</span> {currentPlan.price}
                  </li>
                )}
              </ul>
              
              <div style={{ marginTop: '1.5rem', display: 'flex', gap: '12px', flexWrap: 'wrap' }}>
                {currentPlan.tier === 'free' ? (
                  <button onClick={() => handleUpgradePlan()} style={buttonStyle}>
                    Upgrade Plan
                  </button>
                ) : (
                  <>
                    <button onClick={handleManageBilling} style={buttonStyle}>
                      Manage Billing
                    </button>
                    <button onClick={() => handleUpgradePlan()} style={{...buttonStyle, background: 'var(--color-white)', color: 'var(--color-dark-grey)', border: '1px solid var(--color-border-light)'}}>
                      Change Plan
                    </button>
                    <button onClick={handleCancelSubscription} style={{...buttonStyle, background: 'var(--color-error)', marginLeft: 'auto'}}>
                      Cancel Subscription
                    </button>
                  </>
                )}
              </div>
            </div>

            {/* Usage Statistics */}
            {currentPlan.usage && (
              <div style={{ marginTop: '2rem' }}>
                <h3 style={{ marginBottom: '1rem', color: 'var(--color-text-primary)', fontSize: '1.125rem', fontWeight: '600' }}>Usage This Month</h3>
                <div style={planCardStyle}>
                  <div style={{ display: 'grid', gap: '1rem' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <span>Iterations used:</span>
                      <span style={{ fontWeight: 'bold' }}>
                        {currentPlan.usage.total_iterations?.toLocaleString() || 0} / {currentPlan.limits?.max_iterations === -1 ? 'Unlimited' : currentPlan.limits?.max_iterations?.toLocaleString() || 'N/A'}
                      </span>
                    </div>
                    {currentPlan.limits?.max_iterations !== -1 && (
                      <div style={{ width: '100%', height: '8px', background: '#e6e6e6', borderRadius: '4px' }}>
                        <div style={{ 
                          width: `${Math.min((currentPlan.usage.total_iterations || 0) / (currentPlan.limits?.max_iterations || 1) * 100, 100)}%`, 
                          height: '100%', 
                          background: 'var(--color-braun-orange)', 
                          borderRadius: '4px',
                          transition: 'all 0.3s ease'
                        }}></div>
                      </div>
                    )}
                    
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <span>Concurrent simulations:</span>
                      <span style={{ fontWeight: 'bold' }}>
                        {currentPlan.usage.concurrent_simulations || 0} / {currentPlan.limits?.concurrent_simulations === -1 ? 'Unlimited' : currentPlan.limits?.concurrent_simulations || 'N/A'}
                      </span>
                    </div>
                    
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <span>API calls:</span>
                      <span style={{ fontWeight: 'bold' }}>
                        {currentPlan.usage.api_calls?.toLocaleString() || 0} / {currentPlan.limits?.api_calls_per_month === -1 ? 'Unlimited' : currentPlan.limits?.api_calls_per_month === 0 ? 'Not included' : currentPlan.limits?.api_calls_per_month?.toLocaleString() || 'N/A'}
                      </span>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Plan Comparison */}
            <div style={{ marginTop: '2rem' }}>
              <h3 style={{ marginBottom: '1rem', color: 'var(--color-text-primary)', fontSize: '1.125rem', fontWeight: '600' }}>Available Plans</h3>
              <div className="plans-grid">
                {availablePlans.map((plan, index) => (
                  <div key={index} style={{
                    ...planCardStyle,
                    border: plan.popular ? '2px solid var(--color-braun-orange)' : '1px solid var(--color-border-light)',
                    position: 'relative'
                  }}>
                    {plan.popular && (
                      <div style={{
                        position: 'absolute',
                        top: '-12px',
                        left: '50%',
                        transform: 'translateX(-50%)',
                        background: 'var(--color-braun-orange)',
                        color: 'white',
                        padding: '4px 16px',
                        borderRadius: '12px',
                        fontSize: '12px',
                        fontWeight: '600'
                      }}>
                        Most Popular
                      </div>
                    )}
                    
                    <h4 style={{ fontSize: '1.1rem', fontWeight: '600', marginBottom: '0.5rem', color: 'var(--color-text-primary)' }}>
                      {plan.name}
                    </h4>
                    <p style={{ fontSize: '0.875rem', color: 'var(--color-text-secondary)', marginBottom: '1rem' }}>
                      {plan.description}
                    </p>
                    
                    <div style={{ marginBottom: '1rem' }}>
                      <span style={{ fontSize: '2rem', fontWeight: 'bold', color: 'var(--color-text-primary)' }}>
                        ${plan.price.monthly}
                      </span>
                      <span style={{ fontSize: '0.875rem', color: 'var(--color-text-secondary)' }}>/month</span>
                    </div>
                    
                    <ul style={{...featureListStyle, fontSize: '0.8rem'}}>
                      {plan.features.slice(0, 3).map((feature, idx) => (
                        <li key={idx} style={{...featureItemStyle, fontSize: '0.8rem'}}>
                          <span style={{ color: 'var(--color-success)' }}>✓</span> {feature}
                        </li>
                      ))}
                    </ul>
                    
                    {currentPlan.tier !== plan.tier && (
                      <button 
                        onClick={() => handleSubscribeToPlan(plan.tier)}
                        style={{
                          ...buttonStyle,
                          width: '100%',
                          marginTop: '1rem',
                          background: plan.popular ? 'var(--color-braun-orange)' : 'var(--color-white)',
                          color: plan.popular ? 'white' : 'var(--color-charcoal)',
                          border: plan.popular ? 'none' : '2px solid var(--color-border-light)'
                        }}
                        disabled={loading}
                      >
                        {loading ? 'Processing...' : `Upgrade to ${plan.name}`}
                      </button>
                    )}
                    
                    {currentPlan.tier === plan.tier && (
                      <div style={{
                        padding: '8px 16px',
                        background: 'var(--color-success-bg)',
                        color: 'var(--color-success)',
                        borderRadius: '6px',
                        textAlign: 'center',
                        fontSize: '0.875rem',
                        fontWeight: '500',
                        marginTop: '1rem'
                      }}>
                        Current Plan
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
    </>
  );
};

export default UserAccountPage;
