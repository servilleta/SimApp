import React, { useEffect, useState } from 'react';
import { useSelector } from 'react-redux';
import { useNavigate } from 'react-router-dom';
import './DashboardPage.css';
import TrialStatus from '../components/trial/TrialStatus';
import BillingService from '../services/billingService';
import apiClient from '../services/api';




const UserDashboardPage = () => {
  const navigate = useNavigate();
  const { user } = useSelector((state) => state.auth);
  const [selectedTab, setSelectedTab] = useState('overview');
  const [recentSimulations, setRecentSimulations] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  
  // Usage data state
  const [usageData, setUsageData] = useState(null);
  const [subscriptionData, setSubscriptionData] = useState(null);
  const [usageLoading, setUsageLoading] = useState(true);
  const [usageError, setUsageError] = useState(null);

  // Temporary function to fetch simulation history (inline to fix import issue)
  const getUserSimulationHistory = async (limit = 10) => {
    try {
      console.log('🔍 Fetching user simulation history...');
      const response = await apiClient.get(`/simulation/history?limit=${limit}`);
      console.log('📊 User simulation history received:', response.data);
      return response.data;
    } catch (error) {
      console.error('🚨 Failed to fetch user simulation history:', error);
      if (error.response && error.response.data) {
        throw new Error(error.response.data.detail || 'Failed to fetch simulation history.');
      }
      throw new Error(error.message || 'Network error or failed to fetch simulation history.');
    }
  };

  // Fetch user simulation history and usage data on component mount
  useEffect(() => {
    const fetchDashboardData = async () => {
      // Fetch simulation history
      try {
        setLoading(true);
        setError(null);
        const history = await getUserSimulationHistory(5); // Get 5 most recent
        setRecentSimulations(history);
      } catch (err) {
        console.error('Failed to load simulation history:', err);
        setError(err.message);
        setRecentSimulations([]); // Fallback to empty array
      } finally {
        setLoading(false);
      }
      
      // Fetch usage data
      try {
        setUsageLoading(true);
        setUsageError(null);
        const [subscription, usage] = await Promise.all([
          BillingService.getCurrentSubscription(),
          BillingService.getUsageInfo()
        ]);
        setSubscriptionData(subscription);
        setUsageData(usage);
      } catch (err) {
        console.error('Failed to load usage data:', err);
        setUsageError(err.message);
      } finally {
        setUsageLoading(false);
      }
    };

    fetchDashboardData();
  }, []);

  // Helper function to format timestamp
  const formatTimestamp = (dateString) => {
    if (!dateString) return 'Unknown';
    
    const date = new Date(dateString);
    const now = new Date();
    const diffInMinutes = Math.floor((now - date) / (1000 * 60));
    
    if (diffInMinutes < 1) return 'Just now';
    if (diffInMinutes < 60) return `${diffInMinutes} minute${diffInMinutes === 1 ? '' : 's'} ago`;
    
    const diffInHours = Math.floor(diffInMinutes / 60);
    if (diffInHours < 24) return `${diffInHours} hour${diffInHours === 1 ? '' : 's'} ago`;
    
    const diffInDays = Math.floor(diffInHours / 24);
    return `${diffInDays} day${diffInDays === 1 ? '' : 's'} ago`;
  };

  // Helper function to format simulation name from filename
  const formatSimulationName = (filename, simulationId) => {
    if (filename) {
      // Remove file extension and clean up the name
      return filename.replace(/\.[^/.]+$/, "").replace(/_/g, ' ');
    }
    return `Simulation ${simulationId?.slice(-8) || 'Unknown'}`;
  };
  
  // Helper function to calculate usage percentage
  const getUsagePercentage = (used, limit) => {
    if (limit === -1) return 0; // Unlimited
    if (limit === 0) return 100; // No limit means 100% if any usage
    return Math.min((used / limit) * 100, 100);
  };
  
  // Helper function to format plan name
  const formatPlanName = (tier) => {
    switch(tier) {
      case 'trial': return '7-Day Trial (Professional Features)';
      case 'professional': return 'Professional';
      case 'starter': return 'Starter';
      case 'enterprise': return 'Enterprise';
      case 'free': return 'Free Plan';
      default: return 'Unknown Plan';
    }
  };

  const getStatusColor = (status) => {
    switch (status) {
      case 'completed': return 'var(--color-success)';
      case 'running': return 'var(--color-braun-orange)';
      case 'failed': return 'var(--color-error)';
      default: return 'var(--color-medium-grey)';
    }
  };

  const getStatusBg = (status) => {
    switch (status) {
      case 'completed': return 'var(--color-success-bg)';
      case 'running': return 'rgba(255, 107, 53, 0.1)';
      case 'failed': return 'var(--color-error-bg)';
      default: return 'var(--color-light-grey)';
    }
  };

  return (
    <div className="page-container">
      <div className="dashboard-container">
        
        {/* Header */}
        <div className="dashboard-header">
          <div className="welcome-section">
            <h1 className="text-display">Your Dashboard</h1>
            <p className="text-subheadline">Welcome back, {user?.full_name || user?.username || 'User'}! Here's your simulation activity.</p>
          </div>
        </div>
        
        <TrialStatus onTrialExpired={() => navigate('/account')} />


        {/* Recent Activity */}
        <div className="dashboard-section">
          <h2 className="section-title">Recent Activity</h2>
          <div className="card-braun">
            <div className="activity-list">
              {loading ? (
                <div className="activity-item">
                  <div className="activity-info">
                    <div className="activity-name">Loading simulation history...</div>
                    <div className="activity-timestamp">Please wait</div>
                  </div>
                </div>
              ) : error ? (
                <div className="activity-item">
                  <div className="activity-info">
                    <div className="activity-name" style={{ color: 'var(--color-error)' }}>Failed to load simulation history</div>
                    <div className="activity-timestamp">{error}</div>
                  </div>
                </div>
              ) : recentSimulations.length === 0 ? (
                <div className="activity-item">
                  <div className="activity-info">
                    <div className="activity-name">No simulations yet</div>
                    <div className="activity-timestamp">Start your first simulation to see activity here</div>
                  </div>
                </div>
              ) : (
                recentSimulations.map((sim) => (
                  <div key={sim.simulation_id} className="activity-item">
                    <div className="activity-info">
                      <div className="activity-name">{formatSimulationName(sim.file_name, sim.simulation_id)}</div>
                      <div className="activity-timestamp">{formatTimestamp(sim.created_at)}</div>
                    </div>
                    <div className="activity-status">
                      <span 
                        className="status-badge"
                        style={{ 
                          color: getStatusColor(sim.status),
                          backgroundColor: getStatusBg(sim.status)
                        }}
                      >
                        {sim.status}
                      </span>
                      {sim.status === 'running' && (
                        <div className="progress-bar">
                          <div 
                            className="progress-fill"
                            style={{ width: '75%' }}
                          ></div>
                        </div>
                      )}
                    </div>
                  </div>
                ))
              )}
            </div>
          </div>
        </div>

        {/* Usage Statistics */}
        <div className="dashboard-section">
          <h2 className="section-title">Usage This Month</h2>
          <div className="card-braun">
            {usageLoading ? (
              <div style={{ padding: '2rem', textAlign: 'center' }}>
                <div>Loading usage data...</div>
              </div>
            ) : usageError ? (
              <div style={{ padding: '2rem', textAlign: 'center', color: 'var(--color-error)' }}>
                <div>Failed to load usage data</div>
                <div style={{ fontSize: '0.875rem', marginTop: '0.5rem' }}>{usageError}</div>
              </div>
            ) : usageData && subscriptionData ? (
              <div style={{ padding: '1.5rem' }}>
                {/* Plan Information */}
                <div style={{ marginBottom: '1.5rem', paddingBottom: '1rem', borderBottom: '1px solid var(--color-border)' }}>
                  <h3 style={{ margin: '0 0 0.5rem 0', color: 'var(--color-text-primary)', fontSize: '1.125rem' }}>
                    {formatPlanName(subscriptionData.tier)}
                  </h3>
                  <div style={{ fontSize: '0.875rem', color: 'var(--color-text-secondary)' }}>
                    Status: {subscriptionData.status}
                  </div>
                </div>
                
                {/* Usage Metrics */}
                <div style={{ display: 'grid', gap: '1.5rem' }}>
                  {/* Iterations Usage */}
                  <div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.5rem' }}>
                      <span style={{ fontWeight: '500' }}>Iterations used:</span>
                      <span style={{ fontWeight: 'bold' }}>
                        {usageData.current_usage.total_iterations?.toLocaleString() || 0} / {usageData.limits?.max_iterations === -1 ? 'Unlimited' : usageData.limits?.max_iterations?.toLocaleString() || 'N/A'}
                      </span>
                    </div>
                    {usageData.limits?.max_iterations !== -1 && (
                      <div style={{ width: '100%', height: '8px', background: 'var(--color-light-grey)', borderRadius: '4px' }}>
                        <div style={{ 
                          width: `${getUsagePercentage(usageData.current_usage.total_iterations || 0, usageData.limits?.max_iterations || 1)}%`, 
                          height: '100%', 
                          background: 'var(--color-braun-orange)', 
                          borderRadius: '4px',
                          transition: 'all 0.3s ease'
                        }}></div>
                      </div>
                    )}
                  </div>
                  
                  {/* Simulations Run */}
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <span style={{ fontWeight: '500' }}>Simulations this month:</span>
                    <span style={{ fontWeight: 'bold' }}>
                      {usageData.current_usage.simulations_run || 0}
                    </span>
                  </div>
                  
                  {/* API Calls */}
                  {usageData.limits?.api_calls_per_month !== undefined && (
                    <div>
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.5rem' }}>
                        <span style={{ fontWeight: '500' }}>API calls:</span>
                        <span style={{ fontWeight: 'bold' }}>
                          {usageData.current_usage.api_calls?.toLocaleString() || 0} / {usageData.limits?.api_calls_per_month === -1 ? 'Unlimited' : usageData.limits?.api_calls_per_month === 0 ? 'Not included' : usageData.limits?.api_calls_per_month?.toLocaleString() || 'N/A'}
                        </span>
                      </div>
                      {usageData.limits?.api_calls_per_month > 0 && usageData.limits?.api_calls_per_month !== -1 && (
                        <div style={{ width: '100%', height: '8px', background: 'var(--color-light-grey)', borderRadius: '4px' }}>
                          <div style={{ 
                            width: `${getUsagePercentage(usageData.current_usage.api_calls || 0, usageData.limits?.api_calls_per_month || 1)}%`, 
                            height: '100%', 
                            background: 'var(--color-braun-orange)', 
                            borderRadius: '4px',
                            transition: 'all 0.3s ease'
                          }}></div>
                        </div>
                      )}
                    </div>
                  )}
                  
                  {/* File Storage */}
                  {usageData.current_usage.total_file_size_mb !== undefined && (
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <span style={{ fontWeight: '500' }}>Files uploaded:</span>
                      <span style={{ fontWeight: 'bold' }}>
                        {usageData.current_usage.files_uploaded || 0} files ({(usageData.current_usage.total_file_size_mb || 0).toFixed(1)} MB)
                      </span>
                    </div>
                  )}
                </div>
                
                {/* Period Information */}
                {usageData.current_usage.period_start && (
                  <div style={{ marginTop: '1.5rem', paddingTop: '1rem', borderTop: '1px solid var(--color-border)', fontSize: '0.875rem', color: 'var(--color-text-secondary)' }}>
                    Billing period: {new Date(usageData.current_usage.period_start).toLocaleDateString()} - {usageData.current_usage.period_end ? new Date(usageData.current_usage.period_end).toLocaleDateString() : 'End of month'}
                  </div>
                )}
              </div>
            ) : (
              <div style={{ padding: '2rem', textAlign: 'center' }}>
                <div>No usage data available</div>
              </div>
            )}
          </div>
        </div>


        {/* Getting Started Section */}
        <div className="dashboard-section">
          <h2 className="section-title">Getting Started</h2>
          <div className="getting-started-grid">
            <div className="guide-card card-braun hover-lift">
              <h3>Tutorial & Onboarding</h3>
              <p>Learn the basics of Monte Carlo simulation with interactive tutorials</p>
              <button className="btn-braun-secondary">Start Tutorial</button>
            </div>
            <div className="guide-card card-braun hover-lift">
              <h3>Documentation</h3>
              <p>Comprehensive guides and API documentation for advanced users</p>
              <button className="btn-braun-secondary">View Docs</button>
            </div>
          </div>
        </div>

      </div>
    </div>
  );
};

export default UserDashboardPage;