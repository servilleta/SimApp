import React, { useEffect, useState } from 'react';
import { Link, useOutletContext, useNavigate } from 'react-router-dom';
import { useSelector } from 'react-redux';
import './DashboardPage.css';

// Mock data for dashboard sections
const mockRecentSimulations = [
  { id: 1, name: 'Revenue Forecast Q4', status: 'completed', timestamp: '2 hours ago', progress: 100 },
  { id: 2, name: 'Supply Chain Risk', status: 'running', timestamp: '5 minutes ago', progress: 75 },
  { id: 3, name: 'Market Analysis', status: 'completed', timestamp: '1 day ago', progress: 100 },
  { id: 4, name: 'Cost Optimization', status: 'failed', timestamp: '2 days ago', progress: 45 },
  { id: 5, name: 'Investment Portfolio', status: 'completed', timestamp: '3 days ago', progress: 100 }
];

const mockTemplates = [
  { id: 1, name: 'Financial Risk Model', category: 'Finance', uses: 25, favorite: true },
  { id: 2, name: 'Supply Chain Optimizer', category: 'Operations', uses: 18, favorite: false },
  { id: 3, name: 'Sales Forecasting', category: 'Marketing', uses: 32, favorite: true },
  { id: 4, name: 'Project Timeline', category: 'Management', uses: 12, favorite: false }
];



const DashboardPage = () => {
  const { setSidebarCollapsed } = useOutletContext();
  const navigate = useNavigate();
  const { user } = useSelector((state) => state.auth);
  const [selectedTab, setSelectedTab] = useState('overview');

  useEffect(() => {
    setSidebarCollapsed(false); // Keep sidebar visible for navigation
  }, [setSidebarCollapsed]);

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

        {/* Quick Actions & Navigation */}
        <div className="dashboard-section">
          <h2 className="section-title">Quick Actions & Navigation</h2>
          <div className="quick-actions-grid">
            <button 
              className="action-card primary-action"
              onClick={() => navigate('/simulate')}
            >
              <h3>New Simulation</h3>
              <p>Start a new Monte Carlo simulation with your Excel model</p>
            </button>
            <div className="action-card">
              <h3>Recent Templates</h3>
              <p>Quick access to your most used simulation templates</p>
            </div>
            <div className="action-card">
              <h3>Import Data</h3>
              <p>Upload new datasets or connect to data sources</p>
            </div>
          </div>
        </div>

        {/* Recent Activity */}
        <div className="dashboard-section">
          <h2 className="section-title">Recent Activity</h2>
          <div className="card-braun">
            <div className="activity-list">
              {mockRecentSimulations.map((sim) => (
                <div key={sim.id} className="activity-item">
                  <div className="activity-info">
                    <div className="activity-name">{sim.name}</div>
                    <div className="activity-timestamp">{sim.timestamp}</div>
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
                          style={{ width: `${sim.progress}%` }}
                        ></div>
                      </div>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Simulation Library */}
        <div className="dashboard-section">
          <h2 className="section-title">Simulation Library</h2>
          <div className="templates-grid">
            {mockTemplates.map((template) => (
              <div key={template.id} className="template-card card-braun hover-lift">
                <div className="template-header">
                  <h3>{template.name}</h3>
                  <span className="category-badge">{template.category}</span>
                </div>
                <div className="template-stats">
                  <span>Used {template.uses} times</span>
                  {template.favorite && <span className="favorite-indicator">Favorite</span>}
                </div>
              </div>
            ))}
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
              <h3>Sample Datasets</h3>
              <p>Try example simulations with pre-loaded datasets and templates</p>
              <button className="btn-braun-secondary">Browse Samples</button>
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

export default DashboardPage; 