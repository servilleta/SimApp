import React, { useEffect, useState } from 'react';
import axios from 'axios';
import { getToken } from '../services/authService';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const AdminSupportPage = () => {
  const [supportData, setSupportData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [showCreateTicket, setShowCreateTicket] = useState(false);
  const [ticketForm, setTicketForm] = useState({
    title: '',
    description: '',
    priority: 'medium'
  });

  const fetchSupportData = async () => {
    setLoading(true);
    setError('');
    try {
      // Use mock data for support system demo
      setSupportData({
        total_tickets: 12,
        open_tickets: 3,
        resolved_tickets: 9,
        avg_resolution_time_hours: 4.2,
        sla_compliance_percent: 98.5,
        engineers_available: 3,
        tickets_by_priority: {
          critical: 1,
          high: 2,
          medium: 5,
          low: 4
        },
        tickets_by_tier: {
          enterprise: 3,
          professional: 6,
          standard: 3
        }
      });
    } catch (err) {
      console.error('Error fetching support data:', err);
      setError(err.response?.data?.detail || 'Failed to fetch support data');
    }
    setLoading(false);
  };

  const createSupportTicket = async (e) => {
    e.preventDefault();
    setLoading(true);
    try {
      const response = await axios.post(
        `${API_BASE_URL}/enterprise/support/tickets`,
        ticketForm,
        {
          headers: { 
            'Authorization': `Bearer ${getToken()}`,
            'Content-Type': 'application/json'
          }
        }
      );
      
      alert(`Support ticket created successfully!\nTicket ID: ${response.data.ticket_id}\nSLA: ${response.data.sla_hours} hours`);
      setTicketForm({ title: '', description: '', priority: 'medium' });
      setShowCreateTicket(false);
      await fetchSupportData();
    } catch (err) {
      console.error('Error creating support ticket:', err);
      alert(err.response?.data?.message || 'Failed to create support ticket');
    }
    setLoading(false);
  };

  useEffect(() => {
    fetchSupportData();
  }, []);

  const SupportMetricCard = ({ title, value, subtitle, icon, color = 'var(--color-braun-orange)' }) => (
    <div className="card hover-lift" style={{ 
      padding: '1.5rem',
      background: 'linear-gradient(135deg, var(--color-white) 0%, var(--color-warm-white) 100%)',
      border: '1px solid var(--color-border-light)',
      borderRadius: '12px',
      textAlign: 'center',
      position: 'relative',
      overflow: 'hidden'
    }}>
      {/* Background decoration */}
      <div style={{
        position: 'absolute',
        top: '-20px',
        right: '-20px',
        width: '80px',
        height: '80px',
        background: `linear-gradient(135deg, ${color}10, ${color}05)`,
        borderRadius: '50%'
      }} />
      
      <div style={{ position: 'relative', zIndex: 1 }}>
        <div style={{ fontSize: '2rem', marginBottom: '0.75rem' }}>{icon}</div>
        <div style={{ 
          fontSize: '2.5rem', 
          fontWeight: '700',
          color: color,
          marginBottom: '0.5rem'
        }}>
          {value}
        </div>
        <h3 style={{ 
          color: 'var(--color-charcoal)', 
          margin: '0 0 0.25rem 0',
          fontSize: '1rem',
          fontWeight: '600'
        }}>
          {title}
        </h3>
        {subtitle && (
          <p style={{ 
            color: 'var(--color-medium-grey)', 
            margin: 0,
            fontSize: '0.875rem'
          }}>
            {subtitle}
          </p>
        )}
      </div>
    </div>
  );

  const SLACard = ({ tier, critical, high, medium, low }) => (
    <div className="card" style={{ 
      padding: '1.5rem',
      border: '1px solid var(--color-border-light)',
      borderRadius: '12px'
    }}>
      <h3 style={{ 
        color: 'var(--color-charcoal)', 
        marginBottom: '1rem',
        fontSize: '1.1rem',
        fontWeight: '600',
        textTransform: 'capitalize'
      }}>
        {tier} Tier
      </h3>
      
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '0.75rem' }}>
        <div style={{ textAlign: 'center' }}>
          <div style={{ 
            fontSize: '1.5rem', 
            fontWeight: '700', 
            color: 'var(--color-error)',
            marginBottom: '0.25rem'
          }}>
            {critical}h
          </div>
          <div style={{ fontSize: '0.75rem', color: 'var(--color-medium-grey)' }}>
            Critical
          </div>
        </div>
        
        <div style={{ textAlign: 'center' }}>
          <div style={{ 
            fontSize: '1.5rem', 
            fontWeight: '700', 
            color: 'var(--color-warning)',
            marginBottom: '0.25rem'
          }}>
            {high}h
          </div>
          <div style={{ fontSize: '0.75rem', color: 'var(--color-medium-grey)' }}>
            High
          </div>
        </div>
        
        <div style={{ textAlign: 'center' }}>
          <div style={{ 
            fontSize: '1.5rem', 
            fontWeight: '700', 
            color: 'var(--color-info)',
            marginBottom: '0.25rem'
          }}>
            {medium}h
          </div>
          <div style={{ fontSize: '0.75rem', color: 'var(--color-medium-grey)' }}>
            Medium
          </div>
        </div>
        
        <div style={{ textAlign: 'center' }}>
          <div style={{ 
            fontSize: '1.5rem', 
            fontWeight: '700', 
            color: 'var(--color-success)',
            marginBottom: '0.25rem'
          }}>
            {low}h
          </div>
          <div style={{ fontSize: '0.75rem', color: 'var(--color-medium-grey)' }}>
            Low
          </div>
        </div>
      </div>
    </div>
  );

  if (loading && !supportData) {
    return (
      <div className="page-container">
        <div className="page-header">
          <h1 className="page-title">🎯 Enterprise Support</h1>
          <p className="page-subtitle">
            AI-powered support system with SLA-based response guarantees
          </p>
        </div>
        <div className="card" style={{ textAlign: 'center', padding: '3rem' }}>
          <div style={{ 
            color: 'var(--color-medium-grey)',
            fontSize: '1.1rem'
          }}>
            Loading support data...
          </div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="page-container">
        <div className="page-header">
          <h1 className="page-title">🎯 Enterprise Support</h1>
          <p className="page-subtitle">
            AI-powered support system with SLA-based response guarantees
          </p>
        </div>
        <div className="card error-card">
          <strong>Error:</strong> {error}
        </div>
      </div>
    );
  }

  return (
    <div className="page-container">
      {/* Header Section */}
      <div className="page-header">
        <h1 className="page-title">🎯 Enterprise Support</h1>
        <p className="page-subtitle">
          AI-powered support system with SLA-based response guarantees
        </p>
      </div>

      {/* Action Button */}
      <div style={{ 
        display: 'flex', 
        justifyContent: 'flex-end', 
        marginBottom: '2rem' 
      }}>
        <button 
          onClick={() => setShowCreateTicket(!showCreateTicket)}
          className="btn-braun-primary"
          style={{
            padding: '0.75rem 1.5rem',
            fontSize: '0.9rem',
            fontWeight: '500'
          }}
        >
          {showCreateTicket ? '✕ Cancel' : '🎫 Create Support Ticket'}
        </button>
      </div>

      {/* Create Ticket Form */}
      {showCreateTicket && (
        <div className="card" style={{ 
          padding: '2rem', 
          marginBottom: '2rem',
          border: '2px solid var(--color-braun-orange)',
          borderRadius: '12px'
        }}>
          <h3 style={{ 
            color: 'var(--color-charcoal)', 
            marginBottom: '1.5rem',
            fontSize: '1.3rem',
            fontWeight: '600'
          }}>
            Create Support Ticket
          </h3>
          
          <form onSubmit={createSupportTicket}>
            <div style={{ marginBottom: '1rem' }}>
              <label style={{ 
                display: 'block', 
                marginBottom: '0.5rem',
                color: 'var(--color-charcoal)',
                fontWeight: '500'
              }}>
                Title:
              </label>
              <input
                type="text"
                value={ticketForm.title}
                onChange={(e) => setTicketForm({ ...ticketForm, title: e.target.value })}
                required
                placeholder="Brief description of the issue"
                style={{
                  width: '100%',
                  padding: '0.75rem',
                  border: '1px solid var(--color-border-light)',
                  borderRadius: '4px',
                  fontSize: '0.875rem'
                }}
              />
            </div>
            
            <div style={{ marginBottom: '1rem' }}>
              <label style={{ 
                display: 'block', 
                marginBottom: '0.5rem',
                color: 'var(--color-charcoal)',
                fontWeight: '500'
              }}>
                Description:
              </label>
              <textarea
                value={ticketForm.description}
                onChange={(e) => setTicketForm({ ...ticketForm, description: e.target.value })}
                required
                rows="4"
                placeholder="Detailed description of the issue, steps to reproduce, expected vs actual behavior"
                style={{
                  width: '100%',
                  padding: '0.75rem',
                  border: '1px solid var(--color-border-light)',
                  borderRadius: '4px',
                  fontSize: '0.875rem',
                  resize: 'vertical'
                }}
              />
            </div>
            
            <div style={{ marginBottom: '1.5rem' }}>
              <label style={{ 
                display: 'block', 
                marginBottom: '0.5rem',
                color: 'var(--color-charcoal)',
                fontWeight: '500'
              }}>
                Priority:
              </label>
              <select
                value={ticketForm.priority}
                onChange={(e) => setTicketForm({ ...ticketForm, priority: e.target.value })}
                style={{
                  padding: '0.75rem',
                  border: '1px solid var(--color-border-light)',
                  borderRadius: '4px',
                  fontSize: '0.875rem',
                  backgroundColor: 'white'
                }}
              >
                <option value="low">Low - General questions, feature requests</option>
                <option value="medium">Medium - Non-critical issues</option>
                <option value="high">High - Important functionality affected</option>
                <option value="critical">Critical - System down or major impact</option>
              </select>
            </div>
            
            <div style={{ display: 'flex', gap: '1rem', justifyContent: 'flex-end' }}>
              <button 
                type="button" 
                onClick={() => setShowCreateTicket(false)}
                className="btn-braun-secondary"
              >
                Cancel
              </button>
              <button 
                type="submit"
                className="btn-braun-primary"
                disabled={loading}
              >
                {loading ? 'Creating...' : 'Create Ticket'}
              </button>
            </div>
          </form>
        </div>
      )}

      {/* Support Metrics */}
      <div style={{ marginBottom: '2rem' }}>
        <h2 style={{ 
          color: 'var(--color-charcoal)', 
          marginBottom: '1.5rem',
          fontSize: '1.5rem',
          fontWeight: '600'
        }}>
          Support Metrics
        </h2>
        
        <div style={{ 
          display: 'grid', 
          gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', 
          gap: '1.5rem',
          marginBottom: '2rem'
        }}>
          <SupportMetricCard
            title="Total Tickets"
            value={supportData?.total_tickets || 0}
            icon="🎫"
            color="var(--color-braun-orange)"
          />
          
          <SupportMetricCard
            title="Open Tickets"
            value={supportData?.open_tickets || 0}
            icon="📋"
            color="var(--color-warning)"
          />
          
          <SupportMetricCard
            title="Resolved Tickets"
            value={supportData?.resolved_tickets || 0}
            icon="✅"
            color="var(--color-success)"
          />
          
          <SupportMetricCard
            title="Avg Resolution"
            value={supportData?.avg_resolution_time_hours?.toFixed(1) || '0.0'}
            subtitle="hours"
            icon="⏱️"
            color="var(--color-info)"
          />
          
          <SupportMetricCard
            title="SLA Compliance"
            value={`${supportData?.sla_compliance_percent?.toFixed(1) || '100.0'}%`}
            icon="📊"
            color="var(--color-success)"
          />
          
          <SupportMetricCard
            title="Engineers Available"
            value={supportData?.engineers_available || 0}
            icon="👥"
            color="var(--color-medium-grey)"
          />
        </div>
      </div>

      {/* SLA Response Times */}
      <div style={{ marginBottom: '2rem' }}>
        <h2 style={{ 
          color: 'var(--color-charcoal)', 
          marginBottom: '1.5rem',
          fontSize: '1.5rem',
          fontWeight: '600'
        }}>
          SLA Response Times
        </h2>
        
        <div style={{ 
          display: 'grid', 
          gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', 
          gap: '1.5rem'
        }}>
          <SLACard
            tier="enterprise"
            critical={2}
            high={4}
            medium={8}
            low={24}
          />
          
          <SLACard
            tier="professional"
            critical={4}
            high={8}
            medium={24}
            low={72}
          />
          
          <SLACard
            tier="standard"
            critical={8}
            high={24}
            medium={72}
            low={168}
          />
        </div>
      </div>

      {/* Support Features */}
      <div>
        <h2 style={{ 
          color: 'var(--color-charcoal)', 
          marginBottom: '1.5rem',
          fontSize: '1.5rem',
          fontWeight: '600'
        }}>
          Support Features
        </h2>
        
        <div className="card" style={{ padding: '2rem' }}>
          <div style={{ 
            display: 'grid', 
            gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', 
            gap: '2rem'
          }}>
            <div>
              <h3 style={{ 
                color: 'var(--color-braun-orange)', 
                marginBottom: '1rem',
                fontSize: '1.1rem',
                fontWeight: '600',
                display: 'flex',
                alignItems: 'center',
                gap: '0.5rem'
              }}>
                🤖 AI-Powered Classification
              </h3>
              <ul style={{ 
                color: 'var(--color-medium-grey)', 
                fontSize: '0.875rem',
                lineHeight: '1.6',
                paddingLeft: '1rem'
              }}>
                <li>Automatic issue categorization</li>
                <li>Smart routing to specialists</li>
                <li>Priority assessment based on keywords</li>
                <li>Knowledge base suggestions</li>
              </ul>
            </div>
            
            <div>
              <h3 style={{ 
                color: 'var(--color-braun-orange)', 
                marginBottom: '1rem',
                fontSize: '1.1rem',
                fontWeight: '600',
                display: 'flex',
                alignItems: 'center',
                gap: '0.5rem'
              }}>
                ⏰ SLA Management
              </h3>
              <ul style={{ 
                color: 'var(--color-medium-grey)', 
                fontSize: '0.875rem',
                lineHeight: '1.6',
                paddingLeft: '1rem'
              }}>
                <li>Tier-based response times</li>
                <li>Automatic escalation on SLA breach</li>
                <li>Proactive monitoring and alerts</li>
                <li>Performance tracking and reporting</li>
              </ul>
            </div>
            
            <div>
              <h3 style={{ 
                color: 'var(--color-braun-orange)', 
                marginBottom: '1rem',
                fontSize: '1.1rem',
                fontWeight: '600',
                display: 'flex',
                alignItems: 'center',
                gap: '0.5rem'
              }}>
                👥 Expert Assignment
              </h3>
              <ul style={{ 
                color: 'var(--color-medium-grey)', 
                fontSize: '0.875rem',
                lineHeight: '1.6',
                paddingLeft: '1rem'
              }}>
                <li>Skill-based engineer matching</li>
                <li>Workload balancing</li>
                <li>Tier access controls</li>
                <li>Specialty area routing</li>
              </ul>
            </div>
          </div>
          
          <div style={{ 
            marginTop: '2rem', 
            padding: '1rem',
            backgroundColor: 'var(--color-warm-white)',
            borderRadius: '8px',
            fontSize: '0.75rem',
            color: 'var(--color-medium-grey)',
            textAlign: 'center'
          }}>
            <strong>Enterprise Support System</strong> - Providing guaranteed response times and expert assistance for your Monte Carlo platform
          </div>
        </div>
      </div>
    </div>
  );
};

export default AdminSupportPage;
