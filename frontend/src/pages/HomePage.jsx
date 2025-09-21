import React from 'react';
import { Link } from 'react-router-dom';

const HomePage = () => {
  const heroStyle = {
    height: '90vh',
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    textAlign: 'center',
    color: '#ffffff',
    backgroundImage: `linear-gradient(rgba(0, 0, 0, 0.6), rgba(0, 0, 0, 0.6)), url('/quantum-hero.jpg')`,
    backgroundSize: 'cover',
    backgroundPosition: 'center',
    padding: '0 20px',
    fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", "Roboto", "Oxygen", "Ubuntu", "Cantarell", sans-serif',
  };

  const heroTitleStyle = {
    fontSize: 'clamp(2.5rem, 5vw, 4.5rem)',
    fontWeight: 700,
    letterSpacing: '-0.02em',
    lineHeight: 1.2,
    marginBottom: '24px',
    textShadow: '0 2px 4px rgba(0,0,0,0.5)',
  };

  const heroButtonStyle = {
    display: 'inline-block',
    padding: '16px 40px',
    background: '#3b82f6',
    color: '#ffffff',
    borderRadius: '50px',
    fontSize: '20px',
    fontWeight: 600,
    textDecoration: 'none',
    transition: 'transform 0.2s ease, background-color 0.2s ease',
    boxShadow: '0 4px 15px rgba(0,0,0,0.2)',
  };

  const contentContainerStyle = {
    padding: '60px 20px',
    maxWidth: '900px',
    margin: '0 auto',
    fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", "Roboto", "Oxygen", "Ubuntu", "Cantarell", sans-serif',
  };
  
  const sectionStyle = { marginBottom: '40px', lineHeight: 1.7, color: '#374151' };
  
  const h2Style = { fontSize: '2rem', marginBottom: '1rem', color: '#1f2937' };
  
  const fadedStyle = { ...sectionStyle, opacity: 0.6 };

  return (
    <div>
      <div style={heroStyle}>
        <h1 style={heroTitleStyle}>Next-Generation Simulation and Forecasting</h1>
        <p style={{fontSize: '1.25rem', marginBottom: '40px', maxWidth: '600px'}}>
          Harness the power of Monte Carlo methods and advanced forecasting in one unified cloud platform.
        </p>
        <Link 
          to="/simulate" 
          style={heroButtonStyle}
          onMouseOver={e => e.currentTarget.style.transform = 'translateY(-2px)'}
          onMouseOut={e => e.currentTarget.style.transform = 'translateY(0)'}
        >
          Launch Simulation →
        </Link>
      </div>

      <div style={contentContainerStyle}>
        <div style={sectionStyle}>
          <h2 style={h2Style}>Monte Carlo Simulations</h2>
          <p>
            Monte Carlo simulations use random sampling to quantify how uncertainty impacts results. They are
            indispensable for risk-aware decision making.
          </p>
          <ul>
            <li><strong>Finance:</strong> Value-at-Risk, option pricing, P&L distributions</li>
            <li><strong>Science:</strong> Particle transport, quantum path-integrals, climate modeling</li>
            <li><strong>Industry:</strong> Project schedule & cost-risk, supply-chain stress tests</li>
            <li><strong>Healthcare:</strong> Pharmacokinetic variability, clinical-trial outcomes</li>
          </ul>
        </div>

        <div style={fadedStyle}>
          <h2 style={h2Style}>Forecasting (Coming Soon)</h2>
          <p>
            Build predictive time-series models with modern techniques such as ARIMA, Prophet, Temporal Fusion
            Transformers (TFT) and N-Beats. Stay tuned!
          </p>
        </div>
      </div>
    </div>
  );
};

export default HomePage; 