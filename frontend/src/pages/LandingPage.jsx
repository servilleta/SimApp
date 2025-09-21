import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import '../styles/colors.css'; // Import the Braun color system
import logoFull from '../assets/images/noBgColor.png';

const LandingPage = () => {
  const navigate = useNavigate();
  const [currentSlide, setCurrentSlide] = useState(0);

  // Auto-rotate hero slides
  useEffect(() => {
    const timer = setInterval(() => {
      setCurrentSlide((prev) => (prev + 1) % 4);
    }, 10000);
    return () => clearInterval(timer);
  }, []);

  // Use a static version string for cache busting - update this when images change
  const imageVersion = "v1.0";
  
  const heroSlides = [
    {
      title: "Complex Made Simple",
      subtitle: "Transform uncertainty into confident decisions with Monte Carlo intelligence that reveals hidden insights in your data",
      image: `/quantum-hero.jpg?v=${imageVersion}`
    },
    {
      title: "True Excel Intelligence",
      subtitle: "67+ Excel functions with complete formula fidelity - VLOOKUP, financial modeling, conditional analysis, and complex dependency chains",
      image: `/excel.jpg?v=${imageVersion}`
    },
    {
      title: "Enterprise Analytics Platform",
      subtitle: "Real-time sensitivity analysis and professional-grade statistical outputs",
      image: `/analitics.jpg?v=${imageVersion}`
    },
    {
      title: "Lightning-Fast GPU Acceleration",
      subtitle: "10x faster Monte Carlo simulations with world-class CUDA engines and JIT compilation",
      image: `/fast.jpg?v=${imageVersion}`
    }
  ];

  const industries = [
    {
      title: "Financial Services",
      description: "Risk assessment, portfolio optimization, VaR calculations, stress testing, and regulatory compliance modeling.",
      image: "https://images.unsplash.com/photo-1551434678-e076c223a692?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=2070&q=80",
      useCases: ["Portfolio Risk Analysis", "Credit Default Modeling", "Regulatory Capital", "Stress Testing"]
    },
    {
      title: "Energy & Utilities",
      description: "Resource planning, demand forecasting, renewable energy optimization, and grid stability analysis.",
      image: "/energy-hero.jpg",
      useCases: ["Demand Forecasting", "Grid Optimization", "Renewable Planning", "Price Modeling"]
    },
    {
      title: "Healthcare & Pharma",
      description: "Clinical trial optimization, drug development ROI, epidemiological modeling, and resource allocation.",
      image: "https://images.unsplash.com/photo-1559757148-5c350d0d3c56?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=2069&q=80",
      useCases: ["Clinical Trials", "Drug Development", "Epidemiology", "Resource Planning"]
    },
    {
      title: "Manufacturing",
      description: "Supply chain optimization, quality control, capacity planning, and operational risk management.",
      image: "https://images.unsplash.com/photo-1581091226825-a6a2a5aee158?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=2070&q=80",
      useCases: ["Supply Chain", "Quality Control", "Capacity Planning", "Risk Management"]
    },
    {
      title: "Real Estate",
      description: "Property valuation, investment analysis, market forecasting, and development feasibility studies.",
      image: "https://images.unsplash.com/photo-1560518883-ce09059eeffa?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=2073&q=80",
      useCases: ["Property Valuation", "Investment Analysis", "Market Forecasting", "Development ROI"]
    },
    {
      title: "Technology",
      description: "Product launch planning, user growth modeling, infrastructure scaling, and market penetration analysis.",
      image: "https://images.unsplash.com/photo-1488229297570-58520851e868?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=2069&q=80",
      useCases: ["Growth Modeling", "Launch Planning", "Scaling Analysis", "Market Penetration"]
    }
  ];

  const features = [
    {
      title: "GPU Acceleration",
      description: "10x faster simulations with CUDA-powered Monte Carlo engines"
    },
    {
      title: "Advanced Analytics",
      description: "Sensitivity analysis, correlation matrices, and statistical insights"
    },
    {
      title: "Enterprise Security",
      description: "GDPR compliant, SOC 2 ready, with comprehensive audit trails"
    },
    {
      title: "Real-time Results",
      description: "Live progress tracking with interactive visualizations"
    },
    {
      title: "Excel Formula Mastery",
      description: "67+ Excel functions including VLOOKUP, financial formulas, conditional analysis, and text processing"
    },
    {
      title: "Cloud Native",
      description: "Scalable infrastructure with automatic backup and recovery"
    }
  ];

  const styles = {
    container: {
      minHeight: '100vh',
      backgroundColor: 'var(--color-warm-white)',
      fontFamily: 'system-ui, -apple-system, sans-serif',
      color: 'var(--color-text-primary)'
    },
    nav: {
      position: 'fixed',
      top: 0,
      width: '100%',
      zIndex: 50,
      backgroundColor: 'var(--color-white)',
      borderBottom: '1px solid var(--color-border-light)',
      boxShadow: 'var(--shadow-xs)'
    },
    navContent: {
      maxWidth: '1280px',
      margin: '0 auto',
      padding: '0 24px',
      display: 'flex',
      justifyContent: 'space-between',
      alignItems: 'center',
      height: '64px'
    },
    logo: {
      fontSize: '24px',
      fontWeight: 'bold',
      color: 'var(--color-charcoal)',
      marginTop: '8px'
    },
    navButtons: {
      display: 'flex',
      gap: '16px'
    },
    hero: {
      paddingTop: '120px',
      paddingBottom: '80px',
      backgroundColor: 'var(--color-white)'
    },
    heroContent: {
      maxWidth: '1280px',
      margin: '0 auto',
      padding: '0 24px',
      display: 'grid',
      gridTemplateColumns: '1fr 1fr',
      gap: '48px',
      alignItems: 'center'
    },
    heroText: {
      textAlign: 'left'
    },
    heroTitle: {
      fontSize: '56px',
      fontWeight: 'bold',
      color: 'var(--color-charcoal)',
      marginBottom: '24px',
      lineHeight: '1.1',
      opacity: 0,
      animation: 'fadeInUp 1s ease forwards'
    },
    heroSubtitle: {
      fontSize: '20px',
      color: 'var(--color-text-secondary)',
      marginBottom: '32px',
      lineHeight: '1.6',
      opacity: 0,
      animation: 'fadeInUp 1s ease 0.2s forwards'
    },
    heroButtons: {
      display: 'flex',
      gap: '16px',
      opacity: 0,
      animation: 'fadeInUp 1s ease 0.4s forwards'
    },
    heroImage: {
      position: 'relative',
      opacity: 0,
      animation: 'fadeInRight 1s ease 0.3s forwards'
    },
    imageContainer: {
      width: '100%',
      height: '400px',
      borderRadius: '8px',
      overflow: 'hidden',
      border: '1px solid var(--color-border-light)',
      backgroundColor: 'var(--color-white)',
      transition: 'all var(--transition-base)'
    },
    image: {
      width: '100%',
      height: '100%',
      objectFit: 'cover'
    },
    section: {
      padding: '80px 0'
    },
    sectionAlternate: {
      backgroundColor: 'var(--color-warm-white)'
    },
    sectionContent: {
      maxWidth: '1280px',
      margin: '0 auto',
      padding: '0 24px'
    },
    sectionTitle: {
      fontSize: '40px',
      fontWeight: 'bold',
      color: 'var(--color-charcoal)',
      textAlign: 'center',
      marginBottom: '24px'
    },
    sectionDescription: {
      fontSize: '20px',
      color: 'var(--color-text-secondary)',
      textAlign: 'center',
      maxWidth: '800px',
      margin: '0 auto 64px',
      lineHeight: '1.6'
    },
    grid: {
      display: 'grid',
      gap: '32px'
    },
    grid3: {
      gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))'
    },
    cta: {
      backgroundColor: 'var(--color-white)',
      border: '1px solid var(--color-border-light)',
      borderRadius: '8px',
      padding: '48px',
      textAlign: 'center',
      maxWidth: '800px',
      margin: '0 auto',
      boxShadow: 'var(--shadow-sm)'
    },
    footer: {
      backgroundColor: 'var(--color-white)',
      borderTop: '1px solid var(--color-border-light)',
      padding: '48px 0'
    },
    footerGrid: {
      display: 'grid',
      gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
      gap: '32px'
    },
    footerTitle: {
      color: 'var(--color-dark-grey)',
      fontWeight: '600',
      marginBottom: '16px'
    },
    footerLink: {
      color: 'var(--color-text-tertiary)',
      textDecoration: 'none',
      display: 'block',
      marginBottom: '8px',
      transition: 'color var(--transition-base)',
      fontSize: '14px'
    },
    // CSS animations
    keyframes: `
      @keyframes fadeInUp {
        from {
          opacity: 0;
          transform: translateY(30px);
        }
        to {
          opacity: 1;
          transform: translateY(0);
        }
      }
      
      @keyframes fadeInRight {
        from {
          opacity: 0;
          transform: translateX(30px);
        }
        to {
          opacity: 1;
          transform: translateX(0);
        }
      }
    `
  };

  // Add CSS animations to the document
  useEffect(() => {
    const style = document.createElement('style');
    style.textContent = styles.keyframes;
    document.head.appendChild(style);
    return () => document.head.removeChild(style);
  }, []);

  return (
    <div style={styles.container}>
      {/* Navigation */}
      <nav style={styles.nav}>
        <div style={styles.navContent}>
          <div style={styles.logo}>
            <img 
              src={logoFull} 
              alt="SimApp Logo" 
              style={{ 
                height: '56px', 
                width: 'auto', 
                objectFit: 'contain' 
              }} 
            />
          </div>
          <div style={styles.navButtons}>
            <button
              className="btn-braun-secondary"
              onClick={() => navigate('/login')}
            >
              Sign In
            </button>
            <button
              className="btn-braun-primary"
              onClick={() => navigate('/get-started')}
            >
              Get Started
            </button>
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <section style={styles.hero}>
        <div style={styles.heroContent}>
          <div style={styles.heroText}>
            <h1 style={styles.heroTitle}>
              {heroSlides[currentSlide].title}
            </h1>
            <p style={styles.heroSubtitle}>
              {heroSlides[currentSlide].subtitle}
            </p>
            <div style={styles.heroButtons}>
              <button
                className="btn-braun-primary"
                style={{ padding: '16px 32px', fontSize: '16px' }}
                onClick={() => navigate('/get-started')}
              >
                Get Started
              </button>
              <button
                className="btn-braun-secondary"
                style={{ padding: '16px 32px', fontSize: '16px' }}
              >
                Watch Demo
              </button>
            </div>
          </div>
          <div style={styles.heroImage}>
            <div 
              style={styles.imageContainer}
              className="hover-lift"
            >
              <img
                src={heroSlides[currentSlide].image}
                alt="Monte Carlo Simulation"
                style={styles.image}
              />
            </div>
            {/* Slide indicators */}
            <div style={{ display: 'flex', justifyContent: 'center', marginTop: '24px', gap: '8px' }}>
              {heroSlides.map((_, index) => (
                <button
                  key={index}
                  onClick={() => setCurrentSlide(index)}
                  style={{
                    width: '12px',
                    height: '12px',
                    borderRadius: '50%',
                    border: 'none',
                    cursor: 'pointer',
                    backgroundColor: index === currentSlide ? 'var(--color-braun-orange)' : 'var(--color-light-grey)',
                    transition: 'all var(--transition-base)'
                  }}
                />
              ))}
            </div>
          </div>
        </div>
      </section>

      {/* Monte Carlo Explanation */}
      <section style={{ ...styles.section, ...styles.sectionAlternate }}>
        <div style={styles.sectionContent}>
          <h2 style={styles.sectionTitle}>What is Monte Carlo Simulation?</h2>
          <div style={{ 
            ...styles.sectionDescription, 
            textAlign: 'left', 
            maxWidth: '900px',
            fontSize: '18px',
            lineHeight: '1.8',
            marginBottom: '48px'
          }}>
            <p style={{ marginBottom: '24px' }}>
              Monte Carlo simulation is a computational technique that uses thousands of random scenarios to model uncertainty and show the full range of possible outcomes for a broad scope of challenges in many industries.
            </p>
            <p style={{ marginBottom: '24px' }}>
              Rather than relying on single-point estimates that rarely materialize, it varies your key assumptions within realistic ranges to generate probability distributions of results.
            </p>
            <p style={{ marginBottom: '24px' }}>
              For example in P&L forecasting, this means instead of saying "we expect $2M profit," you can present:
            </p>
            <p style={{ 
              fontSize: '24px',
              fontStyle: 'italic',
              color: 'var(--color-braun-orange)',
              fontWeight: '600',
              marginBottom: '24px',
              textAlign: 'center',
              lineHeight: '1.4'
            }}>
              "70% chance of $1.5M-$2.5M profit, with only 10% risk below $1M"
            </p>
            <p>
              This approach gives managers better methods to manage risk, make credible commitments, better decision-making capabilities, and the ability to proactively communicate uncertainty rather than defend unrealistic projections.
            </p>
          </div>
          
          <div style={{ ...styles.grid, ...styles.grid3 }}>
            <div className="card-braun hover-lift">
              <div style={{ 
                width: '72px', 
                height: '72px', 
                marginBottom: '20px',
                backgroundColor: 'var(--color-braun-orange)',
                borderRadius: '12px',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                boxShadow: '0 4px 12px rgba(255, 122, 47, 0.3)'
              }}>
                <svg width="36" height="36" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                  <path d="M12 2L15.09 8.26L22 9L17 14L18.18 21L12 17.77L5.82 21L7 14L2 9L8.91 8.26L12 2Z" fill="white"/>
                  <path d="M12 6L13.5 10.5L18 11L15 14L15.75 18.5L12 16.5L8.25 18.5L9 14L6 11L10.5 10.5L12 6Z" fill="var(--color-braun-orange)"/>
                </svg>
              </div>
              <h3 style={{ fontSize: '24px', fontWeight: 'bold', color: 'var(--color-charcoal)', marginBottom: '16px' }}>
                Risk Management
              </h3>
              <p style={{ color: 'var(--color-text-secondary)', lineHeight: '1.6' }}>
                Monte Carlo simulation <strong>quantifies your exposure</strong> by showing not just average outcomes but the <strong>probability of extreme losses</strong>, enabling you to set appropriate reserves and communicate risk tolerance clearly to leadership.
              </p>
            </div>
            
            <div className="card-braun hover-lift">
              <div style={{ 
                width: '72px', 
                height: '72px', 
                marginBottom: '20px',
                backgroundColor: 'var(--color-braun-orange)',
                borderRadius: '12px',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                boxShadow: '0 4px 12px rgba(255, 122, 47, 0.3)'
              }}>
                <svg width="36" height="36" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                  <rect x="3" y="3" width="18" height="18" rx="2" stroke="white" strokeWidth="2" fill="none"/>
                  <path d="M8 12L12 8L16 12L20 8" stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                  <circle cx="8" cy="12" r="2" fill="white"/>
                  <circle cx="12" cy="8" r="2" fill="white"/>
                  <circle cx="16" cy="12" r="2" fill="white"/>
                  <circle cx="20" cy="8" r="2" fill="white"/>
                </svg>
              </div>
              <h3 style={{ fontSize: '24px', fontWeight: 'bold', color: 'var(--color-charcoal)', marginBottom: '16px' }}>
                Sensitivity Analysis
              </h3>
              <p style={{ color: 'var(--color-text-secondary)', lineHeight: '1.6' }}>
                The technique <strong>automatically identifies which variables</strong> have the biggest impact on your P&L by tracking how changes in each input affect the final results, helping you <strong>focus management attention</strong> on the factors that matter most.
              </p>
            </div>
            
            <div className="card-braun hover-lift">
              <div style={{ 
                width: '72px', 
                height: '72px', 
                marginBottom: '20px',
                backgroundColor: 'var(--color-braun-orange)',
                borderRadius: '12px',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                boxShadow: '0 4px 12px rgba(255, 122, 47, 0.3)'
              }}>
                <svg width="36" height="36" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                  <circle cx="12" cy="12" r="3" stroke="white" strokeWidth="2"/>
                  <path d="M9 12L11 14L15 10" stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                  <path d="M21 12C21 7.02944 16.9706 3 12 3C7.02944 3 3 7.02944 3 12C3 16.9706 7.02944 21 12 21" stroke="white" strokeWidth="2" strokeLinecap="round"/>
                  <path d="M21 12C19.5 12 18 13.5 18 15C18 16.5 19.5 18 21 18" stroke="white" strokeWidth="2" strokeLinecap="round"/>
                </svg>
              </div>
              <h3 style={{ fontSize: '24px', fontWeight: 'bold', color: 'var(--color-charcoal)', marginBottom: '16px' }}>
                What-If Scenarios
              </h3>
              <p style={{ color: 'var(--color-text-secondary)', lineHeight: '1.6' }}>
                Rather than manually calculating dozens of scenarios, Monte Carlo <strong>generates thousands of possible combinations simultaneously</strong>, revealing unlikely but <strong>high-impact situations</strong> that traditional planning might miss.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Industry Use Cases */}
      <section style={styles.section}>
        <div style={styles.sectionContent}>
          <h2 style={styles.sectionTitle}>Trusted by Leading Industries</h2>
          <p style={styles.sectionDescription}>
            From Fortune 500 companies to innovative startups, organizations across industries 
            rely on Monte Carlo simulation for critical decision making.
          </p>
          
          <div style={{ ...styles.grid, ...styles.grid3 }}>
            {industries.map((industry, index) => (
              <div
                key={industry.title}
                className="card-braun hover-lift"
              >
                <div style={{ 
                  width: '100%', 
                  height: '200px', 
                  marginBottom: '20px',
                  borderRadius: '8px',
                  overflow: 'hidden',
                  border: '1px solid var(--color-border-light)'
                }}>
                  <img 
                    src={industry.image} 
                    alt={industry.title}
                    style={{
                      width: '100%',
                      height: '100%',
                      objectFit: 'cover',
                      transition: 'transform 0.3s ease'
                    }}
                    onMouseOver={(e) => e.target.style.transform = 'scale(1.05)'}
                    onMouseOut={(e) => e.target.style.transform = 'scale(1)'}
                  />
                </div>
                <h3 style={{ fontSize: '24px', fontWeight: 'bold', color: 'var(--color-charcoal)', marginBottom: '16px' }}>
                  {industry.title}
                </h3>
                <p style={{ color: 'var(--color-text-secondary)', lineHeight: '1.6', marginBottom: '24px' }}>
                  {industry.description}
                </p>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                  {industry.useCases.map((useCase, idx) => (
                    <div key={idx} style={{ display: 'flex', alignItems: 'center', fontSize: '14px', color: 'var(--color-text-tertiary)' }}>
                      <div style={{ 
                        width: '6px', 
                        height: '6px', 
                        backgroundColor: 'var(--color-braun-orange)', 
                        borderRadius: '50%', 
                        marginRight: '12px' 
                      }}></div>
                      {useCase}
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section style={{ ...styles.section, ...styles.sectionAlternate }}>
        <div style={styles.sectionContent}>
          <h2 style={styles.sectionTitle}>World-Class Platform Features</h2>
          <p style={styles.sectionDescription}>
            Built for enterprise scale with cutting-edge technology and uncompromising security.
          </p>
          
          <div style={{ ...styles.grid, ...styles.grid3 }}>
            {features.map((feature, index) => {
              // Define SVG icons for each feature
              const getFeatureIcon = (title) => {
                switch(title) {
                  case "GPU Acceleration":
                    return (
                      <svg width="36" height="36" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <path d="M13 3L4 14H11L10 21L19 10H12L13 3Z" fill="white"/>
                        <circle cx="18" cy="6" r="2" fill="white"/>
                        <circle cx="20" cy="18" r="2" fill="white"/>
                        <circle cx="6" cy="18" r="2" fill="white"/>
                      </svg>
                    );
                  case "Advanced Analytics":
                    return (
                      <svg width="36" height="36" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <rect x="3" y="3" width="18" height="18" rx="2" stroke="white" strokeWidth="2" fill="none"/>
                        <path d="M7 14L9 12L13 16L17 8" stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                        <circle cx="7" cy="14" r="1" fill="white"/>
                        <circle cx="9" cy="12" r="1" fill="white"/>
                        <circle cx="13" cy="16" r="1" fill="white"/>
                        <circle cx="17" cy="8" r="1" fill="white"/>
                      </svg>
                    );
                  case "Enterprise Security":
                    return (
                      <svg width="36" height="36" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <path d="M12 2L3 7L12 12L21 7L12 2Z" fill="white"/>
                        <path d="M3 7V17L12 22L21 17V7" stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" fill="none"/>
                        <path d="M9 12L11 14L15 10" stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                      </svg>
                    );
                  case "Real-time Results":
                    return (
                      <svg width="36" height="36" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <circle cx="12" cy="12" r="9" stroke="white" strokeWidth="2"/>
                        <path d="M12 7V12L16 14" stroke="white" strokeWidth="2" strokeLinecap="round"/>
                        <circle cx="12" cy="12" r="2" fill="white"/>
                        <path d="M2 12H4M20 12H22M12 2V4M12 20V22" stroke="white" strokeWidth="2" strokeLinecap="round"/>
                      </svg>
                    );
                  case "Multi-Engine Support":
                    return (
                      <svg width="36" height="36" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <rect x="3" y="3" width="6" height="6" rx="1" fill="white"/>
                        <rect x="15" y="3" width="6" height="6" rx="1" fill="white"/>
                        <rect x="3" y="15" width="6" height="6" rx="1" fill="white"/>
                        <rect x="15" y="15" width="6" height="6" rx="1" fill="white"/>
                        <circle cx="12" cy="12" r="3" stroke="white" strokeWidth="2" fill="none"/>
                        <path d="M9 6H15M9 18H15M6 9V15M18 9V15" stroke="white" strokeWidth="2" strokeLinecap="round"/>
                      </svg>
                    );
                  case "Cloud Native":
                    return (
                      <svg width="36" height="36" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <path d="M18 10H20C21.1046 10 22 10.8954 22 12C22 13.1046 21.1046 14 20 14H18M6 10H4C2.89543 10 2 10.8954 2 12C2 13.1046 2.89543 14 4 14H6" stroke="white" strokeWidth="2" strokeLinecap="round"/>
                        <path d="M8 14C6.89543 14 6 13.1046 6 12C6 9.79086 7.79086 8 10 8H14C16.2091 8 18 9.79086 18 12C18 13.1046 17.1046 14 16 14" stroke="white" strokeWidth="2" strokeLinecap="round"/>
                        <circle cx="12" cy="12" r="2" fill="white"/>
                        <path d="M10 16V20M14 16V20M8 20H16" stroke="white" strokeWidth="2" strokeLinecap="round"/>
                      </svg>
                    );
                  case "Excel Formula Mastery":
                    return (
                      <svg width="36" height="36" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <rect x="4" y="4" width="16" height="16" rx="2" stroke="white" strokeWidth="2" fill="none"/>
                        <path d="M8 8H16M8 12H16M8 16H16" stroke="white" strokeWidth="2" strokeLinecap="round"/>
                        <path d="M4 8H8M4 12H8M4 16H8" stroke="white" strokeWidth="1" strokeLinecap="round"/>
                        <text x="12" y="15" textAnchor="middle" fill="white" fontSize="8" fontWeight="bold">fx</text>
                      </svg>
                    );
                  default:
                    return null;
                }
              };

              return (
                <div
                  key={feature.title}
                  className="card-braun hover-lift"
                >
                  <div style={{ 
                    width: '72px', 
                    height: '72px', 
                    marginBottom: '20px',
                    backgroundColor: 'var(--color-braun-orange)',
                    borderRadius: '12px',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    boxShadow: '0 4px 12px rgba(255, 122, 47, 0.3)'
                  }}>
                    {getFeatureIcon(feature.title)}
                  </div>
                  <h3 style={{ fontSize: '24px', fontWeight: 'bold', color: 'var(--color-charcoal)', marginBottom: '16px' }}>
                    {feature.title}
                  </h3>
                  <p style={{ color: 'var(--color-text-secondary)', lineHeight: '1.6' }}>
                    {feature.description}
                  </p>
                </div>
              );
            })}
          </div>
        </div>
      </section>


      {/* CTA Section */}
      <section style={styles.section}>
        <div style={styles.sectionContent}>
          <div style={styles.cta}>
            <h2 style={{ ...styles.sectionTitle, marginBottom: '24px' }}>
              Ready to Transform Your Analysis?
            </h2>
            <p style={{ ...styles.sectionDescription, marginBottom: '32px' }}>
              Join thousands of professionals who trust our platform for critical decision making.
            </p>
            <div style={{ display: 'flex', gap: '16px', justifyContent: 'center', flexWrap: 'wrap' }}>
              <button
                className="btn-braun-primary"
                style={{ padding: '16px 32px', fontSize: '16px' }}
                onClick={() => navigate('/get-started')}
              >
                Get Started
              </button>
              <button
                className="btn-braun-secondary"
                style={{ padding: '16px 32px', fontSize: '16px' }}
                onClick={() => navigate('/login')}
              >
                Sign In
              </button>
            </div>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer style={styles.footer}>
        <div style={styles.sectionContent}>
          <div style={styles.footerGrid}>
            <div>
              <div style={{ ...styles.logo, marginBottom: '16px' }}>SimApp</div>
              <p style={{ color: 'var(--color-text-tertiary)', fontSize: '14px' }}>
                Enterprise-grade Monte Carlo simulation platform for data-driven decision making.
              </p>
            </div>
            <div>
              <h4 style={styles.footerTitle}>Product</h4>
              <button 
                onClick={() => navigate('/features')}
                style={{
                  ...styles.footerLink, 
                  background: 'none', 
                  border: 'none', 
                  textAlign: 'left', 
                  cursor: 'pointer',
                  padding: 0
                }} 
                onMouseOver={(e) => e.target.style.color = 'var(--color-braun-orange)'} 
                onMouseOut={(e) => e.target.style.color = 'var(--color-text-tertiary)'}
              >
                Features
              </button>
              <button 
                onClick={() => navigate('/pricing')}
                style={{
                  ...styles.footerLink, 
                  background: 'none', 
                  border: 'none', 
                  textAlign: 'left', 
                  cursor: 'pointer',
                  padding: 0
                }} 
                onMouseOver={(e) => e.target.style.color = 'var(--color-braun-orange)'} 
                onMouseOut={(e) => e.target.style.color = 'var(--color-text-tertiary)'}
              >
                Pricing
              </button>
              <a href="#api" style={styles.footerLink} onMouseOver={(e) => e.target.style.color = 'var(--color-braun-orange)'} onMouseOut={(e) => e.target.style.color = 'var(--color-text-tertiary)'}>API</a>
              <a href="#docs" style={styles.footerLink} onMouseOver={(e) => e.target.style.color = 'var(--color-braun-orange)'} onMouseOut={(e) => e.target.style.color = 'var(--color-text-tertiary)'}>Documentation</a>
            </div>
            <div>
              <h4 style={styles.footerTitle}>Company</h4>
              <button 
                onClick={() => navigate('/about')}
                style={{
                  ...styles.footerLink, 
                  background: 'none', 
                  border: 'none', 
                  textAlign: 'left', 
                  cursor: 'pointer',
                  padding: 0
                }} 
                onMouseOver={(e) => e.target.style.color = 'var(--color-braun-orange)'} 
                onMouseOut={(e) => e.target.style.color = 'var(--color-text-tertiary)'}
              >
                About
              </button>
              <a href="#blog" style={styles.footerLink} onMouseOver={(e) => e.target.style.color = 'var(--color-braun-orange)'} onMouseOut={(e) => e.target.style.color = 'var(--color-text-tertiary)'}>Blog</a>
              <a href="#careers" style={styles.footerLink} onMouseOver={(e) => e.target.style.color = 'var(--color-braun-orange)'} onMouseOut={(e) => e.target.style.color = 'var(--color-text-tertiary)'}>Careers</a>
              <button 
                onClick={() => navigate('/contact')}
                style={{
                  ...styles.footerLink, 
                  background: 'none', 
                  border: 'none', 
                  textAlign: 'left', 
                  cursor: 'pointer',
                  padding: 0
                }} 
                onMouseOver={(e) => e.target.style.color = 'var(--color-braun-orange)'} 
                onMouseOut={(e) => e.target.style.color = 'var(--color-text-tertiary)'}
              >
                Contact
              </button>
            </div>
            <div>
              <h4 style={styles.footerTitle}>Legal</h4>
              <button 
                onClick={() => navigate('/privacy')}
                style={{
                  ...styles.footerLink, 
                  background: 'none', 
                  border: 'none', 
                  textAlign: 'left', 
                  cursor: 'pointer',
                  padding: 0
                }} 
                onMouseOver={(e) => e.target.style.color = 'var(--color-braun-orange)'} 
                onMouseOut={(e) => e.target.style.color = 'var(--color-text-tertiary)'}
              >
                Privacy Policy
              </button>
              <button 
                onClick={() => navigate('/terms')}
                style={{
                  ...styles.footerLink, 
                  background: 'none', 
                  border: 'none', 
                  textAlign: 'left', 
                  cursor: 'pointer',
                  padding: 0
                }} 
                onMouseOver={(e) => e.target.style.color = 'var(--color-braun-orange)'} 
                onMouseOut={(e) => e.target.style.color = 'var(--color-text-tertiary)'}
              >
                Terms of Service
              </button>
              <button 
                onClick={() => navigate('/cookie-policy')}
                style={{
                  ...styles.footerLink, 
                  background: 'none', 
                  border: 'none', 
                  textAlign: 'left', 
                  cursor: 'pointer',
                  padding: 0
                }} 
                onMouseOver={(e) => e.target.style.color = 'var(--color-braun-orange)'} 
                onMouseOut={(e) => e.target.style.color = 'var(--color-text-tertiary)'}
              >
                Cookie Policy
              </button>
            </div>
          </div>
          <div style={{ 
            borderTop: '1px solid var(--color-border-light)', 
            marginTop: '32px', 
            paddingTop: '32px', 
            textAlign: 'center', 
            color: 'var(--color-text-tertiary)',
            fontSize: '14px' 
          }}>
            <p>&copy; 2025 SimApp. All rights reserved.</p>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default LandingPage; 