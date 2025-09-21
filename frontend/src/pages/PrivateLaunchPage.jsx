import React, { useEffect } from 'react';
import { useNavigate } from 'react-router-dom';

const PrivateLaunchPage = () => {
  const navigate = useNavigate();

  useEffect(() => {
    // Redirect to the new get started page
    navigate('/get-started', { replace: true });
  }, [navigate]);

  // Return null since we're redirecting
  return null;

  /* Legacy styles kept for reference
  const styles = {
    container: {
      minHeight: '100vh',
      background: 'var(--color-warm-white)',
      fontFamily: 'system-ui, -apple-system, sans-serif',
      position: 'relative'
    },
    header: {
      background: 'var(--color-white)',
      borderBottom: '1px solid var(--color-border-light)',
      padding: '16px 0',
      position: 'sticky',
      top: 0,
      zIndex: 10
    },
    headerContent: {
      maxWidth: '1200px',
      margin: '0 auto',
      padding: '0 20px',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'space-between'
    },
    logo: {
      fontSize: '24px',
      fontWeight: 'bold',
      color: 'var(--color-charcoal)',
      textDecoration: 'none'
    },
    backButton: {
      background: 'var(--color-white)',
      border: '1px solid var(--color-border-light)',
      borderRadius: '8px',
      padding: '8px 16px',
      color: 'var(--color-dark-grey)',
      textDecoration: 'none',
      fontSize: '14px',
      fontWeight: '500',
      transition: 'all var(--transition-base)',
      display: 'flex',
      alignItems: 'center',
      gap: '8px'
    },
    content: {
      padding: '80px 20px',
      maxWidth: '800px',
      margin: '0 auto',
      textAlign: 'center'
    },
    card: {
      background: 'var(--color-white)',
      border: '1px solid var(--color-border-light)',
      borderRadius: '16px',
      padding: '48px 40px',
      boxShadow: 'var(--shadow-sm)',
      textAlign: 'center'
    },
    badge: {
      display: 'inline-block',
      background: 'var(--color-braun-orange)',
      color: 'var(--color-white)',
      padding: '8px 20px',
      borderRadius: '20px',
      fontSize: '12px',
      fontWeight: '600',
      textTransform: 'uppercase',
      letterSpacing: '0.5px',
      marginBottom: '24px'
    },
    title: {
      fontSize: '36px',
      fontWeight: '700',
      color: 'var(--color-charcoal)',
      marginBottom: '16px',
      lineHeight: '1.2'
    },
    subtitle: {
      fontSize: '18px',
      color: 'var(--color-text-secondary)',
      marginBottom: '40px',
      lineHeight: '1.6'
    },
    featureList: {
      display: 'grid',
      gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))',
      gap: '20px',
      marginBottom: '40px',
      textAlign: 'left'
    },
    featureItem: {
      display: 'flex',
      alignItems: 'flex-start',
      gap: '12px',
      padding: '16px',
      background: 'var(--color-warm-white)',
      borderRadius: '8px',
      border: '1px solid var(--color-border-light)'
    },
    featureIcon: {
      width: '24px',
      height: '24px',
      borderRadius: '50%',
      background: 'var(--color-braun-orange)',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      fontSize: '12px',
      color: 'white',
      flexShrink: 0,
      marginTop: '2px'
    },
    featureContent: {
      flex: 1
    },
    featureTitle: {
      fontSize: '16px',
      fontWeight: '600',
      color: 'var(--color-charcoal)',
      marginBottom: '4px'
    },
    featureDescription: {
      fontSize: '14px',
      color: 'var(--color-text-secondary)',
      lineHeight: '1.4'
    },
    contactInfo: {
      background: 'var(--color-warm-white)',
      border: '2px solid var(--color-braun-orange)',
      borderRadius: '12px',
      padding: '32px',
      marginBottom: '32px'
    },
    contactTitle: {
      fontSize: '18px',
      fontWeight: '600',
      color: 'var(--color-braun-orange)',
      marginBottom: '12px'
    },
    contactText: {
      fontSize: '16px',
      color: 'var(--color-text-secondary)',
      lineHeight: '1.5',
      marginBottom: '16px'
    },
    emailLink: {
      color: 'var(--color-braun-orange)',
      textDecoration: 'none',
      fontWeight: '600',
      fontSize: '18px'
    },
    loginButton: {
      background: 'var(--color-braun-orange)',
      color: 'var(--color-white)',
      border: 'none',
      padding: '16px 32px',
      borderRadius: '8px',
      fontSize: '16px',
      fontWeight: '600',
      cursor: 'pointer',
      transition: 'all var(--transition-base)',
      textDecoration: 'none',
      display: 'inline-block',
      marginBottom: '16px'
    },
    footer: {
      fontSize: '14px',
      color: 'var(--color-medium-grey)',
      marginTop: '32px',
      lineHeight: '1.6',
      background: 'var(--color-warm-white)',
      padding: '24px',
      borderRadius: '8px',
      border: '1px solid var(--color-border-light)'
    }
  }; */

  /* Legacy JSX - now redirecting to GetStartedPage
  return (
    // ... legacy JSX content
  );
  */
};

export default PrivateLaunchPage; 