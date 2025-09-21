import React from 'react';

const Footer = () => {
  const footerStyle = {
    backgroundColor: '#f9fafb',
    borderTop: '1px solid #e5e7eb',
    padding: '24px 32px',
    textAlign: 'center',
    marginTop: 'auto',
  };

  const textStyle = {
    fontSize: '14px',
    color: '#6b7280',
    margin: 0,
  };

  const linkStyle = {
    color: '#3b82f6',
    textDecoration: 'none',
    fontWeight: '500',
  };

  return (
    <footer style={footerStyle}>
      <p style={textStyle}>
        © 2024 SimApp. Built with ❤️ for better Monte Carlo simulations.
      </p>
    </footer>
  );
};

export default Footer; 