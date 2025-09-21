import React from 'react';

const Button = ({ children, onClick, type = 'button', variant = 'primary', disabled = false, ...props }) => {
  // Basic styling, can be expanded with a CSS module or styled-components
  const baseStyle = {
    padding: '10px 15px',
    border: 'none',
    borderRadius: '4px',
    cursor: 'pointer',
    fontSize: '1rem',
    fontWeight: 'bold',
    transition: 'background-color 0.2s ease-in-out',
  };

  const variants = {
    primary: {
      backgroundColor: '#007bff',
      color: 'white',
    },
    secondary: {
      backgroundColor: '#6c757d',
      color: 'white',
    },
    danger: {
      backgroundColor: '#dc3545',
      color: 'white',
    },
    link: {
      backgroundColor: 'transparent',
      color: '#007bff',
      textDecoration: 'underline',
      padding: 0,
    },
  };

  const style = {
    ...baseStyle,
    ...variants[variant],
    ...(disabled && { opacity: 0.6, cursor: 'not-allowed' }),
  };

  return (
    <button type={type} onClick={onClick} style={style} disabled={disabled} {...props}>
      {children}
    </button>
  );
};

export default Button; 