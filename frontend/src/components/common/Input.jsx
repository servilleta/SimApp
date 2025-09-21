import React from 'react';

const Input = ({ type = 'text', value, onChange, placeholder, label, name, error, ...props }) => {
  const inputStyle = {
    padding: '10px',
    border: '1px solid #ccc',
    borderRadius: '4px',
    fontSize: '1rem',
    width: '100%', // Default to full width, can be overridden by parent
    boxSizing: 'border-box',
    marginBottom: '5px',
  };

  const labelStyle = {
    display: 'block',
    marginBottom: '5px',
    fontWeight: 'bold',
  };

  const errorStyle = {
    color: 'red',
    fontSize: '0.875rem',
    marginTop: '2px',
  };

  return (
    <div style={{ marginBottom: '15px' }}>
      {label && <label htmlFor={name} style={labelStyle}>{label}</label>}
      <input
        id={name}
        name={name}
        type={type}
        value={value}
        onChange={onChange}
        placeholder={placeholder}
        style={{ ...inputStyle, ...(error && { borderColor: 'red' }) }}
        {...props}
      />
      {error && <p style={errorStyle}>{error}</p>}
    </div>
  );
};

export default Input; 