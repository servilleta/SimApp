// Neumorphic Design System
export const neumorphicStyles = {
  // Base colors
  colors: {
    background: '#f0f0f0',
    surface: '#f0f0f0',
    primary: '#667eea',
    secondary: '#764ba2',
    accent: '#ff6b6b',
    text: '#333',
    textSecondary: '#666',
    success: '#4ecdc4',
    warning: '#ffe66d',
    error: '#ff6b6b',
  },

  // Shadow configurations
  shadows: {
    raised: '20px 20px 60px #d1d1d1, -20px -20px 60px #ffffff',
    pressed: 'inset 20px 20px 60px #d1d1d1, inset -20px -20px 60px #ffffff',
    subtle: '8px 8px 16px #d1d1d1, -8px -8px 16px #ffffff',
    subtlePressed: 'inset 8px 8px 16px #d1d1d1, inset -8px -8px 16px #ffffff',
    floating: '25px 25px 75px #d1d1d1, -25px -25px 75px #ffffff',
  },

  // Common component styles
  card: {
    background: 'linear-gradient(145deg, #f0f0f0, #e6e6e6)',
    borderRadius: '20px',
    padding: '2rem',
    margin: '1rem 0',
    boxShadow: '20px 20px 60px #d1d1d1, -20px -20px 60px #ffffff',
    border: 'none',
  },

  button: {
    base: {
      padding: '12px 24px',
      borderRadius: '15px',
      border: 'none',
      cursor: 'pointer',
      fontSize: '1rem',
      fontWeight: '600',
      transition: 'all 0.3s ease',
      background: 'linear-gradient(145deg, #f0f0f0, #e6e6e6)',
      color: '#333',
      boxShadow: '8px 8px 16px #d1d1d1, -8px -8px 16px #ffffff',
    },
    primary: {
      background: 'linear-gradient(145deg, #667eea, #764ba2)',
      color: 'white',
    },
    secondary: {
      background: 'linear-gradient(145deg, #e6e6e6, #f0f0f0)',
      color: '#333',
    },
    accent: {
      background: 'linear-gradient(145deg, #ff6b6b, #ee5a5a)',
      color: 'white',
    },
    success: {
      background: 'linear-gradient(145deg, #4ecdc4, #44a08d)',
      color: 'white',
    },
  },

  input: {
    background: 'linear-gradient(145deg, #e6e6e6, #f0f0f0)',
    border: 'none',
    borderRadius: '15px',
    padding: '12px 16px',
    fontSize: '1rem',
    color: '#333',
    boxShadow: 'inset 8px 8px 16px #d1d1d1, inset -8px -8px 16px #ffffff',
    outline: 'none',
    transition: 'all 0.3s ease',
  },

  container: {
    maxWidth: '1200px',
    margin: '0 auto',
    padding: '2rem',
    background: '#f0f0f0',
    minHeight: 'calc(100vh - 200px)', // Account for header/footer
  },

  section: {
    background: 'linear-gradient(145deg, #f0f0f0, #e6e6e6)',
    borderRadius: '25px',
    padding: '2.5rem',
    margin: '2rem 0',
    boxShadow: '25px 25px 75px #d1d1d1, -25px -25px 75px #ffffff',
  },

  modal: {
    background: 'linear-gradient(145deg, #f0f0f0, #e6e6e6)',
    borderRadius: '25px',
    padding: '2rem',
    boxShadow: '25px 25px 75px #d1d1d1, -25px -25px 75px #ffffff',
    border: 'none',
  },

  dropzone: {
    background: 'linear-gradient(145deg, #e6e6e6, #f0f0f0)',
    border: '3px dashed #667eea',
    borderRadius: '25px',
    padding: '3rem',
    textAlign: 'center',
    transition: 'all 0.3s ease',
    cursor: 'pointer',
    boxShadow: 'inset 15px 15px 30px #d1d1d1, inset -15px -15px 30px #ffffff',
  },

  dropzoneActive: {
    borderColor: '#764ba2',
    background: 'linear-gradient(145deg, #f0f0f0, #e6e6e6)',
    boxShadow: '20px 20px 60px #d1d1d1, -20px -20px 60px #ffffff',
  },

  table: {
    width: '100%',
    borderCollapse: 'separate',
    borderSpacing: '0 8px',
    background: 'transparent',
  },

  tableRow: {
    background: 'linear-gradient(145deg, #f0f0f0, #e6e6e6)',
    borderRadius: '15px',
    boxShadow: '8px 8px 16px #d1d1d1, -8px -8px 16px #ffffff',
  },

  tableCell: {
    padding: '1rem',
    borderRadius: '15px',
    border: 'none',
  },
};

export default neumorphicStyles; 