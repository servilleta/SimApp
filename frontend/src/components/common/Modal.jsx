import React from 'react';
import ReactDOM from 'react-dom';

const Modal = ({ isOpen, onClose, title, children }) => {
  if (!isOpen) return null;

  const modalOverlayStyle = {
    position: 'fixed',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    backgroundColor: 'rgba(0, 0, 0, 0.5)',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    zIndex: 1000, // Ensure modal is on top
  };

  const modalContentStyle = {
    backgroundColor: 'white',
    padding: '20px',
    borderRadius: '8px',
    boxShadow: '0 4px 6px rgba(0, 0, 0, 0.1)',
    minWidth: '300px',
    maxWidth: '80%',
    maxHeight: '90vh',
    overflowY: 'auto',
  };

  const modalHeaderStyle = {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    borderBottom: '1px solid #eee',
    paddingBottom: '10px',
    marginBottom: '15px',
  };

  const closeButtonStyle = {
    background: 'none',
    border: 'none',
    fontSize: '1.5rem',
    cursor: 'pointer',
  };

  return ReactDOM.createPortal(
    <div style={modalOverlayStyle} onClick={onClose}> {/* Close on overlay click */}
      <div style={modalContentStyle} onClick={(e) => e.stopPropagation()}> {/* Prevent closing when clicking inside content */}
        <div style={modalHeaderStyle}>
          {title && <h2>{title}</h2>}
          <button onClick={onClose} style={closeButtonStyle} aria-label="Close modal">&times;</button>
        </div>
        {children}
      </div>
    </div>,
    document.body // Append modal to body to ensure it's outside the main app flow for z-index and styling
  );
};

export default Modal; 