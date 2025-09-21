import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';
import './index.css'; // We will create this for global styles

// Phase 26: Console logger completely removed to eliminate HTTP request competition

ReactDOM.createRoot(document.getElementById('root')).render(
  <App />
); 