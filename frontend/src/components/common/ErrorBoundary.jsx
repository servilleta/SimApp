import React from 'react';

class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  componentDidCatch(error, info) {
    // eslint-disable-next-line no-console
    console.error('[ErrorBoundary] Caught error:', error, info);
  }

  render() {
    if (this.state.hasError) {
      if (this.props.fallback) {
        return typeof this.props.fallback === 'function'
          ? this.props.fallback(this.state.error)
          : this.props.fallback;
      }
      return (
        <div style={{ padding: '20px', color: '#dc2626' }}>
          <h3>Something went wrong.</h3>
          <pre style={{ whiteSpace: 'pre-wrap' }}>{this.state.error?.message || 'Unknown error'}</pre>
        </div>
      );
    }

    return this.props.children;
  }
}

export default ErrorBoundary; 