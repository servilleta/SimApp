import React, { useState, useEffect } from 'react';
import { useSelector } from 'react-redux';

const CacheManager = () => {
  const [cacheStats, setCacheStats] = useState(null);
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState('');
  const auth = useSelector((state) => state.auth);

  const fetchCacheStats = async () => {
    try {
      setLoading(true);
      const response = await fetch('/api/simulations/cache/stats/comprehensive', {
        headers: {
          'Authorization': `Bearer ${auth.accessToken}`,
          'Content-Type': 'application/json'
        }
      });
      
      if (response.ok) {
        const data = await response.json();
        setCacheStats(data.data);
      } else {
        console.error('Failed to fetch cache stats');
      }
    } catch (error) {
      console.error('Error fetching cache stats:', error);
    } finally {
      setLoading(false);
    }
  };

  const clearUserCache = async () => {
    try {
      setLoading(true);
      const response = await fetch('/api/simulations/cache/user/clear', {
        method: 'DELETE',
        headers: {
          'Authorization': `Bearer ${auth.accessToken}`,
          'Content-Type': 'application/json'
        }
      });
      
      if (response.ok) {
        const data = await response.json();
        setMessage(`✅ ${data.message}`);
        fetchCacheStats(); // Refresh stats
      } else {
        setMessage('❌ Failed to clear cache');
      }
    } catch (error) {
      setMessage('❌ Error clearing cache');
      console.error('Error clearing cache:', error);
    } finally {
      setLoading(false);
    }
  };

  const cleanupExpiredCache = async () => {
    try {
      setLoading(true);
      const response = await fetch('/api/simulations/cache/cleanup/expired', {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${auth.accessToken}`,
          'Content-Type': 'application/json'
        }
      });
      
      if (response.ok) {
        const data = await response.json();
        setMessage(`✅ ${data.message}`);
        setCacheStats(data.updated_stats);
      } else {
        setMessage('❌ Failed to cleanup expired cache');
      }
    } catch (error) {
      setMessage('❌ Error cleaning up cache');
      console.error('Error cleaning up cache:', error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (auth.accessToken) {
      fetchCacheStats();
    }
  }, [auth.accessToken]);

  // Auto-refresh every 30 seconds
  useEffect(() => {
    const interval = setInterval(() => {
      if (auth.accessToken && !loading) {
        fetchCacheStats();
      }
    }, 30000);

    return () => clearInterval(interval);
  }, [auth.accessToken, loading]);

  if (!auth.accessToken) {
    return null;
  }

  return (
    <div style={{
      position: 'fixed',
      bottom: '20px',
      right: '20px',
      background: '#f8f9fa',
      border: '1px solid #dee2e6',
      borderRadius: '8px',
      padding: '15px',
      minWidth: '300px',
      boxShadow: '0 4px 6px rgba(0,0,0,0.1)',
      fontSize: '0.875rem',
      zIndex: 1000
    }}>
      <div style={{ display: 'flex', alignItems: 'center', marginBottom: '10px' }}>
        <h4 style={{ margin: 0, fontSize: '1rem', color: '#495057' }}>
          🧹 Cache Manager
        </h4>
        <button
          onClick={fetchCacheStats}
          disabled={loading}
          style={{
            marginLeft: 'auto',
            background: 'none',
            border: 'none',
            cursor: 'pointer',
            fontSize: '16px'
          }}
          title="Refresh stats"
        >
          🔄
        </button>
      </div>

      {loading && <div style={{ color: '#6c757d' }}>Loading...</div>}

      {cacheStats && (
        <div style={{ marginBottom: '10px' }}>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px', fontSize: '0.8rem' }}>
            <div>
              <strong>Cached:</strong> {cacheStats.total_cached_simulations}
            </div>
            <div>
              <strong>Utilization:</strong> {cacheStats.cache_utilization_percent}%
            </div>
            <div>
              <strong>Users:</strong> {cacheStats.users_with_cached_data}
            </div>
            <div>
              <strong>Avg Age:</strong> {cacheStats.average_age_hours ? `${cacheStats.average_age_hours}h` : 'N/A'}
            </div>
          </div>
          
          {cacheStats.cache_utilization_percent > 80 && (
            <div style={{
              background: '#fff3cd',
              border: '1px solid #ffeaa7',
              borderRadius: '4px',
              padding: '8px',
              marginTop: '8px',
              fontSize: '0.75rem',
              color: '#856404'
            }}>
              ⚠️ High cache utilization! Consider cleanup.
            </div>
          )}

          {cacheStats.oldest_simulation_hours > 24 && (
            <div style={{
              background: '#f8d7da',
              border: '1px solid #f5c6cb',
              borderRadius: '4px',
              padding: '8px',
              marginTop: '8px',
              fontSize: '0.75rem',
              color: '#721c24'
            }}>
              🕰️ Old simulations detected (&gt;{cacheStats.oldest_simulation_hours.toFixed(1)}h)
            </div>
          )}
        </div>
      )}

      <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
        <button
          onClick={clearUserCache}
          disabled={loading}
          style={{
            background: '#007bff',
            color: 'white',
            border: 'none',
            borderRadius: '4px',
            padding: '6px 12px',
            fontSize: '0.75rem',
            cursor: 'pointer'
          }}
        >
          Clear My Cache
        </button>
        <button
          onClick={cleanupExpiredCache}
          disabled={loading}
          style={{
            background: '#28a745',
            color: 'white',
            border: 'none',
            borderRadius: '4px',
            padding: '6px 12px',
            fontSize: '0.75rem',
            cursor: 'pointer'
          }}
        >
          Cleanup Expired
        </button>
      </div>

      {message && (
        <div style={{
          marginTop: '8px',
          padding: '6px',
          background: message.includes('✅') ? '#d4edda' : '#f8d7da',
          border: `1px solid ${message.includes('✅') ? '#c3e6cb' : '#f5c6cb'}`,
          borderRadius: '4px',
          fontSize: '0.75rem',
          color: message.includes('✅') ? '#155724' : '#721c24'
        }}>
          {message}
        </div>
      )}
    </div>
  );
};

export default CacheManager;
