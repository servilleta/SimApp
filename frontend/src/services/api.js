import axios from 'axios';
import { getToken, clearAuthData } from './authService';

// Base API URL - use environment variable or fallback to backend port 8000
export const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000/api';

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 60000, // 60 second timeout for general API calls
});

// Create a special client for long-running operations
const longRunningApiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 600000, // 10 minutes timeout for long-running operations
});

// Interceptor to add auth token to requests
const addAuthToken = (config) => {
  const token = getToken();
  if (token) {
    config.headers['Authorization'] = `Bearer ${token}`;
  }
  return config;
};

apiClient.interceptors.request.use(addAuthToken, (error) => Promise.reject(error));
longRunningApiClient.interceptors.request.use(addAuthToken, (error) => Promise.reject(error));

// Enhanced response interceptor for proper 401 handling
const responseInterceptor = (response) => response;
const errorInterceptor = (error) => {
  console.error('API Error:', error.response || error.message);
  
  // Handle authentication errors
  if (error.response?.status === 401) {
    console.warn('🔐 Authentication failed - clearing session and redirecting to login');
    
    // Clear authentication data
    clearAuthData();
    
    // Dispatch logout event for Redux store
    window.dispatchEvent(new CustomEvent('auth:logout', { 
      detail: { reason: 'token_expired' }
    }));
    
    // Only redirect if not already on login page
    if (!window.location.pathname.includes('/login')) {
      window.location.href = '/login';
    }
  }
  
  // Handle gateway timeouts and server errors
  if (error.response?.status === 504) {
    console.error('⏰ Gateway timeout - simulation may have exceeded time limits');
  }
  
  return Promise.reject(error);
};

apiClient.interceptors.response.use(responseInterceptor, errorInterceptor);
longRunningApiClient.interceptors.response.use(responseInterceptor, errorInterceptor);

export default apiClient;
export { longRunningApiClient }; 