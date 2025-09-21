import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000/api';
const AUTH_API_URL = `${API_BASE_URL}/auth`;

// Create axios instance with default config
const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 10000,
});

// Add token to requests automatically
apiClient.interceptors.request.use((config) => {
  const token = getToken();
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

/**
 * Get current auth token
 * @returns {string|null} 
 */
export const getToken = () => {
  return localStorage.getItem('authToken');
};

/**
 * Clear authentication data
 */
export const clearAuthData = () => {
  localStorage.removeItem('authToken');
  localStorage.removeItem('tokenType');
};

/**
 * Login with username and password
 * @param {string} username 
 * @param {string} password 
 * @returns {Promise<object>} Login response with access token
 */
export const login = async (username, password) => {
  try {
    const formData = new FormData();
    formData.append('username', username);
    formData.append('password', password);
    
    const response = await axios.post(`${AUTH_API_URL}/token`, formData, {
      headers: {
        'Content-Type': 'application/x-www-form-urlencoded',
      },
    });
    
    if (response.data.access_token) {
      localStorage.setItem('authToken', response.data.access_token);
      localStorage.setItem('tokenType', response.data.token_type || 'bearer');
    }
    
    return response.data;
  } catch (error) {
    if (error.response && error.response.data) {
      throw new Error(error.response.data.detail || 'Login failed');
    }
    throw new Error(error.message || 'Network error during login');
  }
};

/**
 * Register a new user
 * @param {string} username 
 * @param {string} password 
 * @param {string} email 
 * @returns {Promise<object>} Registration response
 */
export const register = async (username, password, email) => {
  try {
    const response = await axios.post(`${AUTH_API_URL}/register`, {
      username,
      password,
      email,
    });
    return response.data;
  } catch (error) {
    if (error.response && error.response.data) {
      throw new Error(error.response.data.detail || 'Registration failed');
    }
    throw new Error(error.message || 'Network error during registration');
  }
};

/**
 * Logout user by removing token from localStorage
 */
export const logout = () => {
  clearAuthData();
};

/**
 * Check if user is authenticated
 * @returns {boolean} 
 */
export const isAuthenticated = () => {
  return !!getToken();
};

/**
 * Get current user info
 * @returns {Promise<object>} Current user data
 */
export const getCurrentUser = async () => {
  try {
    const token = getToken();
    
    if (!token) {
      throw new Error('No token found');
    }

    // Try Auth0 profile endpoint first (for Auth0 users)
    try {
      const response = await apiClient.get('/auth0/profile');
      return response.data;
    } catch (auth0Error) {
      // If Auth0 endpoint fails, try traditional auth endpoint
      if (auth0Error.response?.status === 404 || auth0Error.response?.status === 401) {
        try {
          const response = await apiClient.get('/auth/me');
          return response.data;
        } catch (traditionalError) {
          // If both fail, clear auth data and throw error
          if (traditionalError.response?.status === 401 || traditionalError.response?.status === 422) {
            clearAuthData();
            throw new Error('Invalid or expired token');
          }
          throw traditionalError;
        }
      }
      throw auth0Error;
    }
  } catch (error) {
    if (error.response?.status === 401 || error.response?.status === 422) {
      clearAuthData();
      throw new Error('Invalid or expired token');
    }
    throw error;
  }
}; 