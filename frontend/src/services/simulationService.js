import axios from 'axios';
import {
  normalizeSimulationId,
  getParentId,
  validateSimulationId,
  logIdValidationWarning
} from '../utils/simulationIdUtils';

import { useDispatch } from 'react-redux';
import { updateSimulationProgress } from '../store/simulationSlice'; // Assume this action exists or add it

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:9090/api';
const SIMULATION_API_URL = `${API_BASE_URL}/simulations`;

// Function to get fresh Auth0 token with optimized strategy
const getAuth0Token = async () => {
  // STEP 1: Try existing localStorage token first (fast path)
  const existingToken = localStorage.getItem('authToken');
  
  // Basic token validation (check if it's not expired)
  if (existingToken) {
    try {
      const payload = JSON.parse(atob(existingToken.split('.')[1]));
      const expiration = payload.exp * 1000; // Convert to milliseconds
      const now = Date.now();
      const bufferTime = 5 * 60 * 1000; // 5 minutes buffer
      
      // If token is valid for at least 5 more minutes, use it
      if (expiration > now + bufferTime) {
        console.log('🔐 [AUTH] Using valid cached token');
        return existingToken;
      } else {
        console.log('🔐 [AUTH] Token expires soon, refreshing...');
      }
    } catch (tokenParseError) {
      console.warn('🔐 [AUTH] Could not parse existing token, will refresh');
    }
  }
  
  // STEP 2: Refresh token with Auth0 (with timeout)
  if (window.auth0Client) {
    try {
      console.log('🔐 [AUTH] Refreshing Auth0 token with timeout...');
      
      // Add timeout to prevent 100-second delays
      const tokenPromise = window.auth0Client.getAccessTokenSilently({
        timeoutInSeconds: 10, // 10 second timeout instead of default 60s
        cacheMode: 'cache-only' // Try cache first
      });
      
      const timeoutPromise = new Promise((_, reject) => 
        setTimeout(() => reject(new Error('Auth0 token refresh timeout')), 15000) // 15s total timeout
      );
      
      const token = await Promise.race([tokenPromise, timeoutPromise]);
      localStorage.setItem('authToken', token);
      console.log('🔐 [AUTH] Successfully refreshed Auth0 token');
      return token;
    } catch (error) {
      console.error('🚨 AUTH ERROR - Auth0 token refresh failed:', error);
      
      // Check if this is a "Missing Refresh Token" error
      if (error.message && error.message.includes('Missing Refresh Token')) {
        console.warn('🔄 Missing refresh token detected in simulation service - triggering fresh login...');
        
        // Clear tokens and trigger logout
        localStorage.removeItem('authToken');
        window.dispatchEvent(new CustomEvent('auth0-logout'));
        
        throw new Error('Authentication session expired. Please log in again.');
      }
      
      // STEP 3: Final fallback to localStorage (even if expired)
      if (existingToken) {
        console.warn('🔐 [AUTH] Using potentially expired token as last resort');
        return existingToken;
      }
      
      throw new Error('Authentication token expired. Please log in again.');
    }
  }
  
  // STEP 4: No Auth0 client, use localStorage
  if (existingToken) {
    return existingToken;
  }
  
  throw new Error('Authentication token not found. Please log in.');
};

/**
 * @param {object} simulationConfig - Matches SimulationRequest schema from backend.
 * Example: {
 *   file_id: "string",
 *   iterations: 10000,
 *   variables: [
 *     { variable_name: "Sales", cell_address: "C5", distribution_type: "NORMAL", mean: 100, std_dev: 10 },
 *     { variable_name: "COGS_Percent", cell_address: "C6", distribution_type: "UNIFORM", min_val: 0.4, max_val: 0.6 },
 *   ],
 *   output_formulas: [
 *     { formula_name: "GrossProfit", cell_address: "C7" }
 *   ],
 *   use_gpu: false // Optional, defaults to false or backend setting
 * }
 */
export const runSimulationAPI = (simulationConfig) => {
  const token = localStorage.getItem('authToken');
  return axios.post(`${SIMULATION_API_URL}/run`, simulationConfig, {
    headers: { 
      Authorization: `Bearer ${token}` 
    },
    timeout: 600000, // 10 minutes timeout for large simulations
  });
};

/**
 * Initiates a Monte Carlo simulation run.
 * @param {object} simulationRequest - The request payload for the simulation.
 * Expected structure: { file_id, result_cell_coordinate, result_cell_sheet_name, variables, iterations }
 * @returns {Promise<object>} A promise that resolves to the backend's initial response.
 * Expected: { simulation_id, status, message, created_at, updated_at }
 */
export const postRunSimulation = async (simulationRequest) => {
  try {
    console.log('🔐 [AUTH] Getting fresh token for simulation request...');
    const startTime = Date.now();
    
    const token = await getAuth0Token();
    const tokenTime = Date.now() - startTime;
    console.log(`🔐 [AUTH] Token obtained in ${tokenTime}ms, sending simulation request...`);
    
    // Extended timeout for simulation runs (10 minutes for large files)
    const response = await axios.post(`${SIMULATION_API_URL}/run`, simulationRequest, {
      headers: { 
        Authorization: `Bearer ${token}` 
      },
      timeout: 600000, // 10 minutes timeout for large simulations
    });
    
    const totalTime = Date.now() - startTime;
    console.log(`✅ [AUTH] Simulation request successful! Total time: ${totalTime}ms (token: ${tokenTime}ms)`);
    return response.data; // This should be SimulationResponse from backend (status 202)
  } catch (error) {
    console.error('🚨 [AUTH] Simulation request failed:', error);
    
    // Check if it's an auth error
    if (error.response && (error.response.status === 401 || error.response.status === 403)) {
      console.error('🚨 [AUTH] Authentication failed - token may be expired');
      // Try to refresh token and retry once
      try {
        console.log('🔄 [AUTH] Attempting token refresh and retry...');
        const freshToken = await getAuth0Token();
        const retryResponse = await axios.post(`${SIMULATION_API_URL}/run`, simulationRequest, {
          headers: { 
            Authorization: `Bearer ${freshToken}` 
          },
          timeout: 600000,
        });
        console.log('✅ [AUTH] Retry successful after token refresh!');
        return retryResponse.data;
      } catch (retryError) {
        console.error('🚨 [AUTH] Retry failed after token refresh:', retryError);
        throw new Error('Authentication failed. Please log in again.');
      }
    }
    
    if (error.response && error.response.data) {
      throw new Error(error.response.data.detail || 'Failed to start simulation.');
    }
    throw new Error(error.message || 'Network error or failed to start simulation.');
  }
};

/**
 * Fetches the status or results of a specific simulation run.
 * @param {string} simulationId - The ID of the simulation.
 * @returns {Promise<object>} A promise that resolves to the simulation status/results.
 * Expected: { simulation_id, status, message, results?, created_at, updated_at }
 */
/**
 * Fetches the simulation history for the current user.
 * @param {number} limit - Number of simulations to fetch (default 10)
 * @returns {Promise<Array>} A promise that resolves to an array of simulation history.
 */
export const getUserSimulationHistory = async (limit = 10) => {
  try {
    console.log('🔍 Fetching user simulation history...');
    
    const token = await getAuth0Token();
    
    const response = await axios.get(`${API_BASE_URL}/simulation/history?limit=${limit}`, {
      headers: { 
        Authorization: `Bearer ${token}` 
      },
      timeout: 30000, // 30 seconds timeout
    });
    
    console.log('📊 User simulation history received:', response.data);
    return response.data;
  } catch (error) {
    console.error('🚨 Failed to fetch user simulation history:', error);
    
    // Handle authentication error
    if (error.response?.status === 401) {
      console.log('🔐 Token expired, attempting retry with fresh token...');
      try {
        const token = await getAuth0Token();
        const response = await axios.get(`${API_BASE_URL}/simulation/history?limit=${limit}`, {
          headers: { 
            Authorization: `Bearer ${token}` 
          },
          timeout: 30000,
        });
        return response.data;
      } catch (retryError) {
        console.error('🚨 Retry failed after token refresh:', retryError);
        throw new Error('Authentication failed. Please log in again.');
      }
    }
    
    if (error.response && error.response.data) {
      throw new Error(error.response.data.detail || 'Failed to fetch simulation history.');
    }
    throw new Error(error.message || 'Network error or failed to fetch simulation history.');
  }
};

export const getSimulationStatus = async (simulationId) => {
  // CRITICAL FIX: Validate and normalize simulation ID before API call
  const validation = validateSimulationId(simulationId);
  if (!validation.isValid) {
    console.warn(`[simulationService] Invalid simulation ID for status request: ${validation.error}`);
    if (validation.isCorrupted && validation.suggestedFix) {
      console.log(`[simulationService] Using normalized ID: ${validation.suggestedFix}`);
      simulationId = validation.suggestedFix;
    }
  }
  
  const normalizedId = getParentId(simulationId);
  logIdValidationWarning(simulationId, 'getSimulationStatus API call');
  
  try {
    console.log('🔐 [AUTH] Getting fresh token for simulation status request...');
    console.log(`🔧 [ID_NORMALIZATION] Original ID: ${simulationId}, Normalized ID: ${normalizedId}`);
    const token = await getAuth0Token();
    console.log('🔐 [AUTH] Token obtained, fetching simulation status...');
    
    const response = await axios.get(`${SIMULATION_API_URL}/${normalizedId}`, {
      headers: { 
        Authorization: `Bearer ${token}` 
      }
    });
    
    console.log('✅ [AUTH] Simulation status request successful!');
    return response.data; // This should be SimulationResponse from backend
  } catch (error) {
    console.error('🚨 [AUTH] Simulation status request failed:', error);
    
    // Check if it's an auth error
    if (error.response && (error.response.status === 401 || error.response.status === 403)) {
      console.error('🚨 [AUTH] Authentication failed - token may be expired');
      // Try to refresh token and retry once
      try {
        console.log('🔄 [AUTH] Attempting token refresh and retry...');
        const freshToken = await getAuth0Token();
        const retryResponse = await axios.get(`${SIMULATION_API_URL}/${normalizedId}`, {
          headers: { 
            Authorization: `Bearer ${freshToken}` 
          }
        });
        console.log('✅ [AUTH] Retry successful after token refresh!');
        return retryResponse.data;
      } catch (retryError) {
        console.error('🚨 [AUTH] Retry failed after token refresh:', retryError);
        throw new Error('Authentication failed. Please log in again.');
      }
    }
    
    // Handle 404s specifically for child simulations
    if (error.response && error.response.status === 404) {
      throw new Error('404 - Simulation not found');
    }
    
    if (error.response && error.response.data) {
      throw new Error(error.response.data.detail || 'Failed to fetch simulation status.');
    }
    throw new Error(error.message || 'Network error or failed to fetch simulation status.');
  }
};

export const getAllResultsSummariesAPI = (page = 1, size = 20) => {
  const token = localStorage.getItem('authToken');
  return axios.get(`${SIMULATION_API_URL}/results?page=${page}&size=${size}`, {
    headers: { 
      Authorization: `Bearer ${token}` 
    }
  });
};

export const getResultDetailsAPI = (simulationId) => {
  const token = localStorage.getItem('authToken');
  return axios.get(`${SIMULATION_API_URL}/results/${simulationId}`, {
    headers: { 
      Authorization: `Bearer ${token}` 
    }
  });
};

// CRITICAL FIX: Enhanced request deduplication map to handle both parent and child IDs
const progressRequestCache = new Map();

// Connection health tracking
let lastHealthCheck = 0;
let isBackendHealthy = true;
const HEALTH_CHECK_INTERVAL = 30000; // 30 seconds

/**
 * Check backend connectivity with a lightweight ping
 */
const checkBackendHealth = async () => {
  const now = Date.now();
  if (now - lastHealthCheck < HEALTH_CHECK_INTERVAL) {
    return isBackendHealthy;
  }
  
  try {
    // Quick health check to the progress health endpoint
    // 🚀 OPTIMIZED: Faster health check timeout for real-time progress
    const healthEndpoint = import.meta.env.VITE_HEALTH_ENDPOINT || `${API_BASE_URL}/health/progress`;
    await axios.get(healthEndpoint, { timeout: 1000 }); // 🚀 FAST: 1 second timeout
    isBackendHealthy = true;
    lastHealthCheck = now;
    return true;
  } catch (error) {
    isBackendHealthy = false;
    lastHealthCheck = now;
    console.warn('Backend health check failed:', error.message);
    return false;
  }
};

/**
 * Progressive timeout implementation for retry logic
 * 🚀 OPTIMIZED: Faster timeouts for real-time progress updates during simulation
 */
const getProgressiveTimeout = (attempt) => {
  const baseTimeout = 800; // 🚀 FAST: Start with 800ms for instant progress updates
  const maxTimeout = 3000;  // 🚀 FAST: Cap at 3 seconds instead of 12
  return Math.min(baseTimeout * Math.pow(1.5, attempt), maxTimeout);
};

/**
 * Exponential backoff delay calculation
 */
const getRetryDelay = (attempt) => {
  return Math.min(1000 * Math.pow(2, attempt), 5000); // Cap at 5 seconds
};

/**
 * Enhanced progress fetching with retry logic and request deduplication
 * @param {string} simulationId - The ID of the simulation.
 * @param {number} maxRetries - Maximum number of retry attempts (default: 3)
 * @returns {Promise<object|null>} A promise that resolves to the progress data or null if not found.
 */
export const getSimulationProgress = async (simulationId, maxRetries = 3) => {
  // CRITICAL FIX: Validate and normalize simulation ID before processing
  const validation = validateSimulationId(simulationId);
  if (!validation.isValid) {
    console.warn(`[simulationService] Invalid simulation ID for progress request: ${validation.error}`);
    if (validation.isCorrupted && validation.suggestedFix) {
      console.log(`[simulationService] Using normalized ID for progress: ${validation.suggestedFix}`);
      simulationId = validation.suggestedFix;
    }
  }
  
  const parentId = getParentId(simulationId);
  logIdValidationWarning(simulationId, 'getSimulationProgress API call');
  console.log(`🔧 [PROGRESS_ID_NORMALIZATION] Original ID: ${simulationId}, Parent ID: ${parentId}`);
  
  // CRITICAL FIX: Use parent ID as cache key to prevent duplicate requests
  const cacheKey = parentId;
  
  // Check for existing request to prevent duplication
  if (progressRequestCache.has(cacheKey)) {
    console.debug(`Deduplicating progress request for ${cacheKey} (original: ${simulationId})`);
    return progressRequestCache.get(cacheKey);
  }
  
  // Create the request promise using parent ID
  const requestPromise = _fetchProgressWithRetry(parentId, maxRetries);
  
  // Cache the promise using parent ID
  progressRequestCache.set(cacheKey, requestPromise);
  
  try {
    const result = await requestPromise;
    return result;
  } finally {
    // CRITICAL FIX: Clean up cache using the correct cache key
    setTimeout(() => {
      progressRequestCache.delete(cacheKey);
    }, 1000); // Clear after 1 second
  }
};

/**
 * Internal function to fetch progress with retry logic
 */
const _fetchProgressWithRetry = async (simulationId, maxRetries) => {
  let lastError = null;
  
  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    // Progressive timeout: start short, increase on retry (declare outside try block)
    const timeout = getProgressiveTimeout(attempt);
    
    try {
      // Check backend health before making request
      if (attempt > 0) {
        const healthCheck = await checkBackendHealth();
        if (!healthCheck) {
          console.warn(`Backend unhealthy, attempt ${attempt + 1}/${maxRetries + 1} for ${simulationId}`);
          if (attempt === maxRetries) {
            return {
              status: 'connection_error',
              simulation_id: simulationId,
              message: 'Backend temporarily unavailable'
            };
          }
        }
      }
      
      console.debug(`Progress request attempt ${attempt + 1}/${maxRetries + 1} for ${simulationId} (timeout: ${timeout}ms)`);
      
      // CRITICAL FIX: Add ID validation for progress endpoint
      console.log(`🔧 [PROGRESS_REQUEST] Making progress request for simulation ID: ${simulationId}`);
      
      // Add optional auth header when token is available
      const headers = {};
      try {
        const token = localStorage.getItem('authToken');
        if (token) {
          headers.Authorization = `Bearer ${token}`;
        }
      } catch (error) {
        // Ignore auth token errors for progress endpoint
      }
      
      const response = await axios.get(`${SIMULATION_API_URL}/${simulationId}/progress`, {
        timeout: timeout,
        headers: headers
      });
      
      // Handle 202 status - backend is processing but progress unavailable
      if (response.status === 202) {
        console.debug(`Progress temporarily unavailable for ${simulationId}, backend still processing`);
        return response.data || {
          status: 'processing',
          simulation_id: simulationId,
          message: 'Still processing, please wait'
        };
      }
      
      // Success - reset health status
      isBackendHealthy = true;
      
      console.debug(`Progress retrieved successfully for ${simulationId} on attempt ${attempt + 1}`);
      return response.data;
      
    } catch (error) {
      lastError = error;
      
      // Handle 404 gracefully - simulation may not exist or be completed
      if (error.response && error.response.status === 404) {
        return { status: 'not_found', simulation_id: simulationId };
      }
      
      // Check for timeout errors - handle both axios timeout and browser timeout
      const isTimeoutError = (
        error.code === 'ECONNABORTED' || 
        error.message?.includes('timeout') ||
        error.message?.includes('Network Error') ||
        (error.name && error.name === 'TimeoutError')
      );
      
      if (isTimeoutError) {
        console.warn(`Timeout on attempt ${attempt + 1}/${maxRetries + 1} for ${simulationId} (${timeout}ms):`, error.message);
        isBackendHealthy = false;
        
        // If this is the last attempt, return timeout error
        if (attempt === maxRetries) {
          return {
            status: 'timeout',
            simulation_id: simulationId,
            message: `Request timed out after ${maxRetries + 1} attempts`,
            last_timeout: timeout
          };
        }
      } else {
        // For non-timeout errors, log and potentially break early
        console.warn(`Network error on attempt ${attempt + 1} for ${simulationId}:`, error.message);
        
        // For server errors (5xx), retry. For client errors (4xx except 404), don't retry
        if (error.response && error.response.status >= 400 && error.response.status < 500 && error.response.status !== 404) {
          console.warn(`Client error ${error.response.status}, not retrying for ${simulationId}`);
          break;
        }
      }
      
      // Wait before retrying (except on last attempt)
      if (attempt < maxRetries) {
        const delay = getRetryDelay(attempt);
        console.debug(`Waiting ${delay}ms before retry ${attempt + 2} for ${simulationId}`);
        await new Promise(resolve => setTimeout(resolve, delay));
      }
    }
  }
  
  // All retries failed
  console.error(`All ${maxRetries + 1} attempts failed for simulation ${simulationId}:`, lastError?.message);
  
  // Return more informative error state instead of null
  return {
    status: 'error',
    simulation_id: simulationId,
    message: `Progress fetch failed: ${lastError?.message || 'Unknown error'}`,
    error_type: 'network_failure',
    attempts: maxRetries + 1
  };
};

/**
 * Cancels a running simulation with retry logic.
 * @param {string} simulationId - The ID of the simulation to cancel.
 * @returns {Promise<object>} A promise that resolves to the cancellation response.
 */
export const cancelSimulation = async (simulationId) => {
  try {
    const token = localStorage.getItem('authToken');
    if (!token) {
      throw new Error('Authentication token not found. Please log in.');
    }
    
    const response = await axios.post(`${SIMULATION_API_URL}/${simulationId}/cancel`, {}, {
      headers: { 
        Authorization: `Bearer ${token}` 
      },
      timeout: 10000 // 10 second timeout for cancellation
    });
    return response.data;
  } catch (error) {
    if (error.response && error.response.data) {
      throw new Error(error.response.data.detail || 'Failed to cancel simulation.');
    }
    throw new Error(error.message || 'Network error or failed to cancel simulation.');
  }
};

/**
 * Get backend connection status for debugging
 */
export const getConnectionStatus = () => {
  return {
    isHealthy: isBackendHealthy,
    lastHealthCheck: lastHealthCheck,
    activeRequests: progressRequestCache.size
  };
};

/**
 * Force a backend health check
 */
export const forceHealthCheck = async () => {
  lastHealthCheck = 0; // Force health check
  return await checkBackendHealth();
};

/**
 * Clear all cached progress requests (for debugging)
 */
export const clearProgressCache = () => {
  progressRequestCache.clear();
  console.log('Progress request cache cleared');
}; 

// 🚀 EARLY CONNECTION: Enhanced WebSocket connection function for pre-simulation connection
export const connectWebSocket = (simulationId, dispatch) => {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = window.location.host;
    const wsUrl = `${protocol}//${host}/ws/simulations/${simulationId}`;
    
    console.log(`🚀 [EARLY_CONNECTION] Connecting WebSocket to: ${wsUrl}`);
    const ws = new WebSocket(wsUrl);
    
    ws.onopen = () => {
        console.log(`✅ [EARLY_CONNECTION] WebSocket connected BEFORE simulation starts: ${simulationId}`);
        console.log(`🎯 [EARLY_CONNECTION] Ready to receive real-time progress updates!`);
    };
    
    ws.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);
            console.log(`📨 [EARLY_CONNECTION] Real-time progress update received:`, data);
            console.log(`🎯 [EARLY_CONNECTION] Progress: ${data.progress_percentage || data.progress || 'N/A'}%`);
            dispatch(updateSimulationProgress({ simulationId, ...data }));
        } catch (e) {
            console.error('❌ [EARLY_CONNECTION] WS Message Error:', e);
        }
    };
    
    ws.onerror = (error) => {
        console.error('WS Error:', error);
        // Fallback to polling
        startPolling(simulationId, dispatch);
    };
    
    ws.onclose = () => {
        console.log('WS Closed');
        // Optional: Reconnect logic
    };
    
    return ws;
};

// Fallback polling function
export const startPolling = (simulationId, dispatch) => {
    const interval = setInterval(async () => {
        try {
            const response = await axios.get(`${SIMULATION_API_URL}/${simulationId}/progress`);
            const data = response.data;
            dispatch(updateSimulationProgress({ simulationId, ...data }));
            
            if (data.progress >= 100 || data.status === 'completed') {
                clearInterval(interval);
            }
        } catch (error) {
            console.error('Polling error:', error);
            clearInterval(interval);
        }
    }, 2000);
    
    return interval;
};

// 🚀 NEW: Create real simulation ID (eliminates temp ID system)
export const createSimulationId = async () => {
  try {
    console.log('🔐 [AUTH] Getting fresh token for ID creation request...');
    const token = await getAuth0Token();
    console.log('🔐 [AUTH] Token obtained, creating simulation ID...');
    
    const response = await axios.post(`${SIMULATION_API_URL}/create-id`, {}, {
      headers: { 
        Authorization: `Bearer ${token}` 
      },
      timeout: 10000 // 10 second timeout
    });
    
    console.log('✅ [AUTH] Simulation ID created successfully:', response.data.simulation_id);
    return response.data;
  } catch (error) {
    console.error('🚨 [AUTH] Failed to create simulation ID:', error);
    
    // Check if it's an auth error and retry once
    if (error.response && (error.response.status === 401 || error.response.status === 403)) {
      try {
        console.log('🔄 [AUTH] Attempting token refresh and retry...');
        const freshToken = await getAuth0Token();
        const retryResponse = await axios.post(`${SIMULATION_API_URL}/create-id`, {}, {
          headers: { 
            Authorization: `Bearer ${freshToken}` 
          },
          timeout: 10000
        });
        console.log('✅ [AUTH] Retry successful after token refresh!');
        return retryResponse.data;
      } catch (retryError) {
        console.error('🚨 [AUTH] Retry failed after token refresh:', retryError);
        throw new Error('Authentication failed. Please log in again.');
      }
    }
    
    if (error.response && error.response.data) {
      throw new Error(error.response.data.detail || 'Failed to create simulation ID.');
    }
    throw new Error(error.message || 'Network error or failed to create simulation ID.');
  }
};

// In the runSimulation function or export, after getting real ID:
// const response = await api.post('/simulations/run', payload);
// const realId = response.data.simulation_id;

// // Connect WS with real ID
// const ws = connectWebSocket(realId, dispatch); 