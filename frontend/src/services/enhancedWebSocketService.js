/**
 * Phase 30 Enhanced WebSocket Service
 * 
 * Optimized for smooth progress tracking with graceful fallback,
 * real-time iteration counting, and improved error handling.
 * Now with progress recovery to eliminate "black hole" periods.
 */

import progressRecovery from './progressRecovery';

class EnhancedWebSocketService {
  constructor() {
    this.connections = new Map(); // simulationId -> WebSocket
    this.progressCallbacks = new Map(); // simulationId -> progress callback
    this.errorCallbacks = new Map(); // simulationId -> error callback
    this.completionCallbacks = new Map(); // simulationId -> completion callback
    
    // Phase 30: Enhanced connection management
    this.reconnectAttempts = new Map();
    this.maxReconnectAttempts = 3;
    this.reconnectDelay = 1000;
    this.connectionTimeouts = new Map();
    this.connectionTimeout = 10000; // 10 seconds
    
    // Phase 30: Progress interpolation support
    this.lastProgressUpdate = new Map(); // simulationId -> { progress, timestamp, iteration }
    this.progressInterpolators = new Map(); // simulationId -> interval
    
    // Phase 30: Fallback HTTP polling
    this.pollingIntervals = new Map(); // simulationId -> interval
    this.pollingRate = 2000; // 2 seconds
    
    console.log('🚀 [Phase30] Enhanced WebSocket Service initialized');
  }

  /**
   * Phase 30: Connect with enhanced progress tracking
   */
  connect(simulationId, options = {}) {
    const {
      onProgress = null,
      onError = null,
      onComplete = null,
      enableInterpolation = true,
      fallbackToPolling = true
    } = options;

    if (this.connections.has(simulationId)) {
      console.log(`🔄 [Phase30] Already connected to simulation ${simulationId}`);
      return Promise.resolve();
    }

    console.log(`🚀 [Phase30] Connecting to simulation ${simulationId} with enhanced tracking`);

    // Store callbacks
    if (onProgress) this.progressCallbacks.set(simulationId, onProgress);
    if (onError) this.errorCallbacks.set(simulationId, onError);
    if (onComplete) this.completionCallbacks.set(simulationId, onComplete);
    
    // 🚀 NEW: Register for progress recovery
    if (onProgress) {
      progressRecovery.registerSimulation(simulationId, onProgress);
      progressRecovery.startRecoveryChecks(simulationId);
    }

    return new Promise((resolve, reject) => {
      const startTime = Date.now();
      
      try {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const host = window.location.host;
        const wsUrl = `${protocol}//${host}/ws/simulations/${simulationId}`;
        
        const ws = new WebSocket(wsUrl);
        
        // Phase 30: Connection timeout
        const timeoutId = setTimeout(() => {
          console.log(`⏰ [Phase30] WebSocket connection timeout for ${simulationId}`);
          ws.close();
          
          if (fallbackToPolling) {
            this.startHttpPolling(simulationId);
            resolve();
          } else {
            this.handleError(simulationId, new Error('Connection timeout'));
            reject(new Error('WebSocket connection timeout'));
          }
        }, this.connectionTimeout);

        ws.onopen = () => {
          const connectionTime = Date.now() - startTime;
          console.log(`✅ [Phase30] Connected to simulation ${simulationId} in ${connectionTime}ms`);
          
          clearTimeout(timeoutId);
          this.connections.set(simulationId, ws);
          this.connectionTimeouts.delete(simulationId);
          this.resetReconnectAttempts(simulationId);
          
          // Phase 30: Start progress interpolation if enabled
          if (enableInterpolation) {
            this.startProgressInterpolation(simulationId);
          }
          
          // 🚀 NEW: Attempt progress recovery when connection is established
          setTimeout(async () => {
            await progressRecovery.recoverProgress(simulationId);
          }, 1000); // Wait 1 second for connection to stabilize
          
          resolve();
        };

        ws.onmessage = (event) => {
          this.handleMessage(simulationId, event.data);
        };

        ws.onerror = (error) => {
          console.log(`❌ [Phase30] WebSocket error for ${simulationId}:`, error);
          clearTimeout(timeoutId);
          
          if (fallbackToPolling) {
            this.startHttpPolling(simulationId);
            resolve();
          } else {
            this.handleError(simulationId, error);
            reject(error);
          }
        };

        ws.onclose = (event) => {
          console.log(`🔌 [Phase30] WebSocket closed for ${simulationId} - Code: ${event.code}`);
          clearTimeout(timeoutId);
          
          this.cleanup(simulationId);
          
          // Phase 30: Auto-reconnect for unexpected closures
          if (event.code !== 1000 && this.shouldReconnect(simulationId)) {
            this.attemptReconnect(simulationId, options);
          } else if (fallbackToPolling) {
            this.startHttpPolling(simulationId);
          }
        };

        this.connectionTimeouts.set(simulationId, timeoutId);

      } catch (error) {
        console.error(`💥 [Phase30] Failed to create WebSocket for ${simulationId}:`, error);
        
        if (fallbackToPolling) {
          this.startHttpPolling(simulationId);
          resolve();
        } else {
          reject(error);
        }
      }
    });
  }

  /**
   * Phase 30: Enhanced message handling with iteration tracking
   */
  handleMessage(simulationId, data) {
    try {
      const message = JSON.parse(data);
      const timestamp = Date.now();
      
      console.log(`📊 [Phase30] Message for ${simulationId}:`, message);
      
      // 🚀 CRITICAL: Handle simulation ID mapping
      if (message.type === 'simulation_id_mapping') {
        console.log(`🚀 [Enhanced WebSocket] ID mapping received: ${message.temp_id} -> ${message.real_id}`);
        
        // Trigger Redux action to update simulation ID mapping
        if (window.store && window.store.dispatch) {
          import('../store/simulationSlice').then(({ updateSimulationId }) => {
            window.store.dispatch(updateSimulationId({
              tempId: message.temp_id,
              realId: message.real_id,
              status: 'running'
            }));
            console.log(`🔄 [Enhanced WebSocket] Dispatched ID mapping to Redux: ${message.temp_id} -> ${message.real_id}`);
          }).catch(err => {
            console.error('❌ [Enhanced WebSocket] Failed to import updateSimulationId:', err);
          });
        }
        
        // Move WebSocket connection from temp_id to real_id
        if (this.connections.has(message.temp_id)) {
          const tempConnection = this.connections.get(message.temp_id);
          const tempCallback = this.progressCallbacks.get(message.temp_id);
          const tempProgress = this.lastProgressUpdate.get(message.temp_id);
          
          // Move all data to real_id
          this.connections.set(message.real_id, tempConnection);
          if (tempCallback) this.progressCallbacks.set(message.real_id, tempCallback);
          if (tempProgress) this.lastProgressUpdate.set(message.real_id, tempProgress);
          
          // Clean up temp_id references
          this.connections.delete(message.temp_id);
          this.progressCallbacks.delete(message.temp_id);
          this.lastProgressUpdate.delete(message.temp_id);
          
          console.log(`🔄 [Enhanced WebSocket] Moved connection: ${message.temp_id} -> ${message.real_id}`);
        }
        
        return; // Don't process as normal progress message
      }
      
      // Update last progress data
      this.lastProgressUpdate.set(simulationId, {
        progress: message.progress || 0,
        iteration: message.iteration || 0,
        totalIterations: message.total_iterations || 1000,
        timestamp: timestamp,
        phase: message.phase || 'simulation',
        status: message.status || 'running'
      });
      
      // Call progress callback
      const callback = this.progressCallbacks.get(simulationId);
      if (callback) {
        const progressData = {
          progress: message.progress || 0,
          iteration: message.iteration || 0,
          totalIterations: message.total_iterations || 1000,
          timestamp: timestamp,
          source: 'websocket',
          ...message
        };
        
        callback(progressData);
        
        // 🚀 NEW: Update progress recovery with latest data
        progressRecovery.updateLastKnownProgress(simulationId, {
          progress_percentage: progressData.progress,
          status: progressData.status || 'running',
          stage_description: progressData.stage_description || progressData.message,
          timestamp: timestamp
        });
      }
      
      // Check for completion
      if (message.progress >= 100 || message.status === 'completed') {
        this.handleCompletion(simulationId, message);
      }
      
    } catch (error) {
      console.error(`💥 [Phase30] Failed to parse WebSocket message for ${simulationId}:`, error);
      console.error('Raw message data:', data);
    }
  }

  /**
   * Phase 30: Progress interpolation for smooth animation
   */
  startProgressInterpolation(simulationId) {
    if (this.progressInterpolators.has(simulationId)) {
      return; // Already interpolating
    }
    
    console.log(`🎬 [Phase30] Starting progress interpolation for ${simulationId}`);
    
    const intervalId = setInterval(() => {
      const lastUpdate = this.lastProgressUpdate.get(simulationId);
      if (!lastUpdate) return;
      
      const now = Date.now();
      const timeSinceUpdate = now - lastUpdate.timestamp;
      
      // If more than 5 seconds since last update, estimate progress
      if (timeSinceUpdate > 5000 && lastUpdate.progress < 100) {
        const estimatedProgress = Math.min(
          lastUpdate.progress + (timeSinceUpdate / 1000) * 1.2, // ~1.2% per second
          99.5 // Don't exceed 99.5% from estimation
        );
        
        const callback = this.progressCallbacks.get(simulationId);
        if (callback) {
          callback({
            progress: estimatedProgress,
            iteration: Math.floor((estimatedProgress / 100) * lastUpdate.totalIterations),
            totalIterations: lastUpdate.totalIterations,
            timestamp: now,
            source: 'interpolation',
            estimated: true
          });
        }
      }
    }, 1000);
    
    this.progressInterpolators.set(simulationId, intervalId);
  }

  /**
   * Phase 30: HTTP polling fallback
   */
  startHttpPolling(simulationId) {
    if (this.pollingIntervals.has(simulationId)) {
      return; // Already polling
    }
    
    console.log(`🔄 [Phase30] Starting HTTP polling fallback for ${simulationId}`);
    
    const poll = async () => {
      try {
        // CRITICAL FIX: Add cache-busting to ensure fresh data
        const cacheBuster = Date.now();
        const response = await fetch(`/api/simulations/${simulationId}/progress?_t=${cacheBuster}`, {
          cache: 'no-cache',
          headers: {
            'Cache-Control': 'no-cache',
            'Pragma': 'no-cache'
          }
        });
        if (response.ok) {
          const data = await response.json();
          const timestamp = Date.now();
          
          // Update progress via callback
          const callback = this.progressCallbacks.get(simulationId);
          if (callback) {
            // CRITICAL FIX: Map API response fields correctly
            callback({
              progress: data.progress_percentage || 0,        // ✅ API uses "progress_percentage" 
              iteration: data.current_iteration || 0,         // ✅ API uses "current_iteration"
              totalIterations: data.total_iterations || 1000, // ✅ API uses "total_iterations"
              timestamp: timestamp,
              source: 'http_polling',
              ...data
            });
          }
          
          // ENHANCED COMPLETION DETECTION
          const isCompleted = data.progress_percentage >= 100 || 
                             data.status === 'completed' || 
                             data.current_iteration >= data.total_iterations;
          
          if (isCompleted) {
            console.log(`🎉 [Phase30] Simulation ${simulationId} completed - triggering results display`);
            
            // Force final 100% progress update
            const callback = this.progressCallbacks.get(simulationId);
            if (callback) {
              callback({
                progress: 100,
                iteration: data.current_iteration || data.total_iterations,
                totalIterations: data.total_iterations || 1000,
                timestamp: timestamp,
                source: 'completion_forced',
                status: 'completed',
                completed: true,
                ...data
              });
            }
            
            this.handleCompletion(simulationId, data);
            
            // Stop polling for completed simulation
            if (this.httpPollingIntervals.has(simulationId)) {
              clearInterval(this.httpPollingIntervals.get(simulationId));
              this.httpPollingIntervals.delete(simulationId);
            }
          }
        }
      } catch (error) {
        console.error(`❌ [Phase30] HTTP polling error for ${simulationId}:`, error);
      }
    };
    
    // Start polling immediately, then continue at intervals
    poll();
    const intervalId = setInterval(poll, this.pollingRate);
    this.pollingIntervals.set(simulationId, intervalId);
  }

  /**
   * Phase 30: Handle simulation completion
   */
  handleCompletion(simulationId, data) {
    console.log(`🎉 [Phase30] Simulation ${simulationId} completed`);
    
    const callback = this.completionCallbacks.get(simulationId);
    if (callback) {
      callback(data);
    }
    
    // 🚀 NEW: Clean up progress recovery
    progressRecovery.cleanupSimulation(simulationId);
    
    // Clean up connections and intervals
    this.disconnect(simulationId);
  }

  /**
   * Phase 30: Handle errors with proper callbacks
   */
  handleError(simulationId, error) {
    console.error(`💥 [Phase30] Error for simulation ${simulationId}:`, error);
    
    const callback = this.errorCallbacks.get(simulationId);
    if (callback) {
      callback(error);
    }
  }

  /**
   * Phase 30: Reconnection logic
   */
  shouldReconnect(simulationId) {
    const attempts = this.reconnectAttempts.get(simulationId) || 0;
    return attempts < this.maxReconnectAttempts;
  }

  async attemptReconnect(simulationId, options) {
    const attempts = this.reconnectAttempts.get(simulationId) || 0;
    this.reconnectAttempts.set(simulationId, attempts + 1);
    
    console.log(`🔄 [Phase30] Reconnection attempt ${attempts + 1}/${this.maxReconnectAttempts} for ${simulationId}`);
    
    await new Promise(resolve => setTimeout(resolve, this.reconnectDelay * attempts));
    
    try {
      await this.connect(simulationId, options);
    } catch (error) {
      console.error(`💥 [Phase30] Reconnection failed for ${simulationId}:`, error);
      this.startHttpPolling(simulationId); // Fallback to polling
    }
  }

  resetReconnectAttempts(simulationId) {
    this.reconnectAttempts.delete(simulationId);
  }

  /**
   * Phase 30: Enhanced cleanup
   */
  cleanup(simulationId) {
    // Clear WebSocket connection
    const ws = this.connections.get(simulationId);
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.close();
    }
    this.connections.delete(simulationId);
    
    // Clear intervals
    const interpolatorId = this.progressInterpolators.get(simulationId);
    if (interpolatorId) {
      clearInterval(interpolatorId);
      this.progressInterpolators.delete(simulationId);
    }
    
    const pollingId = this.pollingIntervals.get(simulationId);
    if (pollingId) {
      clearInterval(pollingId);
      this.pollingIntervals.delete(simulationId);
    }
    
    // Clear timeouts
    const timeoutId = this.connectionTimeouts.get(simulationId);
    if (timeoutId) {
      clearTimeout(timeoutId);
      this.connectionTimeouts.delete(simulationId);
    }
    
    console.log(`🧹 [Phase30] Cleaned up resources for ${simulationId}`);
  }

  /**
   * Phase 30: Disconnect and cleanup
   */
  disconnect(simulationId) {
    console.log(`🔌 [Phase30] Disconnecting simulation ${simulationId}`);
    
    this.cleanup(simulationId);
    
    // Clear callbacks
    this.progressCallbacks.delete(simulationId);
    this.errorCallbacks.delete(simulationId);
    this.completionCallbacks.delete(simulationId);
    this.lastProgressUpdate.delete(simulationId);
    this.reconnectAttempts.delete(simulationId);
  }

  /**
   * Phase 30: Get connection status
   */
  getConnectionStatus(simulationId) {
    const ws = this.connections.get(simulationId);
    const isPolling = this.pollingIntervals.has(simulationId);
    
    return {
      connected: ws && ws.readyState === WebSocket.OPEN,
      connecting: ws && ws.readyState === WebSocket.CONNECTING,
      polling: isPolling,
      lastUpdate: this.lastProgressUpdate.get(simulationId),
      reconnectAttempts: this.reconnectAttempts.get(simulationId) || 0
    };
  }

  /**
   * Phase 30: Disconnect all connections
   */
  disconnectAll() {
    console.log('🔌 [Phase30] Disconnecting all WebSocket connections');
    
    const simulationIds = Array.from(this.connections.keys());
    simulationIds.forEach(id => this.disconnect(id));
  }
}

// Export singleton instance
const enhancedWebSocketService = new EnhancedWebSocketService();
export default enhancedWebSocketService;