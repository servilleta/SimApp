/**
 * WebSocket Service for Real-time Simulation Progress Updates
 * Enhanced for Phase 2 Real-time Progress Reporting
 */

class WebSocketService {
  constructor() {
    this.connections = new Map(); // simulationId -> WebSocket
    this.callbacks = new Map(); // simulationId -> callback function
    this.reconnectAttempts = new Map(); // simulationId -> attempt count
    this.maxReconnectAttempts = 5;
    this.reconnectDelay = 500; // CRITICAL FIX: Reduced from 2000ms to 500ms for faster reconnection
    this.heartbeatInterval = 5000; // 5 seconds - PHASE 22 FIX: Prevent keepalive timeout
    this.heartbeatTimers = new Map();
  }

  /**
   * Connect to WebSocket for a specific simulation
   * @param {string} simulationId - The simulation ID
   * @param {function} onProgressUpdate - Callback for progress updates
   * @param {function} onError - Optional error callback
   * @param {function} onComplete - Optional completion callback
   */
  connect(simulationId, onProgressUpdate, onError = null, onComplete = null) {
    // 🚀 PHASE 27: Add performance monitoring for WebSocket connection timing
    const connectionStartMark = `phase27-ws-${simulationId}-start`;
    const connectionEndMark = `phase27-ws-${simulationId}-connected`;
    const connectionMeasure = `phase27-ws-${simulationId}-duration`;
    
    if (this.connections.has(simulationId)) {
      console.log(`🔄 [WebSocket] Already connected to simulation ${simulationId}`);
      return;
    }

    try {
      // 🚀 PHASE 27: Mark connection start time
      performance.mark(connectionStartMark);
      
      // Use current domain for WebSocket connection
      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      const host = window.location.host;
      const wsUrl = `${protocol}//${host}/ws/simulations/${simulationId}`;
      
      console.log(`🚀 [PHASE27] Connecting to ${wsUrl} for simulation: ${simulationId}`);
      
      const ws = new WebSocket(wsUrl);
      
      ws.onopen = () => {
        // 🚀 PHASE 27: Measure connection time
        performance.mark(connectionEndMark);
        performance.measure(connectionMeasure, connectionStartMark, connectionEndMark);
        
        const connectionTime = performance.getEntriesByName(connectionMeasure)[0]?.duration || 0;
        console.log(`✅ [PHASE27] Connected to simulation ${simulationId} in ${connectionTime.toFixed(2)}ms`);
        
        this.connections.set(simulationId, ws);
        this.callbacks.set(simulationId, onProgressUpdate);
        this.reconnectAttempts.set(simulationId, 0);
        
        // Start heartbeat
        this.startHeartbeat(simulationId);
        
        // 🚀 PHASE 27: Call completion callback with timing info
        if (onComplete) {
          onComplete({ connectionTime, simulationId });
        }
      };
      
      ws.onmessage = (event) => {
        try {
          const message = JSON.parse(event.data);
          
          if (message.type === 'simulation_id_mapping') {
            // 🚀 CRITICAL: Handle instant simulation ID mapping
            console.log(`🚀 [WebSocket] ID mapping received: ${message.temp_id} -> ${message.real_id}`);
            
            // Trigger Redux action to update simulation ID mapping
            if (window.store && window.store.dispatch) {
              // Import the action creator dynamically
              import('../store/simulationSlice').then(({ updateSimulationId }) => {
                window.store.dispatch(updateSimulationId({
                  tempId: message.temp_id,
                  realId: message.real_id,
                  status: 'running'
                }));
                console.log(`🔄 [WebSocket] Dispatched ID mapping to Redux: ${message.temp_id} -> ${message.real_id}`);
              }).catch(err => {
                console.error('❌ [WebSocket] Failed to import updateSimulationId:', err);
              });
            }
            
            // Move WebSocket connection from temp_id to real_id
            if (this.connections.has(message.temp_id)) {
              const tempConnection = this.connections.get(message.temp_id);
              this.connections.set(message.real_id, tempConnection);
              this.connections.delete(message.temp_id);
              console.log(`🔄 [WebSocket] Moved connection: ${message.temp_id} -> ${message.real_id}`);
            }
            
          } else if (message.type === 'progress_update' && message.data) {
            // Add simulation_id to progress data for frontend component matching
            const progressData = {
              ...message.data,
              simulation_id: simulationId,
              timestamp: message.timestamp
            };
            
            // 🚀 PHASE 27: Enhanced progress logging with timing
            const progress = progressData.progress || progressData.progress_percentage;
            console.log(`🚀 [PHASE27] Progress update for ${simulationId}: ${progress}% at ${new Date().toISOString()}`);
            
            if (onProgressUpdate) {
              onProgressUpdate(progressData);
            }
          } else if (message.type === 'simulation_complete') {
            console.log(`🎉 [WebSocket] Simulation ${simulationId} completed`);
            if (onComplete) {
              onComplete(message.data);
            }
            this.disconnect(simulationId);
          } else if (message.type === 'simulation_error') {
            console.error(`❌ [WebSocket] Simulation ${simulationId} error:`, message.error);
            if (onError) {
              onError(message.error);
            }
            this.disconnect(simulationId);
          }
        } catch (error) {
          console.error(`❌ [WebSocket] Failed to parse message:`, error);
        }
      };
      
      ws.onerror = (error) => {
        console.error(`❌ [WebSocket] Error for simulation ${simulationId}:`, error);
        if (onError) {
          onError(`WebSocket connection error: ${error.message || 'Unknown error'}`);
        }
      };
      
      ws.onclose = (event) => {
        console.log(`🔌 [WebSocket] Connection closed for simulation ${simulationId}`, event);
        this.stopHeartbeat(simulationId);
        
        // Attempt reconnection if not a clean close
        if (event.code !== 1000) {
          this.attemptReconnect(simulationId, onProgressUpdate, onError, onComplete);
        } else {
          this.cleanup(simulationId);
        }
      };
      
    } catch (error) {
      console.error(`❌ [WebSocket] Failed to create connection:`, error);
      if (onError) {
        onError(`Failed to establish WebSocket connection: ${error.message}`);
      }
    }
  }

  /**
   * Disconnect from a simulation's WebSocket
   * @param {string} simulationId - The simulation ID
   */
  disconnect(simulationId) {
    const ws = this.connections.get(simulationId);
    if (ws) {
      console.log(`🔌 [WebSocket] Disconnecting from simulation ${simulationId}`);
      ws.close(1000, 'Client disconnect');
    }
    this.cleanup(simulationId);
  }

  /**
   * Disconnect from all WebSocket connections
   */
  disconnectAll() {
    console.log(`🔌 [WebSocket] Disconnecting from all simulations`);
    for (const simulationId of this.connections.keys()) {
      this.disconnect(simulationId);
    }
  }

  /**
   * Check if connected to a simulation
   * @param {string} simulationId - The simulation ID
   * @returns {boolean} - Connection status
   */
  isConnected(simulationId) {
    const ws = this.connections.get(simulationId);
    return ws && ws.readyState === WebSocket.OPEN;
  }

  /**
   * Get connection status for a simulation
   * @param {string} simulationId - The simulation ID
   * @returns {string} - Connection status
   */
  getConnectionStatus(simulationId) {
    const ws = this.connections.get(simulationId);
    if (!ws) return 'disconnected';
    
    switch (ws.readyState) {
      case WebSocket.CONNECTING: return 'connecting';
      case WebSocket.OPEN: return 'connected';
      case WebSocket.CLOSING: return 'closing';
      case WebSocket.CLOSED: return 'closed';
      default: return 'unknown';
    }
  }

  /**
   * Attempt to reconnect to a simulation
   * @private
   */
  attemptReconnect(simulationId, onProgressUpdate, onError, onComplete) {
    const attempts = this.reconnectAttempts.get(simulationId) || 0;
    
    if (attempts >= this.maxReconnectAttempts) {
      console.error(`❌ [WebSocket] Max reconnection attempts reached for ${simulationId}`);
      if (onError) {
        onError('WebSocket connection lost and max reconnection attempts reached');
      }
      this.cleanup(simulationId);
      return;
    }

    const delay = this.reconnectDelay * Math.pow(2, attempts); // Exponential backoff
    console.log(`🔄 [WebSocket] Reconnecting to ${simulationId} in ${delay}ms (attempt ${attempts + 1})`);
    
    setTimeout(() => {
      this.reconnectAttempts.set(simulationId, attempts + 1);
      this.connect(simulationId, onProgressUpdate, onError, onComplete);
    }, delay);
  }

  /**
   * Start heartbeat for a connection
   * @private
   */
  startHeartbeat(simulationId) {
    const timer = setInterval(() => {
      const ws = this.connections.get(simulationId);
      if (ws && ws.readyState === WebSocket.OPEN) {
        try {
          ws.send(JSON.stringify({ type: 'ping', timestamp: Date.now() }));
        } catch (error) {
          console.error(`❌ [WebSocket] Heartbeat failed for ${simulationId}:`, error);
          this.disconnect(simulationId);
        }
      } else {
        this.stopHeartbeat(simulationId);
      }
    }, this.heartbeatInterval);
    
    this.heartbeatTimers.set(simulationId, timer);
  }

  /**
   * Stop heartbeat for a connection
   * @private
   */
  stopHeartbeat(simulationId) {
    const timer = this.heartbeatTimers.get(simulationId);
    if (timer) {
      clearInterval(timer);
      this.heartbeatTimers.delete(simulationId);
    }
  }

  /**
   * Clean up resources for a simulation
   * @private
   */
  cleanup(simulationId) {
    this.connections.delete(simulationId);
    this.callbacks.delete(simulationId);
    this.reconnectAttempts.delete(simulationId);
    this.stopHeartbeat(simulationId);
  }

  /**
   * 🚀 PHASE 27: Check if WebSocket is connected for a simulation
   * @param {string} simulationId - The simulation ID
   * @returns {boolean} - True if connected and ready
   */
  isConnected(simulationId) {
    const ws = this.connections.get(simulationId);
    return ws && ws.readyState === WebSocket.OPEN;
  }

  /**
   * 🚀 PHASE 27: Set progress handler for existing connection
   * @param {string} simulationId - The simulation ID
   * @param {function} handler - Progress update handler
   */
  setProgressHandler(simulationId, handler) {
    if (this.isConnected(simulationId)) {
      console.log(`🚀 [PHASE27] Setting progress handler for existing connection: ${simulationId}`);
      this.callbacks.set(simulationId, handler);
      return true;
    } else {
      console.warn(`🚀 [PHASE27] Cannot set handler - no active connection for: ${simulationId}`);
      return false;
    }
  }

  /**
   * Get all active connections
   * @returns {Array<string>} - Array of simulation IDs
   */
  getActiveConnections() {
    return Array.from(this.connections.keys()).filter(id => this.isConnected(id));
  }
}

// Export singleton instance
const websocketService = new WebSocketService();

// Cleanup on page unload
window.addEventListener('beforeunload', () => {
  websocketService.disconnectAll();
});

export default websocketService; 