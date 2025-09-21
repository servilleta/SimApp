"""
WebSocket Manager for Real-time Simulation Progress Updates
Enhanced for Phase 2 Real-time Progress Reporting
"""

import asyncio
import json
import logging
from typing import Dict, List, Set
from fastapi import WebSocket, WebSocketDisconnect
import time
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class WebSocketManager:
    """Manages WebSocket connections for real-time simulation progress updates"""
    
    def __init__(self):
        # Store active connections by simulation_id
        self.active_connections: Dict[str, List[WebSocket]] = {}
        # Track connection metadata
        self.connection_metadata: Dict[WebSocket, Dict] = {}
        # Cleanup task
        self.cleanup_task = None
        self.cleanup_interval = 30  # seconds
        
    async def connect(self, websocket: WebSocket, simulation_id: str, user_id: str = None):
        """Accept a new WebSocket connection for a simulation"""
        try:
            await websocket.accept()
            
            # Initialize connections list for simulation if not exists
            if simulation_id not in self.active_connections:
                self.active_connections[simulation_id] = []
            
            # Add connection
            self.active_connections[simulation_id].append(websocket)
            
            # Store metadata
            self.connection_metadata[websocket] = {
                'simulation_id': simulation_id,
                'user_id': user_id,
                'connected_at': datetime.now(),
                'last_ping': datetime.now()
            }
            
            logger.info(f"✅ [WebSocket] Client connected to simulation {simulation_id} (User: {user_id})")
            logger.info(f"📊 [WebSocket] Active connections: {len(self.active_connections[simulation_id])} for {simulation_id}")
            
            # Start cleanup task if not already running
            if self.cleanup_task is None or self.cleanup_task.done():
                self.cleanup_task = asyncio.create_task(self._periodic_cleanup())
                
        except Exception as e:
            logger.error(f"❌ [WebSocket] Failed to connect: {e}")
            raise
    
    async def disconnect(self, websocket: WebSocket):
        """Remove a WebSocket connection"""
        try:
            metadata = self.connection_metadata.get(websocket)
            if metadata:
                simulation_id = metadata['simulation_id']
                user_id = metadata['user_id']
                
                # Remove from active connections
                if simulation_id in self.active_connections:
                    if websocket in self.active_connections[simulation_id]:
                        self.active_connections[simulation_id].remove(websocket)
                        
                    # Clean up empty simulation entries
                    if not self.active_connections[simulation_id]:
                        del self.active_connections[simulation_id]
                
                # Remove metadata
                del self.connection_metadata[websocket]
                
                logger.info(f"🔌 [WebSocket] Client disconnected from simulation {simulation_id} (User: {user_id})")
            else:
                logger.warning(f"🔌 [WebSocket] Attempted to disconnect unknown WebSocket")
                
        except Exception as e:
            logger.error(f"❌ [WebSocket] Error during disconnect: {e}")
    
    async def send_progress_update(self, simulation_id: str, progress_data: dict):
        """Send progress update to all connected clients for a simulation"""
        if simulation_id not in self.active_connections:
            logger.debug(f"📡 [WebSocket] No active connections for simulation {simulation_id}")
            return
        
        # Prepare message
        message = {
            'type': 'progress_update',
            'simulation_id': simulation_id,
            'data': progress_data,
            'timestamp': time.time()
        }
        
        message_str = json.dumps(message)
        connections = self.active_connections[simulation_id].copy()
        
        # Send to all connected clients
        disconnected_websockets = []
        for websocket in connections:
            try:
                await websocket.send_text(message_str)
                # Update last ping
                if websocket in self.connection_metadata:
                    self.connection_metadata[websocket]['last_ping'] = datetime.now()
                    
            except WebSocketDisconnect:
                logger.info(f"🔌 [WebSocket] Client disconnected during send for {simulation_id}")
                disconnected_websockets.append(websocket)
            except Exception as e:
                logger.error(f"❌ [WebSocket] Failed to send message: {e}")
                disconnected_websockets.append(websocket)
        
        # Clean up disconnected websockets
        for websocket in disconnected_websockets:
            await self.disconnect(websocket)
    
    async def send_simulation_complete(self, simulation_id: str, results_data: dict):
        """Send simulation completion notification"""
        if simulation_id not in self.active_connections:
            return
        
        message = {
            'type': 'simulation_complete',
            'simulation_id': simulation_id,
            'data': results_data,
            'timestamp': time.time()
        }
        
        await self.send_progress_update(simulation_id, message)
    
    async def send_simulation_error(self, simulation_id: str, error_message: str):
        """Send simulation error notification"""
        if simulation_id not in self.active_connections:
            return
        
        message = {
            'type': 'simulation_error',
            'simulation_id': simulation_id,
            'error': error_message,
            'timestamp': time.time()
        }
        
        await self.send_progress_update(simulation_id, message)
    
    async def send_simulation_id_mapping(self, temp_id: str, real_id: str):
        """
        🚀 CRITICAL: Send immediate simulation ID mapping via WebSocket
        
        This enables instant progress tracking by broadcasting the real simulation ID
        to all clients listening on the temporary ID WebSocket connection.
        """
        # Send to both temp_id and real_id connections if they exist
        temp_connections = self.active_connections.get(temp_id, [])
        real_connections = self.active_connections.get(real_id, [])
        
        if not temp_connections and not real_connections:
            logger.debug(f"📡 [WebSocket] No active connections for ID mapping {temp_id} -> {real_id}")
            return
        
        # Prepare ID mapping message
        message = {
            'type': 'simulation_id_mapping',
            'temp_id': temp_id,
            'real_id': real_id,
            'timestamp': time.time(),
            'message': f'Simulation ID ready: {real_id}'
        }
        
        message_str = json.dumps(message)
        logger.info(f"🚀 [WebSocket] Broadcasting ID mapping: {temp_id} -> {real_id}")
        
        # Send to all connections for both IDs
        all_connections = list(set(temp_connections + real_connections))
        disconnected_websockets = []
        
        for websocket in all_connections:
            try:
                await websocket.send_text(message_str)
                logger.debug(f"📡 [WebSocket] Sent ID mapping to client")
                
                # Update last ping
                if websocket in self.connection_metadata:
                    self.connection_metadata[websocket]['last_ping'] = datetime.now()
                    
            except WebSocketDisconnect:
                logger.info(f"🔌 [WebSocket] Client disconnected during ID mapping send")
                disconnected_websockets.append(websocket)
            except Exception as e:
                logger.error(f"❌ [WebSocket] Failed to send ID mapping: {e}")
                disconnected_websockets.append(websocket)
        
        # Clean up disconnected websockets
        for websocket in disconnected_websockets:
            await self.disconnect(websocket)
        
        # CRITICAL: Move connections from temp_id to real_id
        if temp_id in self.active_connections and real_id not in self.active_connections:
            self.active_connections[real_id] = self.active_connections[temp_id]
            del self.active_connections[temp_id]
            
            # Update metadata for all moved connections
            for websocket in self.active_connections[real_id]:
                if websocket in self.connection_metadata:
                    self.connection_metadata[websocket]['simulation_id'] = real_id
            
            logger.info(f"🔄 [WebSocket] Moved {len(self.active_connections[real_id])} connections from {temp_id} to {real_id}")
        elif temp_id in self.active_connections and real_id in self.active_connections:
            # Merge connections if both exist
            self.active_connections[real_id].extend(self.active_connections[temp_id])
            del self.active_connections[temp_id]
            logger.info(f"🔗 [WebSocket] Merged connections: {temp_id} -> {real_id}")
    
    def get_connection_count(self, simulation_id: str = None) -> int:
        """Get number of active connections"""
        if simulation_id:
            return len(self.active_connections.get(simulation_id, []))
        return sum(len(connections) for connections in self.active_connections.values())
    
    def get_active_simulations(self) -> List[str]:
        """Get list of simulation IDs with active connections"""
        return list(self.active_connections.keys())
    
    async def _periodic_cleanup(self):
        """Periodically clean up stale connections"""
        while True:
            try:
                await asyncio.sleep(self.cleanup_interval)
                await self._cleanup_stale_connections()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"❌ [WebSocket] Cleanup error: {e}")
    
    async def _cleanup_stale_connections(self):
        """Remove connections that haven't been active recently"""
        now = datetime.now()
        stale_threshold = timedelta(minutes=10)  # 10 minutes
        
        stale_websockets = []
        for websocket, metadata in self.connection_metadata.items():
            if now - metadata['last_ping'] > stale_threshold:
                stale_websockets.append(websocket)
        
        for websocket in stale_websockets:
            logger.info(f"🧹 [WebSocket] Removing stale connection")
            await self.disconnect(websocket)

# Global WebSocket manager instance
websocket_manager = WebSocketManager() 