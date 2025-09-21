"""
WebSocket Manager for Real-Time Progress Updates
Handles WebSocket connections and broadcasts for simulation progress.
"""

import json
import logging
from typing import Dict, Set
from fastapi import WebSocket
import asyncio

logger = logging.getLogger(__name__)

class WebSocketManager:
    """Manages WebSocket connections for real-time simulation progress updates"""
    
    def __init__(self):
        # Map of simulation_id -> set of WebSocket connections
        self.connections: Dict[str, Set[WebSocket]] = {}
        
    def connect(self, simulation_id: str, websocket: WebSocket):
        """Register a WebSocket connection for a simulation"""
        if simulation_id not in self.connections:
            self.connections[simulation_id] = set()
        
        self.connections[simulation_id].add(websocket)
        logger.info(f"📡 WebSocket connected for simulation {simulation_id}. Total connections: {len(self.connections[simulation_id])}")
        
    def disconnect(self, simulation_id: str, websocket: WebSocket):
        """Remove a WebSocket connection"""
        if simulation_id in self.connections:
            self.connections[simulation_id].discard(websocket)
            if not self.connections[simulation_id]:
                # No more connections for this simulation
                del self.connections[simulation_id]
            logger.info(f"📡 WebSocket disconnected for simulation {simulation_id}")
            
    async def send_progress_update(self, simulation_id: str, progress_data: dict):
        """Send progress update to all connected clients for a simulation"""
        if simulation_id not in self.connections:
            logger.debug(f"No WebSocket connections for simulation {simulation_id}")
            return
            
        # Convert progress data to JSON
        message = json.dumps({
            "type": "progress_update",
            "simulation_id": simulation_id,
            "progress": progress_data.get("progress_percentage", 0),
            "stage": progress_data.get("stage", "unknown"),
            "iteration": progress_data.get("current_iteration", 0),
            "total": progress_data.get("total_iterations", 0),
            "status": progress_data.get("status", "running")
        })
        
        # Send to all connected clients
        disconnected = set()
        for websocket in self.connections[simulation_id].copy():
            try:
                await websocket.send_text(message)
                logger.debug(f"📨 Sent progress update to WebSocket for {simulation_id}: {progress_data.get('progress_percentage', 0)}%")
            except Exception as e:
                logger.warning(f"Failed to send WebSocket message to {simulation_id}: {e}")
                disconnected.add(websocket)
                
        # Clean up disconnected websockets
        for websocket in disconnected:
            self.disconnect(simulation_id, websocket)

# Global instance
websocket_manager = WebSocketManager()
