#!/usr/bin/env python3
"""
Paperspace API Manager for SimApp Multi-Server Architecture
===========================================================

This module provides programmatic control over Paperspace machines for:
- On-demand server startup/shutdown
- Cost optimization through intelligent scaling
- Integration with Kubernetes auto-scaling
- Blue-Green deployment support

Author: SimApp DevOps Team
Date: September 21, 2025
"""

import os
import sys
import json
import time
import logging
import requests
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/home/paperspace/SimApp/logs/paperspace_api.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class MachineState(Enum):
    """Paperspace machine states"""
    STARTING = "starting"
    RUNNING = "running" 
    STOPPING = "stopping"
    STOPPED = "stopped"
    RESTARTING = "restarting"
    PENDING = "pending"
    ERROR = "error"

@dataclass
class PaperspaceMachine:
    """Represents a Paperspace machine instance"""
    id: str
    name: str
    state: str
    public_ip: Optional[str]
    private_ip: Optional[str]
    region: str
    machine_type: str
    os: str
    created_at: str
    updated_at: str

class PaperspaceAPIManager:
    """
    Manages Paperspace machines via API for SimApp scaling
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Paperspace API Manager
        
        Args:
            api_key: Paperspace API key (defaults to env var PAPERSPACE_API_KEY)
        """
        self.api_key = api_key or os.getenv('PAPERSPACE_API_KEY')
        if not self.api_key:
            raise ValueError("❌ PAPERSPACE_API_KEY environment variable not set!")
        
        self.base_url = "https://api.paperspace.io"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # SimApp server configuration
        self.primary_server_ip = "64.71.146.187"
        self.secondary_server_ip = "72.52.107.230"
        self.secondary_machine_id = None  # To be discovered
        
        logger.info("🚀 Paperspace API Manager initialized")

    def _make_request(self, method: str, endpoint: str, data: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Make authenticated API request to Paperspace
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint path
            data: Request payload for POST requests
            
        Returns:
            API response as dictionary
            
        Raises:
            requests.RequestException: If API request fails
        """
        url = f"{self.base_url}{endpoint}"
        
        try:
            if method.upper() == "GET":
                response = requests.get(url, headers=self.headers, timeout=30)
            elif method.upper() == "POST":
                response = requests.post(url, headers=self.headers, json=data, timeout=30)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ API request failed: {e}")
            raise

    def list_machines(self) -> List[PaperspaceMachine]:
        """
        List all machines in the account
        
        Returns:
            List of PaperspaceMachine objects
        """
        logger.info("📋 Fetching machine list from Paperspace API...")
        
        try:
            response = self._make_request("GET", "/machines/getMachines")
            machines = []
            
            for machine_data in response:
                machine = PaperspaceMachine(
                    id=machine_data.get('id'),
                    name=machine_data.get('name'),
                    state=machine_data.get('state'),
                    public_ip=machine_data.get('publicIpAddress'),
                    private_ip=machine_data.get('privateIpAddress'),
                    region=machine_data.get('region'),
                    machine_type=machine_data.get('machineType'),
                    os=machine_data.get('os'),
                    created_at=machine_data.get('dtCreated'),
                    updated_at=machine_data.get('dtLastRun')
                )
                machines.append(machine)
            
            logger.info(f"✅ Found {len(machines)} machines")
            return machines
            
        except Exception as e:
            logger.error(f"❌ Failed to list machines: {e}")
            return []

    def find_secondary_server(self) -> Optional[str]:
        """
        Find the machine ID for the secondary server (72.52.107.230)
        
        Returns:
            Machine ID if found, None otherwise
        """
        logger.info(f"🔍 Searching for secondary server: {self.secondary_server_ip}")
        
        machines = self.list_machines()
        for machine in machines:
            if machine.public_ip == self.secondary_server_ip:
                self.secondary_machine_id = machine.id
                logger.info(f"✅ Found secondary server: {machine.id} ({machine.name})")
                return machine.id
        
        logger.warning(f"⚠️ Secondary server {self.secondary_server_ip} not found")
        return None

    def get_machine_status(self, machine_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed status of a specific machine
        
        Args:
            machine_id: Paperspace machine ID
            
        Returns:
            Machine status dictionary
        """
        logger.info(f"📊 Getting status for machine: {machine_id}")
        
        try:
            response = self._make_request("GET", f"/machines/getMachinePublic?machineId={machine_id}")
            logger.info(f"✅ Machine {machine_id} status: {response.get('state')}")
            return response
            
        except Exception as e:
            logger.error(f"❌ Failed to get machine status: {e}")
            return None

    def start_machine(self, machine_id: str) -> bool:
        """
        Start a Paperspace machine
        
        Args:
            machine_id: Paperspace machine ID
            
        Returns:
            True if successful, False otherwise
        """
        logger.info(f"🚀 Starting machine: {machine_id}")
        
        try:
            response = self._make_request("POST", "/machines/startMachine", {"machineId": machine_id})
            
            if response.get('message') == 'Machine started':
                logger.info(f"✅ Machine {machine_id} start command sent successfully")
                return True
            else:
                logger.warning(f"⚠️ Unexpected response: {response}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to start machine: {e}")
            return False

    def stop_machine(self, machine_id: str) -> bool:
        """
        Stop a Paperspace machine
        
        Args:
            machine_id: Paperspace machine ID
            
        Returns:
            True if successful, False otherwise
        """
        logger.info(f"🛑 Stopping machine: {machine_id}")
        
        try:
            response = self._make_request("POST", "/machines/stopMachine", {"machineId": machine_id})
            
            if response.get('message') == 'Machine stopped':
                logger.info(f"✅ Machine {machine_id} stop command sent successfully")
                return True
            else:
                logger.warning(f"⚠️ Unexpected response: {response}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to stop machine: {e}")
            return False

    def wait_for_state(self, machine_id: str, target_state: str, timeout: int = 300) -> bool:
        """
        Wait for machine to reach target state
        
        Args:
            machine_id: Paperspace machine ID
            target_state: Target state to wait for
            timeout: Maximum wait time in seconds
            
        Returns:
            True if target state reached, False if timeout
        """
        logger.info(f"⏳ Waiting for machine {machine_id} to reach state: {target_state}")
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            status = self.get_machine_status(machine_id)
            if status and status.get('state') == target_state:
                logger.info(f"✅ Machine {machine_id} reached state: {target_state}")
                return True
            
            logger.info(f"⏳ Current state: {status.get('state') if status else 'unknown'} - waiting...")
            time.sleep(10)
        
        logger.error(f"❌ Timeout waiting for machine {machine_id} to reach state: {target_state}")
        return False

    def scale_up_secondary_server(self) -> bool:
        """
        Scale up by starting the secondary server
        
        Returns:
            True if successful, False otherwise
        """
        logger.info("📈 SCALE UP: Starting secondary server for increased capacity")
        
        if not self.secondary_machine_id:
            machine_id = self.find_secondary_server()
            if not machine_id:
                logger.error("❌ Cannot find secondary server machine ID")
                return False
        else:
            machine_id = self.secondary_machine_id
        
        # Check current status
        status = self.get_machine_status(machine_id)
        if status and status.get('state') == 'running':
            logger.info("✅ Secondary server is already running")
            return True
        
        # Start the machine
        if self.start_machine(machine_id):
            # Wait for it to be running
            if self.wait_for_state(machine_id, 'running', timeout=600):
                logger.info("🎉 Secondary server successfully started and running!")
                return True
        
        return False

    def scale_down_secondary_server(self) -> bool:
        """
        Scale down by stopping the secondary server
        
        Returns:
            True if successful, False otherwise
        """
        logger.info("📉 SCALE DOWN: Stopping secondary server to save costs")
        
        if not self.secondary_machine_id:
            machine_id = self.find_secondary_server()
            if not machine_id:
                logger.error("❌ Cannot find secondary server machine ID")
                return False
        else:
            machine_id = self.secondary_machine_id
        
        # Check current status
        status = self.get_machine_status(machine_id)
        if status and status.get('state') == 'stopped':
            logger.info("✅ Secondary server is already stopped")
            return True
        
        # Stop the machine
        if self.stop_machine(machine_id):
            # Wait for it to be stopped
            if self.wait_for_state(machine_id, 'stopped', timeout=300):
                logger.info("🎉 Secondary server successfully stopped!")
                return True
        
        return False

    def get_cluster_status(self) -> Dict[str, Any]:
        """
        Get comprehensive status of the SimApp cluster
        
        Returns:
            Dictionary with cluster status information
        """
        logger.info("📊 Getting comprehensive cluster status...")
        
        machines = self.list_machines()
        primary = None
        secondary = None
        
        for machine in machines:
            if machine.public_ip == self.primary_server_ip:
                primary = machine
            elif machine.public_ip == self.secondary_server_ip:
                secondary = machine
                self.secondary_machine_id = machine.id
        
        status = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "primary_server": {
                "ip": self.primary_server_ip,
                "status": primary.state if primary else "unknown",
                "machine_id": primary.id if primary else None,
                "name": primary.name if primary else None
            },
            "secondary_server": {
                "ip": self.secondary_server_ip,
                "status": secondary.state if secondary else "unknown", 
                "machine_id": secondary.id if secondary else None,
                "name": secondary.name if secondary else None
            },
            "total_machines": len(machines),
            "running_machines": len([m for m in machines if m.state == 'running'])
        }
        
        return status

def main():
    """
    CLI interface for Paperspace API Manager
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="SimApp Paperspace API Manager")
    parser.add_argument("action", choices=[
        "list", "status", "start", "stop", "scale-up", "scale-down", "cluster-status"
    ], help="Action to perform")
    parser.add_argument("--machine-id", help="Specific machine ID for start/stop actions")
    
    args = parser.parse_args()
    
    try:
        api_manager = PaperspaceAPIManager()
        
        if args.action == "list":
            machines = api_manager.list_machines()
            print(f"\n📋 Found {len(machines)} machines:")
            for machine in machines:
                print(f"  - {machine.name} ({machine.id}): {machine.state} @ {machine.public_ip}")
        
        elif args.action == "status":
            if not args.machine_id:
                print("❌ --machine-id required for status action")
                sys.exit(1)
            status = api_manager.get_machine_status(args.machine_id)
            print(f"\n📊 Machine Status: {json.dumps(status, indent=2)}")
        
        elif args.action == "start":
            if not args.machine_id:
                print("❌ --machine-id required for start action")
                sys.exit(1)
            success = api_manager.start_machine(args.machine_id)
            sys.exit(0 if success else 1)
        
        elif args.action == "stop":
            if not args.machine_id:
                print("❌ --machine-id required for stop action")
                sys.exit(1)
            success = api_manager.stop_machine(args.machine_id)
            sys.exit(0 if success else 1)
        
        elif args.action == "scale-up":
            success = api_manager.scale_up_secondary_server()
            sys.exit(0 if success else 1)
        
        elif args.action == "scale-down":
            success = api_manager.scale_down_secondary_server()
            sys.exit(0 if success else 1)
        
        elif args.action == "cluster-status":
            status = api_manager.get_cluster_status()
            print(f"\n🏗️ SimApp Cluster Status:")
            print(json.dumps(status, indent=2))
    
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
