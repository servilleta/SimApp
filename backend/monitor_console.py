#!/usr/bin/env python3
"""
Real-time browser console log monitor for development

Usage:
    python monitor_console.py [--clear] [--follow]

Options:
    --clear     Clear stored logs before starting
    --follow    Follow logs in real-time (default)
    --no-follow Just show current logs and exit
"""

import requests
import time
import json
import argparse
import sys
from datetime import datetime
from typing import List, Dict

class ConsoleLogMonitor:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.api_url = f"{base_url}/api/dev/console-logs"
        self.last_seen_count = 0
        
        # Color codes for different log levels
        self.colors = {
            'log': '\033[0m',      # Default
            'info': '\033[94m',    # Blue
            'warn': '\033[93m',    # Yellow
            'error': '\033[91m',   # Red
            'debug': '\033[90m',   # Gray
            'reset': '\033[0m'     # Reset
        }

    def format_log_entry(self, log: Dict) -> str:
        """Format a log entry for console display"""
        timestamp = datetime.fromisoformat(log['timestamp'].replace('Z', '+00:00'))
        local_time = timestamp.strftime('%H:%M:%S.%f')[:-3]  # Include milliseconds
        
        level = log['level'].lower()
        color = self.colors.get(level, self.colors['log'])
        reset = self.colors['reset']
        
        # Extract page from URL for cleaner display
        url_parts = log['url'].split('/')
        page = url_parts[-1] if url_parts else 'unknown'
        
        return f"{color}[{local_time}] [{level.upper():5}] {page:15} | {log['message']}{reset}"

    def get_logs(self, limit: int = 100) -> List[Dict]:
        """Fetch console logs from the server"""
        try:
            response = requests.get(f"{self.api_url}?limit={limit}")
            response.raise_for_status()
            return response.json()['logs']
        except requests.exceptions.RequestException as e:
            print(f"❌ Error fetching logs: {e}")
            return []

    def clear_logs(self) -> bool:
        """Clear stored console logs"""
        try:
            response = requests.delete(self.api_url)
            response.raise_for_status()
            print("🧹 Console logs cleared")
            return True
        except requests.exceptions.RequestException as e:
            print(f"❌ Error clearing logs: {e}")
            return False

    def show_logs(self, logs: List[Dict]):
        """Display logs in the console"""
        if not logs:
            print("📭 No console logs available")
            return

        for log in logs:
            print(self.format_log_entry(log))

    def follow_logs(self, interval: float = 1.0):
        """Follow logs in real-time"""
        print(f"🔍 Monitoring browser console logs at {self.api_url}")
        print("📱 Open your browser and start using the app to see logs here...")
        print("🛑 Press Ctrl+C to stop monitoring\n")

        try:
            while True:
                logs = self.get_logs(1000)  # Get recent logs
                
                # Only show new logs
                new_logs = logs[self.last_seen_count:]
                if new_logs:
                    self.show_logs(new_logs)
                    self.last_seen_count = len(logs)
                
                time.sleep(interval)
                
        except KeyboardInterrupt:
            print(f"\n👋 Stopped monitoring console logs")

    def check_server_connection(self) -> bool:
        """Check if the server is accessible"""
        try:
            response = requests.get(f"{self.base_url}/api")
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False

def main():
    parser = argparse.ArgumentParser(description='Monitor browser console logs')
    parser.add_argument('--clear', action='store_true', help='Clear stored logs before starting')
    parser.add_argument('--no-follow', action='store_true', help='Just show current logs and exit')
    parser.add_argument('--url', default='http://localhost:8000', help='Backend server URL')
    
    args = parser.parse_args()
    
    monitor = ConsoleLogMonitor(args.url)
    
    # Check server connection
    if not monitor.check_server_connection():
        print(f"❌ Cannot connect to server at {args.url}")
        print("💡 Make sure Docker containers are running and SSH port forwarding is active")
        sys.exit(1)
    
    # Clear logs if requested
    if args.clear:
        monitor.clear_logs()
    
    # Show current logs
    current_logs = monitor.get_logs()
    if current_logs:
        print("📋 Current console logs:")
        monitor.show_logs(current_logs)
        print()
    
    # Follow logs unless --no-follow is specified
    if not args.no_follow:
        monitor.last_seen_count = len(current_logs)
        monitor.follow_logs()

if __name__ == "__main__":
    main() 