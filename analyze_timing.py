#!/usr/bin/env python3
"""
Analyze frontend timing for WebSocket connections during simulations
Usage: python3 analyze_timing.py
"""

import requests
import json
import time
from datetime import datetime
import re

class TimingAnalyzer:
    def __init__(self, api_url="http://localhost:9090/api/dev/console-logs"):
        self.api_url = api_url
        self.simulation_events = []
        
    def get_logs(self, limit=1000):
        """Get console logs from backend"""
        try:
            response = requests.get(f"{self.api_url}?limit={limit}")
            response.raise_for_status()
            return response.json().get('logs', [])
        except Exception as e:
            print(f"Error fetching logs: {e}")
            return []
    
    def analyze_simulation_timing(self):
        """Analyze timing of simulation events"""
        logs = self.get_logs(500)
        
        events = {
            'simulation_start': [],
            'websocket_connect': [],
            'progress_updates': [],
            'simulation_complete': []
        }
        
        print("🔍 Analyzing console logs for simulation timing...")
        print("=" * 60)
        
        for log in logs:
            message = log.get('message', '')
            timestamp = log.get('timestamp', '')
            
            # Parse timestamp
            try:
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                time_str = dt.strftime('%H:%M:%S.%f')[:-3]
            except:
                time_str = timestamp
            
            # Categorize events
            if 'handleRunSimulation' in message or 'Run Simulation' in message:
                events['simulation_start'].append((time_str, message))
                print(f"🚀 {time_str}: SIMULATION START - {message[:80]}...")
                
            elif 'WebSocket' in message and 'Connect' in message:
                events['websocket_connect'].append((time_str, message))
                print(f"🔌 {time_str}: WEBSOCKET CONNECT - {message[:80]}...")
                
            elif 'RENDERING PROGRESS BAR' in message:
                events['progress_updates'].append((time_str, message))
                progress_match = re.search(r'(\d+)%', message)
                if progress_match:
                    progress = progress_match.group(1)
                    print(f"📊 {time_str}: PROGRESS UPDATE - {progress}%")
                    
            elif 'simulation.*completed' in message.lower() or 'status.*completed' in message.lower():
                events['simulation_complete'].append((time_str, message))
                print(f"✅ {time_str}: SIMULATION COMPLETE - {message[:80]}...")
                
            elif 'currentSimulationId' in message or 'simulation_id' in message:
                # Extract simulation ID
                sim_id_match = re.search(r'([a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12})', message)
                if sim_id_match:
                    sim_id = sim_id_match.group(1)
                    print(f"🆔 {time_str}: SIMULATION ID - {sim_id}")
        
        print("\n" + "=" * 60)
        print("📈 TIMING ANALYSIS SUMMARY:")
        print(f"Simulation starts: {len(events['simulation_start'])}")
        print(f"WebSocket connects: {len(events['websocket_connect'])}")
        print(f"Progress updates: {len(events['progress_updates'])}")
        print(f"Simulation completes: {len(events['simulation_complete'])}")
        
        return events
    
    def monitor_next_simulation(self):
        """Monitor for the next simulation and analyze timing"""
        print("🔍 Monitoring for next simulation...")
        print("Please start a simulation now and I'll analyze the timing.")
        print("Press Ctrl+C to stop monitoring\n")
        
        initial_log_count = len(self.get_logs(10))
        
        try:
            while True:
                current_logs = self.get_logs(100)
                new_logs = current_logs[initial_log_count:]
                
                for log in new_logs:
                    message = log.get('message', '')
                    timestamp = log.get('timestamp', '')
                    
                    try:
                        dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                        time_str = dt.strftime('%H:%M:%S.%f')[:-3]
                    except:
                        time_str = timestamp
                    
                    # Real-time event detection
                    if any(keyword in message for keyword in ['handleRunSimulation', 'WebSocket', 'RENDERING PROGRESS', 'simulation']):
                        print(f"⚡ {time_str}: {message[:100]}...")
                
                initial_log_count = len(current_logs)
                time.sleep(0.5)
                
        except KeyboardInterrupt:
            print(f"\n👋 Stopped monitoring")

if __name__ == "__main__":
    analyzer = TimingAnalyzer()
    
    print("Choose analysis mode:")
    print("1. Analyze recent simulation timing")
    print("2. Monitor next simulation in real-time")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        analyzer.analyze_simulation_timing()
    elif choice == "2":
        analyzer.monitor_next_simulation()
    else:
        print("Invalid choice. Analyzing recent timing...")
        analyzer.analyze_simulation_timing()