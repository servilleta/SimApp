#!/usr/bin/env python3
"""
Debug script to test simulation progress endpoints
"""

import requests
import json
import time
from datetime import datetime

# Test configuration
BASE_URL = "http://localhost:8000/api"
NGINX_URL = "http://localhost:9090/api"

def test_endpoint(url, description):
    """Test an endpoint and print results"""
    print(f"\n🔍 Testing {description}")
    print(f"URL: {url}")
    
    try:
        response = requests.get(url, timeout=5)
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            try:
                data = response.json()
                print(f"Response: {json.dumps(data, indent=2)[:500]}...")
            except:
                print(f"Response (text): {response.text[:200]}...")
        else:
            print(f"Error response: {response.text}")
            
    except requests.exceptions.ConnectionError as e:
        print(f"❌ Connection Error: {e}")
    except requests.exceptions.Timeout as e:
        print(f"⏰ Timeout Error: {e}")
    except Exception as e:
        print(f"❌ Error: {e}")

def main():
    print("🚀 Simulation Progress Endpoint Diagnostic")
    print(f"Time: {datetime.now()}")
    
    # Test backend directly
    test_endpoint(f"{BASE_URL}/simulations/queue/status", "Backend Queue Status (Direct)")
    
    # Test nginx proxy
    test_endpoint(f"{NGINX_URL}/simulations/queue/status", "Backend Queue Status (via Nginx)")
    
    # Try to get recent simulation IDs from progress store
    try:
        print(f"\n🔍 Checking for active simulations...")
        
        # Import the progress store to see what simulations exist
        import sys
        sys.path.append('/app')
        
        from shared.progress_store import _progress_store
        
        if hasattr(_progress_store, 'store') and _progress_store.store:
            print(f"Found {len(_progress_store.store)} items in progress store:")
            
            for sim_id, data in list(_progress_store.store.items())[:5]:  # Show first 5
                if isinstance(data, dict):
                    status = data.get('status', 'unknown')
                    progress = data.get('progress_percentage', 0)
                    print(f"  {sim_id}: {status} - {progress}%")
                    
                    # Test the specific simulation status endpoint
                    test_endpoint(f"{BASE_URL}/simulations/{sim_id}/status", f"Simulation {sim_id} Status (Direct)")
                    test_endpoint(f"{NGINX_URL}/simulations/{sim_id}/status", f"Simulation {sim_id} Status (via Nginx)")
                    
                    break  # Test only the first one
        else:
            print("No items found in progress store")
            
    except Exception as e:
        print(f"❌ Error checking progress store: {e}")
        
        # Fallback: try some common simulation IDs
        test_sim_ids = ["sim_test", "I6", "J6", "K6"]
        for sim_id in test_sim_ids:
            test_endpoint(f"{BASE_URL}/simulations/{sim_id}/status", f"Test Simulation {sim_id} (Direct)")

if __name__ == "__main__":
    main() 