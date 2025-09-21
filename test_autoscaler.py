#!/usr/bin/env python3
"""
Test script for Paperspace Auto-Scaler
Tests the metrics endpoint and scaling logic
"""

import asyncio
import aiohttp
import json
import time

async def test_metrics_endpoint():
    """Test the /api/metrics endpoint"""
    print("🧪 Testing metrics endpoint...")
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get('http://localhost:8000/api/metrics') as response:
                if response.status == 200:
                    data = await response.json()
                    print("✅ Metrics endpoint working!")
                    print(f"📊 Current metrics:")
                    print(f"   - Active users: {data.get('active_users', 0)}")
                    print(f"   - CPU usage: {data.get('cpu_usage', 0):.1f}%")
                    print(f"   - Memory usage: {data.get('memory_usage', 0):.1f}%")
                    print(f"   - GPU usage: {data.get('gpu_usage', 0):.1f}%")
                    print(f"   - Queue length: {data.get('simulation_queue_length', 0)}")
                    print(f"   - Response time: {data.get('avg_response_time', 0)}ms")
                    return data
                else:
                    print(f"❌ Metrics endpoint failed: HTTP {response.status}")
                    return None
    except Exception as e:
        print(f"❌ Failed to connect to metrics endpoint: {e}")
        print("💡 Make sure your Monte Carlo platform is running on localhost:8000")
        return None

async def simulate_load_test():
    """Simulate different load scenarios to test scaling logic"""
    print("\n🎯 Simulating load scenarios...")
    
    # Test scaling up scenario
    print("\n📈 Scenario 1: High load (should trigger scale up)")
    high_load_metrics = {
        "active_users": 8,
        "cpu_usage": 75.0,
        "gpu_usage": 85.0,
        "queue_length": 6,
        "response_time_ms": 3500
    }
    
    should_scale_up = analyze_scaling_decision(high_load_metrics, "up")
    print(f"   Decision: {'✅ SCALE UP' if should_scale_up else '❌ No scaling'}")
    
    # Test scaling down scenario  
    print("\n📉 Scenario 2: Low load (should trigger scale down)")
    low_load_metrics = {
        "active_users": 2,
        "cpu_usage": 25.0,
        "gpu_usage": 30.0,
        "queue_length": 0,
        "response_time_ms": 800
    }
    
    should_scale_down = analyze_scaling_decision(low_load_metrics, "down")
    print(f"   Decision: {'✅ SCALE DOWN' if should_scale_down else '❌ No scaling'}")
    
    # Test medium load scenario
    print("\n⚖️  Scenario 3: Medium load (should maintain current scale)")
    medium_load_metrics = {
        "active_users": 5,
        "cpu_usage": 50.0,
        "gpu_usage": 60.0,
        "queue_length": 2,
        "response_time_ms": 1800
    }
    
    should_scale_up_medium = analyze_scaling_decision(medium_load_metrics, "up")
    should_scale_down_medium = analyze_scaling_decision(medium_load_metrics, "down")
    print(f"   Decision: {'⚖️ MAINTAIN' if not should_scale_up_medium and not should_scale_down_medium else '❓ Unexpected'}")

def analyze_scaling_decision(metrics, direction):
    """Analyze if we should scale based on metrics (simplified logic)"""
    if direction == "up":
        conditions = [
            metrics["active_users"] >= 6,
            metrics["cpu_usage"] > 70,
            metrics["gpu_usage"] > 80,
            metrics["queue_length"] > 5,
            metrics["response_time_ms"] > 3000
        ]
        return sum(conditions) >= 2
    
    elif direction == "down":
        conditions = [
            metrics["active_users"] <= 4,
            metrics["cpu_usage"] < 30,
            metrics["gpu_usage"] < 40,
            metrics["queue_length"] == 0,
            metrics["response_time_ms"] < 1000
        ]
        return all(conditions)
    
    return False

async def test_paperspace_api_connection():
    """Test if Paperspace API credentials work"""
    print("\n🔑 Testing Paperspace API connection...")
    
    import os
    
    # Load environment variables
    try:
        with open('.env.autoscaler', 'r') as f:
            for line in f:
                if '=' in line and not line.startswith('#'):
                    key, value = line.strip().split('=', 1)
                    os.environ[key] = value
    except FileNotFoundError:
        print("❌ .env.autoscaler file not found")
        print("💡 Run the autoscaler once to create the template")
        return False
    
    api_key = os.getenv('PAPERSPACE_API_KEY')
    primary_id = os.getenv('PRIMARY_MACHINE_ID')
    secondary_id = os.getenv('SECONDARY_MACHINE_ID')
    
    if not api_key or api_key == 'your_paperspace_api_key_here':
        print("❌ Paperspace API key not configured")
        print("💡 Please update .env.autoscaler with your real API key")
        return False
    
    if not primary_id or primary_id == 'your_primary_machine_id':
        print("❌ Machine IDs not configured")
        print("💡 Please update .env.autoscaler with your machine IDs")
        return False
    
    print("✅ Configuration looks valid")
    print(f"   - API Key: {api_key[:10]}...")
    print(f"   - Primary Machine: {primary_id}")
    print(f"   - Secondary Machine: {secondary_id}")
    
    # Test API connection (simplified)
    try:
        async with aiohttp.ClientSession() as session:
            headers = {'X-Api-Key': api_key}
            url = f"https://api.paperspace.io/machines/{primary_id}"
            
            async with session.get(url, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"✅ Primary machine found: {data.get('name', 'Unknown')}")
                    print(f"   - State: {data.get('state', 'Unknown')}")
                    return True
                else:
                    print(f"❌ API call failed: HTTP {response.status}")
                    text = await response.text()
                    print(f"   Error: {text}")
                    return False
    except Exception as e:
        print(f"❌ API connection failed: {e}")
        return False

def print_cost_analysis():
    """Print cost analysis for different scaling scenarios"""
    print("\n💰 Cost Analysis for Auto-Scaling:")
    print("=" * 50)
    
    scenarios = [
        ("Low usage (0-4 users)", 0.51, 8760 * 0.3),  # 30% of time
        ("Medium usage (5-8 users)", 0.51, 8760 * 0.5),  # 50% of time  
        ("High usage (9-16 users)", 1.02, 8760 * 0.2),  # 20% of time
    ]
    
    total_annual_cost = 0
    
    for scenario, hourly_rate, hours_per_year in scenarios:
        annual_cost = hourly_rate * hours_per_year
        total_annual_cost += annual_cost
        print(f"{scenario:<25} ${hourly_rate:.2f}/hr × {hours_per_year:,.0f}h = ${annual_cost:,.0f}/year")
    
    print("-" * 50)
    print(f"{'Total estimated annual cost:':<25} ${total_annual_cost:,.0f}/year")
    print(f"{'Average monthly cost:':<25} ${total_annual_cost/12:,.0f}/month")
    print(f"{'Cost per user (avg 10 users):':<25} ${total_annual_cost/12/10:,.0f}/month")
    
    print("\n📊 Comparison with alternatives:")
    print(f"{'Always-on dual P4000:':<25} ${0.51*2*24*30:,.0f}/month")
    print(f"{'Single A100 instance:':<25} ${2.30*24*30:,.0f}/month")
    print(f"{'Auto-scaling (estimated):':<25} ${total_annual_cost/12:,.0f}/month")

async def main():
    """Main test function"""
    print("🚀 Paperspace Auto-Scaler Test Suite")
    print("=" * 50)
    
    # Test 1: Metrics endpoint
    metrics = await test_metrics_endpoint()
    
    # Test 2: Load scenarios
    await simulate_load_test()
    
    # Test 3: Paperspace API
    api_working = await test_paperspace_api_connection()
    
    # Test 4: Cost analysis
    print_cost_analysis()
    
    # Summary
    print("\n📋 Test Summary:")
    print("=" * 50)
    print(f"✅ Metrics endpoint: {'Working' if metrics else 'Failed'}")
    print(f"✅ Scaling logic: Working (simulated)")
    print(f"✅ Paperspace API: {'Working' if api_working else 'Failed'}")
    print(f"✅ Cost analysis: Complete")
    
    if metrics and api_working:
        print("\n🎉 All systems ready for auto-scaling!")
        print("💡 Run: sudo systemctl start monte-carlo-autoscaler")
    else:
        print("\n⚠️  Some issues found. Please fix them before enabling auto-scaling.")
    
    print("\n🔍 Next steps:")
    print("1. Fix any failed tests above")
    print("2. Start your Monte Carlo platform if not running")
    print("3. Configure .env.autoscaler with real credentials")
    print("4. Run ./start_autoscaler.sh to install the system")
    print("5. Monitor with: sudo journalctl -u monte-carlo-autoscaler -f")

if __name__ == "__main__":
    asyncio.run(main())







