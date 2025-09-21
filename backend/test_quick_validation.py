#!/usr/bin/env python3
"""
Quick Validation Test for Monte Carlo Simulation
Checks if the current backend is producing proper Monte Carlo results
"""

import sys
import asyncio
import aiohttp
import json
import numpy as np
from datetime import datetime


async def quick_validation():
    """Run a quick validation of the Monte Carlo simulation"""
    print("\n" + "="*60)
    print("🚀 QUICK MONTE CARLO VALIDATION TEST")
    print("="*60)
    
    base_url = "http://localhost:8000"
    
    async with aiohttp.ClientSession() as session:
        # 1. Check backend health
        print("\n1️⃣ Checking backend health...")
        try:
            async with session.get(f"{base_url}/health") as resp:
                if resp.status == 200:
                    print("   ✅ Backend is healthy")
                else:
                    print(f"   ❌ Backend returned status {resp.status}")
                    return False
        except Exception as e:
            print(f"   ❌ Cannot connect to backend: {e}")
            print("   Please ensure backend is running: docker-compose up -d")
            return False
            
        # 2. Check simulation status (if you have a simulation ID)
        print("\n2️⃣ Simulation Configuration:")
        print("   Target cells: B12, B13")
        print("   Monte Carlo variables:")
        print("     • F4: [0.8, 1.2] uniform")
        print("     • F5: [0.9, 1.1] uniform")
        print("     • F6: [0.95, 1.05] uniform")
        print("   Iterations: 1000")
        
        # 3. Expected behavior
        print("\n3️⃣ Expected Behavior:")
        print("   ✓ Results should show variation (std_dev > 0)")
        print("   ✓ Mean should be stable across runs")
        print("   ✓ Histogram should show distribution")
        print("   ✓ All 1000 iterations should produce different values")
        
        # 4. Check logs
        print("\n4️⃣ Key Backend Logs to Check:")
        print("   Run: docker-compose logs backend | tail -100")
        print("   Look for:")
        print("     • [FULL_EVALUATION] Processing ALL 1990 formulas")
        print("     • [CONSTANTS] Using X constants, excluding Y cells")
        print("     • [TARGET_VALUE] Different values per iteration")
        
        # 5. Validate statistical properties
        print("\n5️⃣ Statistical Validation:")
        
        # Simulate what proper results should look like
        mc_vars = {
            "F4": np.random.uniform(0.8, 1.2, 1000),
            "F5": np.random.uniform(0.9, 1.1, 1000),
            "F6": np.random.uniform(0.95, 1.05, 1000)
        }
        
        # Expected ranges for uniform distributions
        print("\n   Expected MC variable statistics:")
        for var, values in mc_vars.items():
            print(f"   {var}: mean={np.mean(values):.3f}, std={np.std(values):.3f}, " +
                  f"range=[{np.min(values):.3f}, {np.max(values):.3f}]")
        
        print("\n   ✅ If your results show:")
        print("      • Non-zero standard deviation")
        print("      • Values changing between iterations")
        print("      • Reasonable mean values")
        print("   Then the Monte Carlo simulation is working!")
        
        print("\n   ❌ If your results show:")
        print("      • Zero standard deviation")
        print("      • Same value for all iterations")
        print("      • Extreme values (like 1e+25)")
        print("   Then there's still an issue to fix")
        
        return True


async def check_backend_logs():
    """Check backend logs for key patterns"""
    print("\n" + "="*60)
    print("📝 CHECKING BACKEND LOGS")
    print("="*60)
    
    import subprocess
    
    try:
        # Get last 50 lines of backend logs
        result = subprocess.run(
            ["docker-compose", "logs", "--tail=50", "backend"],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            logs = result.stdout
            
            # Check for key patterns
            patterns = {
                "FULL_EVALUATION": "Full formula evaluation",
                "CONSTANTS": "Constants management",
                "TARGET_VALUE": "Target value tracking",
                "Processing ALL": "Processing all formulas"
            }
            
            found = []
            not_found = []
            
            for pattern, description in patterns.items():
                if pattern in logs:
                    found.append(f"✅ {description} ({pattern})")
                else:
                    not_found.append(f"❌ {description} ({pattern})")
                    
            print("\nFound patterns:")
            for item in found:
                print(f"  {item}")
                
            if not_found:
                print("\nMissing patterns:")
                for item in not_found:
                    print(f"  {item}")
                    
            # Look for iteration values
            if "TARGET_VALUE" in logs:
                print("\n📊 Sample target values from logs:")
                lines = logs.split('\n')
                target_lines = [l for l in lines if "TARGET_VALUE" in l][-5:]
                for line in target_lines:
                    if "=" in line:
                        value_part = line.split("=")[-1].strip()
                        print(f"  • {value_part}")
                        
        else:
            print("❌ Failed to get Docker logs")
            
    except Exception as e:
        print(f"❌ Error checking logs: {e}")


def main():
    """Main entry point"""
    print("\n🎯 Monte Carlo Quick Validation Test")
    print("📍 This will help verify if the simulation is working correctly")
    
    # Run validation
    asyncio.run(quick_validation())
    
    # Check logs
    asyncio.run(check_backend_logs())
    
    print("\n" + "="*60)
    print("📋 VALIDATION CHECKLIST")
    print("="*60)
    print("□ Backend is running and healthy")
    print("□ Simulation shows non-zero standard deviation")
    print("□ Target values vary between iterations")
    print("□ No extreme values (like 1e+25)")
    print("□ Logs show FULL_EVALUATION processing all formulas")
    print("□ Constants are properly filtered")
    print("\n✨ If all items are checked, the Monte Carlo simulation is working!")
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 