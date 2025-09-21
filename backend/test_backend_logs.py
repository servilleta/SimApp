#!/usr/bin/env python3
"""
Check Backend Logs for Monte Carlo Simulation Status
"""

import subprocess
import sys


def check_backend_logs():
    """Check backend logs for Monte Carlo simulation patterns"""
    print("\n" + "="*60)
    print("📝 BACKEND LOGS ANALYSIS")
    print("="*60)
    
    try:
        # Get all backend logs
        result = subprocess.run(
            ["docker-compose", "logs", "backend"],
            capture_output=True,
            text=True,
            cwd=".."  # Run from project root
        )
        
        if result.returncode == 0:
            logs = result.stdout
            lines = logs.split('\n')
            
            print(f"📊 Total log lines: {len(lines)}")
            
            # Analysis
            patterns = {
                "FULL_EVALUATION": 0,
                "ULTRA-SELECTIVE": 0,
                "TARGET_VALUE": 0,
                "Processing ALL": 0,
                "CONSTANTS": 0,
                "DIRECT_DEPENDENTS": 0
            }
            
            target_values = []
            
            for line in lines:
                for pattern in patterns:
                    if pattern in line:
                        patterns[pattern] += 1
                        
                if "TARGET_VALUE" in line and "=" in line:
                    try:
                        value_part = line.split("=")[-1].strip()
                        target_values.append(value_part)
                    except:
                        pass
                        
            # Report findings
            print("\n📈 Pattern Analysis:")
            for pattern, count in patterns.items():
                if count > 0:
                    status = "✅" if pattern != "ULTRA-SELECTIVE" else "❌"
                    print(f"   {status} {pattern}: {count} occurrences")
                else:
                    status = "⚠️" if pattern == "FULL_EVALUATION" else "  "
                    print(f"   {status} {pattern}: Not found")
                    
            # Check if we're using the right approach
            print("\n🔍 Evaluation Approach:")
            if patterns["FULL_EVALUATION"] > 0:
                print("   ✅ FULL_EVALUATION approach is active!")
                # Find and show a sample
                for line in lines[-100:]:  # Check last 100 lines
                    if "FULL_EVALUATION" in line:
                        print(f"      Sample: {line.strip()[:100]}...")
                        break
            elif patterns["ULTRA-SELECTIVE"] > 0:
                print("   ❌ ULTRA-SELECTIVE approach is still active!")
                print("   🔧 Need to rebuild Docker or check code deployment")
            else:
                print("   ⚠️  No clear evaluation approach found in logs")
                print("   💡 Try running a simulation to generate logs")
                
            # Check Monte Carlo variation
            if target_values:
                print(f"\n📊 Monte Carlo Target Values:")
                print(f"   Total values logged: {len(target_values)}")
                unique_values = set(target_values)
                print(f"   Unique values: {len(unique_values)}")
                
                if len(unique_values) > 1:
                    print("   ✅ Values are varying - Monte Carlo is working!")
                    # Show sample
                    print("   Sample values:")
                    for i, val in enumerate(list(unique_values)[:5]):
                        print(f"      {i+1}. {val}")
                elif len(unique_values) == 1:
                    print("   ❌ All values are the same - Monte Carlo not varying!")
                    print(f"   Single value: {list(unique_values)[0]}")
            else:
                print("\n⚠️  No target values found in logs")
                print("   Run a simulation to see Monte Carlo in action")
                
            # Final verdict
            print("\n" + "="*60)
            print("📋 VERDICT")
            print("="*60)
            
            if patterns["FULL_EVALUATION"] > 0 and patterns["ULTRA-SELECTIVE"] == 0:
                print("✅ Backend is using the correct FULL_EVALUATION approach!")
                if len(set(target_values)) > 1:
                    print("✅ Monte Carlo values are properly varying!")
                    print("\n🎉 The Monte Carlo simulation platform is working correctly!")
                else:
                    print("⚠️  Need to run a simulation to verify Monte Carlo variation")
            else:
                print("❌ Backend needs attention:")
                if patterns["ULTRA-SELECTIVE"] > 0:
                    print("   • Still using old ULTRA-SELECTIVE approach")
                if patterns["FULL_EVALUATION"] == 0:
                    print("   • FULL_EVALUATION not found in logs")
                print("\n🔧 Actions needed:")
                print("   1. Verify code changes are saved")
                print("   2. Rebuild Docker: docker-compose build --no-cache backend")
                print("   3. Restart: docker-compose up -d")
                
        else:
            print(f"❌ Failed to get Docker logs: {result.stderr}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        

def main():
    """Main entry point"""
    print("\n🎯 Backend Logs Analysis for Monte Carlo Simulation")
    print("📍 Checking if FULL_EVALUATION approach is active")
    
    check_backend_logs()
    
    print("\n💡 Tips:")
    print("   • To see live logs: docker-compose logs -f backend")
    print("   • To trigger simulation: Use the frontend or API")
    print("   • To rebuild: docker-compose build --no-cache backend")
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 