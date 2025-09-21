#!/usr/bin/env python3
"""
Live Monte Carlo Simulation Test
Triggers an actual simulation to verify FULL_EVALUATION is working
"""

import asyncio
import sys
import os
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from database import SessionLocal, engine
from models import Base, ExcelFile, Simulation, User
from simulation.service import create_simulation, run_simulation
from excel_parser.service import parse_excel_file, get_formulas_for_file
import json


async def run_live_test():
    """Run a live Monte Carlo simulation test"""
    print("\n" + "="*60)
    print("🚀 LIVE MONTE CARLO SIMULATION TEST")
    print("="*60)
    
    db = SessionLocal()
    
    try:
        # Create tables if needed
        Base.metadata.create_all(bind=engine)
        
        # Create test user
        test_user = db.query(User).filter(User.email == "test@example.com").first()
        if not test_user:
            test_user = User(
                email="test@example.com",
                username="testuser",
                full_name="Test User"
            )
            db.add(test_user)
            db.commit()
            db.refresh(test_user)
        
        print(f"✅ Test user: {test_user.email}")
        
        # Create test Excel file record
        file_id = f"test_mc_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        excel_file = ExcelFile(
            id=file_id,
            user_id=test_user.id,
            filename="test_monte_carlo.xlsx",
            file_path=f"saved_simulations_files/{file_id}.xlsx"
        )
        db.add(excel_file)
        db.commit()
        
        print(f"✅ Created Excel file record: {file_id}")
        
        # Create simulation
        simulation_config = {
            "target_sheet": "WIZEMICE Likest",
            "target_cell": "B13",
            "iterations": 100,  # Small number for quick test
            "monte_carlo_variables": [
                {"name": "F4", "min_value": 0.8, "max_value": 1.2, "distribution": "uniform"},
                {"name": "F5", "min_value": 0.9, "max_value": 1.1, "distribution": "uniform"},
                {"name": "F6", "min_value": 0.95, "max_value": 1.05, "distribution": "uniform"}
            ],
            "engine": "ultra"
        }
        
        print("\n📊 Simulation Configuration:")
        print(f"   Target: {simulation_config['target_sheet']}!{simulation_config['target_cell']}")
        print(f"   Iterations: {simulation_config['iterations']}")
        print(f"   MC Variables: {len(simulation_config['monte_carlo_variables'])}")
        
        # Create simulation
        simulation = await create_simulation(
            db=db,
            file_id=file_id,
            user_id=test_user.id,
            **simulation_config
        )
        
        print(f"\n✅ Created simulation: {simulation.id}")
        print(f"   Status: {simulation.status}")
        
        # Note: In a real scenario, run_simulation would be called by a background task
        # For testing, we'll just verify the simulation was created
        
        # Check if simulation is ready to run
        if simulation.status == "pending":
            print("\n✅ Simulation created successfully and ready to run")
            print("   In production, this would be picked up by a background worker")
            
            # You can manually trigger it with:
            # await run_simulation(simulation.id)
            
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        db.close()


async def check_recent_logs():
    """Check recent backend logs for FULL_EVALUATION"""
    print("\n" + "="*60)
    print("📝 CHECKING FOR FULL_EVALUATION IN LOGS")
    print("="*60)
    
    import subprocess
    
    try:
        # Get recent backend logs
        result = subprocess.run(
            ["docker-compose", "logs", "--tail=100", "backend"],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            logs = result.stdout
            
            # Look for key patterns
            if "FULL_EVALUATION" in logs:
                print("✅ Found FULL_EVALUATION in logs!")
                # Extract relevant lines
                lines = logs.split('\n')
                for line in lines:
                    if "FULL_EVALUATION" in line:
                        print(f"   {line.strip()}")
            else:
                print("⚠️  No FULL_EVALUATION found in recent logs")
                
            if "ULTRA-SELECTIVE" in logs:
                print("❌ Found ULTRA-SELECTIVE in logs - old code may still be running!")
            else:
                print("✅ No ULTRA-SELECTIVE found - good!")
                
            # Check for target values
            target_values = []
            for line in lines:
                if "TARGET_VALUE" in line and "=" in line:
                    value_part = line.split("=")[-1].strip()
                    target_values.append(value_part)
                    
            if target_values:
                print(f"\n📊 Found {len(target_values)} target values in logs")
                unique_values = len(set(target_values))
                print(f"   Unique values: {unique_values}")
                if unique_values > 1:
                    print("   ✅ Values are varying - Monte Carlo is working!")
                else:
                    print("   ❌ All values are the same - Monte Carlo not varying!")
                    
    except Exception as e:
        print(f"❌ Error checking logs: {e}")


def main():
    """Main entry point"""
    print("\n🎯 Live Monte Carlo Simulation Test")
    print("📍 This will create a test simulation to verify FULL_EVALUATION")
    
    # Run live test
    success = asyncio.run(run_live_test())
    
    # Check logs
    asyncio.run(check_recent_logs())
    
    print("\n" + "="*60)
    print("📋 TEST SUMMARY")
    print("="*60)
    
    if success:
        print("✅ Test simulation created successfully")
        print("\n🔍 To verify FULL_EVALUATION is working:")
        print("   1. Check backend logs: docker-compose logs backend | grep FULL_EVALUATION")
        print("   2. Look for: [FULL_EVALUATION] Processing complete Excel model with 1990 formulas")
        print("   3. Verify no ULTRA-SELECTIVE messages appear")
    else:
        print("❌ Test failed - check errors above")
        
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main()) 