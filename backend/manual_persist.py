#!/usr/bin/env python3
"""
Manual persistence script for the completed simulation
"""
import asyncio
import sys
import os

# Add the backend directory to Python path
sys.path.insert(0, '/app')

async def manual_persist():
    try:
        from shared.result_store import get_result
        from persistence_logging.persistence import persist_simulation_run, build_simulation_summary
        from shared.progress_store import get_progress
        
        sim_id = '2649af6e-c842-4380-9798-530977af5bc9'
        
        print(f"📦 Attempting to manually persist simulation: {sim_id}")
        
        # Get the simulation result from memory
        result = get_result(sim_id)
        if not result:
            print(f"❌ Simulation {sim_id} not found in memory")
            return False
            
        # Get progress data for additional metadata
        progress_data = get_progress(sim_id)
        
        print(f"✅ Found simulation in memory: {result.get('status')}")
        print(f"📋 Original filename: {result.get('original_filename')}")
        print(f"👤 User: {result.get('user')}")
        
        # Build simulation summary for persistence
        summary = build_simulation_summary(
            simulation_id=sim_id,
            results=result,  # Pass the full result data
            status="completed",
            message=result.get('message', 'Multi-target simulation completed'),
            engine_type=result.get('engine_type', 'ultra'),
            iterations_requested=1000,  # Default from the simulation we saw
            target_cell=result.get('result_cell_coordinate'),
            started_at=progress_data.get('start_time') if progress_data else None
        )
        
        # Add additional fields that build_simulation_summary doesn't handle
        summary['original_filename'] = result.get('original_filename')
        summary['user_identifier'] = result.get('user')
        summary['file_id'] = None  # Not available from memory
        
        print(f"📋 Built summary for user: {summary.get('user_identifier')}")
        
        # Persist to database
        persist_success = await persist_simulation_run(summary)
        if persist_success:
            print(f"✅ Successfully persisted simulation {sim_id} to database")
            return True
        else:
            print(f"❌ Failed to persist simulation {sim_id} to database")
            return False
            
    except Exception as e:
        print(f"❌ Error during manual persistence: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    result = asyncio.run(manual_persist())
    sys.exit(0 if result else 1)
