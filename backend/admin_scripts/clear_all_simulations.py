#!/usr/bin/env python3
"""
Admin script to safely delete all simulation data from the system.
This will clear:
1. PostgreSQL simulation_results table
2. PostgreSQL saved_simulations table  
3. Ultra Engine SQLite database tables
4. Redis progress cache
5. Temporary simulation files

CAUTION: This will permanently delete ALL simulation data for ALL users!
"""

import os
import sys
import sqlite3
import redis
import logging
from pathlib import Path

# Add backend to path so we can import modules
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

from database import SessionLocal, engine
from models import SimulationResult
from saved_simulations.models import SavedSimulation
from sqlalchemy import text

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def clear_postgresql_tables():
    """Clear all simulation data from database tables"""
    logger.info("🗑️ Clearing database simulation tables...")
    
    db = SessionLocal()
    try:
        # Use raw SQL to safely clear tables and get counts
        sim_results_count = db.execute(text("SELECT COUNT(*) FROM simulation_results")).scalar()
        
        # Check if saved_simulations table exists and get count
        try:
            saved_sims_count = db.execute(text("SELECT COUNT(*) FROM saved_simulations")).scalar()
        except Exception:
            saved_sims_count = 0
        
        logger.info(f"Found {sim_results_count} simulation results and {saved_sims_count or 0} saved simulations")
        
        if sim_results_count == 0 and (saved_sims_count or 0) == 0:
            logger.info("No simulation data found to delete")
            return
        
        # Delete all simulation results using raw SQL
        db.execute(text("DELETE FROM simulation_results"))
        
        # Delete all saved simulations if the table exists
        try:
            db.execute(text("DELETE FROM saved_simulations"))
        except Exception as e:
            logger.warning(f"Could not clear saved_simulations table (may not exist): {e}")
        
        # Commit the changes
        db.commit()
        
        logger.info(f"✅ Deleted {sim_results_count} simulation results and {saved_sims_count or 0} saved simulations from database")
        
    except Exception as e:
        logger.error(f"❌ Error clearing database tables: {e}")
        db.rollback()
        raise
    finally:
        db.close()

def clear_ultra_engine_database():
    """Clear Ultra Engine SQLite database tables"""
    logger.info("🗑️ Clearing Ultra Engine SQLite database...")
    
    # Ultra Engine database path (check multiple possible locations)
    possible_paths = [
        "/tmp/ultra_simulation.db",
        "/home/paperspace/PROJECT/backend/ultra_simulation.db",
        ":memory:"  # In-memory database won't need clearing
    ]
    
    for db_path in possible_paths:
        if db_path == ":memory:":
            continue
            
        if os.path.exists(db_path):
            try:
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                
                # Get counts before deletion
                tables = ['simulations', 'target_cells', 'histogram_data', 'tornado_data', 'dependency_tree']
                total_rows = 0
                
                for table in tables:
                    try:
                        cursor.execute(f"SELECT COUNT(*) FROM {table}")
                        count = cursor.fetchone()[0]
                        total_rows += count
                        logger.info(f"Found {count} rows in {table}")
                    except sqlite3.OperationalError:
                        # Table doesn't exist
                        continue
                
                if total_rows == 0:
                    logger.info("No data found in Ultra Engine database")
                    conn.close()
                    continue
                
                # Clear all tables
                for table in tables:
                    try:
                        cursor.execute(f"DELETE FROM {table}")
                        logger.info(f"Cleared {table} table")
                    except sqlite3.OperationalError:
                        # Table doesn't exist
                        continue
                
                conn.commit()
                conn.close()
                
                logger.info(f"✅ Cleared {total_rows} rows from Ultra Engine database: {db_path}")
                
            except Exception as e:
                logger.error(f"❌ Error clearing Ultra Engine database {db_path}: {e}")
                
def clear_redis_cache():
    """Clear simulation progress data from Redis"""
    logger.info("🗑️ Clearing Redis simulation cache...")
    
    try:
        # Try multiple Redis connection options
        redis_hosts = ['redis', 'localhost', '127.0.0.1']
        r = None
        
        for host in redis_hosts:
            try:
                r = redis.Redis(host=host, port=6379, decode_responses=True, socket_timeout=5)
                r.ping()  # Test connection
                logger.info(f"Connected to Redis at {host}")
                break
            except Exception:
                continue
        
        if r is None:
            logger.warning("Could not connect to Redis - skipping cache cleanup")
            return
        
        # Find all simulation-related keys
        sim_keys = []
        
        # Common patterns for simulation keys
        patterns = [
            'progress:*',
            'simulation:*', 
            'sim:*',
            '*simulation*'
        ]
        
        for pattern in patterns:
            keys = r.keys(pattern)
            sim_keys.extend(keys)
        
        # Remove duplicates
        sim_keys = list(set(sim_keys))
        
        if not sim_keys:
            logger.info("No simulation keys found in Redis")
            return
            
        logger.info(f"Found {len(sim_keys)} simulation-related keys in Redis")
        
        # Delete all simulation keys
        if sim_keys:
            deleted_count = r.delete(*sim_keys)
            logger.info(f"✅ Deleted {deleted_count} keys from Redis cache")
            
    except Exception as e:
        logger.warning(f"Could not clear Redis cache (this is ok if Redis is not accessible): {e}")

def clear_temp_files():
    """Clear temporary simulation files"""
    logger.info("🗑️ Clearing temporary simulation files...")
    
    temp_dirs = [
        "/tmp/simulations",
        "/home/paperspace/PROJECT/backend/temp_files",
        "/home/paperspace/PROJECT/backend/uploads"
    ]
    
    total_files_deleted = 0
    
    for temp_dir in temp_dirs:
        if os.path.exists(temp_dir):
            try:
                files = list(Path(temp_dir).rglob("*"))
                file_count = len([f for f in files if f.is_file()])
                
                if file_count > 0:
                    logger.info(f"Found {file_count} files in {temp_dir}")
                    
                    # Delete all files in the directory
                    for file_path in files:
                        if file_path.is_file():
                            try:
                                file_path.unlink()
                                total_files_deleted += 1
                            except Exception as e:
                                logger.warning(f"Could not delete {file_path}: {e}")
                                
            except Exception as e:
                logger.error(f"❌ Error clearing temp directory {temp_dir}: {e}")
    
    if total_files_deleted > 0:
        logger.info(f"✅ Deleted {total_files_deleted} temporary files")
    else:
        logger.info("No temporary files found to delete")

def main():
    """Main function to clear all simulation data"""
    logger.info("🚨 STARTING COMPLETE SIMULATION DATA CLEANUP")
    logger.info("This will permanently delete ALL simulation data for ALL users!")
    
    try:
        # Clear all data sources
        clear_postgresql_tables()
        clear_ultra_engine_database() 
        clear_redis_cache()
        clear_temp_files()
        
        logger.info("🎉 CLEANUP COMPLETE! All simulation data has been cleared.")
        logger.info("The system is now ready for fresh simulations.")
        
    except Exception as e:
        logger.error(f"❌ CLEANUP FAILED: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
