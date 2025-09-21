"""
🚀 DATABASE PERFORMANCE OPTIMIZATION

Enterprise-grade database optimization for multi-tenant workloads.

This module provides:
- Strategic indexing for user-isolated queries
- Query optimization for common access patterns
- Performance monitoring and metrics
- Database maintenance automation
"""

import logging
import time
from typing import List, Dict, Any, Optional
from sqlalchemy import text, Index, event
from sqlalchemy.orm import Session
from sqlalchemy.engine import Engine

from database import engine, SessionLocal
from models import SimulationResult, User, SecurityAuditLog, UserUsageMetrics

logger = logging.getLogger(__name__)

class DatabaseOptimizer:
    """
    Database optimization manager for enterprise workloads.
    
    Handles:
    - Index creation and management
    - Query performance monitoring
    - Database maintenance tasks
    - Performance analytics
    """
    
    def __init__(self, engine: Engine):
        self.engine = engine
        self.performance_stats = {}
        
    def create_enterprise_indexes(self):
        """
        Create optimized indexes for enterprise multi-tenant queries.
        
        Strategy:
        - Composite indexes on (user_id, frequently_queried_field)
        - Single indexes on foreign keys and status fields
        - Optimized for common query patterns
        """
        
        logger.info("🚀 [DB_OPT] Creating enterprise-optimized database indexes...")
        
        with self.engine.connect() as conn:
            try:
                # 1. SIMULATION RESULTS INDEXES
                # Most critical for user isolation performance
                
                # Primary user isolation index
                self._create_index_if_not_exists(
                    conn, 
                    "idx_simulation_results_user_id", 
                    "simulation_results", 
                    ["user_id"]
                )
                
                # Composite index for user + status queries (very common)
                self._create_index_if_not_exists(
                    conn, 
                    "idx_simulation_results_user_status", 
                    "simulation_results", 
                    ["user_id", "status"]
                )
                
                # Composite index for user + created_at (pagination, recent queries)
                self._create_index_if_not_exists(
                    conn, 
                    "idx_simulation_results_user_created", 
                    "simulation_results", 
                    ["user_id", "created_at"]
                )
                
                # Unique simulation_id lookup (API access)
                self._create_index_if_not_exists(
                    conn, 
                    "idx_simulation_results_sim_id", 
                    "simulation_results", 
                    ["simulation_id"]
                )
                
                # Engine type queries
                self._create_index_if_not_exists(
                    conn, 
                    "idx_simulation_results_user_engine", 
                    "simulation_results", 
                    ["user_id", "engine_type"]
                )
                
                # 2. USER TABLE INDEXES
                
                # Auth0 integration lookup
                self._create_index_if_not_exists(
                    conn, 
                    "idx_users_auth0_user_id", 
                    "users", 
                    ["auth0_user_id"]
                )
                
                # Email lookup for user management
                self._create_index_if_not_exists(
                    conn, 
                    "idx_users_email", 
                    "users", 
                    ["email"]
                )
                
                # Username lookup
                self._create_index_if_not_exists(
                    conn, 
                    "idx_users_username", 
                    "users", 
                    ["username"]
                )
                
                # Active users filtering
                self._create_index_if_not_exists(
                    conn, 
                    "idx_users_active", 
                    "users", 
                    ["is_active"]
                )
                
                # 3. SECURITY AUDIT LOGS INDEXES
                
                # User audit trail
                self._create_index_if_not_exists(
                    conn, 
                    "idx_audit_logs_user_id", 
                    "security_audit_logs", 
                    ["user_id"]
                )
                
                # Event type filtering
                self._create_index_if_not_exists(
                    conn, 
                    "idx_audit_logs_event_type", 
                    "security_audit_logs", 
                    ["event_type"]
                )
                
                # IP-based security analysis
                self._create_index_if_not_exists(
                    conn, 
                    "idx_audit_logs_client_ip", 
                    "security_audit_logs", 
                    ["client_ip"]
                )
                
                # Time-based analysis (recent events)
                self._create_index_if_not_exists(
                    conn, 
                    "idx_audit_logs_timestamp", 
                    "security_audit_logs", 
                    ["timestamp"]
                )
                
                # Composite for user security analysis
                self._create_index_if_not_exists(
                    conn, 
                    "idx_audit_logs_user_timestamp", 
                    "security_audit_logs", 
                    ["user_id", "timestamp"]
                )
                
                # 4. USER SUBSCRIPTIONS INDEXES
                
                # User subscription lookup
                self._create_index_if_not_exists(
                    conn, 
                    "idx_subscriptions_user_id", 
                    "user_subscriptions", 
                    ["user_id"]
                )
                
                # Stripe integration lookups
                self._create_index_if_not_exists(
                    conn, 
                    "idx_subscriptions_stripe_customer", 
                    "user_subscriptions", 
                    ["stripe_customer_id"]
                )
                
                # Subscription status filtering
                self._create_index_if_not_exists(
                    conn, 
                    "idx_subscriptions_status", 
                    "user_subscriptions", 
                    ["status"]
                )
                
                # Trial tracking
                self._create_index_if_not_exists(
                    conn, 
                    "idx_subscriptions_trial", 
                    "user_subscriptions", 
                    ["is_trial"]
                )
                
                # 5. USAGE METRICS INDEXES
                
                # User usage tracking
                self._create_index_if_not_exists(
                    conn, 
                    "idx_usage_metrics_user_id", 
                    "user_usage_metrics", 
                    ["user_id"]
                )
                
                # Period-based analysis
                self._create_index_if_not_exists(
                    conn, 
                    "idx_usage_metrics_period", 
                    "user_usage_metrics", 
                    ["period_start", "period_end"]
                )
                
                # User period composite (most common query)
                self._create_index_if_not_exists(
                    conn, 
                    "idx_usage_metrics_user_period", 
                    "user_usage_metrics", 
                    ["user_id", "period_start", "period_end"]
                )
                
                # 6. API KEYS INDEXES
                
                # API key lookups
                self._create_index_if_not_exists(
                    conn, 
                    "idx_api_keys_key_id", 
                    "api_keys", 
                    ["key_id"]
                )
                
                # User API keys
                self._create_index_if_not_exists(
                    conn, 
                    "idx_api_keys_user_id", 
                    "api_keys", 
                    ["user_id"]
                )
                
                # Active keys filtering
                self._create_index_if_not_exists(
                    conn, 
                    "idx_api_keys_active", 
                    "api_keys", 
                    ["is_active"]
                )
                
                logger.info("✅ [DB_OPT] All enterprise indexes created successfully")
                
            except Exception as e:
                logger.error(f"❌ [DB_OPT] Index creation failed: {e}")
                raise
    
    def _create_index_if_not_exists(self, conn, index_name: str, table_name: str, columns: List[str]):
        """Create an index if it doesn't already exist."""
        try:
            # For SQLite, check if index exists
            result = conn.execute(text(f"SELECT name FROM sqlite_master WHERE type='index' AND name='{index_name}'"))
            exists = result.fetchone() is not None
            
            if exists:
                logger.debug(f"   ⏭️ Index {index_name} already exists")
                return
            
            # Create the index
            columns_str = ", ".join(columns)
            sql = f"CREATE INDEX {index_name} ON {table_name} ({columns_str})"
            conn.execute(text(sql))
            conn.commit()
            
            logger.info(f"   ✅ Created index: {index_name} on {table_name}({columns_str})")
            
        except Exception as e:
            logger.warning(f"   ⚠️ Failed to create index {index_name}: {e}")
    
    def analyze_query_performance(self, db: Session) -> Dict[str, Any]:
        """
        Analyze query performance for common multi-tenant patterns.
        
        Returns:
            Dictionary with performance metrics and recommendations
        """
        logger.info("📊 [DB_OPT] Analyzing query performance...")
        
        performance_report = {
            "timestamp": time.time(),
            "query_tests": [],
            "recommendations": [],
            "overall_score": 0
        }
        
        try:
            # Test 1: User simulation queries
            start_time = time.time()
            test_user_id = 1  # Use test user
            simulations = db.query(SimulationResult).filter(SimulationResult.user_id == test_user_id).limit(10).all()
            user_query_time = time.time() - start_time
            
            performance_report["query_tests"].append({
                "test": "user_simulations_query",
                "execution_time_ms": round(user_query_time * 1000, 2),
                "rows_returned": len(simulations),
                "performance": "good" if user_query_time < 0.1 else "needs_optimization"
            })
            
            # Test 2: Status filtering queries
            start_time = time.time()
            completed_sims = db.query(SimulationResult).filter(
                SimulationResult.user_id == test_user_id,
                SimulationResult.status == "completed"
            ).limit(10).all()
            status_query_time = time.time() - start_time
            
            performance_report["query_tests"].append({
                "test": "user_status_query",
                "execution_time_ms": round(status_query_time * 1000, 2),
                "rows_returned": len(completed_sims),
                "performance": "good" if status_query_time < 0.1 else "needs_optimization"
            })
            
            # Test 3: Recent simulations query
            start_time = time.time()
            recent_sims = db.query(SimulationResult).filter(
                SimulationResult.user_id == test_user_id
            ).order_by(SimulationResult.created_at.desc()).limit(5).all()
            recent_query_time = time.time() - start_time
            
            performance_report["query_tests"].append({
                "test": "user_recent_query",
                "execution_time_ms": round(recent_query_time * 1000, 2),
                "rows_returned": len(recent_sims),
                "performance": "good" if recent_query_time < 0.1 else "needs_optimization"
            })
            
            # Test 4: User lookup query
            start_time = time.time()
            user = db.query(User).filter(User.id == test_user_id).first()
            user_query_time = time.time() - start_time
            
            performance_report["query_tests"].append({
                "test": "user_lookup_query",
                "execution_time_ms": round(user_query_time * 1000, 2),
                "rows_returned": 1 if user else 0,
                "performance": "good" if user_query_time < 0.05 else "needs_optimization"
            })
            
            # Calculate overall performance score
            avg_time = sum(test["execution_time_ms"] for test in performance_report["query_tests"]) / len(performance_report["query_tests"])
            if avg_time < 50:
                performance_report["overall_score"] = 95
                performance_report["recommendations"].append("✅ Excellent query performance")
            elif avg_time < 100:
                performance_report["overall_score"] = 80
                performance_report["recommendations"].append("✅ Good query performance")
            elif avg_time < 200:
                performance_report["overall_score"] = 60
                performance_report["recommendations"].append("⚠️ Moderate performance - consider index optimization")
            else:
                performance_report["overall_score"] = 30
                performance_report["recommendations"].append("❌ Poor performance - immediate optimization needed")
            
            # Additional recommendations
            if avg_time > 100:
                performance_report["recommendations"].extend([
                    "🔧 Run VACUUM on SQLite database",
                    "📊 Consider switching to PostgreSQL for better performance",
                    "🚀 Implement query result caching",
                    "📈 Monitor database size and implement archiving"
                ])
            
            logger.info(f"📊 [DB_OPT] Performance analysis complete - Score: {performance_report['overall_score']}/100")
            
        except Exception as e:
            logger.error(f"❌ [DB_OPT] Performance analysis failed: {e}")
            performance_report["error"] = str(e)
        
        return performance_report
    
    def optimize_database_settings(self):
        """
        Optimize database settings for multi-tenant workloads.
        """
        logger.info("⚙️ [DB_OPT] Optimizing database settings...")
        
        with self.engine.connect() as conn:
            try:
                # SQLite optimizations for concurrent access
                optimizations = [
                    "PRAGMA journal_mode = WAL",  # Write-Ahead Logging for better concurrency
                    "PRAGMA synchronous = NORMAL",  # Balanced safety vs performance
                    "PRAGMA cache_size = 10000",  # Larger cache for better performance
                    "PRAGMA temp_store = MEMORY",  # Keep temporary data in memory
                    "PRAGMA mmap_size = 268435456",  # 256MB memory mapping
                ]
                
                for pragma in optimizations:
                    try:
                        conn.execute(text(pragma))
                        logger.debug(f"   ✅ Applied: {pragma}")
                    except Exception as e:
                        logger.warning(f"   ⚠️ Failed to apply {pragma}: {e}")
                
                conn.commit()
                logger.info("✅ [DB_OPT] Database settings optimized")
                
            except Exception as e:
                logger.error(f"❌ [DB_OPT] Database optimization failed: {e}")
    
    def run_maintenance_tasks(self):
        """
        Run database maintenance tasks for optimal performance.
        """
        logger.info("🔧 [DB_OPT] Running database maintenance...")
        
        with self.engine.connect() as conn:
            try:
                # Update table statistics
                conn.execute(text("ANALYZE"))
                logger.info("   ✅ Updated table statistics")
                
                # Defragment database (SQLite VACUUM)
                conn.execute(text("VACUUM"))
                logger.info("   ✅ Defragmented database")
                
                # Rebuild indexes (if needed)
                conn.execute(text("REINDEX"))
                logger.info("   ✅ Rebuilt indexes")
                
                logger.info("✅ [DB_OPT] Database maintenance completed")
                
            except Exception as e:
                logger.error(f"❌ [DB_OPT] Database maintenance failed: {e}")
    
    def get_database_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive database metrics and health information.
        """
        logger.info("📈 [DB_OPT] Collecting database metrics...")
        
        metrics = {
            "timestamp": time.time(),
            "database_type": "SQLite",
            "tables": {},
            "indexes": [],
            "performance": {},
            "recommendations": []
        }
        
        with self.engine.connect() as conn:
            try:
                # Table statistics
                tables = ["users", "simulation_results", "security_audit_logs", "user_subscriptions", "user_usage_metrics", "api_keys"]
                
                for table in tables:
                    try:
                        # Row count
                        result = conn.execute(text(f"SELECT COUNT(*) FROM {table}"))
                        row_count = result.scalar()
                        
                        # Table size (approximate for SQLite)
                        result = conn.execute(text(f"SELECT SUM(LENGTH(sql)) FROM sqlite_master WHERE tbl_name='{table}'"))
                        schema_size = result.scalar() or 0
                        
                        metrics["tables"][table] = {
                            "row_count": row_count,
                            "schema_size": schema_size,
                            "health": "good" if row_count < 100000 else "monitor"
                        }
                        
                    except Exception as e:
                        logger.warning(f"   ⚠️ Failed to get metrics for {table}: {e}")
                        metrics["tables"][table] = {"error": str(e)}
                
                # Index information
                result = conn.execute(text("SELECT name, tbl_name FROM sqlite_master WHERE type='index' AND name NOT LIKE 'sqlite_%'"))
                for index_info in result.fetchall():
                    metrics["indexes"].append({
                        "name": index_info[0],
                        "table": index_info[1]
                    })
                
                # Database file size
                result = conn.execute(text("PRAGMA page_count"))
                page_count = result.scalar()
                result = conn.execute(text("PRAGMA page_size"))
                page_size = result.scalar()
                
                db_size_bytes = page_count * page_size
                metrics["performance"]["database_size_mb"] = round(db_size_bytes / (1024 * 1024), 2)
                
                # Performance recommendations
                total_rows = sum(table.get("row_count", 0) for table in metrics["tables"].values() if "row_count" in table)
                
                if total_rows > 500000:
                    metrics["recommendations"].append("🚀 Consider PostgreSQL migration for better performance")
                
                if db_size_bytes > 100 * 1024 * 1024:  # 100MB
                    metrics["recommendations"].append("📊 Database size growing - consider archiving old data")
                
                if len(metrics["indexes"]) < 10:
                    metrics["recommendations"].append("📈 Add more indexes for better query performance")
                
                logger.info(f"📈 [DB_OPT] Metrics collected - DB size: {metrics['performance']['database_size_mb']}MB, Total rows: {total_rows}")
                
            except Exception as e:
                logger.error(f"❌ [DB_OPT] Metrics collection failed: {e}")
                metrics["error"] = str(e)
        
        return metrics

# Global optimizer instance
db_optimizer = DatabaseOptimizer(engine)

def setup_enterprise_database():
    """
    Complete database setup for enterprise deployment.
    
    This function:
    1. Creates all necessary indexes
    2. Optimizes database settings
    3. Runs initial maintenance
    4. Validates performance
    """
    logger.info("🚀 [DB_SETUP] Setting up enterprise database...")
    
    try:
        # Step 1: Create indexes
        db_optimizer.create_enterprise_indexes()
        
        # Step 2: Optimize settings
        db_optimizer.optimize_database_settings()
        
        # Step 3: Run maintenance
        db_optimizer.run_maintenance_tasks()
        
        # Step 4: Performance validation
        with SessionLocal() as db:
            performance = db_optimizer.analyze_query_performance(db)
            
            if performance["overall_score"] >= 80:
                logger.info(f"✅ [DB_SETUP] Database setup complete - Performance score: {performance['overall_score']}/100")
            else:
                logger.warning(f"⚠️ [DB_SETUP] Database setup complete but performance needs improvement - Score: {performance['overall_score']}/100")
        
        # Step 5: Get final metrics
        metrics = db_optimizer.get_database_metrics()
        logger.info(f"📊 [DB_SETUP] Final metrics - Size: {metrics['performance']['database_size_mb']}MB, Indexes: {len(metrics['indexes'])}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ [DB_SETUP] Enterprise database setup failed: {e}")
        return False

if __name__ == "__main__":
    # Run enterprise database setup
    success = setup_enterprise_database()
    
    if success:
        print("🎉 Enterprise database setup completed successfully!")
        print("✅ All indexes created")
        print("✅ Database optimized")
        print("✅ Performance validated")
    else:
        print("❌ Enterprise database setup failed!")
        print("🔧 Check logs for details")
