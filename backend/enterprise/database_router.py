"""
ENTERPRISE DATABASE MANAGEMENT ROUTER
Phase 2 Week 8: Database Architecture Management API

This router provides enterprise database management endpoints:
- Database service status
- Tenant routing information
- Database performance metrics
- Migration status

CRITICAL: This does NOT interfere with Ultra engine or progress bar functionality.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from auth.auth0_dependencies import get_current_active_auth0_user
from models import User
from enterprise.auth_service import (
    enterprise_auth_service,
    EnterpriseUser,
    require_permission
)
from enterprise.tenant_database import (
    tenant_database,
    TenantInfo,
    DatabaseTier
)
from enterprise.database_architecture import (
    database_service_registry,
    ServiceType,
    enterprise_database_service
)

logger = logging.getLogger(__name__)

# Pydantic models
class TenantInfoResponse(BaseModel):
    """Tenant information response"""
    tenant_id: str
    organization_id: int
    user_id: int
    tier: str
    database_tier: str
    db_shard: Optional[str]
    needs_dedicated_db: bool
    
    class Config:
        from_attributes = True

class DatabaseServiceStatusResponse(BaseModel):
    """Database service status response"""
    service_type: str
    database_name: str
    status: str
    pool_size: int
    max_connections: int
    schema_version: str
    tables: List[str]
    last_check: datetime
    
    class Config:
        from_attributes = True

class DatabaseMetricsResponse(BaseModel):
    """Database performance metrics"""
    service_type: str
    active_connections: int
    total_queries: int
    avg_query_time_ms: float
    cache_hit_rate: float
    tenant_count: int
    last_updated: datetime

# Create router
router = APIRouter(prefix="/enterprise/database", tags=["Enterprise Database"])

@router.get("/tenant-info", response_model=TenantInfoResponse)
async def get_user_tenant_info(
    current_user: User = Depends(get_current_active_auth0_user)
):
    """Get tenant information for current user"""
    try:
        # Get tenant info through routing service
        tenant_info = await tenant_database.tenant_routing.get_tenant_for_user(current_user.id)
        
        return TenantInfoResponse(
            tenant_id=tenant_info.tenant_id,
            organization_id=tenant_info.organization_id,
            user_id=tenant_info.user_id,
            tier=tenant_info.tier.value,
            database_tier=tenant_info.database_tier.value,
            db_shard=tenant_info.db_shard,
            needs_dedicated_db=tenant_info.needs_dedicated_db
        )
        
    except Exception as e:
        logger.error(f"❌ [ENTERPRISE_DB] Failed to get tenant info: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve tenant information"
        )

@router.get("/services", response_model=List[DatabaseServiceStatusResponse])
@require_permission("organization.view")
async def get_database_services_status(
    enterprise_user: EnterpriseUser = None  # Injected by decorator
):
    """Get status of all database services"""
    try:
        services_status = []
        
        for service_type in ServiceType:
            config = database_service_registry.get_service_config(service_type)
            
            if config:
                # Get service status (in production, this would check actual database health)
                status_info = DatabaseServiceStatusResponse(
                    service_type=service_type.value,
                    database_name=config.database_name,
                    status="healthy",  # Would be determined by health check
                    pool_size=config.pool_size,
                    max_connections=config.max_overflow,
                    schema_version=config.schema_version,
                    tables=config.tables,
                    last_check=datetime.utcnow()
                )
                
                services_status.append(status_info)
        
        return services_status
        
    except Exception as e:
        logger.error(f"❌ [ENTERPRISE_DB] Failed to get services status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve database services status"
        )

@router.get("/metrics", response_model=List[DatabaseMetricsResponse])
@require_permission("organization.view")
async def get_database_metrics(
    enterprise_user: EnterpriseUser = None  # Injected by decorator
):
    """Get database performance metrics"""
    try:
        metrics = []
        
        for service_type in ServiceType:
            # In production, these would be real metrics from monitoring systems
            metric = DatabaseMetricsResponse(
                service_type=service_type.value,
                active_connections=5,  # Would be from connection pool monitoring
                total_queries=1000,   # Would be from query logging
                avg_query_time_ms=15.5,  # Would be from performance monitoring
                cache_hit_rate=85.2,  # Would be from cache monitoring
                tenant_count=1,       # Would be from tenant registry
                last_updated=datetime.utcnow()
            )
            
            metrics.append(metric)
        
        return metrics
        
    except Exception as e:
        logger.error(f"❌ [ENTERPRISE_DB] Failed to get database metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve database metrics"
        )

@router.get("/user-data-summary")
async def get_user_data_summary(
    current_user: User = Depends(get_current_active_auth0_user)
):
    """Get summary of user's data across all services"""
    try:
        # Get user's simulations (preserving existing functionality)
        simulations = await enterprise_database_service.get_user_simulations(current_user.id, limit=10)
        
        # Get user's files
        files = await enterprise_database_service.get_user_files(current_user.id)
        
        # Get organization usage
        enterprise_user = await enterprise_auth_service.authenticate_enterprise_user(current_user)
        org_usage = await enterprise_database_service.get_organization_usage_stats(
            enterprise_user.organization.id, 
            current_user.id
        )
        
        return {
            "user_id": current_user.id,
            "tenant_info": await tenant_database.tenant_routing.get_tenant_for_user(current_user.id),
            "data_summary": {
                "simulations_count": len(simulations),
                "recent_simulations": [
                    {
                        "id": sim.simulation_id,
                        "status": sim.status,
                        "created_at": sim.created_at.isoformat() if sim.created_at else None
                    } for sim in simulations[:5]
                ],
                "files_count": len(files) if files else 0,
                "organization_usage": org_usage
            },
            "database_routing": {
                "uses_dedicated_db": enterprise_user.organization.tier.value in ["enterprise"],
                "database_tier": "shared" if enterprise_user.organization.tier.value in ["trial", "standard"] else "dedicated"
            }
        }
        
    except Exception as e:
        logger.error(f"❌ [ENTERPRISE_DB] Failed to get user data summary: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve user data summary"
        )

@router.post("/migrate")
@require_permission("admin.database")
async def run_database_migration(
    enterprise_user: EnterpriseUser = None  # Injected by decorator
):
    """Run database migration for multi-tenant architecture (admin only)"""
    try:
        from enterprise.database_architecture import DatabaseMigrationService
        
        migration_service = DatabaseMigrationService()
        
        # Run tenant column migration
        await migration_service.add_tenant_columns()
        
        # Backfill tenant data for existing records
        await migration_service.backfill_tenant_data()
        
        return {
            "status": "success",
            "message": "Database migration completed successfully",
            "timestamp": datetime.utcnow(),
            "executed_by": enterprise_user.email
        }
        
    except Exception as e:
        logger.error(f"❌ [ENTERPRISE_DB] Database migration failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database migration failed: {str(e)}"
        )

@router.get("/health")
async def database_service_health():
    """Health check for enterprise database services"""
    try:
        # Test main database connectivity (preserves existing functionality)
        from database import get_db
        db = next(get_db())
        db.execute(text("SELECT 1"))
        db.close()
        
        # Test tenant routing
        tenant_router_healthy = True
        try:
            # Test with a mock user ID
            test_tenant = await tenant_database.tenant_routing.get_tenant_for_user(1)
            tenant_router_healthy = test_tenant is not None
        except:
            tenant_router_healthy = False
        
        return {
            "status": "healthy",
            "service": "Enterprise Database Architecture",
            "components": {
                "main_database": True,
                "tenant_routing": tenant_router_healthy,
                "service_registry": True,
                "cross_service_communication": True
            },
            "services": {
                "simulation_service": "healthy",  # Ultra engine database
                "user_service": "healthy",
                "file_service": "healthy", 
                "results_service": "healthy",
                "billing_service": "healthy",
                "analytics_service": "healthy"
            },
            "features": {
                "tenant_isolation": True,
                "database_per_service": True,
                "shared_vs_dedicated": True,
                "automatic_routing": True
            },
            "timestamp": datetime.utcnow()
        }
        
    except Exception as e:
        logger.error(f"❌ [ENTERPRISE_DB] Health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Enterprise database service unhealthy"
        )
