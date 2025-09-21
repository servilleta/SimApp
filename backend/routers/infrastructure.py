"""
Infrastructure Management API Endpoints
======================================

Provides REST API endpoints for managing Paperspace infrastructure:
- Manual server start/stop operations
- Automatic scaling management  
- Cost optimization reporting
- Cluster status monitoring

Author: SimApp DevOps Team
Date: September 21, 2025
"""

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from fastapi.security import HTTPBearer
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime

from auth.auth_middleware import get_current_user
from services.paperspace_service import get_paperspace_service, PaperspaceScalingService
from pydantic import BaseModel

logger = logging.getLogger(__name__)
security = HTTPBearer()

router = APIRouter(
    prefix="/infrastructure",
    tags=["Infrastructure"],
    dependencies=[Depends(security)]
)

# Pydantic models for request/response
class ScalingAction(BaseModel):
    action: str
    reason: Optional[str] = None
    force: bool = False

class ServerControl(BaseModel):
    server_type: str  # "primary" or "secondary"
    action: str       # "start" or "stop"

class ClusterStatusResponse(BaseModel):
    timestamp: str
    primary_server: Dict[str, Any]
    secondary_server: Dict[str, Any]
    total_machines: int
    running_machines: int
    simulation_load: Dict[str, int]

@router.get("/status", response_model=ClusterStatusResponse)
async def get_cluster_status(
    current_user=Depends(get_current_user),
    paperspace_service: PaperspaceScalingService = Depends(get_paperspace_service)
):
    """
    Get comprehensive cluster status including Paperspace machines and simulation load
    """
    try:
        logger.info(f"📊 User {current_user.get('email')} requested cluster status")
        
        # Get Paperspace cluster status
        cluster_status = paperspace_service.api_manager.get_cluster_status()
        
        # Get simulation load
        simulation_load = await paperspace_service.get_simulation_load()
        
        response = ClusterStatusResponse(
            timestamp=cluster_status["timestamp"],
            primary_server=cluster_status["primary_server"],
            secondary_server=cluster_status["secondary_server"],
            total_machines=cluster_status["total_machines"],
            running_machines=cluster_status["running_machines"],
            simulation_load=simulation_load
        )
        
        return response
        
    except Exception as e:
        logger.error(f"❌ Failed to get cluster status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get cluster status: {str(e)}")

@router.post("/scale")
async def manual_scaling_action(
    scaling_request: ScalingAction,
    background_tasks: BackgroundTasks,
    current_user=Depends(get_current_user),
    paperspace_service: PaperspaceScalingService = Depends(get_paperspace_service)
):
    """
    Perform manual scaling action (scale-up or scale-down)
    """
    try:
        user_email = current_user.get('email', 'unknown')
        logger.info(f"🎛️ User {user_email} requested manual scaling: {scaling_request.action}")
        
        if scaling_request.action not in ["scale-up", "scale-down"]:
            raise HTTPException(
                status_code=400, 
                detail="Invalid action. Must be 'scale-up' or 'scale-down'"
            )
        
        # Check cooldown unless force=True
        if not scaling_request.force and not paperspace_service.can_perform_scaling_action():
            time_remaining = paperspace_service.scaling_cooldown - \
                (datetime.utcnow() - paperspace_service.last_scale_action).total_seconds()
            raise HTTPException(
                status_code=429,
                detail=f"Scaling cooldown active. Try again in {int(time_remaining)} seconds or use force=true"
            )
        
        # Perform scaling action
        if scaling_request.action == "scale-up":
            success = paperspace_service.api_manager.scale_up_secondary_server()
            action_description = "Starting secondary server"
        else:
            success = paperspace_service.api_manager.scale_down_secondary_server()
            action_description = "Stopping secondary server"
        
        if success:
            paperspace_service.last_scale_action = datetime.utcnow()
            logger.info(f"✅ {action_description} successful for user {user_email}")
        
        return {
            "success": success,
            "action": scaling_request.action,
            "description": action_description,
            "timestamp": datetime.utcnow().isoformat(),
            "user": user_email,
            "reason": scaling_request.reason or "manual_request"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Manual scaling failed: {e}")
        raise HTTPException(status_code=500, detail=f"Scaling operation failed: {str(e)}")

@router.post("/auto-scale")
async def trigger_auto_scaling(
    background_tasks: BackgroundTasks,
    current_user=Depends(get_current_user),
    paperspace_service: PaperspaceScalingService = Depends(get_paperspace_service)
):
    """
    Trigger automatic scaling check based on current workload
    """
    try:
        user_email = current_user.get('email', 'unknown')
        logger.info(f"🤖 User {user_email} triggered auto-scaling check")
        
        # Run auto-scaling in background
        def run_auto_scale():
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(paperspace_service.auto_scale_cluster())
            loop.close()
            return result
        
        background_tasks.add_task(run_auto_scale)
        
        return {
            "message": "Auto-scaling check triggered",
            "timestamp": datetime.utcnow().isoformat(),
            "user": user_email,
            "status": "running_in_background"
        }
        
    except Exception as e:
        logger.error(f"❌ Auto-scaling trigger failed: {e}")
        raise HTTPException(status_code=500, detail=f"Auto-scaling trigger failed: {str(e)}")

@router.get("/cost-optimization")
async def get_cost_optimization_report(
    current_user=Depends(get_current_user),
    paperspace_service: PaperspaceScalingService = Depends(get_paperspace_service)
):
    """
    Get cost optimization analysis and recommendations
    """
    try:
        user_email = current_user.get('email', 'unknown')
        logger.info(f"💰 User {user_email} requested cost optimization report")
        
        report = await paperspace_service.get_cost_optimization_report()
        
        return report
        
    except Exception as e:
        logger.error(f"❌ Failed to generate cost report: {e}")
        raise HTTPException(status_code=500, detail=f"Cost report generation failed: {str(e)}")

@router.post("/server-control")
async def control_server(
    control_request: ServerControl,
    background_tasks: BackgroundTasks,
    current_user=Depends(get_current_user),
    paperspace_service: PaperspaceScalingService = Depends(get_paperspace_service)
):
    """
    Direct server control (start/stop specific servers)
    """
    try:
        user_email = current_user.get('email', 'unknown')
        logger.info(f"🎛️ User {user_email} requested server control: {control_request.server_type} -> {control_request.action}")
        
        if control_request.server_type not in ["secondary"]:
            raise HTTPException(
                status_code=400,
                detail="Only 'secondary' server control is supported"
            )
        
        if control_request.action not in ["start", "stop"]:
            raise HTTPException(
                status_code=400,
                detail="Action must be 'start' or 'stop'"
            )
        
        # Get machine ID for secondary server
        machine_id = paperspace_service.api_manager.find_secondary_server()
        if not machine_id:
            raise HTTPException(
                status_code=404,
                detail="Secondary server not found in Paperspace account"
            )
        
        # Perform action
        if control_request.action == "start":
            success = paperspace_service.api_manager.start_machine(machine_id)
        else:
            success = paperspace_service.api_manager.stop_machine(machine_id)
        
        return {
            "success": success,
            "server_type": control_request.server_type,
            "action": control_request.action,
            "machine_id": machine_id,
            "timestamp": datetime.utcnow().isoformat(),
            "user": user_email
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Server control failed: {e}")
        raise HTTPException(status_code=500, detail=f"Server control failed: {str(e)}")

@router.get("/machines")
async def list_paperspace_machines(
    current_user=Depends(get_current_user),
    paperspace_service: PaperspaceScalingService = Depends(get_paperspace_service)
):
    """
    List all Paperspace machines in the account
    """
    try:
        user_email = current_user.get('email', 'unknown')
        logger.info(f"📋 User {user_email} requested machine list")
        
        machines = paperspace_service.api_manager.list_machines()
        
        return {
            "machines": [
                {
                    "id": machine.id,
                    "name": machine.name,
                    "state": machine.state,
                    "public_ip": machine.public_ip,
                    "private_ip": machine.private_ip,
                    "region": machine.region,
                    "machine_type": machine.machine_type,
                    "os": machine.os,
                    "created_at": machine.created_at,
                    "updated_at": machine.updated_at
                }
                for machine in machines
            ],
            "total_count": len(machines),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to list machines: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list machines: {str(e)}")

@router.get("/health")
async def infrastructure_health_check():
    """
    Health check endpoint for infrastructure management
    """
    return {
        "status": "healthy",
        "service": "infrastructure_management",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0"
    }
