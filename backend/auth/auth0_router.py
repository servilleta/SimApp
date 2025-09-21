from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from sqlalchemy import text
from pydantic import BaseModel
from typing import Optional, List
import logging

from config import settings
from database import get_db
from models import User as UserModel
from auth.auth0_dependencies import get_current_active_auth0_user, get_current_admin_auth0_user
from auth.auth0_management import auth0_management

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/auth0", tags=["auth0"])

class Auth0ConfigResponse(BaseModel):
    domain: str
    clientId: str
    audience: str
    redirectUri: str
    logoutUrl: str

class UserProfileResponse(BaseModel):
    id: int
    username: str
    email: Optional[str]
    full_name: Optional[str]
    is_admin: bool
    auth0_user_id: Optional[str]

@router.get("/config", response_model=Auth0ConfigResponse)
async def get_auth0_config():
    """
    Get Auth0 configuration for frontend
    """
    return Auth0ConfigResponse(
        domain=settings.AUTH0_DOMAIN,
        clientId=settings.AUTH0_CLIENT_ID,
        audience=settings.AUTH0_AUDIENCE,
        redirectUri=settings.AUTH0_CALLBACK_URL,
        logoutUrl=settings.AUTH0_LOGOUT_URL
    )

@router.get("/profile", response_model=UserProfileResponse)
async def get_user_profile(
    current_user: UserModel = Depends(get_current_active_auth0_user)
):
    """
    Get current user profile
    """
    return UserProfileResponse(
        id=current_user.id,
        username=current_user.username,
        email=current_user.email,
        full_name=current_user.full_name,
        is_admin=current_user.is_admin,
        auth0_user_id=current_user.auth0_user_id
    )

@router.post("/verify")
async def verify_token(
    current_user: UserModel = Depends(get_current_active_auth0_user)
):
    """
    Verify Auth0 token and return user info
    """
    return {
        "valid": True,
        "user": {
            "id": current_user.id,
            "username": current_user.username,
            "email": current_user.email,
            "is_admin": current_user.is_admin
        }
    }

@router.post("/admin/promote/{user_id}")
async def promote_user_to_admin(
    user_id: int,
    current_admin: UserModel = Depends(get_current_admin_auth0_user),
    db: Session = Depends(get_db)
):
    """
    Promote a user to admin (admin only)
    """
    user = db.query(UserModel).filter(UserModel.id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    if user.is_admin:
        return {"message": f"User {user.username} is already an admin"}
    
    user.is_admin = True
    db.commit()
    
    logger.info(f"Admin {current_admin.username} promoted user {user.username} to admin")
    
    return {"message": f"User {user.username} promoted to admin successfully"}

@router.get("/users")
async def list_users(
    current_admin: UserModel = Depends(get_current_admin_auth0_user),
    db: Session = Depends(get_db)
):
    """
    List all users (admin only)
    """
    users = db.query(UserModel).all()
    return [
        {
            "id": user.id,
            "username": user.username,
            "email": user.email,
            "full_name": user.full_name,
            "is_admin": user.is_admin,
            "disabled": user.disabled,
            "auth0_user_id": user.auth0_user_id,
            "created_at": user.created_at.isoformat() if user.created_at else None
        }
        for user in users
    ]

@router.delete("/users/{user_id}")
async def delete_user_complete(
    user_id: int,
    delete_from_auth0: bool = True,
    current_admin: UserModel = Depends(get_current_admin_auth0_user),
    db: Session = Depends(get_db)
):
    """
    Delete a user completely (admin only)
    
    This endpoint deletes the user from both the local database and Auth0.
    
    Args:
        user_id: Local database user ID
        delete_from_auth0: Whether to also delete from Auth0 (default: True)
    """
    logger.info(f"🗑️ Admin '{current_admin.username}' attempting to delete user {user_id} (Auth0: {delete_from_auth0})")
    
    # Get user from local database
    user = db.query(UserModel).filter(UserModel.id == user_id).first()
    if not user:
        logger.warning(f"Admin '{current_admin.username}' tried to delete non-existent user {user_id}")
        raise HTTPException(status_code=404, detail="User not found")
    
    auth0_user_id = user.auth0_user_id
    username = user.username
    email = user.email
    
    # Track deletion results
    deletion_results = {
        "local_database": False,
        "auth0": False,
        "errors": []
    }
    
    # Delete from Auth0 first (if requested and user has Auth0 ID)
    if delete_from_auth0 and auth0_user_id:
        try:
            logger.info(f"🔄 Deleting user from Auth0: {auth0_user_id}")
            await auth0_management.delete_user(auth0_user_id)
            deletion_results["auth0"] = True
            logger.info(f"✅ Successfully deleted user from Auth0: {auth0_user_id}")
        except Exception as e:
            error_msg = f"Failed to delete user from Auth0: {str(e)}"
            logger.error(f"❌ {error_msg}")
            deletion_results["errors"].append(error_msg)
            
            # Check if it's a permissions issue
            if "403" in str(e) or "unauthorized" in str(e).lower():
                deletion_results["errors"].append("Auth0 Management API needs proper permissions. Please configure 'delete:users' scope for your Auth0 application.")
            
            # Continue with local deletion even if Auth0 deletion fails
    
    # Delete from local database (with related records)
    try:
        logger.info(f"🔄 Deleting user from local database: {user.username} (id: {user.id})")
        
        # Delete related records first to avoid foreign key constraints
        from models import UserSubscription, SimulationResult
        
        # Delete user subscriptions
        user_subscriptions = db.query(UserSubscription).filter(UserSubscription.user_id == user.id).all()
        for subscription in user_subscriptions:
            logger.info(f"🗑️ Deleting subscription {subscription.id} for user {user.id}")
            db.delete(subscription)
        
        # Delete simulation results (if table exists)
        try:
            simulation_results = db.query(SimulationResult).filter(SimulationResult.user_id == user.id).all()
            for result in simulation_results:
                logger.info(f"🗑️ Deleting simulation result {result.id} for user {user.id}")
                db.delete(result)
        except Exception as sim_error:
            logger.warning(f"Could not delete simulation results: {sim_error}")
        
        # Delete any other related records that might exist
        try:
            # Check for saved simulations
            db.execute(text("DELETE FROM saved_simulations WHERE user_id = :user_id"), {"user_id": user.id})
        except Exception as saved_error:
            logger.warning(f"Could not delete saved simulations: {saved_error}")
        
        # Now delete the user
        db.delete(user)
        db.commit()
        deletion_results["local_database"] = True
        logger.info(f"✅ Successfully deleted user and related records from local database: {username}")
        
    except Exception as e:
        error_msg = f"Failed to delete user from local database: {str(e)}"
        logger.error(f"❌ {error_msg}")
        deletion_results["errors"].append(error_msg)
        db.rollback()
    
    # Prepare response
    success_parts = []
    if deletion_results["local_database"]:
        success_parts.append("local database")
    if deletion_results["auth0"]:
        success_parts.append("Auth0")
    
    if success_parts:
        message = f"User '{username}' successfully deleted from: {', '.join(success_parts)}"
        status_code = 200
    else:
        message = f"Failed to delete user '{username}'"
        status_code = 500
    
    if deletion_results["errors"]:
        message += f". Errors: {'; '.join(deletion_results['errors'])}"
    
    response = {
        "message": message,
        "user_id": user_id,
        "username": username,
        "email": email,
        "auth0_user_id": auth0_user_id,
        "deletion_results": deletion_results,
        "deleted_by": current_admin.username
    }
    
    if not success_parts:
        raise HTTPException(status_code=status_code, detail=response)
    
    return response

@router.post("/users")
async def create_user_complete(
    user_data: dict,
    create_in_auth0: bool = True,
    current_admin: UserModel = Depends(get_current_admin_auth0_user),
    db: Session = Depends(get_db)
):
    """
    Create a new user in both Auth0 and local database (admin only)
    
    Args:
        user_data: User information (email, password, username, name)
        create_in_auth0: Whether to create in Auth0 (default: True)
    """
    logger.info(f"➕ Admin '{current_admin.username}' creating new user: {user_data.get('email')}")
    
    email = user_data.get("email")
    password = user_data.get("password")
    username = user_data.get("username")
    name = user_data.get("name")
    
    if not email or not password:
        raise HTTPException(status_code=400, detail="Email and password are required")
    
    creation_results = {
        "auth0": False,
        "local_database": False,
        "errors": []
    }
    
    auth0_user_id = None
    
    # Create in Auth0 first
    if create_in_auth0:
        try:
            logger.info(f"🔄 Creating user in Auth0: {email}")
            auth0_user = await auth0_management.create_user(
                email=email,
                password=password,
                username=username,
                name=name
            )
            auth0_user_id = auth0_user.user_id
            creation_results["auth0"] = True
            logger.info(f"✅ Successfully created user in Auth0: {auth0_user_id}")
        except Exception as e:
            error_msg = f"Failed to create user in Auth0: {str(e)}"
            logger.error(f"❌ {error_msg}")
            creation_results["errors"].append(error_msg)
            # Continue with local creation even if Auth0 creation fails
    
    # Create in local database
    try:
        logger.info(f"🔄 Creating user in local database: {email}")
        
        # Check if user already exists locally
        existing_user = db.query(UserModel).filter(
            (UserModel.email == email) | (UserModel.username == username)
        ).first()
        
        if existing_user:
            raise Exception(f"User with email or username already exists locally")
        
        # Create local user
        from auth.service import get_password_hash
        
        local_user = UserModel(
            username=username or email.split('@')[0],
            email=email,
            full_name=name,
            hashed_password=get_password_hash(password),
            auth0_user_id=auth0_user_id,
            is_admin=False,
            disabled=False
        )
        
        db.add(local_user)
        db.commit()
        db.refresh(local_user)
        
        creation_results["local_database"] = True
        logger.info(f"✅ Successfully created user in local database: {local_user.id}")
        
    except Exception as e:
        error_msg = f"Failed to create user in local database: {str(e)}"
        logger.error(f"❌ {error_msg}")
        creation_results["errors"].append(error_msg)
        db.rollback()
    
    # Prepare response
    success_parts = []
    if creation_results["auth0"]:
        success_parts.append("Auth0")
    if creation_results["local_database"]:
        success_parts.append("local database")
    
    if success_parts:
        message = f"User '{email}' successfully created in: {', '.join(success_parts)}"
        status_code = 201
    else:
        message = f"Failed to create user '{email}'"
        status_code = 500
    
    if creation_results["errors"]:
        message += f". Errors: {'; '.join(creation_results['errors'])}"
    
    response = {
        "message": message,
        "email": email,
        "username": username,
        "auth0_user_id": auth0_user_id,
        "local_user_id": local_user.id if creation_results["local_database"] else None,
        "creation_results": creation_results,
        "created_by": current_admin.username
    }
    
    if not success_parts:
        raise HTTPException(status_code=status_code, detail=response)
    
    return response

@router.get("/users/sync")
async def sync_users_from_auth0(
    current_admin: UserModel = Depends(get_current_admin_auth0_user),
    db: Session = Depends(get_db)
):
    """
    Sync users from Auth0 to local database (admin only)
    
    This will fetch all users from Auth0 and create/update them in the local database.
    """
    logger.info(f"🔄 Admin '{current_admin.username}' syncing users from Auth0")
    
    try:
        # Get all users from Auth0
        auth0_users = await auth0_management.get_users(per_page=100)
        
        sync_results = {
            "total_auth0_users": len(auth0_users),
            "created": 0,
            "updated": 0,
            "errors": []
        }
        
        for auth0_user in auth0_users:
            try:
                # Check if user exists locally
                local_user = db.query(UserModel).filter(
                    UserModel.auth0_user_id == auth0_user.user_id
                ).first()
                
                if local_user:
                    # Update existing user
                    local_user.email = auth0_user.email or local_user.email
                    local_user.full_name = auth0_user.name or local_user.full_name
                    local_user.username = auth0_user.username or local_user.username
                    sync_results["updated"] += 1
                else:
                    # Create new user
                    username = auth0_user.username or auth0_user.nickname
                    if not username and auth0_user.email:
                        username = auth0_user.email.split('@')[0]
                    
                    local_user = UserModel(
                        username=username or f"user_{auth0_user.user_id.split('|')[-1]}",
                        email=auth0_user.email or f"unknown_{auth0_user.user_id}@auth0.local",
                        full_name=auth0_user.name,
                        auth0_user_id=auth0_user.user_id,
                        is_admin=False,
                        disabled=False,
                        hashed_password=""  # Auth0 users don't need local passwords
                    )
                    db.add(local_user)
                    sync_results["created"] += 1
                
                db.commit()
                
            except Exception as e:
                error_msg = f"Failed to sync user {auth0_user.user_id}: {str(e)}"
                logger.error(f"❌ {error_msg}")
                sync_results["errors"].append(error_msg)
                db.rollback()
        
        message = f"Sync completed: {sync_results['created']} created, {sync_results['updated']} updated"
        if sync_results["errors"]:
            message += f", {len(sync_results['errors'])} errors"
        
        return {
            "message": message,
            "sync_results": sync_results,
            "synced_by": current_admin.username
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to sync users from Auth0: {e}")
        raise HTTPException(status_code=500, detail=f"User sync failed: {str(e)}")

@router.post("/users/{user_id}/block")
async def block_user(
    user_id: int,
    current_admin: UserModel = Depends(get_current_admin_auth0_user),
    db: Session = Depends(get_db)
):
    """
    Block a user in Auth0 (admin only)
    """
    logger.info(f"🚫 Admin '{current_admin.username}' blocking user {user_id}")
    
    user = db.query(UserModel).filter(UserModel.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    if not user.auth0_user_id:
        raise HTTPException(status_code=400, detail="User is not managed by Auth0")
    
    try:
        await auth0_management.block_user(user.auth0_user_id)
        return {
            "message": f"User '{user.username}' blocked successfully",
            "user_id": user_id,
            "auth0_user_id": user.auth0_user_id,
            "blocked_by": current_admin.username
        }
    except Exception as e:
        logger.error(f"❌ Failed to block user {user_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to block user: {str(e)}")

@router.post("/users/{user_id}/unblock")
async def unblock_user(
    user_id: int,
    current_admin: UserModel = Depends(get_current_admin_auth0_user),
    db: Session = Depends(get_db)
):
    """
    Unblock a user in Auth0 (admin only)
    """
    logger.info(f"✅ Admin '{current_admin.username}' unblocking user {user_id}")
    
    user = db.query(UserModel).filter(UserModel.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    if not user.auth0_user_id:
        raise HTTPException(status_code=400, detail="User is not managed by Auth0")
    
    try:
        await auth0_management.unblock_user(user.auth0_user_id)
        return {
            "message": f"User '{user.username}' unblocked successfully",
            "user_id": user_id,
            "auth0_user_id": user.auth0_user_id,
            "unblocked_by": current_admin.username
        }
    except Exception as e:
        logger.error(f"❌ Failed to unblock user {user_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to unblock user: {str(e)}")

@router.post("/link-auth0-account")
async def link_auth0_account_to_existing_user(
    email: str,
    current_user: UserModel = Depends(get_current_active_auth0_user),
    db: Session = Depends(get_db)
):
    """
    Link current Auth0 user to an existing local user account (self-service)
    
    This allows users to link their Auth0 authentication to their existing local account.
    """
    logger.info(f"🔗 User '{current_user.username}' requesting to link Auth0 account to email: {email}")
    
    # Find the existing local user
    existing_user = db.query(UserModel).filter(UserModel.email == email).first()
    if not existing_user:
        raise HTTPException(status_code=404, detail="No local user found with that email address")
    
    if existing_user.auth0_user_id:
        raise HTTPException(status_code=400, detail="That user is already linked to an Auth0 account")
    
    # Verify the current user has the same email (security check)
    if current_user.email != email:
        raise HTTPException(status_code=403, detail="You can only link your own email address")
    
    try:
        # Link the accounts by updating the existing user with Auth0 info
        existing_user.auth0_user_id = current_user.auth0_user_id
        existing_user.full_name = current_user.full_name or existing_user.full_name
        
        # Remove the temporary Auth0-created user
        db.delete(current_user)
        db.commit()
        
        logger.info(f"✅ Successfully linked Auth0 account to existing user: {existing_user.username}")
        
        return {
            "message": f"Successfully linked your Auth0 account to existing user '{existing_user.username}'",
            "user_id": existing_user.id,
            "username": existing_user.username,
            "email": existing_user.email,
            "is_admin": existing_user.is_admin
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to link Auth0 account: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to link accounts: {str(e)}")

@router.get("/debug/current-user")
async def debug_current_user(
    current_user: UserModel = Depends(get_current_active_auth0_user)
):
    """Debug endpoint to see current user information"""
    return {
        "id": current_user.id,
        "username": current_user.username,
        "email": current_user.email,
        "is_admin": current_user.is_admin,
        "auth0_user_id": current_user.auth0_user_id,
        "full_name": current_user.full_name
    } 