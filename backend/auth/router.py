from fastapi import APIRouter, Depends, HTTPException, status, Request, Body
from fastapi.security import OAuth2PasswordRequestForm
from typing import Annotated
from sqlalchemy.orm import Session
import logging

from core.rate_limiter import limiter
from auth import service as auth_service
from auth import schemas as auth_schemas
from auth.dependencies import get_current_active_user, get_current_admin_user
from database import get_db

logger = logging.getLogger(__name__)

router = APIRouter(
    tags=["auth"],
    responses={404: {"description": "Not found"}}
)

@router.post("/register", response_model=auth_schemas.User)
async def register_user(user_in: auth_schemas.UserCreate, db: Session = Depends(get_db)):
    """
    Register a new user. (Endpoint is now disabled again)
    """
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="New user registrations are currently disabled."
    )
    # Original registration logic (now disabled again):
    # if user_in.password != user_in.password_confirm:
    #     raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Passwords do not match")
    # db_user = auth_service.create_user(db=db, user_in=user_in)
    # return db_user

@router.post("/token", response_model=auth_schemas.Token)
@limiter.limit("5 per minute")
async def login_for_access_token(request: Request, form_data: Annotated[OAuth2PasswordRequestForm, Depends()], db: Session = Depends(get_db)):
    """
    Standard OAuth2 password flow. Provides an access token.
    """
    user = auth_service.authenticate_user(db=db, username=form_data.username, password=form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token = auth_service.create_access_token(
        data={"sub": user.username} # "sub" is a standard claim for subject (the user)
    )
    return {"access_token": access_token, "token_type": "bearer"}

@router.get("/users/me", response_model=auth_schemas.User)
async def read_users_me(current_user: Annotated[auth_schemas.User, Depends(get_current_active_user)]):
    """
    Get current authenticated user's details.
    """
    logger.info(f"User '{current_user.username}' requesting own profile. is_admin: {current_user.is_admin}")
    return current_user 

@router.get("/users", response_model=list[auth_schemas.User])
async def list_users(db: Session = Depends(get_db), admin: auth_schemas.UserInDB = Depends(get_current_admin_user)):
    logger.info(f"Admin user '{admin.username}' is listing all users")
    users = db.query(auth_service.UserModel).all()
    logger.info(f"Found {len(users)} users in database")
    return users

@router.post("/users", response_model=auth_schemas.User)
async def admin_create_user(user_in: auth_schemas.UserCreateAdmin, db: Session = Depends(get_db), admin: auth_schemas.UserInDB = Depends(get_current_admin_user)):
    logger.info(f"Admin '{admin.username}' attempting to create user: username='{user_in.username}', email='{user_in.email}', is_admin={user_in.is_admin}")
    try:
        db_user = auth_service.create_user(db=db, user_in=user_in)
        logger.info(f"User '{user_in.username}' created successfully by admin '{admin.username}'")
        return db_user
    except HTTPException as e:
        logger.error(f"Failed to create user '{user_in.username}': {e.detail}")
        raise e
    except Exception as e:
        logger.error(f"Unexpected error creating user '{user_in.username}': {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error during user creation")

@router.patch("/users/{user_id}", response_model=auth_schemas.User)
async def admin_edit_user(user_id: int, user_update: auth_schemas.UserUpdate, db: Session = Depends(get_db), admin: auth_schemas.UserInDB = Depends(get_current_admin_user)):
    logger.info(f"Admin '{admin.username}' attempting to edit user {user_id}: {user_update.dict(exclude_unset=True)}")
    user = db.query(auth_service.UserModel).filter(auth_service.UserModel.id == user_id).first()
    if not user:
        logger.warning(f"Admin '{admin.username}' tried to edit non-existent user {user_id}")
        raise HTTPException(status_code=404, detail="User not found")
    try:
        updated_user = auth_service.update_user(db, user, user_update.dict(exclude_unset=True))
        logger.info(f"User {user_id} updated successfully by admin '{admin.username}'")
        return updated_user
    except Exception as e:
        logger.error(f"Error updating user {user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error during user update")

@router.delete("/users/{user_id}", status_code=204)
async def admin_delete_user(user_id: int, db: Session = Depends(get_db), admin: auth_schemas.UserInDB = Depends(get_current_admin_user)):
    logger.info(f"Admin '{admin.username}' attempting to delete user {user_id}")
    user = db.query(auth_service.UserModel).filter(auth_service.UserModel.id == user_id).first()
    if not user:
        logger.warning(f"Admin '{admin.username}' tried to delete non-existent user {user_id}")
        raise HTTPException(status_code=404, detail="User not found")
    logger.info(f"Deleting user: {user.username} (id: {user.id})")
    db.delete(user)
    db.commit()
    logger.info(f"User {user_id} ('{user.username}') deleted successfully by admin '{admin.username}'")
    return Response(status_code=204) 