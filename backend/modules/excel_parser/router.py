"""
Excel Parser router module

Handles HTTP endpoints for Excel parsing functionality.
Uses dependency injection to get the ExcelParserService instance.
"""

from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Annotated, Dict, Any

from ..base import ExcelParserServiceProtocol, AuthServiceProtocol


# Security scheme for JWT tokens
security = HTTPBearer()


def create_excel_parser_router(
    excel_parser_service: ExcelParserServiceProtocol,
    auth_service: AuthServiceProtocol
) -> APIRouter:
    """Create excel parser router with injected services"""
    
    router = APIRouter(prefix="/api/excel-parser", tags=["excel-parser"])
    
    async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
        """Get current user from JWT token"""
        user = await auth_service.verify_token(credentials.credentials)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return user
    
    @router.post("/upload")
    async def upload_excel_file(
        file: UploadFile = File(...),
        current_user: Dict = Depends(get_current_user)
    ):
        """Upload and parse Excel file"""
        
        # Validate file type
        if not file.filename.endswith(('.xlsx', '.xls')):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Only Excel files (.xlsx, .xls) are allowed"
            )
        
        try:
            result = await excel_parser_service.parse_excel_file(
                file=file,
                user_id=current_user["id"]
            )
            
            return result
            
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to parse Excel file: {str(e)}"
            )
    
    @router.get("/files/{file_id}")
    async def get_file_info(
        file_id: str,
        current_user: Dict = Depends(get_current_user)
    ):
        """Get information about a parsed Excel file"""
        
        try:
            result = await excel_parser_service.get_file_info(
                file_id=file_id,
                user_id=current_user["id"]
            )
            
            if not result:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="File not found"
                )
            
            return result
            
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to get file info: {str(e)}"
            )
    
    @router.get("/files/{file_id}/sheets")
    async def get_file_sheets(
        file_id: str,
        current_user: Dict = Depends(get_current_user)
    ):
        """Get sheet information for a parsed Excel file"""
        
        try:
            result = await excel_parser_service.get_file_sheets(
                file_id=file_id,
                user_id=current_user["id"]
            )
            
            if not result:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="File not found"
                )
            
            return result
            
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to get file sheets: {str(e)}"
            )
    
    @router.get("/files")
    async def list_user_files(
        current_user: Dict = Depends(get_current_user),
        skip: int = 0,
        limit: int = 10
    ):
        """List user's uploaded Excel files"""
        
        try:
            files = await excel_parser_service.list_user_files(
                user_id=current_user["id"],
                skip=skip,
                limit=limit
            )
            
            return files
            
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to list files: {str(e)}"
            )
    
    @router.delete("/files/{file_id}")
    async def delete_file(
        file_id: str,
        current_user: Dict = Depends(get_current_user)
    ):
        """Delete an uploaded Excel file"""
        
        try:
            success = await excel_parser_service.delete_file(
                file_id=file_id,
                user_id=current_user["id"]
            )
            
            if not success:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="File not found"
                )
            
            return {"message": "File deleted successfully"}
            
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to delete file: {str(e)}"
            )
    
    return router 