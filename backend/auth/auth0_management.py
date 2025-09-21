"""
Auth0 Management API Service

This service provides methods to interact with Auth0's Management API
for complete user lifecycle management (create, read, update, delete).
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
import httpx
import json
from datetime import datetime, timedelta
from dataclasses import dataclass
import time

from config import settings

logger = logging.getLogger(__name__)

@dataclass
class Auth0User:
    """Represents an Auth0 user"""
    user_id: str
    email: Optional[str] = None
    username: Optional[str] = None
    name: Optional[str] = None
    nickname: Optional[str] = None
    picture: Optional[str] = None
    email_verified: bool = False
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    last_login: Optional[str] = None
    connection: Optional[str] = None
    app_metadata: Optional[Dict] = None
    user_metadata: Optional[Dict] = None

class Auth0ManagementService:
    """
    Auth0 Management API Service
    
    Provides methods to manage users in Auth0 using the Management API.
    Handles authentication, rate limiting, and error handling.
    """
    
    def __init__(self):
        self.domain = settings.AUTH0_DOMAIN
        self.client_id = settings.AUTH0_MANAGEMENT_CLIENT_ID
        self.client_secret = settings.AUTH0_MANAGEMENT_CLIENT_SECRET
        self.audience = settings.AUTH0_MANAGEMENT_AUDIENCE
        
        self.base_url = f"https://{self.domain}"
        self.api_url = f"{self.base_url}/api/v2"
        
        # Token caching
        self._access_token = None
        self._token_expires_at = None
        
        # Rate limiting
        self._last_request_time = 0
        self._min_request_interval = 0.1  # 100ms between requests
        
        logger.info(f"🔧 Auth0 Management Service initialized for domain: {self.domain}")
    
    async def _get_management_token(self) -> str:
        """
        Get a Management API access token using client credentials flow.
        Caches the token until it expires.
        """
        # Check if we have a valid cached token
        if (self._access_token and self._token_expires_at and 
            datetime.now() < self._token_expires_at - timedelta(minutes=5)):
            return self._access_token
        
        logger.info("🔐 Fetching new Auth0 Management API token...")
        
        token_url = f"{self.base_url}/oauth/token"
        
        payload = {
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "audience": self.audience,
            "grant_type": "client_credentials"
        }
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    token_url,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=30.0
                )
                response.raise_for_status()
                
                token_data = response.json()
                self._access_token = token_data["access_token"]
                expires_in = token_data.get("expires_in", 3600)
                self._token_expires_at = datetime.now() + timedelta(seconds=expires_in)
                
                logger.info(f"✅ Auth0 Management API token obtained, expires in {expires_in}s")
                return self._access_token
                
            except httpx.HTTPStatusError as e:
                logger.error(f"❌ HTTP error getting Auth0 token: {e.response.status_code} - {e.response.text}")
                raise Exception(f"Failed to get Auth0 Management API token: {e.response.status_code}")
            except Exception as e:
                logger.error(f"❌ Error getting Auth0 Management API token: {e}")
                raise Exception(f"Failed to get Auth0 Management API token: {str(e)}")
    
    async def _make_request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """
        Make an authenticated request to the Auth0 Management API.
        Handles rate limiting and token refresh.
        """
        # Rate limiting
        current_time = time.time()
        time_since_last = current_time - self._last_request_time
        if time_since_last < self._min_request_interval:
            await asyncio.sleep(self._min_request_interval - time_since_last)
        
        token = await self._get_management_token()
        url = f"{self.api_url}{endpoint}"
        
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.request(
                    method=method,
                    url=url,
                    headers=headers,
                    timeout=30.0,
                    **kwargs
                )
                
                self._last_request_time = time.time()
                
                # Handle rate limiting
                if response.status_code == 429:
                    retry_after = int(response.headers.get('Retry-After', 1))
                    logger.warning(f"⚠️ Rate limited by Auth0, waiting {retry_after}s...")
                    await asyncio.sleep(retry_after)
                    return await self._make_request(method, endpoint, **kwargs)
                
                response.raise_for_status()
                
                # Return empty dict for 204 No Content
                if response.status_code == 204:
                    return {}
                
                return response.json()
                
            except httpx.HTTPStatusError as e:
                logger.error(f"❌ Auth0 API error {method} {endpoint}: {e.response.status_code} - {e.response.text}")
                raise Exception(f"Auth0 API error: {e.response.status_code} - {e.response.text}")
            except Exception as e:
                logger.error(f"❌ Request error {method} {endpoint}: {e}")
                raise Exception(f"Request failed: {str(e)}")
    
    async def get_users(self, page: int = 0, per_page: int = 50, search: Optional[str] = None) -> List[Auth0User]:
        """
        Get users from Auth0.
        
        Args:
            page: Page number (0-based)
            per_page: Users per page (max 100)
            search: Optional search query
        
        Returns:
            List of Auth0User objects
        """
        logger.info(f"📋 Fetching Auth0 users (page {page}, per_page {per_page})")
        
        params = {
            "page": page,
            "per_page": min(per_page, 100),  # Auth0 limit
            "include_totals": "true"
        }
        
        if search:
            params["q"] = search
            params["search_engine"] = "v3"
        
        try:
            response = await self._make_request("GET", "/users", params=params)
            
            users = []
            for user_data in response.get("users", []):
                users.append(Auth0User(
                    user_id=user_data["user_id"],
                    email=user_data.get("email"),
                    username=user_data.get("username"),
                    name=user_data.get("name"),
                    nickname=user_data.get("nickname"),
                    picture=user_data.get("picture"),
                    email_verified=user_data.get("email_verified", False),
                    created_at=user_data.get("created_at"),
                    updated_at=user_data.get("updated_at"),
                    last_login=user_data.get("last_login"),
                    connection=user_data.get("identities", [{}])[0].get("connection"),
                    app_metadata=user_data.get("app_metadata", {}),
                    user_metadata=user_data.get("user_metadata", {})
                ))
            
            logger.info(f"✅ Retrieved {len(users)} users from Auth0")
            return users
            
        except Exception as e:
            logger.error(f"❌ Failed to get users from Auth0: {e}")
            raise
    
    async def get_user(self, user_id: str) -> Optional[Auth0User]:
        """
        Get a specific user from Auth0.
        
        Args:
            user_id: Auth0 user ID
        
        Returns:
            Auth0User object or None if not found
        """
        logger.info(f"👤 Fetching Auth0 user: {user_id}")
        
        try:
            user_data = await self._make_request("GET", f"/users/{user_id}")
            
            return Auth0User(
                user_id=user_data["user_id"],
                email=user_data.get("email"),
                username=user_data.get("username"),
                name=user_data.get("name"),
                nickname=user_data.get("nickname"),
                picture=user_data.get("picture"),
                email_verified=user_data.get("email_verified", False),
                created_at=user_data.get("created_at"),
                updated_at=user_data.get("updated_at"),
                last_login=user_data.get("last_login"),
                connection=user_data.get("identities", [{}])[0].get("connection"),
                app_metadata=user_data.get("app_metadata", {}),
                user_metadata=user_data.get("user_metadata", {})
            )
            
        except Exception as e:
            if "404" in str(e):
                logger.warning(f"⚠️ User {user_id} not found in Auth0")
                return None
            logger.error(f"❌ Failed to get user {user_id} from Auth0: {e}")
            raise
    
    async def create_user(self, email: str, password: str, username: Optional[str] = None, 
                         name: Optional[str] = None, connection: str = "Username-Password-Authentication") -> Auth0User:
        """
        Create a new user in Auth0.
        
        Args:
            email: User email
            password: User password
            username: Optional username
            name: Optional full name
            connection: Auth0 connection (default: Username-Password-Authentication)
        
        Returns:
            Created Auth0User object
        """
        logger.info(f"➕ Creating Auth0 user: {email}")
        
        user_data = {
            "email": email,
            "password": password,
            "connection": connection,
            "email_verified": False
        }
        
        if username:
            user_data["username"] = username
        if name:
            user_data["name"] = name
        
        try:
            response = await self._make_request("POST", "/users", json=user_data)
            
            logger.info(f"✅ Created Auth0 user: {response['user_id']}")
            
            return Auth0User(
                user_id=response["user_id"],
                email=response.get("email"),
                username=response.get("username"),
                name=response.get("name"),
                nickname=response.get("nickname"),
                picture=response.get("picture"),
                email_verified=response.get("email_verified", False),
                created_at=response.get("created_at"),
                updated_at=response.get("updated_at"),
                connection=connection
            )
            
        except Exception as e:
            logger.error(f"❌ Failed to create user {email} in Auth0: {e}")
            raise
    
    async def update_user(self, user_id: str, **updates) -> Auth0User:
        """
        Update a user in Auth0.
        
        Args:
            user_id: Auth0 user ID
            **updates: Fields to update
        
        Returns:
            Updated Auth0User object
        """
        logger.info(f"📝 Updating Auth0 user: {user_id}")
        
        try:
            response = await self._make_request("PATCH", f"/users/{user_id}", json=updates)
            
            logger.info(f"✅ Updated Auth0 user: {user_id}")
            
            return Auth0User(
                user_id=response["user_id"],
                email=response.get("email"),
                username=response.get("username"),
                name=response.get("name"),
                nickname=response.get("nickname"),
                picture=response.get("picture"),
                email_verified=response.get("email_verified", False),
                created_at=response.get("created_at"),
                updated_at=response.get("updated_at"),
                last_login=response.get("last_login"),
                connection=response.get("identities", [{}])[0].get("connection"),
                app_metadata=response.get("app_metadata", {}),
                user_metadata=response.get("user_metadata", {})
            )
            
        except Exception as e:
            logger.error(f"❌ Failed to update user {user_id} in Auth0: {e}")
            raise
    
    async def delete_user(self, user_id: str) -> bool:
        """
        Delete a user from Auth0.
        
        Args:
            user_id: Auth0 user ID
        
        Returns:
            True if deletion was successful
        """
        logger.info(f"🗑️ Deleting Auth0 user: {user_id}")
        
        try:
            await self._make_request("DELETE", f"/users/{user_id}")
            logger.info(f"✅ Deleted Auth0 user: {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to delete user {user_id} from Auth0: {e}")
            raise
    
    async def block_user(self, user_id: str) -> bool:
        """
        Block a user in Auth0.
        
        Args:
            user_id: Auth0 user ID
        
        Returns:
            True if blocking was successful
        """
        logger.info(f"🚫 Blocking Auth0 user: {user_id}")
        
        try:
            await self._make_request("PATCH", f"/users/{user_id}", json={"blocked": True})
            logger.info(f"✅ Blocked Auth0 user: {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to block user {user_id} in Auth0: {e}")
            raise
    
    async def unblock_user(self, user_id: str) -> bool:
        """
        Unblock a user in Auth0.
        
        Args:
            user_id: Auth0 user ID
        
        Returns:
            True if unblocking was successful
        """
        logger.info(f"✅ Unblocking Auth0 user: {user_id}")
        
        try:
            await self._make_request("PATCH", f"/users/{user_id}", json={"blocked": False})
            logger.info(f"✅ Unblocked Auth0 user: {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to unblock user {user_id} in Auth0: {e}")
            raise

# Global instance
auth0_management = Auth0ManagementService()
