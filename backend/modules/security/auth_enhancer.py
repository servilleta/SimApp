"""
Authentication Enhancement Service - Security Module

Provides enhanced authentication features including:
- OAuth2 providers (Google, Microsoft, GitHub)
- Two-Factor Authentication (2FA) with TOTP
- Refresh token support
- Email verification
- Account lockout policies
- Session management
"""

import logging
import secrets
import pyotp
import qrcode
from io import BytesIO
import base64
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from authlib.integrations.requests_client import OAuth2Session
from authlib.integrations.base_client import OAuthError
import httpx

from ..base import BaseService
from ..auth.service import AuthService


logger = logging.getLogger(__name__)


class AuthEnhancerService(BaseService):
    """Enhanced authentication service with OAuth2 and 2FA"""
    
    # OAuth2 provider configurations
    OAUTH_PROVIDERS = {
        'google': {
            'client_id': None,  # Set from environment
            'client_secret': None,  # Set from environment
            'authorize_url': 'https://accounts.google.com/o/oauth2/auth',
            'token_url': 'https://oauth2.googleapis.com/token',
            'userinfo_url': 'https://www.googleapis.com/oauth2/v2/userinfo',
            'scope': 'openid email profile'
        },
        'microsoft': {
            'client_id': None,  # Set from environment
            'client_secret': None,  # Set from environment
            'authorize_url': 'https://login.microsoftonline.com/common/oauth2/v2.0/authorize',
            'token_url': 'https://login.microsoftonline.com/common/oauth2/v2.0/token',
            'userinfo_url': 'https://graph.microsoft.com/v1.0/me',
            'scope': 'openid email profile'
        },
        'github': {
            'client_id': None,  # Set from environment
            'client_secret': None,  # Set from environment
            'authorize_url': 'https://github.com/login/oauth/authorize',
            'token_url': 'https://github.com/login/oauth/access_token',
            'userinfo_url': 'https://api.github.com/user',
            'scope': 'user:email'
        }
    }
    
    # Account lockout settings
    LOCKOUT_SETTINGS = {
        'max_attempts': 5,
        'lockout_duration': 3600,  # 1 hour in seconds
        'reset_time': 900  # 15 minutes to reset attempt counter
    }
    
    def __init__(self, auth_service: AuthService):
        super().__init__("auth_enhancer")
        self.auth_service = auth_service
        self.failed_attempts = {}  # In-memory storage for failed attempts
        self.refresh_tokens = {}  # In-memory storage for refresh tokens
        self.verification_codes = {}  # In-memory storage for email verification
        
    async def initialize(self) -> None:
        """Initialize the auth enhancer service"""
        await super().initialize()
        
        # Load OAuth2 configurations from environment
        import os
        for provider in self.OAUTH_PROVIDERS:
            self.OAUTH_PROVIDERS[provider]['client_id'] = os.getenv(f'{provider.upper()}_CLIENT_ID')
            self.OAUTH_PROVIDERS[provider]['client_secret'] = os.getenv(f'{provider.upper()}_CLIENT_SECRET')
        
        logger.info("Auth enhancer service initialized")
    
    # OAuth2 Authentication
    async def get_oauth_authorization_url(self, provider: str, redirect_uri: str) -> Dict[str, str]:
        """Get OAuth2 authorization URL for provider"""
        if provider not in self.OAUTH_PROVIDERS:
            raise ValueError(f"Unsupported OAuth provider: {provider}")
        
        config = self.OAUTH_PROVIDERS[provider]
        if not config['client_id']:
            raise ValueError(f"OAuth provider {provider} not configured")
        
        try:
            client = OAuth2Session(
                client_id=config['client_id'],
                redirect_uri=redirect_uri,
                scope=config['scope']
            )
            
            authorization_url, state = client.create_authorization_url(config['authorize_url'])
            
            return {
                'authorization_url': authorization_url,
                'state': state,
                'provider': provider
            }
            
        except Exception as e:
            logger.error(f"Failed to create OAuth authorization URL for {provider}: {e}")
            raise
    
    async def handle_oauth_callback(self, provider: str, code: str, state: str, 
                                   redirect_uri: str) -> Dict[str, any]:
        """Handle OAuth2 callback and create/login user"""
        if provider not in self.OAUTH_PROVIDERS:
            raise ValueError(f"Unsupported OAuth provider: {provider}")
        
        config = self.OAUTH_PROVIDERS[provider]
        
        try:
            # Exchange code for token
            client = OAuth2Session(
                client_id=config['client_id'],
                redirect_uri=redirect_uri
            )
            
            token = client.fetch_token(
                config['token_url'],
                code=code,
                client_secret=config['client_secret']
            )
            
            # Get user info
            client.token = token
            async with httpx.AsyncClient() as http_client:
                response = await http_client.get(
                    config['userinfo_url'],
                    headers={'Authorization': f"Bearer {token['access_token']}"}
                )
                response.raise_for_status()
                user_info = response.json()
            
            # Extract user data based on provider
            user_data = await self._extract_user_data(provider, user_info)
            
            # Create or login user
            user = await self._create_or_login_oauth_user(user_data, provider)
            
            # Generate tokens
            access_token = await self.auth_service.create_access_token(data={"sub": user.email})
            refresh_token = await self._create_refresh_token(user.email)
            
            return {
                'access_token': access_token,
                'refresh_token': refresh_token,
                'token_type': 'bearer',
                'user': user,
                'provider': provider
            }
            
        except Exception as e:
            logger.error(f"OAuth callback failed for {provider}: {e}")
            raise
    
    async def _extract_user_data(self, provider: str, user_info: Dict) -> Dict:
        """Extract standardized user data from provider-specific response"""
        if provider == 'google':
            return {
                'email': user_info.get('email'),
                'full_name': user_info.get('name'),
                'first_name': user_info.get('given_name'),
                'last_name': user_info.get('family_name'),
                'picture': user_info.get('picture'),
                'verified_email': user_info.get('verified_email', False)
            }
        elif provider == 'microsoft':
            return {
                'email': user_info.get('mail') or user_info.get('userPrincipalName'),
                'full_name': user_info.get('displayName'),
                'first_name': user_info.get('givenName'),
                'last_name': user_info.get('surname'),
                'picture': None,  # Would need separate Graph API call
                'verified_email': True  # Microsoft accounts are pre-verified
            }
        elif provider == 'github':
            return {
                'email': user_info.get('email'),
                'full_name': user_info.get('name'),
                'first_name': user_info.get('name', '').split(' ')[0] if user_info.get('name') else None,
                'last_name': ' '.join(user_info.get('name', '').split(' ')[1:]) if user_info.get('name') else None,
                'picture': user_info.get('avatar_url'),
                'verified_email': user_info.get('email') is not None
            }
        else:
            raise ValueError(f"Unknown provider: {provider}")
    
    async def _create_or_login_oauth_user(self, user_data: Dict, provider: str):
        """Create new user or login existing user from OAuth"""
        # Try to find existing user by email
        user = await self.auth_service.get_user_by_email(user_data['email'])
        
        if user:
            # Update user info if needed
            if not user.full_name and user_data.get('full_name'):
                user.full_name = user_data['full_name']
            if not user.is_verified and user_data.get('verified_email'):
                user.is_verified = True
            
            # Add OAuth provider info
            if not hasattr(user, 'oauth_providers'):
                user.oauth_providers = []
            if provider not in user.oauth_providers:
                user.oauth_providers.append(provider)
            
            await self.auth_service.update_user(user)
        else:
            # Create new user
            user = await self.auth_service.create_user(
                email=user_data['email'],
                password=None,  # OAuth users don't need passwords
                full_name=user_data.get('full_name'),
                is_verified=user_data.get('verified_email', False),
                oauth_providers=[provider]
            )
        
        return user
    
    # Two-Factor Authentication (2FA)
    async def setup_2fa(self, user_email: str) -> Dict[str, str]:
        """Setup 2FA for user and return QR code"""
        try:
            # Generate secret key
            secret = pyotp.random_base32()
            
            # Create TOTP instance
            totp = pyotp.TOTP(secret)
            
            # Generate QR code
            qr_uri = totp.provisioning_uri(
                name=user_email,
                issuer_name="Monte Carlo Platform"
            )
            
            # Create QR code image
            qr = qrcode.QRCode(version=1, box_size=10, border=5)
            qr.add_data(qr_uri)
            qr.make(fit=True)
            
            qr_image = qr.make_image(fill_color="black", back_color="white")
            
            # Convert to base64
            buffer = BytesIO()
            qr_image.save(buffer, format='PNG')
            qr_base64 = base64.b64encode(buffer.getvalue()).decode()
            
            # Store secret temporarily (should be stored in database)
            # For now, we'll use in-memory storage
            if not hasattr(self, 'temp_2fa_secrets'):
                self.temp_2fa_secrets = {}
            self.temp_2fa_secrets[user_email] = secret
            
            return {
                'secret': secret,
                'qr_code': f"data:image/png;base64,{qr_base64}",
                'manual_entry_key': secret
            }
            
        except Exception as e:
            logger.error(f"Failed to setup 2FA for {user_email}: {e}")
            raise
    
    async def verify_2fa_setup(self, user_email: str, token: str) -> bool:
        """Verify 2FA setup with user-provided token"""
        try:
            if not hasattr(self, 'temp_2fa_secrets'):
                return False
            
            secret = self.temp_2fa_secrets.get(user_email)
            if not secret:
                return False
            
            totp = pyotp.TOTP(secret)
            is_valid = totp.verify(token)
            
            if is_valid:
                # Save 2FA secret to user (should be in database)
                user = await self.auth_service.get_user_by_email(user_email)
                if user:
                    user.totp_secret = secret
                    user.is_2fa_enabled = True
                    await self.auth_service.update_user(user)
                
                # Clean up temporary secret
                del self.temp_2fa_secrets[user_email]
            
            return is_valid
            
        except Exception as e:
            logger.error(f"Failed to verify 2FA setup for {user_email}: {e}")
            return False
    
    async def verify_2fa_token(self, user_email: str, token: str) -> bool:
        """Verify 2FA token for login"""
        try:
            user = await self.auth_service.get_user_by_email(user_email)
            if not user or not user.is_2fa_enabled or not user.totp_secret:
                return False
            
            totp = pyotp.TOTP(user.totp_secret)
            return totp.verify(token)
            
        except Exception as e:
            logger.error(f"Failed to verify 2FA token for {user_email}: {e}")
            return False
    
    async def disable_2fa(self, user_email: str) -> bool:
        """Disable 2FA for user"""
        try:
            user = await self.auth_service.get_user_by_email(user_email)
            if user:
                user.is_2fa_enabled = False
                user.totp_secret = None
                await self.auth_service.update_user(user)
                return True
            return False
            
        except Exception as e:
            logger.error(f"Failed to disable 2FA for {user_email}: {e}")
            return False
    
    # Refresh Token Management
    async def _create_refresh_token(self, user_email: str) -> str:
        """Create refresh token for user"""
        refresh_token = secrets.token_urlsafe(32)
        expires_at = datetime.utcnow() + timedelta(days=30)  # 30 days
        
        self.refresh_tokens[refresh_token] = {
            'user_email': user_email,
            'expires_at': expires_at,
            'created_at': datetime.utcnow()
        }
        
        return refresh_token
    
    async def refresh_access_token(self, refresh_token: str) -> Dict[str, str]:
        """Refresh access token using refresh token"""
        token_data = self.refresh_tokens.get(refresh_token)
        if not token_data:
            raise ValueError("Invalid refresh token")
        
        if datetime.utcnow() > token_data['expires_at']:
            del self.refresh_tokens[refresh_token]
            raise ValueError("Refresh token expired")
        
        # Create new access token
        access_token = await self.auth_service.create_access_token(
            data={"sub": token_data['user_email']}
        )
        
        # Optionally create new refresh token (refresh token rotation)
        new_refresh_token = await self._create_refresh_token(token_data['user_email'])
        del self.refresh_tokens[refresh_token]  # Invalidate old refresh token
        
        return {
            'access_token': access_token,
            'refresh_token': new_refresh_token,
            'token_type': 'bearer'
        }
    
    async def revoke_refresh_token(self, refresh_token: str) -> bool:
        """Revoke refresh token"""
        if refresh_token in self.refresh_tokens:
            del self.refresh_tokens[refresh_token]
            return True
        return False
    
    # Account Lockout Protection
    async def record_failed_login(self, identifier: str) -> Dict[str, any]:
        """Record failed login attempt and check if account should be locked"""
        now = datetime.utcnow()
        
        if identifier not in self.failed_attempts:
            self.failed_attempts[identifier] = {
                'attempts': 0,
                'first_attempt': now,
                'last_attempt': now,
                'locked_until': None
            }
        
        attempt_data = self.failed_attempts[identifier]
        
        # Reset counter if enough time has passed
        if (now - attempt_data['last_attempt']).seconds > self.LOCKOUT_SETTINGS['reset_time']:
            attempt_data['attempts'] = 0
            attempt_data['first_attempt'] = now
        
        attempt_data['attempts'] += 1
        attempt_data['last_attempt'] = now
        
        # Check if account should be locked
        if attempt_data['attempts'] >= self.LOCKOUT_SETTINGS['max_attempts']:
            attempt_data['locked_until'] = now + timedelta(seconds=self.LOCKOUT_SETTINGS['lockout_duration'])
            
            return {
                'locked': True,
                'locked_until': attempt_data['locked_until'],
                'attempts': attempt_data['attempts']
            }
        
        return {
            'locked': False,
            'attempts': attempt_data['attempts'],
            'remaining_attempts': self.LOCKOUT_SETTINGS['max_attempts'] - attempt_data['attempts']
        }
    
    async def is_account_locked(self, identifier: str) -> Dict[str, any]:
        """Check if account is currently locked"""
        if identifier not in self.failed_attempts:
            return {'locked': False}
        
        attempt_data = self.failed_attempts[identifier]
        
        if attempt_data.get('locked_until'):
            if datetime.utcnow() < attempt_data['locked_until']:
                return {
                    'locked': True,
                    'locked_until': attempt_data['locked_until'],
                    'attempts': attempt_data['attempts']
                }
            else:
                # Lock expired, reset attempts
                del self.failed_attempts[identifier]
                return {'locked': False}
        
        return {'locked': False}
    
    async def reset_failed_attempts(self, identifier: str) -> bool:
        """Reset failed login attempts for identifier"""
        if identifier in self.failed_attempts:
            del self.failed_attempts[identifier]
            return True
        return False
    
    # Email Verification
    async def send_verification_email(self, user_email: str) -> str:
        """Generate verification code and send email (mock implementation)"""
        verification_code = secrets.token_urlsafe(32)
        expires_at = datetime.utcnow() + timedelta(hours=24)  # 24 hours
        
        self.verification_codes[verification_code] = {
            'email': user_email,
            'expires_at': expires_at,
            'created_at': datetime.utcnow()
        }
        
        # In a real implementation, send email here
        logger.info(f"Verification email sent to {user_email} with code: {verification_code}")
        
        return verification_code
    
    async def verify_email(self, verification_code: str) -> Dict[str, any]:
        """Verify email with verification code"""
        code_data = self.verification_codes.get(verification_code)
        if not code_data:
            return {'verified': False, 'error': 'Invalid verification code'}
        
        if datetime.utcnow() > code_data['expires_at']:
            del self.verification_codes[verification_code]
            return {'verified': False, 'error': 'Verification code expired'}
        
        # Mark user as verified
        user = await self.auth_service.get_user_by_email(code_data['email'])
        if user:
            user.is_verified = True
            user.verified_at = datetime.utcnow()
            await self.auth_service.update_user(user)
        
        # Clean up verification code
        del self.verification_codes[verification_code]
        
        return {
            'verified': True,
            'email': code_data['email'],
            'user': user
        }
    
    def health_check(self) -> Dict[str, any]:
        """Health check for auth enhancer service"""
        health = super().health_check()
        
        # Check OAuth provider configurations
        configured_providers = []
        for provider, config in self.OAUTH_PROVIDERS.items():
            if config['client_id'] and config['client_secret']:
                configured_providers.append(provider)
        
        health.update({
            'oauth_providers_configured': configured_providers,
            'failed_attempts_tracked': len(self.failed_attempts),
            'active_refresh_tokens': len(self.refresh_tokens),
            'pending_verifications': len(self.verification_codes),
            'lockout_settings': self.LOCKOUT_SETTINGS
        })
        
        return health 