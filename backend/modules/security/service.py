"""
Security Service Container - Security Module

Main security service that orchestrates all security components:
- File scanning and validation
- Input sanitization and validation
- Authentication enhancement
- Rate limiting
- Security middleware
- Audit logging and monitoring
"""

import logging
from typing import Dict, List, Optional, Any
from fastapi import Request, Response, HTTPException, UploadFile

from ..base import BaseService
from ..auth.service import AuthService
from modules.security.file_scanner import FileScannerService
from modules.security.input_validator import InputValidatorService
from modules.security.auth_enhancer import AuthEnhancerService
from modules.security.rate_limiter import RateLimiterService
from modules.security.security_middleware import SecurityMiddlewareService


logger = logging.getLogger(__name__)


class SecurityService(BaseService):
    """Main security service orchestrating all security components"""
    
    def __init__(self, auth_service: AuthService, redis_url: str = "redis://localhost:6379"):
        super().__init__("security")
        self.auth_service = auth_service
        
        # Initialize security services
        self.file_scanner = FileScannerService()
        self.input_validator = InputValidatorService()
        self.auth_enhancer = AuthEnhancerService(auth_service)
        self.rate_limiter = RateLimiterService(redis_url)
        self.security_middleware = SecurityMiddlewareService()
        
    async def initialize(self) -> None:
        """Initialize all security services"""
        await super().initialize()
        
        # Initialize all security components
        await self.file_scanner.initialize()
        await self.input_validator.initialize()
        await self.auth_enhancer.initialize()
        await self.rate_limiter.initialize()
        await self.security_middleware.initialize()
        
        logger.info("Security service container initialized with all components")
    
    # File Security
    async def scan_uploaded_file(self, file_content: bytes, filename: str, 
                                user_tier: str = 'free') -> Dict[str, Any]:
        """Comprehensive file security scan"""
        try:
            return await self.file_scanner.scan_file(file_content, filename, user_tier)
        except Exception as e:
            logger.error(f"File scan failed: {e}")
            return {
                'safe': False,
                'issues': [f'File scan error: {str(e)}'],
                'error': True
            }
    
    async def get_file_info(self, file_content: bytes, filename: str) -> Dict[str, Any]:
        """Get detailed file information"""
        return await self.file_scanner.get_file_info(file_content, filename)
    
    # Input Security
    def validate_user_input(self, input_data: Any, validation_type: str = 'general') -> Dict[str, Any]:
        """Validate and sanitize user input"""
        return self.input_validator.validate_input(input_data, validation_type)
    
    def sanitize_filename(self, filename: str) -> str:
        """Sanitize filename for safe storage"""
        return self.input_validator.sanitize_filename(filename)
    
    def validate_email(self, email: str) -> Dict[str, Any]:
        """Validate email address"""
        return self.input_validator.validate_email(email)
    
    def validate_password(self, password: str) -> Dict[str, Any]:
        """Validate password strength"""
        return self.input_validator.validate_password(password)
    
    # Authentication Enhancement
    async def setup_2fa(self, user_email: str) -> Dict[str, str]:
        """Setup 2FA for user"""
        return await self.auth_enhancer.setup_2fa(user_email)
    
    async def verify_2fa_setup(self, user_email: str, token: str) -> bool:
        """Verify 2FA setup"""
        return await self.auth_enhancer.verify_2fa_setup(user_email, token)
    
    async def verify_2fa_token(self, user_email: str, token: str) -> bool:
        """Verify 2FA token for login"""
        return await self.auth_enhancer.verify_2fa_token(user_email, token)
    
    async def get_oauth_authorization_url(self, provider: str, redirect_uri: str) -> Dict[str, str]:
        """Get OAuth2 authorization URL"""
        return await self.auth_enhancer.get_oauth_authorization_url(provider, redirect_uri)
    
    async def handle_oauth_callback(self, provider: str, code: str, state: str, 
                                   redirect_uri: str) -> Dict[str, Any]:
        """Handle OAuth2 callback"""
        return await self.auth_enhancer.handle_oauth_callback(provider, code, state, redirect_uri)
    
    async def refresh_access_token(self, refresh_token: str) -> Dict[str, str]:
        """Refresh access token"""
        return await self.auth_enhancer.refresh_access_token(refresh_token)
    
    async def record_failed_login(self, identifier: str) -> Dict[str, Any]:
        """Record failed login attempt"""
        return await self.auth_enhancer.record_failed_login(identifier)
    
    async def is_account_locked(self, identifier: str) -> Dict[str, Any]:
        """Check if account is locked"""
        return await self.auth_enhancer.is_account_locked(identifier)
    
    # Rate Limiting
    async def check_rate_limit(self, identifier: str, limit_type: str, 
                              user_tier: str = 'free') -> Dict[str, Any]:
        """Check rate limits"""
        return await self.rate_limiter.check_rate_limit(identifier, limit_type, user_tier)
    
    async def check_daily_limit(self, identifier: str, limit_type: str, 
                               user_tier: str = 'free') -> Dict[str, Any]:
        """Check daily limits"""
        return await self.rate_limiter.check_daily_limit(identifier, limit_type, user_tier)
    
    async def get_usage_stats(self, identifier: str, hours: int = 24) -> Dict[str, Any]:
        """Get usage statistics"""
        return await self.rate_limiter.get_usage_stats(identifier, hours)
    
    async def check_burst_protection(self, identifier: str) -> bool:
        """Check burst protection"""
        return await self.rate_limiter.check_burst_protection(identifier)
    
    # Security Middleware
    def get_security_middleware(self):
        """Get security middleware for FastAPI"""
        return self.security_middleware.get_security_middleware()
    
    def get_cors_middleware(self):
        """Get CORS middleware for FastAPI"""
        return self.security_middleware.get_cors_middleware()
    
    async def validate_request(self, request: Request):
        """Validate incoming request"""
        return await self.security_middleware.validate_request(request)
    
    def add_security_headers(self, response: Response) -> Response:
        """Add security headers to response"""
        return self.security_middleware.add_security_headers(response)
    
    # IP and Host Management
    def block_ip(self, ip_address: str, reason: str = "Manual block") -> bool:
        """Block IP address"""
        # Block in both rate limiter and middleware
        rate_limiter_result = self.rate_limiter.block_ip(ip_address, 3600, reason)
        middleware_result = self.security_middleware.block_ip(ip_address)
        return rate_limiter_result and middleware_result
    
    def unblock_ip(self, ip_address: str) -> bool:
        """Unblock IP address"""
        rate_limiter_result = self.rate_limiter.unblock_ip(ip_address)
        middleware_result = self.security_middleware.unblock_ip(ip_address)
        return rate_limiter_result and middleware_result
    
    def is_ip_blocked(self, ip_address: str) -> bool:
        """Check if IP is blocked"""
        return (self.rate_limiter.is_ip_blocked(ip_address)['blocked'] or 
                self.security_middleware.is_ip_blocked(ip_address))
    
    def add_trusted_host(self, host: str) -> bool:
        """Add trusted host"""
        return self.security_middleware.add_trusted_host(host)
    
    def remove_trusted_host(self, host: str) -> bool:
        """Remove trusted host"""
        return self.security_middleware.remove_trusted_host(host)
    
    # Security Analytics and Monitoring
    def get_security_events(self, event_type: str = None, hours: int = 24) -> List[Dict]:
        """Get security events"""
        return self.security_middleware.get_security_events(event_type, hours)
    
    def get_security_stats(self, hours: int = 24) -> Dict[str, Any]:
        """Get comprehensive security statistics"""
        middleware_stats = self.security_middleware.get_security_stats(hours)
        
        # Combine with other security stats
        combined_stats = {
            'middleware': middleware_stats,
            'rate_limiting': {
                'tier_limits': self.rate_limiter.TIER_LIMITS,
                'redis_connected': self.rate_limiter.redis_client is not None
            },
            'file_scanning': {
                'allowed_extensions': list(self.file_scanner.ALLOWED_EXTENSIONS),
                'max_file_sizes': self.file_scanner.MAX_FILE_SIZES,
                'clamav_available': self.file_scanner.clamav_client is not None
            },
            'auth_enhancement': {
                'oauth_providers': list(self.auth_enhancer.OAUTH_PROVIDERS.keys()),
                'lockout_settings': self.auth_enhancer.LOCKOUT_SETTINGS
            }
        }
        
        return combined_stats
    
    # Comprehensive Security Check
    async def comprehensive_security_check(self, request: Request, 
                                         file_data: Optional[bytes] = None,
                                         filename: Optional[str] = None,
                                         user_tier: str = 'free') -> Dict[str, Any]:
        """Perform comprehensive security check on request"""
        results = {
            'passed': True,
            'issues': [],
            'warnings': [],
            'request_validation': None,
            'file_scan': None,
            'rate_limit_check': None
        }
        
        try:
            # 1. Request validation
            request_validation = await self.validate_request(request)
            if request_validation:
                results['passed'] = False
                results['issues'].append('Request validation failed')
                results['request_validation'] = 'failed'
            else:
                results['request_validation'] = 'passed'
            
            # 2. File scanning (if file provided)
            if file_data and filename:
                file_scan = await self.scan_uploaded_file(file_data, filename, user_tier)
                results['file_scan'] = file_scan
                if not file_scan['safe']:
                    results['passed'] = False
                    results['issues'].extend(file_scan['issues'])
            
            # 3. Rate limiting check
            user_email = getattr(request.state, 'user_email', None)
            identifier = user_email or request.client.host if request.client else 'unknown'
            
            rate_check = await self.check_rate_limit(identifier, 'api_calls', user_tier)
            results['rate_limit_check'] = rate_check
            if not rate_check['allowed']:
                results['passed'] = False
                results['issues'].append('Rate limit exceeded')
            
            # 4. Burst protection
            if not await self.check_burst_protection(identifier):
                results['passed'] = False
                results['issues'].append('Burst protection triggered')
            
            # 5. IP blocking check
            client_ip = request.client.host if request.client else 'unknown'
            if self.is_ip_blocked(client_ip):
                results['passed'] = False
                results['issues'].append('IP address is blocked')
            
        except Exception as e:
            logger.error(f"Comprehensive security check failed: {e}")
            results['passed'] = False
            results['issues'].append(f'Security check error: {str(e)}')
        
        return results
    
    # Health Check
    def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check for all security services"""
        health = super().health_check()
        
        # Get health from all components
        health.update({
            'file_scanner': self.file_scanner.health_check(),
            'input_validator': self.input_validator.health_check(),
            'auth_enhancer': self.auth_enhancer.health_check(),
            'rate_limiter': self.rate_limiter.health_check(),
            'security_middleware': self.security_middleware.health_check()
        })
        
        # Overall security status
        all_healthy = all(
            component.get('status') == 'healthy' 
            for component in health.values() 
            if isinstance(component, dict) and 'status' in component
        )
        
        health['overall_security_status'] = 'healthy' if all_healthy else 'degraded'
        
        return health
    
    # Cleanup and Maintenance
    async def cleanup_old_data(self, hours: int = 24) -> Dict[str, int]:
        """Clean up old security data"""
        results = {}
        
        try:
            # Clean up rate limiting data
            rate_limit_cleaned = await self.rate_limiter.cleanup_old_entries(hours)
            results['rate_limit_entries_cleaned'] = rate_limit_cleaned
            
            # Clean up security events (keep last 1000)
            if len(self.security_middleware.audit_logs) > 1000:
                old_count = len(self.security_middleware.audit_logs)
                self.security_middleware.audit_logs = self.security_middleware.audit_logs[-1000:]
                results['audit_logs_cleaned'] = old_count - len(self.security_middleware.audit_logs)
            else:
                results['audit_logs_cleaned'] = 0
            
            logger.info(f"Security cleanup completed: {results}")
            
        except Exception as e:
            logger.error(f"Security cleanup failed: {e}")
            results['error'] = str(e)
        
        return results 