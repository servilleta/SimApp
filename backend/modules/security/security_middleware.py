"""
Security Middleware Service - Security Module

Provides comprehensive security middleware including:
- Security headers (HSTS, CSP, X-Frame-Options, etc.)
- CORS configuration
- Request ID tracking
- Audit logging
- Trusted host validation
- IP filtering
- Request/response sanitization
"""

import logging
import uuid
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Callable
from fastapi import Request, Response, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import JSONResponse
import secure

from ..base import BaseService


logger = logging.getLogger(__name__)


class SecurityMiddlewareService(BaseService):
    """Security middleware and headers service"""
    
    # Security headers configuration
    SECURITY_HEADERS = {
        'X-Content-Type-Options': 'nosniff',
        'X-Frame-Options': 'DENY',
        'X-XSS-Protection': '1; mode=block',
        'Referrer-Policy': 'strict-origin-when-cross-origin',
        'Permissions-Policy': 'geolocation=(), microphone=(), camera=()',
        'Cross-Origin-Embedder-Policy': 'require-corp',
        'Cross-Origin-Opener-Policy': 'same-origin',
        'Cross-Origin-Resource-Policy': 'cross-origin'
    }
    
    # Secure Content Security Policy - removed unsafe directives
    CSP_POLICY = {
        'default-src': ["'self'"],
        'script-src': ["'self'", "https://cdn.jsdelivr.net"],  # Removed unsafe-inline and unsafe-eval
        'style-src': ["'self'", "https://fonts.googleapis.com"],  # Removed unsafe-inline - use hashes instead
        'font-src': ["'self'", "https://fonts.gstatic.com"],
        'img-src': ["'self'", "data:", "https:"],
        'connect-src': ["'self'", "https://api.stripe.com", "https://accounts.google.com"],
        'frame-src': ["'none'"],
        'frame-ancestors': ["'none'"],
        'object-src': ["'none'"],
        'base-uri': ["'self'"],
        'form-action': ["'self'"],
        'upgrade-insecure-requests': []
    }
    
    # CORS settings
    CORS_SETTINGS = {
        'allow_origins': ["http://localhost:3000", "http://localhost:80", "https://yourdomain.com"],
        'allow_credentials': True,
        'allow_methods': ["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
        'allow_headers': [
            "Accept", "Accept-Language", "Content-Language", "Content-Type",
            "Authorization", "X-Requested-With", "X-Request-ID", "X-API-Key"
        ],
        'expose_headers': [
            "X-Request-ID", "X-RateLimit-Limit", "X-RateLimit-Remaining", 
            "X-RateLimit-Reset", "X-Total-Count"
        ]
    }
    
    def __init__(self, trusted_hosts: List[str] = None, blocked_ips: Set[str] = None):
        super().__init__("security_middleware")
        self.trusted_hosts = set(trusted_hosts or ["localhost", "127.0.0.1"])
        self.blocked_ips = blocked_ips or set()
        self.audit_logs = []  # In-memory audit log (should use database in production)
        
        # Initialize secure headers
        self.secure_headers = secure.Secure(
            hsts=secure.StrictTransportSecurity().include_subdomains().preload().max_age(31536000),
            csp=self._build_csp_header(),
            referrer=secure.ReferrerPolicy().strict_origin_when_cross_origin(),
            permissions=secure.PermissionsPolicy().geolocation("none").microphone("none").camera("none"),
            cache=secure.CacheControl().no_cache().no_store().must_revalidate(),
            xcto=secure.XContentTypeOptions().nosniff(),
            xfo=secure.XFrameOptions().deny()
        )
    
    async def initialize(self) -> None:
        """Initialize the security middleware service"""
        await super().initialize()
        logger.info("Security middleware service initialized")
    
    def _build_csp_header(self) -> secure.ContentSecurityPolicy:
        """Build Content Security Policy header"""
        csp = secure.ContentSecurityPolicy()
        
        for directive, sources in self.CSP_POLICY.items():
            method_name = directive.replace('-', '_')
            if hasattr(csp, method_name):
                method = getattr(csp, method_name)
                method(*sources)
        
        return csp
    
    def get_security_middleware(self) -> BaseHTTPMiddleware:
        """Get security middleware for FastAPI"""
        return SecurityHeadersMiddleware(self)
    
    def get_cors_middleware(self) -> CORSMiddleware:
        """Get CORS middleware for FastAPI"""
        return CORSMiddleware(**self.CORS_SETTINGS)
    
    async def validate_request(self, request: Request) -> Optional[JSONResponse]:
        """Validate incoming request for security issues"""
        try:
            # 1. Check blocked IPs
            client_ip = self._get_client_ip(request)
            if client_ip in self.blocked_ips:
                await self._log_security_event(request, "blocked_ip", {"ip": client_ip})
                return JSONResponse(
                    status_code=403,
                    content={"error": "Access denied", "code": "IP_BLOCKED"}
                )
            
            # 2. Validate trusted hosts
            if not self._is_trusted_host(request):
                await self._log_security_event(request, "untrusted_host", {"host": request.headers.get("host")})
                return JSONResponse(
                    status_code=400,
                    content={"error": "Invalid host header", "code": "UNTRUSTED_HOST"}
                )
            
            # 3. Check for suspicious patterns in headers
            suspicious_headers = await self._check_suspicious_headers(request)
            if suspicious_headers:
                await self._log_security_event(request, "suspicious_headers", {"headers": suspicious_headers})
                # Log but don't block (might be false positive)
            
            # 4. Validate request size
            if await self._is_request_too_large(request):
                await self._log_security_event(request, "request_too_large", {"size": request.headers.get("content-length")})
                return JSONResponse(
                    status_code=413,
                    content={"error": "Request too large", "code": "REQUEST_TOO_LARGE"}
                )
            
            return None  # Request is valid
            
        except Exception as e:
            logger.error(f"Request validation failed: {e}")
            return None  # Allow request on validation error
    
    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address from request"""
        # Check for forwarded headers (reverse proxy)
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        # Fall back to direct connection
        return request.client.host if request.client else "unknown"
    
    def _is_trusted_host(self, request: Request) -> bool:
        """Check if request host is trusted"""
        host = request.headers.get("host", "").lower()
        
        # Remove port if present
        if ":" in host:
            host = host.split(":")[0]
        
        return host in self.trusted_hosts
    
    async def _check_suspicious_headers(self, request: Request) -> List[str]:
        """Check for suspicious patterns in request headers"""
        suspicious = []
        
        # Check User-Agent
        user_agent = request.headers.get("user-agent", "").lower()
        suspicious_ua_patterns = [
            "sqlmap", "nikto", "nmap", "masscan", "nessus", "openvas",
            "burpsuite", "owasp", "w3af", "skipfish", "grabber"
        ]
        
        for pattern in suspicious_ua_patterns:
            if pattern in user_agent:
                suspicious.append(f"suspicious_user_agent: {pattern}")
        
        # Check for unusual headers
        unusual_headers = [
            "x-originating-ip", "x-forwarded-host", "x-remote-ip",
            "x-remote-addr", "x-cluster-client-ip"
        ]
        
        for header in unusual_headers:
            if header in request.headers:
                suspicious.append(f"unusual_header: {header}")
        
        return suspicious
    
    async def _is_request_too_large(self, request: Request, max_size: int = 100 * 1024 * 1024) -> bool:
        """Check if request is too large (100MB default)"""
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                return int(content_length) > max_size
            except ValueError:
                return False
        return False
    
    async def _log_security_event(self, request: Request, event_type: str, details: Dict) -> None:
        """Log security event for audit trail"""
        event = {
            'timestamp': datetime.utcnow().isoformat(),
            'request_id': getattr(request.state, 'request_id', 'unknown'),
            'event_type': event_type,
            'client_ip': self._get_client_ip(request),
            'user_agent': request.headers.get('user-agent', 'unknown'),
            'path': str(request.url.path),
            'method': request.method,
            'details': details
        }
        
        self.audit_logs.append(event)
        logger.warning(f"Security event: {event_type} - {details}")
        
        # Keep only last 1000 events in memory
        if len(self.audit_logs) > 1000:
            self.audit_logs = self.audit_logs[-1000:]
    
    def add_security_headers(self, response: Response) -> Response:
        """Add security headers to response"""
        try:
            # Add static security headers
            for header, value in self.SECURITY_HEADERS.items():
                response.headers[header] = value
            
            # Add secure headers using secure library
            secure_headers = self.secure_headers.headers()
            for header, value in secure_headers.items():
                response.headers[header] = value
            
            # Add HSTS for HTTPS
            if response.headers.get("X-Forwarded-Proto") == "https":
                response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains; preload"
            
        except Exception as e:
            logger.error(f"Failed to add security headers: {e}")
        
        return response
    
    def add_request_id(self, request: Request) -> str:
        """Add unique request ID to request"""
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        return request_id
    
    async def log_request(self, request: Request, response: Response, 
                         processing_time: float) -> None:
        """Log request for audit trail"""
        try:
            log_entry = {
                'timestamp': datetime.utcnow().isoformat(),
                'request_id': getattr(request.state, 'request_id', 'unknown'),
                'method': request.method,
                'path': str(request.url.path),
                'query_params': dict(request.query_params),
                'client_ip': self._get_client_ip(request),
                'user_agent': request.headers.get('user-agent', 'unknown'),
                'status_code': response.status_code,
                'processing_time_ms': round(processing_time * 1000, 2),
                'response_size': response.headers.get('content-length', 'unknown'),
                'user_email': getattr(request.state, 'user_email', None)
            }
            
            # Log to structured logger
            logger.info(f"Request processed", extra=log_entry)
            
        except Exception as e:
            logger.error(f"Failed to log request: {e}")
    
    # IP Management
    def block_ip(self, ip_address: str) -> bool:
        """Block IP address"""
        try:
            self.blocked_ips.add(ip_address)
            logger.warning(f"IP {ip_address} blocked")
            return True
        except Exception as e:
            logger.error(f"Failed to block IP {ip_address}: {e}")
            return False
    
    def unblock_ip(self, ip_address: str) -> bool:
        """Unblock IP address"""
        try:
            self.blocked_ips.discard(ip_address)
            logger.info(f"IP {ip_address} unblocked")
            return True
        except Exception as e:
            logger.error(f"Failed to unblock IP {ip_address}: {e}")
            return False
    
    def is_ip_blocked(self, ip_address: str) -> bool:
        """Check if IP is blocked"""
        return ip_address in self.blocked_ips
    
    # Host Management
    def add_trusted_host(self, host: str) -> bool:
        """Add trusted host"""
        try:
            self.trusted_hosts.add(host.lower())
            logger.info(f"Host {host} added to trusted hosts")
            return True
        except Exception as e:
            logger.error(f"Failed to add trusted host {host}: {e}")
            return False
    
    def remove_trusted_host(self, host: str) -> bool:
        """Remove trusted host"""
        try:
            self.trusted_hosts.discard(host.lower())
            logger.info(f"Host {host} removed from trusted hosts")
            return True
        except Exception as e:
            logger.error(f"Failed to remove trusted host {host}: {e}")
            return False
    
    # Audit and Analytics
    def get_security_events(self, event_type: str = None, hours: int = 24) -> List[Dict]:
        """Get security events from audit log"""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        filtered_events = []
        for event in self.audit_logs:
            event_time = datetime.fromisoformat(event['timestamp'])
            if event_time >= cutoff_time:
                if event_type is None or event['event_type'] == event_type:
                    filtered_events.append(event)
        
        return filtered_events
    
    def get_security_stats(self, hours: int = 24) -> Dict[str, any]:
        """Get security statistics"""
        events = self.get_security_events(hours=hours)
        
        stats = {
            'total_events': len(events),
            'event_types': {},
            'top_ips': {},
            'blocked_ips_count': len(self.blocked_ips),
            'trusted_hosts_count': len(self.trusted_hosts)
        }
        
        for event in events:
            # Count event types
            event_type = event['event_type']
            stats['event_types'][event_type] = stats['event_types'].get(event_type, 0) + 1
            
            # Count IPs
            client_ip = event['client_ip']
            stats['top_ips'][client_ip] = stats['top_ips'].get(client_ip, 0) + 1
        
        # Sort top IPs
        stats['top_ips'] = dict(sorted(stats['top_ips'].items(), key=lambda x: x[1], reverse=True)[:10])
        
        return stats
    
    def health_check(self) -> Dict[str, any]:
        """Health check for security middleware service"""
        health = super().health_check()
        
        health.update({
            'blocked_ips_count': len(self.blocked_ips),
            'trusted_hosts_count': len(self.trusted_hosts),
            'audit_logs_count': len(self.audit_logs),
            'security_headers_count': len(self.SECURITY_HEADERS),
            'cors_configured': bool(self.CORS_SETTINGS)
        })
        
        return health


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware for security headers and request validation"""
    
    def __init__(self, security_service: SecurityMiddlewareService):
        super().__init__()
        self.security_service = security_service
    
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """Process request through security middleware"""
        start_time = time.time()
        
        # Add request ID
        request_id = self.security_service.add_request_id(request)
        
        # Validate request
        validation_response = await self.security_service.validate_request(request)
        if validation_response:
            # Add request ID to error response
            validation_response.headers["X-Request-ID"] = request_id
            return validation_response
        
        # Process request
        try:
            response = await call_next(request)
        except Exception as e:
            logger.error(f"Request processing failed: {e}")
            error_response = JSONResponse(
                status_code=500,
                content={"error": "Internal server error", "request_id": request_id}
            )
            error_response.headers["X-Request-ID"] = request_id
            return error_response
        
        # Add security headers
        response = self.security_service.add_security_headers(response)
        
        # Add request ID to response
        response.headers["X-Request-ID"] = request_id
        
        # Log request
        processing_time = time.time() - start_time
        await self.security_service.log_request(request, response, processing_time)
        
        return response 