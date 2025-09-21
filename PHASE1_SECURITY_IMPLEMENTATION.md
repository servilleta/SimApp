# Phase 1: Security Hardening Implementation Guide

## Week 1-3: Security Implementation

### 1. File Upload Security

#### A. Virus Scanning Integration
```python
# backend/security/file_scanner.py
import pyclamd
import hashlib
import magic
from typing import BinaryIO, Tuple
from fastapi import HTTPException

class FileScanner:
    def __init__(self):
        try:
            # Try to connect to ClamAV daemon
            self.clam = pyclamd.ClamdUnixSocket()
            if not self.clam.ping():
                self.clam = None
                print("Warning: ClamAV not available")
        except:
            self.clam = None
            print("Warning: ClamAV connection failed")
    
    async def scan_file(self, file_content: bytes, filename: str) -> Tuple[bool, str]:
        """Scan file for viruses and malicious content"""
        # Check file magic numbers
        file_type = magic.from_buffer(file_content, mime=True)
        allowed_types = [
            'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            'application/vnd.ms-excel.sheet.macroEnabled.12'
        ]
        
        if file_type not in allowed_types:
            return False, f"Invalid file type: {file_type}"
        
        # Check file size
        if len(file_content) > 10 * 1024 * 1024:  # 10MB
            return False, "File too large (max 10MB)"
        
        # Virus scan if ClamAV available
        if self.clam:
            scan_result = self.clam.scan_stream(file_content)
            if scan_result:
                status = scan_result['stream'][0]
                if status != 'OK':
                    return False, f"Virus detected: {scan_result['stream'][1]}"
        
        # Calculate file hash for tracking
        file_hash = hashlib.sha256(file_content).hexdigest()
        
        return True, file_hash

# backend/security/excel_validator.py
import openpyxl
import re
from typing import Dict, List

class ExcelSecurityValidator:
    # Dangerous Excel functions that could be used for attacks
    DANGEROUS_FUNCTIONS = [
        'WEBSERVICE', 'FILTERXML', 'WORKDAY.INTL', 
        'NETWORKDAYS.INTL', 'ENCODEURL', 'HYPERLINK',
        'IMPORTDATA', 'IMPORTXML', 'IMPORTHTML',
        'GETPIVOTDATA', 'CUBEMEMBER', 'CUBEVALUE'
    ]
    
    # Suspicious patterns
    SUSPICIOUS_PATTERNS = [
        r'cmd\.exe', r'powershell', r'bash',
        r'<script', r'javascript:', r'vbscript:',
        r'=cmd\|', r'=powershell\|'
    ]
    
    @staticmethod
    def validate_excel_file(file_path: str) -> Dict[str, any]:
        """Deep validation of Excel file for security threats"""
        issues = []
        
        try:
            wb = openpyxl.load_workbook(file_path, data_only=False)
            
            for sheet_name in wb.sheetnames:
                ws = wb[sheet_name]
                
                # Check for suspicious formulas
                for row in ws.iter_rows():
                    for cell in row:
                        if cell.value and isinstance(cell.value, str):
                            # Check for dangerous functions
                            for func in ExcelSecurityValidator.DANGEROUS_FUNCTIONS:
                                if func.upper() in str(cell.value).upper():
                                    issues.append({
                                        'type': 'dangerous_function',
                                        'cell': f"{sheet_name}!{cell.coordinate}",
                                        'function': func,
                                        'value': cell.value[:100]
                                    })
                            
                            # Check for suspicious patterns
                            for pattern in ExcelSecurityValidator.SUSPICIOUS_PATTERNS:
                                if re.search(pattern, str(cell.value), re.IGNORECASE):
                                    issues.append({
                                        'type': 'suspicious_pattern',
                                        'cell': f"{sheet_name}!{cell.coordinate}",
                                        'pattern': pattern,
                                        'value': cell.value[:100]
                                    })
                
                # Check for hidden sheets
                if ws.sheet_state == 'hidden':
                    issues.append({
                        'type': 'hidden_sheet',
                        'sheet': sheet_name
                    })
            
            # Check for external links
            if wb.external_links:
                for link in wb.external_links:
                    issues.append({
                        'type': 'external_link',
                        'link': str(link)
                    })
            
            return {
                'valid': len(issues) == 0,
                'issues': issues
            }
            
        except Exception as e:
            return {
                'valid': False,
                'error': str(e)
            }
```

#### B. Secure File Upload Endpoint
```python
# backend/excel_parser/secure_upload.py
from fastapi import UploadFile, File, HTTPException, Depends
from sqlalchemy.orm import Session
import aiofiles
import os
import uuid

async def secure_file_upload(
    file: UploadFile = File(...),
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Secure file upload with multiple validation layers"""
    
    # Check user quota
    user_uploads_today = db.query(FileUpload).filter(
        FileUpload.user_id == current_user.id,
        FileUpload.created_at >= datetime.utcnow().date()
    ).count()
    
    if user_uploads_today >= 10:  # Daily limit
        raise HTTPException(429, "Daily upload limit exceeded")
    
    # Read file content
    content = await file.read()
    
    # Security scanning
    scanner = FileScanner()
    is_safe, file_hash = await scanner.scan_file(content, file.filename)
    
    if not is_safe:
        raise HTTPException(400, f"File rejected: {file_hash}")
    
    # Check for duplicate uploads
    existing = db.query(FileUpload).filter(
        FileUpload.file_hash == file_hash,
        FileUpload.user_id == current_user.id
    ).first()
    
    if existing:
        return {"file_id": existing.id, "duplicate": True}
    
    # Save to secure location
    file_id = str(uuid.uuid4())
    safe_filename = f"{file_id}.xlsx"
    upload_path = f"/secure-uploads/{current_user.id}/{safe_filename}"
    
    os.makedirs(os.path.dirname(upload_path), exist_ok=True)
    
    async with aiofiles.open(upload_path, 'wb') as f:
        await f.write(content)
    
    # Validate Excel content
    validator = ExcelSecurityValidator()
    validation_result = validator.validate_excel_file(upload_path)
    
    if not validation_result['valid']:
        os.unlink(upload_path)
        raise HTTPException(400, f"Excel validation failed: {validation_result}")
    
    # Record upload
    upload_record = FileUpload(
        id=file_id,
        user_id=current_user.id,
        filename=file.filename[:255],  # Truncate long names
        file_hash=file_hash,
        file_size=len(content),
        upload_path=upload_path,
        validation_status=validation_result
    )
    db.add(upload_record)
    db.commit()
    
    return {
        "file_id": file_id,
        "filename": file.filename,
        "size": len(content),
        "validation": validation_result
    }
```

### 2. Authentication Enhancement

#### A. Add OAuth2 Providers
```python
# backend/auth/oauth.py
from authlib.integrations.starlette_client import OAuth
from fastapi import Request, HTTPException
import os

oauth = OAuth()

# Configure OAuth providers
oauth.register(
    name='google',
    client_id=os.getenv('GOOGLE_CLIENT_ID'),
    client_secret=os.getenv('GOOGLE_CLIENT_SECRET'),
    server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
    client_kwargs={'scope': 'openid email profile'}
)

oauth.register(
    name='microsoft',
    client_id=os.getenv('MICROSOFT_CLIENT_ID'),
    client_secret=os.getenv('MICROSOFT_CLIENT_SECRET'),
    authorize_url='https://login.microsoftonline.com/common/oauth2/v2.0/authorize',
    token_url='https://login.microsoftonline.com/common/oauth2/v2.0/token',
    client_kwargs={'scope': 'openid email profile'}
)

oauth.register(
    name='github',
    client_id=os.getenv('GITHUB_CLIENT_ID'),
    client_secret=os.getenv('GITHUB_CLIENT_SECRET'),
    access_token_url='https://github.com/login/oauth/access_token',
    authorize_url='https://github.com/login/oauth/authorize',
    api_base_url='https://api.github.com/',
    client_kwargs={'scope': 'user:email'}
)

# OAuth login endpoints
@app.get('/api/auth/login/{provider}')
async def oauth_login(provider: str, request: Request):
    client = oauth.create_client(provider)
    if not client:
        raise HTTPException(404, f"Provider {provider} not found")
    
    redirect_uri = request.url_for('oauth_callback', provider=provider)
    return await client.authorize_redirect(request, redirect_uri)

@app.get('/api/auth/callback/{provider}')
async def oauth_callback(provider: str, request: Request, db: Session = Depends(get_db)):
    client = oauth.create_client(provider)
    if not client:
        raise HTTPException(404, f"Provider {provider} not found")
    
    token = await client.authorize_access_token(request)
    
    # Get user info
    if provider == 'github':
        resp = await client.get('user', token=token)
        user_data = resp.json()
        email = user_data.get('email')
        if not email:
            # GitHub may not provide email, fetch separately
            resp = await client.get('user/emails', token=token)
            emails = resp.json()
            email = next((e['email'] for e in emails if e['primary']), None)
    else:
        # Google and Microsoft use OIDC
        user_data = token.get('userinfo')
        email = user_data.get('email')
    
    if not email:
        raise HTTPException(400, "Email not provided by OAuth provider")
    
    # Find or create user
    user = db.query(User).filter(User.email == email).first()
    if not user:
        user = User(
            email=email,
            full_name=user_data.get('name', ''),
            oauth_provider=provider,
            email_verified=True,
            created_at=datetime.utcnow()
        )
        db.add(user)
        db.commit()
    
    # Create JWT token
    access_token = create_access_token(data={"sub": user.email})
    
    # Redirect to frontend with token
    frontend_url = os.getenv('FRONTEND_URL', 'http://localhost')
    return RedirectResponse(f"{frontend_url}/auth/callback?token={access_token}")
```

#### B. Two-Factor Authentication
```python
# backend/auth/two_factor.py
import pyotp
import qrcode
import io
import base64

class TwoFactorAuth:
    @staticmethod
    def generate_secret():
        """Generate a new TOTP secret"""
        return pyotp.random_base32()
    
    @staticmethod
    def generate_qr_code(email: str, secret: str) -> str:
        """Generate QR code for authenticator app"""
        totp_uri = pyotp.totp.TOTP(secret).provisioning_uri(
            name=email,
            issuer_name='Monte Carlo Simulator'
        )
        
        qr = qrcode.QRCode(version=1, box_size=10, border=5)
        qr.add_data(totp_uri)
        qr.make(fit=True)
        
        img = qr.make_image(fill_color="black", back_color="white")
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        
        return base64.b64encode(buffer.getvalue()).decode()
    
    @staticmethod
    def verify_token(secret: str, token: str) -> bool:
        """Verify TOTP token"""
        totp = pyotp.TOTP(secret)
        return totp.verify(token, valid_window=1)

# Add to user routes
@app.post('/api/auth/2fa/setup')
async def setup_2fa(
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Setup 2FA for user"""
    if current_user.two_factor_enabled:
        raise HTTPException(400, "2FA already enabled")
    
    secret = TwoFactorAuth.generate_secret()
    qr_code = TwoFactorAuth.generate_qr_code(current_user.email, secret)
    
    # Store secret temporarily (not enabled yet)
    current_user.two_factor_secret_temp = secret
    db.commit()
    
    return {
        "qr_code": f"data:image/png;base64,{qr_code}",
        "secret": secret
    }

@app.post('/api/auth/2fa/verify')
async def verify_2fa(
    token: str,
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Verify and enable 2FA"""
    if not current_user.two_factor_secret_temp:
        raise HTTPException(400, "2FA setup not initiated")
    
    if TwoFactorAuth.verify_token(current_user.two_factor_secret_temp, token):
        current_user.two_factor_secret = current_user.two_factor_secret_temp
        current_user.two_factor_secret_temp = None
        current_user.two_factor_enabled = True
        db.commit()
        
        return {"status": "2FA enabled successfully"}
    else:
        raise HTTPException(400, "Invalid token")
```

### 3. Rate Limiting Implementation

```python
# backend/security/rate_limiting.py
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi import Request, Response
from typing import Optional
import redis
import json

# Custom key function that considers user authentication
def get_rate_limit_key(request: Request) -> str:
    # Try to get user from JWT
    try:
        token = request.headers.get("Authorization", "").replace("Bearer ", "")
        if token:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            user_email = payload.get("sub")
            if user_email:
                return f"user:{user_email}"
    except:
        pass
    
    # Fall back to IP address
    return get_remote_address(request)

# Initialize limiter
limiter = Limiter(key_func=get_rate_limit_key)

# Custom rate limit storage with Redis
class RedisRateLimitStorage:
    def __init__(self, redis_client):
        self.redis = redis_client
    
    async def get_usage(self, key: str, limit: str) -> dict:
        """Get current usage for a key"""
        data = await self.redis.get(f"rate_limit:{key}:{limit}")
        if data:
            return json.loads(data)
        return {"count": 0, "window_start": time.time()}
    
    async def increment_usage(self, key: str, limit: str, window: int):
        """Increment usage counter"""
        current = await self.get_usage(key, limit)
        
        now = time.time()
        if now - current["window_start"] > window:
            # Reset window
            current = {"count": 1, "window_start": now}
        else:
            current["count"] += 1
        
        await self.redis.setex(
            f"rate_limit:{key}:{limit}",
            window,
            json.dumps(current)
        )
        
        return current["count"]

# Apply rate limits to endpoints
@app.post("/api/simulations/run")
@limiter.limit("10/minute")  # Free tier
async def run_simulation(request: Request):
    # Check if user is paid for higher limits
    user = get_current_user_optional(request)
    if user and user.subscription_tier == "professional":
        # Override with higher limit
        return await run_simulation_pro(request)
    
    # Free tier logic
    pass

# Different limits for different endpoints
RATE_LIMITS = {
    "auth": {
        "login": "5/minute",
        "register": "3/hour",
        "password_reset": "3/hour"
    },
    "api": {
        "upload": "10/hour",
        "simulate": "100/hour",
        "download": "50/hour"
    },
    "admin": {
        "all": "1000/hour"
    }
}

# Middleware to apply dynamic rate limits
@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    path = request.url.path
    method = request.method
    
    # Determine rate limit based on path
    if path.startswith("/api/auth"):
        limit = RATE_LIMITS["auth"].get(path.split("/")[-1], "10/minute")
    elif path.startswith("/api/admin"):
        limit = RATE_LIMITS["admin"]["all"]
    else:
        limit = RATE_LIMITS["api"].get(path.split("/")[-1], "100/hour")
    
    # Check rate limit
    key = get_rate_limit_key(request)
    # ... implement rate limit check ...
    
    response = await call_next(request)
    return response
```

### 4. Security Headers & Middleware

```python
# backend/security/middleware.py
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
import time
import uuid

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Add request ID for tracking
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        # Process request
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        
        # Add security headers
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time"] = str(process_time)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
        
        # Content Security Policy
        csp = [
            "default-src 'self'",
            "script-src 'self' 'unsafe-inline' 'unsafe-eval' https://cdn.jsdelivr.net",
            "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com",
            "font-src 'self' https://fonts.gstatic.com",
            "img-src 'self' data: https:",
            "connect-src 'self' wss: https:",
            "frame-ancestors 'none'",
            "base-uri 'self'",
            "form-action 'self'"
        ]
        response.headers["Content-Security-Policy"] = "; ".join(csp)
        
        return response

# Apply middleware
app.add_middleware(SecurityHeadersMiddleware)

# CORS configuration for production
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://app.yourdomain.com",
        "https://www.yourdomain.com"
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID", "X-Process-Time"]
)

# Trusted host middleware
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=[
        "app.yourdomain.com",
        "api.yourdomain.com",
        "*.yourdomain.com"
    ]
)

# Request size limiting
from fastapi import Request
from fastapi.exceptions import RequestValidationError

@app.exception_handler(RequestValidationError)
async def request_validation_exception_handler(request: Request, exc: RequestValidationError):
    # Check if error is due to request size
    if "Request body too large" in str(exc):
        return JSONResponse(
            status_code=413,
            content={"detail": "Request body too large. Maximum size is 10MB"}
        )
    
    # Default handling
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors()}
    )
```

### 5. Input Validation & Sanitization

```python
# backend/security/validators.py
import re
import html
from typing import Any, Optional
from pydantic import validator, BaseModel

class SecureInputMixin:
    """Mixin for Pydantic models to add security validation"""
    
    @validator('*', pre=True)
    def sanitize_string_inputs(cls, v):
        if isinstance(v, str):
            # Remove null bytes
            v = v.replace('\x00', '')
            
            # Limit length
            if len(v) > 10000:
                raise ValueError("Input too long")
            
            # Basic HTML escaping
            v = html.escape(v)
            
            # Remove potential SQL injection patterns
            sql_patterns = [
                r'(\b(union|select|insert|update|delete|drop|create|alter|exec|execute)\b)',
                r'(--|#|\/\*|\*\/|;)',
                r'(\bor\b\s*\d+\s*=\s*\d+)',
                r'(\band\b\s*\d+\s*=\s*\d+)'
            ]
            
            for pattern in sql_patterns:
                if re.search(pattern, v, re.IGNORECASE):
                    raise ValueError("Potentially malicious input detected")
        
        return v

# Example usage in models
class SimulationRequest(BaseModel, SecureInputMixin):
    name: str
    description: Optional[str]
    target_cell: str
    iterations: int
    
    @validator('target_cell')
    def validate_cell_reference(cls, v):
        # Only allow valid Excel cell references
        if not re.match(r'^[A-Z]{1,3}\d{1,7}$', v):
            raise ValueError("Invalid cell reference")
        return v
    
    @validator('iterations')
    def validate_iterations(cls, v):
        if v < 100 or v > 10000:
            raise ValueError("Iterations must be between 100 and 10,000")
        return v

# File path validation
def validate_safe_path(path: str) -> bool:
    """Validate that a path is safe and doesn't contain traversal attempts"""
    # Normalize path
    path = os.path.normpath(path)
    
    # Check for path traversal
    if '..' in path or path.startswith('/'):
        return False
    
    # Check for suspicious patterns
    suspicious = ['~', '\\', '..', './']
    for pattern in suspicious:
        if pattern in path:
            return False
    
    return True
```

### 6. Audit Logging

```python
# backend/security/audit.py
from datetime import datetime
from typing import Optional, Dict, Any
import json

class AuditLogger:
    def __init__(self, db_session):
        self.db = db_session
    
    async def log_event(
        self,
        event_type: str,
        user_id: Optional[int],
        ip_address: str,
        user_agent: str,
        request_id: str,
        details: Dict[str, Any],
        severity: str = "info"
    ):
        """Log security-relevant events"""
        
        audit_entry = AuditLog(
            event_type=event_type,
            user_id=user_id,
            ip_address=ip_address,
            user_agent=user_agent[:255],
            request_id=request_id,
            details=json.dumps(details),
            severity=severity,
            created_at=datetime.utcnow()
        )
        
        self.db.add(audit_entry)
        await self.db.commit()
        
        # Also log critical events to file
        if severity in ["warning", "error", "critical"]:
            with open("/var/log/montecarlo/security.log", "a") as f:
                f.write(f"{datetime.utcnow().isoformat()} | {severity.upper()} | {event_type} | {json.dumps(details)}\n")

# Middleware to log all requests
@app.middleware("http")
async def audit_middleware(request: Request, call_next):
    # Get request details
    user = get_current_user_optional(request)
    ip_address = get_remote_address(request)
    user_agent = request.headers.get("User-Agent", "")
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    
    # Log request
    audit = AuditLogger(next(get_db()))
    await audit.log_event(
        event_type="api_request",
        user_id=user.id if user else None,
        ip_address=ip_address,
        user_agent=user_agent,
        request_id=request_id,
        details={
            "method": request.method,
            "path": str(request.url.path),
            "query": str(request.url.query)
        }
    )
    
    # Process request
    response = await call_next(request)
    
    # Log response
    await audit.log_event(
        event_type="api_response",
        user_id=user.id if user else None,
        ip_address=ip_address,
        user_agent=user_agent,
        request_id=request_id,
        details={
            "status_code": response.status_code,
            "process_time": response.headers.get("X-Process-Time", "0")
        }
    )
    
    return response
```

### 7. Database Models for Security

```python
# backend/auth/models.py
from sqlalchemy import Column, Integer, String, Boolean, DateTime, Text, JSON
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=True)  # Null for OAuth users
    full_name = Column(String(255))
    
    # OAuth
    oauth_provider = Column(String(50))  # google, microsoft, github
    oauth_id = Column(String(255))
    
    # 2FA
    two_factor_enabled = Column(Boolean, default=False)
    two_factor_secret = Column(String(255))
    two_factor_secret_temp = Column(String(255))  # During setup
    
    # Security
    email_verified = Column(Boolean, default=False)
    is_active = Column(Boolean, default=True)
    is_superuser = Column(Boolean, default=False)
    failed_login_attempts = Column(Integer, default=0)
    last_failed_login = Column(DateTime)
    locked_until = Column(DateTime)
    
    # Subscription
    subscription_tier = Column(String(50), default="free")  # free, starter, professional, enterprise
    subscription_expires = Column(DateTime)
    
    # Metadata
    created_at = Column(DateTime, nullable=False)
    updated_at = Column(DateTime)
    last_login_at = Column(DateTime)
    last_login_ip = Column(String(45))

class FileUpload(Base):
    __tablename__ = "file_uploads"
    
    id = Column(String(36), primary_key=True)  # UUID
    user_id = Column(Integer, nullable=False, index=True)
    filename = Column(String(255), nullable=False)
    file_hash = Column(String(64), nullable=False, index=True)
    file_size = Column(Integer, nullable=False)
    upload_path = Column(String(500), nullable=False)
    
    # Security validation
    validation_status = Column(JSON)
    virus_scan_result = Column(String(50))
    
    # Metadata
    created_at = Column(DateTime, nullable=False)
    accessed_at = Column(DateTime)
    expires_at = Column(DateTime)

class AuditLog(Base):
    __tablename__ = "audit_logs"
    
    id = Column(Integer, primary_key=True)
    event_type = Column(String(50), nullable=False, index=True)
    user_id = Column(Integer, index=True)
    ip_address = Column(String(45), nullable=False)
    user_agent = Column(String(255))
    request_id = Column(String(36), index=True)
    details = Column(Text)  # JSON
    severity = Column(String(20), default="info")  # info, warning, error, critical
    created_at = Column(DateTime, nullable=False, index=True)

class RateLimitViolation(Base):
    __tablename__ = "rate_limit_violations"
    
    id = Column(Integer, primary_key=True)
    key = Column(String(255), nullable=False, index=True)  # IP or user identifier
    endpoint = Column(String(255), nullable=False)
    limit_exceeded = Column(String(50))  # e.g., "10/minute"
    violation_count = Column(Integer, default=1)
    first_violation = Column(DateTime, nullable=False)
    last_violation = Column(DateTime, nullable=False)
    blocked_until = Column(DateTime)
```

### 8. Environment Configuration

```bash
# .env.production
# Security
SECRET_KEY=your-256-bit-secret-key-here
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30
REFRESH_TOKEN_EXPIRE_DAYS=7

# OAuth Providers
GOOGLE_CLIENT_ID=your-google-client-id
GOOGLE_CLIENT_SECRET=your-google-client-secret
MICROSOFT_CLIENT_ID=your-microsoft-client-id
MICROSOFT_CLIENT_SECRET=your-microsoft-client-secret
GITHUB_CLIENT_ID=your-github-client-id
GITHUB_CLIENT_SECRET=your-github-client-secret

# Database
DATABASE_URL=postgresql://user:password@localhost/montecarlo
DATABASE_POOL_SIZE=20
DATABASE_MAX_OVERFLOW=40

# Redis
REDIS_URL=redis://localhost:6379/0
REDIS_PASSWORD=your-redis-password

# File Storage
UPLOAD_PATH=/secure-uploads
MAX_UPLOAD_SIZE=10485760  # 10MB
ALLOWED_EXTENSIONS=xlsx,xlsm

# Security
BCRYPT_ROUNDS=12
CORS_ORIGINS=https://app.yourdomain.com,https://www.yourdomain.com
TRUSTED_HOSTS=app.yourdomain.com,api.yourdomain.com

# Rate Limiting
RATE_LIMIT_STORAGE=redis
RATE_LIMIT_DEFAULT=100/hour

# Monitoring
SENTRY_DSN=your-sentry-dsn
LOG_LEVEL=INFO
AUDIT_LOG_PATH=/var/log/montecarlo/audit.log
```

### 9. Docker Security Configuration

```dockerfile
# backend/Dockerfile.secure
FROM python:3.11-slim

# Security: Run as non-root user
RUN groupadd -r montecarlo && useradd -r -g montecarlo montecarlo

# Install security tools
RUN apt-get update && apt-get install -y \
    clamav \
    clamav-daemon \
    libmagic1 \
    && rm -rf /var/lib/apt/lists/*

# Update ClamAV database
RUN freshclam

# Create directories with proper permissions
RUN mkdir -p /app /secure-uploads /var/log/montecarlo \
    && chown -R montecarlo:montecarlo /app /secure-uploads /var/log/montecarlo

WORKDIR /app

# Copy requirements first for better caching
COPY --chown=montecarlo:montecarlo requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY --chown=montecarlo:montecarlo . .

# Security: Drop privileges
USER montecarlo

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')"

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 10. Testing Security

```python
# backend/tests/test_security.py
import pytest
from fastapi.testclient import TestClient
import time

def test_rate_limiting(client: TestClient):
    """Test rate limiting works"""
    # Make 11 requests (limit is 10/minute)
    for i in range(11):
        response = client.post("/api/simulations/run", json={})
        if i < 10:
            assert response.status_code != 429
        else:
            assert response.status_code == 429
            assert "rate limit exceeded" in response.text.lower()

def test_sql_injection_prevention(client: TestClient):
    """Test SQL injection is prevented"""
    malicious_inputs = [
        "'; DROP TABLE users; --",
        "1' OR '1'='1",
        "admin'--",
        "1; DELETE FROM users WHERE 1=1"
    ]
    
    for payload in malicious_inputs:
        response = client.post("/api/auth/login", json={
            "email": payload,
            "password": "test"
        })
        assert response.status_code == 422  # Validation error
        assert "malicious input" in response.text.lower()

def test_file_upload_security(client: TestClient, auth_headers):
    """Test file upload security checks"""
    # Test virus file (EICAR test file)
    eicar = b"X5O!P%@AP[4\\PZX54(P^)7CC)7}$EICAR-STANDARD-ANTIVIRUS-TEST-FILE!$H+H*"
    
    response = client.post(
        "/api/excel-parser/upload",
        files={"file": ("virus.xlsx", eicar, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")},
        headers=auth_headers
    )
    assert response.status_code == 400
    assert "virus detected" in response.text.lower()

def test_path_traversal_prevention(client: TestClient, auth_headers):
    """Test path traversal is prevented"""
    response = client.get(
        "/api/files/download",
        params={"path": "../../etc/passwd"},
        headers=auth_headers
    )
    assert response.status_code == 400
    assert "invalid path" in response.text.lower()

def test_xss_prevention(client: TestClient, auth_headers):
    """Test XSS prevention in inputs"""
    response = client.post(
        "/api/simulations/create",
        json={
            "name": "<script>alert('XSS')</script>",
            "description": "Test"
        },
        headers=auth_headers
    )
    # Should escape the input
    assert response.status_code == 200
    result = response.json()
    assert "&lt;script&gt;" in result["name"]
    assert "alert" not in result["name"]

def test_security_headers(client: TestClient):
    """Test security headers are present"""
    response = client.get("/api/health")
    
    assert response.headers.get("X-Content-Type-Options") == "nosniff"
    assert response.headers.get("X-Frame-Options") == "DENY"
    assert response.headers.get("X-XSS-Protection") == "1; mode=block"
    assert "Strict-Transport-Security" in response.headers
    assert "Content-Security-Policy" in response.headers
```

## Deployment Checklist

### Before Launch:
- [ ] All security tests passing
- [ ] ClamAV updated and running
- [ ] SSL certificates installed
- [ ] Environment variables secured
- [ ] Database backed up
- [ ] Rate limits tested under load
- [ ] OAuth providers configured
- [ ] Audit logging verified
- [ ] Security scan completed (OWASP ZAP)
- [ ] Penetration test performed

### Monitoring:
- [ ] Sentry error tracking active
- [ ] Security alerts configured
- [ ] Rate limit violations monitored
- [ ] Failed login attempts tracked
- [ ] File upload patterns analyzed
- [ ] API usage patterns normal
- [ ] No suspicious audit log entries

This implementation provides comprehensive security for the Free MVP launch! 