"""
Input Validation & Sanitization Service - Security Module

Provides comprehensive input security including:
- SQL injection prevention
- XSS prevention and HTML sanitization
- Path traversal prevention
- Command injection prevention
- Data validation decorators
- Secure input processing
"""

import re
import logging
import html
from typing import Any, Dict, List, Optional, Union, Callable
from functools import wraps
from pathlib import Path
import bleach
from pydantic import BaseModel, field_validator
import defusedxml.ElementTree as ET

from ..base import BaseService


logger = logging.getLogger(__name__)


class InputValidatorService(BaseService):
    """Input validation and sanitization service"""
    
    # SQL injection patterns
    SQL_INJECTION_PATTERNS = [
        r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|EXECUTE|UNION|SCRIPT)\b)",
        r"(\b(OR|AND)\s+\d+\s*=\s*\d+)",
        r"(\b(OR|AND)\s+['\"]?\w+['\"]?\s*=\s*['\"]?\w+['\"]?)",
        r"(--|#|/\*|\*/)",
        r"(\bxp_\w+)",
        r"(\bsp_\w+)",
        r"(\bUNION\s+(ALL\s+)?SELECT)",
        r"(\bINTO\s+(OUT|DUMP)FILE)",
        r"(\bLOAD_FILE\s*\()",
        r"(\bINTO\s+OUTFILE)",
        r"(\bSLEEP\s*\()",
        r"(\bBENCHMARK\s*\()"
    ]
    
    # XSS patterns
    XSS_PATTERNS = [
        r"<script[^>]*>.*?</script>",
        r"javascript:",
        r"vbscript:",
        r"onload\s*=",
        r"onerror\s*=",
        r"onclick\s*=",
        r"onmouseover\s*=",
        r"onfocus\s*=",
        r"onblur\s*=",
        r"eval\s*\(",
        r"document\.write",
        r"document\.cookie",
        r"window\.location",
        r"<iframe[^>]*>",
        r"<object[^>]*>",
        r"<embed[^>]*>",
        r"<link[^>]*>"
    ]
    
    # Command injection patterns
    COMMAND_INJECTION_PATTERNS = [
        r"[;&|`$(){}[\]\\]",
        r"\b(cat|ls|dir|type|copy|move|del|rm|mkdir|rmdir|cd|pwd)\b",
        r"\b(wget|curl|nc|netcat|telnet|ssh|ftp)\b",
        r"\b(echo|printf|print)\b.*[>&|]",
        r"\$\{.*\}",
        r"`.*`",
        r"\$\(.*\)"
    ]
    
    # Path traversal patterns
    PATH_TRAVERSAL_PATTERNS = [
        r"\.\.[\\/]",
        r"[\\/]\.\.[\\/]",
        r"\.\.\\",
        r"\.\./",
        r"%2e%2e",
        r"%2f",
        r"%5c",
        r"\\\\",
        r"//+"
    ]
    
    # Allowed HTML tags for sanitization
    ALLOWED_HTML_TAGS = [
        'b', 'i', 'u', 'em', 'strong', 'p', 'br', 'span', 'div',
        'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'ul', 'ol', 'li',
        'blockquote', 'code', 'pre'
    ]
    
    # Allowed HTML attributes
    ALLOWED_HTML_ATTRIBUTES = {
        '*': ['class', 'id'],
        'a': ['href', 'title'],
        'img': ['src', 'alt', 'width', 'height'],
        'span': ['style'],
        'div': ['style']
    }
    
    def __init__(self):
        super().__init__("input_validator")
        self.sql_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.SQL_INJECTION_PATTERNS]
        self.xss_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.XSS_PATTERNS]
        self.cmd_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.COMMAND_INJECTION_PATTERNS]
        self.path_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.PATH_TRAVERSAL_PATTERNS]
    
    async def initialize(self) -> None:
        """Initialize the input validator service"""
        await super().initialize()
        logger.info("Input validator service initialized")
    
    def validate_input(self, input_data: Any, validation_type: str = 'general') -> Dict[str, Any]:
        """
        Comprehensive input validation
        
        Args:
            input_data: Data to validate
            validation_type: Type of validation ('general', 'sql', 'xss', 'path', 'command')
        
        Returns:
            Dict with validation results
        """
        result = {
            'valid': True,
            'issues': [],
            'sanitized_data': input_data,
            'risk_level': 'low'
        }
        
        if input_data is None:
            return result
        
        # Convert to string for pattern matching
        input_str = str(input_data)
        
        try:
            # Check for SQL injection
            if validation_type in ['general', 'sql']:
                sql_check = self._check_sql_injection(input_str)
                if not sql_check['safe']:
                    result['valid'] = False
                    result['issues'].extend(sql_check['issues'])
                    result['risk_level'] = 'high'
            
            # Check for XSS
            if validation_type in ['general', 'xss']:
                xss_check = self._check_xss(input_str)
                if not xss_check['safe']:
                    result['valid'] = False
                    result['issues'].extend(xss_check['issues'])
                    result['risk_level'] = 'high'
                result['sanitized_data'] = xss_check.get('sanitized', input_data)
            
            # Check for command injection
            if validation_type in ['general', 'command']:
                cmd_check = self._check_command_injection(input_str)
                if not cmd_check['safe']:
                    result['valid'] = False
                    result['issues'].extend(cmd_check['issues'])
                    result['risk_level'] = 'high'
            
            # Check for path traversal
            if validation_type in ['general', 'path']:
                path_check = self._check_path_traversal(input_str)
                if not path_check['safe']:
                    result['valid'] = False
                    result['issues'].extend(path_check['issues'])
                    result['risk_level'] = 'medium'
            
        except Exception as e:
            logger.error(f"Input validation failed: {e}")
            result['valid'] = False
            result['issues'].append(f"Validation error: {str(e)}")
            result['risk_level'] = 'high'
        
        return result
    
    def _check_sql_injection(self, input_str: str) -> Dict:
        """Check for SQL injection patterns"""
        issues = []
        
        for pattern in self.sql_patterns:
            matches = pattern.findall(input_str)
            if matches:
                issues.extend([f"SQL injection pattern detected: {match}" for match in matches])
        
        return {
            'safe': len(issues) == 0,
            'issues': issues
        }
    
    def _check_xss(self, input_str: str) -> Dict:
        """Check for XSS patterns and sanitize"""
        issues = []
        
        # Check for XSS patterns
        for pattern in self.xss_patterns:
            matches = pattern.findall(input_str)
            if matches:
                issues.extend([f"XSS pattern detected: {match}" for match in matches])
        
        # Sanitize HTML
        sanitized = bleach.clean(
            input_str,
            tags=self.ALLOWED_HTML_TAGS,
            attributes=self.ALLOWED_HTML_ATTRIBUTES,
            strip=True
        )
        
        # Additional HTML entity encoding
        sanitized = html.escape(sanitized, quote=True)
        
        return {
            'safe': len(issues) == 0,
            'issues': issues,
            'sanitized': sanitized
        }
    
    def _check_command_injection(self, input_str: str) -> Dict:
        """Check for command injection patterns"""
        issues = []
        
        for pattern in self.cmd_patterns:
            matches = pattern.findall(input_str)
            if matches:
                issues.extend([f"Command injection pattern detected: {match}" for match in matches])
        
        return {
            'safe': len(issues) == 0,
            'issues': issues
        }
    
    def _check_path_traversal(self, input_str: str) -> Dict:
        """Check for path traversal patterns"""
        issues = []
        
        for pattern in self.path_patterns:
            if pattern.search(input_str):
                issues.append(f"Path traversal pattern detected: {pattern.pattern}")
        
        return {
            'safe': len(issues) == 0,
            'issues': issues
        }
    
    def sanitize_filename(self, filename: str) -> str:
        """Sanitize filename for safe storage"""
        # Remove path components
        filename = Path(filename).name
        
        # Remove dangerous characters
        safe_filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
        
        # Remove control characters
        safe_filename = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', safe_filename)
        
        # Limit length
        if len(safe_filename) > 255:
            name, ext = Path(safe_filename).stem, Path(safe_filename).suffix
            safe_filename = name[:255-len(ext)] + ext
        
        # Ensure it's not empty
        if not safe_filename or safe_filename == '.':
            safe_filename = 'unnamed_file'
        
        return safe_filename
    
    def validate_email(self, email: str) -> Dict[str, Any]:
        """Validate email address"""
        result = {'valid': True, 'issues': []}
        
        if not email:
            result['valid'] = False
            result['issues'].append('Email is required')
            return result
        
        # Basic email regex
        email_pattern = re.compile(
            r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        )
        
        if not email_pattern.match(email):
            result['valid'] = False
            result['issues'].append('Invalid email format')
        
        # Check for suspicious patterns
        if any(pattern.search(email) for pattern in self.xss_patterns):
            result['valid'] = False
            result['issues'].append('Email contains suspicious patterns')
        
        return result
    
    def validate_password(self, password: str) -> Dict[str, Any]:
        """Validate password strength"""
        result = {'valid': True, 'issues': [], 'strength': 'weak'}
        
        if not password:
            result['valid'] = False
            result['issues'].append('Password is required')
            return result
        
        # Length check
        if len(password) < 8:
            result['valid'] = False
            result['issues'].append('Password must be at least 8 characters')
        
        # Complexity checks
        checks = {
            'lowercase': re.search(r'[a-z]', password),
            'uppercase': re.search(r'[A-Z]', password),
            'digit': re.search(r'\d', password),
            'special': re.search(r'[!@#$%^&*(),.?":{}|<>]', password)
        }
        
        missing = [check for check, found in checks.items() if not found]
        if missing:
            result['issues'].append(f'Password missing: {", ".join(missing)}')
        
        # Determine strength
        complexity_score = sum(1 for found in checks.values() if found)
        if len(password) >= 12 and complexity_score >= 3:
            result['strength'] = 'strong'
        elif len(password) >= 8 and complexity_score >= 2:
            result['strength'] = 'medium'
        
        return result
    
    def validate_json(self, json_str: str) -> Dict[str, Any]:
        """Validate and sanitize JSON input"""
        result = {'valid': True, 'issues': [], 'parsed': None}
        
        try:
            import json
            parsed = json.loads(json_str)
            
            # Check for deeply nested structures (DoS protection)
            def check_depth(obj, max_depth=10, current_depth=0):
                if current_depth > max_depth:
                    return False
                if isinstance(obj, dict):
                    return all(check_depth(v, max_depth, current_depth + 1) for v in obj.values())
                elif isinstance(obj, list):
                    return all(check_depth(item, max_depth, current_depth + 1) for item in obj)
                return True
            
            if not check_depth(parsed):
                result['valid'] = False
                result['issues'].append('JSON structure too deeply nested')
            else:
                result['parsed'] = parsed
                
        except json.JSONDecodeError as e:
            result['valid'] = False
            result['issues'].append(f'Invalid JSON: {str(e)}')
        except Exception as e:
            result['valid'] = False
            result['issues'].append(f'JSON validation error: {str(e)}')
        
        return result
    
    def validate_xml(self, xml_str: str) -> Dict[str, Any]:
        """Validate and sanitize XML input using defusedxml"""
        result = {'valid': True, 'issues': [], 'parsed': None}
        
        try:
            # Use defusedxml to prevent XML bombs and external entity attacks
            root = ET.fromstring(xml_str)
            result['parsed'] = root
            
        except ET.ParseError as e:
            result['valid'] = False
            result['issues'].append(f'Invalid XML: {str(e)}')
        except Exception as e:
            result['valid'] = False
            result['issues'].append(f'XML validation error: {str(e)}')
        
        return result


# Validation decorators
def validate_input_decorator(validation_type: str = 'general'):
    """Decorator to validate function inputs"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            validator = InputValidatorService()
            await validator.initialize()
            
            # Validate all string arguments
            for i, arg in enumerate(args):
                if isinstance(arg, str):
                    validation = validator.validate_input(arg, validation_type)
                    if not validation['valid']:
                        raise ValueError(f"Invalid input at position {i}: {validation['issues']}")
            
            # Validate string keyword arguments
            for key, value in kwargs.items():
                if isinstance(value, str):
                    validation = validator.validate_input(value, validation_type)
                    if not validation['valid']:
                        raise ValueError(f"Invalid input for {key}: {validation['issues']}")
            
            return await func(*args, **kwargs)
        return wrapper
    return decorator


def sanitize_input_decorator(func: Callable) -> Callable:
    """Decorator to sanitize function inputs"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        validator = InputValidatorService()
        await validator.initialize()
        
        # Sanitize string keyword arguments
        sanitized_kwargs = {}
        for key, value in kwargs.items():
            if isinstance(value, str):
                validation = validator.validate_input(value, 'xss')
                sanitized_kwargs[key] = validation['sanitized_data']
            else:
                sanitized_kwargs[key] = value
        
        return await func(*args, **sanitized_kwargs)
    return wrapper


# Pydantic validators for secure input models
class SecureInputMixin(BaseModel):
    """Mixin for Pydantic models with security validation"""
    
    @field_validator('*', mode='before')
    @classmethod
    def validate_strings(cls, v):
        if isinstance(v, str):
            # Note: This is synchronous validation for Pydantic
            # For full async validation, use the service directly
            
            # Basic XSS check
            if any(pattern in v.lower() for pattern in ['<script', 'javascript:', 'vbscript:']):
                raise ValueError('Input contains potentially malicious content')
            
            # Basic SQL injection check
            if any(pattern in v.lower() for pattern in ['union select', 'drop table', '--', ';']):
                raise ValueError('Input contains potentially malicious SQL patterns')
        
        return v