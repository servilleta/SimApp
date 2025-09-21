"""
File Scanner Service - Security Module

Provides comprehensive file security scanning including:
- Virus/malware detection using ClamAV
- File type validation using python-magic
- File size limits enforcement
- Content security scanning
- Malicious file pattern detection
"""

import logging
import os
import tempfile
import magic
from typing import Dict, List, Optional, Tuple, BinaryIO
from pathlib import Path

try:
    import pyclamd
    CLAMAV_AVAILABLE = True
except ImportError:
    CLAMAV_AVAILABLE = False
    pyclamd = None

from ..base import BaseService


logger = logging.getLogger(__name__)


class FileScannerService(BaseService):
    """File security scanner service"""
    
    # Allowed file types for Excel files
    ALLOWED_EXCEL_TYPES = {
        'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',  # .xlsx
        'application/vnd.ms-excel',  # .xls
        'application/vnd.ms-excel.sheet.macroEnabled.12',  # .xlsm
        'text/csv',  # .csv
        'application/csv'  # .csv alternative
    }
    
    # Allowed file extensions
    ALLOWED_EXTENSIONS = {'.xlsx', '.xls', '.xlsm', '.csv'}
    
    # Maximum file sizes by tier (in bytes)
    MAX_FILE_SIZES = {
        'free': 10 * 1024 * 1024,      # 10MB
        'basic': 50 * 1024 * 1024,     # 50MB
        'pro': 200 * 1024 * 1024,      # 200MB
        'enterprise': 1024 * 1024 * 1024  # 1GB
    }
    
    # Suspicious file patterns
    SUSPICIOUS_PATTERNS = [
        b'<script',
        b'javascript:',
        b'vbscript:',
        b'onload=',
        b'onerror=',
        b'eval(',
        b'document.write',
        b'ActiveXObject',
        b'Shell.Application',
        b'WScript.Shell',
        b'cmd.exe',
        b'powershell',
        b'<?php',
        b'<%',
        b'#!/bin/',
        b'exec(',
        b'system(',
        b'passthru(',
        b'shell_exec('
    ]
    
    def __init__(self, clamav_host: str = 'localhost', clamav_port: int = 3310):
        super().__init__("file_scanner")
        self.clamav_host = clamav_host
        self.clamav_port = clamav_port
        self.clamav_client = None
        self.magic_mime = None
        
    async def initialize(self) -> None:
        """Initialize the file scanner service"""
        await super().initialize()
        
        # Initialize python-magic
        try:
            self.magic_mime = magic.Magic(mime=True)
            logger.info("File type detection initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize file type detection: {e}")
        
        # Initialize ClamAV connection
        if CLAMAV_AVAILABLE:
            try:
                self.clamav_client = pyclamd.ClamdUnixSocket()
                if not self.clamav_client.ping():
                    # Try network socket
                    self.clamav_client = pyclamd.ClamdNetworkSocket(
                        host=self.clamav_host, 
                        port=self.clamav_port
                    )
                    
                if self.clamav_client.ping():
                    logger.info("ClamAV virus scanner connected")
                else:
                    logger.warning("ClamAV not available - virus scanning disabled")
                    self.clamav_client = None
            except Exception as e:
                logger.warning(f"Failed to connect to ClamAV: {e}")
                self.clamav_client = None
        else:
            logger.warning("pyclamd not installed - virus scanning disabled")
    
    async def scan_file(self, file_content: bytes, filename: str, 
                       user_tier: str = 'free') -> Dict[str, any]:
        """
        Comprehensive file security scan
        
        Returns:
            Dict with scan results including:
            - safe: bool
            - issues: List[str]
            - file_type: str
            - size: int
            - virus_scan_result: Optional[str]
        """
        scan_result = {
            'safe': True,
            'issues': [],
            'file_type': None,
            'size': len(file_content),
            'virus_scan_result': None,
            'filename': filename
        }
        
        try:
            # 1. File size check
            size_check = await self._check_file_size(file_content, user_tier)
            if not size_check['safe']:
                scan_result['safe'] = False
                scan_result['issues'].extend(size_check['issues'])
            
            # 2. File extension check
            ext_check = await self._check_file_extension(filename)
            if not ext_check['safe']:
                scan_result['safe'] = False
                scan_result['issues'].extend(ext_check['issues'])
            
            # 3. File type validation
            type_check = await self._check_file_type(file_content)
            scan_result['file_type'] = type_check.get('file_type')
            if not type_check['safe']:
                scan_result['safe'] = False
                scan_result['issues'].extend(type_check['issues'])
            
            # 4. Content security scan
            content_check = await self._check_file_content(file_content)
            if not content_check['safe']:
                scan_result['safe'] = False
                scan_result['issues'].extend(content_check['issues'])
            
            # 5. Virus scan (if available)
            if self.clamav_client:
                virus_check = await self._virus_scan(file_content)
                scan_result['virus_scan_result'] = virus_check.get('result')
                if not virus_check['safe']:
                    scan_result['safe'] = False
                    scan_result['issues'].extend(virus_check['issues'])
            else:
                scan_result['issues'].append('Virus scanning not available')
            
            logger.info(f"File scan completed for {filename}: {'SAFE' if scan_result['safe'] else 'UNSAFE'}")
            
        except Exception as e:
            logger.error(f"File scan failed for {filename}: {e}")
            scan_result['safe'] = False
            scan_result['issues'].append(f"Scan failed: {str(e)}")
        
        return scan_result
    
    async def _check_file_size(self, file_content: bytes, user_tier: str) -> Dict:
        """Check if file size is within limits"""
        file_size = len(file_content)
        max_size = self.MAX_FILE_SIZES.get(user_tier, self.MAX_FILE_SIZES['free'])
        
        if file_size > max_size:
            return {
                'safe': False,
                'issues': [f'File size ({file_size:,} bytes) exceeds {user_tier} tier limit ({max_size:,} bytes)']
            }
        
        return {'safe': True, 'issues': []}
    
    async def _check_file_extension(self, filename: str) -> Dict:
        """Check if file extension is allowed"""
        file_ext = Path(filename).suffix.lower()
        
        if file_ext not in self.ALLOWED_EXTENSIONS:
            return {
                'safe': False,
                'issues': [f'File extension {file_ext} not allowed. Allowed: {", ".join(self.ALLOWED_EXTENSIONS)}']
            }
        
        return {'safe': True, 'issues': []}
    
    async def _check_file_type(self, file_content: bytes) -> Dict:
        """Validate file type using magic numbers"""
        if not self.magic_mime:
            return {'safe': True, 'issues': [], 'file_type': 'unknown'}
        
        try:
            # Save to temporary file for magic detection
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(file_content)
                tmp_file.flush()
                
                detected_type = self.magic_mime.from_file(tmp_file.name)
                
                # Clean up
                os.unlink(tmp_file.name)
            
            if detected_type not in self.ALLOWED_EXCEL_TYPES:
                return {
                    'safe': False,
                    'issues': [f'File type {detected_type} not allowed for Excel files'],
                    'file_type': detected_type
                }
            
            return {'safe': True, 'issues': [], 'file_type': detected_type}
            
        except Exception as e:
            logger.error(f"File type detection failed: {e}")
            return {
                'safe': False,
                'issues': [f'File type detection failed: {str(e)}'],
                'file_type': 'unknown'
            }
    
    async def _check_file_content(self, file_content: bytes) -> Dict:
        """Scan file content for suspicious patterns"""
        issues = []
        
        # Check for suspicious patterns
        for pattern in self.SUSPICIOUS_PATTERNS:
            if pattern in file_content:
                issues.append(f'Suspicious pattern detected: {pattern.decode("utf-8", errors="ignore")}')
        
        # Check for embedded files or executables
        if b'PK\x03\x04' in file_content and b'META-INF' in file_content:
            # This could be a ZIP file masquerading as Excel
            if not (b'xl/' in file_content or b'[Content_Types].xml' in file_content):
                issues.append('Suspicious ZIP structure detected')
        
        # Check for embedded objects
        if b'oleObject' in file_content or b'OLE_LINK' in file_content:
            issues.append('Embedded OLE objects detected - potential security risk')
        
        # Check for macros in non-macro files
        if b'vbaProject' in file_content or b'macros' in file_content.lower():
            issues.append('Macro content detected in file')
        
        return {
            'safe': len(issues) == 0,
            'issues': issues
        }
    
    async def _virus_scan(self, file_content: bytes) -> Dict:
        """Scan file for viruses using ClamAV"""
        if not self.clamav_client:
            return {'safe': True, 'issues': [], 'result': 'scanner_unavailable'}
        
        try:
            # Scan the file content
            scan_result = self.clamav_client.scan_stream(file_content)
            
            if scan_result is None:
                return {'safe': True, 'issues': [], 'result': 'clean'}
            
            # ClamAV found something
            return {
                'safe': False,
                'issues': [f'Virus detected: {scan_result}'],
                'result': scan_result
            }
            
        except Exception as e:
            logger.error(f"Virus scan failed: {e}")
            return {
                'safe': False,
                'issues': [f'Virus scan failed: {str(e)}'],
                'result': 'scan_failed'
            }
    
    async def get_file_info(self, file_content: bytes, filename: str) -> Dict:
        """Get detailed file information"""
        info = {
            'filename': filename,
            'size': len(file_content),
            'extension': Path(filename).suffix.lower(),
            'file_type': 'unknown',
            'is_excel': False,
            'has_macros': False,
            'has_external_links': False
        }
        
        # Detect file type
        if self.magic_mime:
            try:
                with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                    tmp_file.write(file_content)
                    tmp_file.flush()
                    info['file_type'] = self.magic_mime.from_file(tmp_file.name)
                    os.unlink(tmp_file.name)
            except:
                pass
        
        # Check if it's an Excel file
        info['is_excel'] = info['file_type'] in self.ALLOWED_EXCEL_TYPES
        
        # Check for macros
        info['has_macros'] = b'vbaProject' in file_content or b'macros' in file_content.lower()
        
        # Check for external links
        info['has_external_links'] = (
            b'http://' in file_content or 
            b'https://' in file_content or 
            b'ftp://' in file_content or
            b'externalLink' in file_content
        )
        
        return info
    
    def health_check(self) -> Dict[str, any]:
        """Health check for file scanner service"""
        health = super().health_check()
        
        health.update({
            'clamav_available': self.clamav_client is not None,
            'magic_available': self.magic_mime is not None,
            'allowed_extensions': list(self.ALLOWED_EXTENSIONS),
            'max_file_sizes': self.MAX_FILE_SIZES
        })
        
        if self.clamav_client:
            try:
                health['clamav_ping'] = self.clamav_client.ping()
                health['clamav_version'] = self.clamav_client.version()
            except:
                health['clamav_ping'] = False
                health['clamav_version'] = 'unknown'
        
        return health 