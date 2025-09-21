#!/usr/bin/env python3
"""
File Upload Security Testing Suite
Tests for virus injection, malicious file uploads, and file processing vulnerabilities
"""

import requests
import os
import tempfile
import logging
import json
from datetime import datetime
from typing import Dict, List, Any
import mimetypes
import hashlib

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("file_upload_security")

class FileUploadSecurityTester:
    """Comprehensive file upload security testing"""
    
    def __init__(self, base_url: str = "http://localhost:9090"):
        self.base_url = base_url
        self.api_base = f"{base_url}/api"
        self.session = requests.Session()
        self.vulnerabilities = []
        self.test_results = {
            "timestamp": datetime.utcnow().isoformat(),
            "target": self.api_base,
            "vulnerabilities": [],
            "file_upload_endpoints": [],
            "security_measures": []
        }
        
        # Get authentication token (admin credentials)
        self.auth_token = None
        self.get_auth_token()
        
        logger.info(f"🔍 Initialized File Upload Security Tester for {self.api_base}")

    def get_auth_token(self):
        """Attempt to get authentication token for testing"""
        try:
            # Try common authentication endpoints
            auth_endpoints = [
                "/auth/login",
                "/api/auth/login", 
                "/login"
            ]
            
            # Common credentials for testing
            credentials = [
                {"username": "admin", "password": "Demo123!MonteCarlo"},
                {"email": "admin@montecarlo.com", "password": "Demo123!MonteCarlo"},
                {"username": "test", "password": "test"}
            ]
            
            for endpoint in auth_endpoints:
                for creds in credentials:
                    try:
                        url = f"{self.base_url}{endpoint}"
                        response = self.session.post(url, json=creds, timeout=5)
                        
                        if response.status_code == 200:
                            data = response.json()
                            if "access_token" in data or "token" in data:
                                self.auth_token = data.get("access_token") or data.get("token")
                                logger.info("✅ Authentication successful for testing")
                                return
                    except:
                        continue
            
            logger.warning("⚠️ Could not authenticate - testing without auth token")
            
        except Exception as e:
            logger.warning(f"Authentication failed: {e}")

    def add_vulnerability(self, endpoint: str, vuln_type: str, severity: str, 
                         description: str, evidence: str, remediation: str):
        """Add a vulnerability finding"""
        vuln = {
            "endpoint": endpoint,
            "type": vuln_type,
            "severity": severity,
            "description": description,
            "evidence": evidence,
            "remediation": remediation,
            "timestamp": datetime.utcnow().isoformat()
        }
        self.vulnerabilities.append(vuln)
        self.test_results["vulnerabilities"].append(vuln)
        
        severity_emoji = {"critical": "🚨", "high": "⚠️", "medium": "🟡", "low": "ℹ️"}
        logger.warning(f"{severity_emoji.get(severity, '❔')} {severity.upper()} - {vuln_type}: {description}")

    def create_malicious_files(self):
        """Create various malicious test files"""
        test_files = {}
        
        # 1. Executable disguised as Excel
        exe_content = b"MZ\x90\x00" + b"A" * 100  # PE header + padding
        test_files["malicious.xlsx"] = exe_content
        
        # 2. Script injection in filename
        script_content = b"PK\x03\x04" + b"<script>alert('xss')</script>" * 10
        test_files["<script>alert('xss')</script>.xlsx"] = script_content
        
        # 3. Path traversal filename
        legitimate_excel = b"PK\x03\x04\x14\x00\x08\x00\x08\x00" + b"X" * 200  # ZIP signature
        test_files["../../../etc/passwd.xlsx"] = legitimate_excel
        
        # 4. Zip bomb (nested zip)
        zip_bomb = b"PK\x03\x04" + b"\x00" * 1000  # Simulated zip bomb
        test_files["zipbomb.xlsx"] = zip_bomb
        
        # 5. Large file (DoS test)
        test_files["huge_file.xlsx"] = b"A" * (50 * 1024 * 1024)  # 50MB
        
        # 6. Null byte injection
        test_files["malicious\x00.exe.xlsx"] = legitimate_excel
        
        # 7. Double extension
        test_files["document.xlsx.exe"] = exe_content
        
        # 8. Polyglot file (valid Excel + embedded script)
        polyglot = b"PK\x03\x04" + b"<?php system($_GET['cmd']); ?>" + b"X" * 500
        test_files["polyglot.xlsx"] = polyglot
        
        # 9. EICAR test string (antivirus test)
        eicar = b"X5O!P%@AP[4\\PZX54(P^)7CC)7}$EICAR-STANDARD-ANTIVIRUS-TEST-FILE!$H+H*"
        test_files["virus_test.xlsx"] = b"PK\x03\x04" + eicar
        
        # 10. Macro-enabled file simulation
        macro_file = b"PK\x03\x04" + b"Sub AutoOpen()\nShell \"calc.exe\"\nEnd Sub" + b"X" * 300
        test_files["macro_document.xlsm"] = macro_file
        
        return test_files

    def test_file_upload_endpoints(self):
        """Discover and test file upload endpoints"""
        logger.info("🔍 Discovering file upload endpoints...")
        
        # Common file upload endpoints
        upload_endpoints = [
            "/api/excel-parser/upload",
            "/api/files/upload",
            "/api/upload",
            "/upload",
            "/api/documents/upload",
            "/api/simulations/upload"
        ]
        
        headers = {}
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"
        
        working_endpoints = []
        
        for endpoint in upload_endpoints:
            url = f"{self.base_url}{endpoint}"
            
            try:
                # Test with simple file
                test_file = {"file": ("test.xlsx", b"PK\x03\x04test", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")}
                response = self.session.post(url, files=test_file, headers=headers, timeout=10)
                
                if response.status_code not in [404, 405]:  # Endpoint exists
                    working_endpoints.append((endpoint, response.status_code))
                    self.test_results["file_upload_endpoints"].append({
                        "endpoint": endpoint,
                        "status": response.status_code,
                        "response_size": len(response.content)
                    })
                    logger.info(f"📁 Found upload endpoint: {endpoint} (Status: {response.status_code})")
                    
            except Exception as e:
                logger.debug(f"Error testing {endpoint}: {e}")
        
        return working_endpoints

    def test_malicious_file_uploads(self, endpoints):
        """Test uploading malicious files to discovered endpoints"""
        logger.info("🦠 Testing malicious file uploads...")
        
        malicious_files = self.create_malicious_files()
        headers = {}
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"
        
        for endpoint, _ in endpoints:
            url = f"{self.base_url}{endpoint}"
            
            for filename, content in malicious_files.items():
                try:
                    # Test file upload
                    files = {"file": (filename, content, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")}
                    response = self.session.post(url, files=files, headers=headers, timeout=30)
                    
                    # Check if malicious file was accepted
                    if response.status_code == 200:
                        self.add_vulnerability(
                            endpoint, "Malicious File Upload", "high",
                            f"Malicious file '{filename}' was accepted by the server",
                            f"Status: {response.status_code}, File: {filename[:50]}",
                            "Implement file type validation, content scanning, and size limits"
                        )
                    
                    # Check for path traversal success
                    if "../" in filename and response.status_code == 200:
                        self.add_vulnerability(
                            endpoint, "Path Traversal via Filename", "critical",
                            f"Path traversal attack successful with filename: {filename}",
                            f"Server accepted file with traversal path",
                            "Sanitize filenames and validate file paths"
                        )
                    
                    # Check for XSS in filename
                    if "<script>" in filename and filename in response.text:
                        self.add_vulnerability(
                            endpoint, "XSS via Filename", "high",
                            f"XSS payload in filename reflected in response",
                            f"Filename with script tag reflected: {filename}",
                            "Sanitize and encode filenames in responses"
                        )
                    
                    # Check for executable upload
                    if filename.endswith(".exe") and response.status_code == 200:
                        self.add_vulnerability(
                            endpoint, "Executable File Upload", "critical",
                            f"Executable file upload allowed: {filename}",
                            f"Server accepted .exe file",
                            "Block executable file extensions"
                        )
                    
                    # Check for large file DoS
                    if len(content) > 10 * 1024 * 1024 and response.status_code == 200:
                        self.add_vulnerability(
                            endpoint, "Large File DoS", "medium",
                            f"Large file upload accepted (DoS risk)",
                            f"File size: {len(content)} bytes accepted",
                            "Implement file size limits"
                        )
                        
                except requests.exceptions.Timeout:
                    self.add_vulnerability(
                        endpoint, "File Upload Timeout", "medium",
                        f"File upload caused server timeout with {filename}",
                        f"Request timed out processing file",
                        "Implement proper timeout handling and file size limits"
                    )
                    
                except Exception as e:
                    logger.debug(f"Error uploading {filename} to {endpoint}: {e}")

    def test_file_processing_vulnerabilities(self, endpoints):
        """Test file processing and parsing vulnerabilities"""
        logger.info("⚙️ Testing file processing vulnerabilities...")
        
        headers = {}
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"
        
        # Create files that could exploit parsing vulnerabilities
        processing_test_files = {
            # XML External Entity (XXE) attack
            "xxe_attack.xlsx": b"""PK\x03\x04<?xml version="1.0"?>
<!DOCTYPE root [
<!ENTITY xxe SYSTEM "file:///etc/passwd">
]>
<root>&xxe;</root>""" + b"X" * 200,
            
            # Zip slip vulnerability
            "zip_slip.xlsx": b"PK\x03\x04" + b"../../../etc/passwd" + b"\x00" * 100,
            
            # Formula injection
            "formula_injection.xlsx": b"PK\x03\x04=cmd|' /C calc'!A0" + b"X" * 200,
            
            # Memory exhaustion
            "memory_bomb.xlsx": b"PK\x03\x04" + b"A" * (1024 * 1024),  # 1MB
        }
        
        for endpoint, _ in endpoints:
            url = f"{self.base_url}{endpoint}"
            
            for filename, content in processing_test_files.items():
                try:
                    files = {"file": (filename, content, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")}
                    response = self.session.post(url, files=files, headers=headers, timeout=15)
                    
                    # Check for XXE vulnerability
                    if "root:" in response.text or "/etc/passwd" in response.text:
                        self.add_vulnerability(
                            endpoint, "XML External Entity (XXE)", "critical",
                            f"XXE vulnerability detected - system files accessible",
                            f"Response contains system file content",
                            "Disable external entity processing in XML parsers"
                        )
                    
                    # Check for formula injection
                    if "=cmd" in filename and response.status_code == 200:
                        self.add_vulnerability(
                            endpoint, "Formula Injection", "high",
                            f"Formula injection may be possible in Excel processing",
                            f"Formula payload accepted for processing",
                            "Sanitize Excel formulas and disable macro execution"
                        )
                    
                    # Check for zip slip
                    if "zip_slip" in filename and response.status_code == 200:
                        self.add_vulnerability(
                            endpoint, "Zip Slip Vulnerability", "high",
                            f"Potential zip slip vulnerability in file processing",
                            f"Archive with traversal paths accepted",
                            "Validate and sanitize archive file paths"
                        )
                        
                except requests.exceptions.Timeout:
                    if "memory_bomb" in filename:
                        self.add_vulnerability(
                            endpoint, "Memory Exhaustion DoS", "medium",
                            f"File processing timeout suggests memory exhaustion",
                            f"Large file caused processing timeout",
                            "Implement memory limits and streaming processing"
                        )
                        
                except Exception as e:
                    logger.debug(f"Error testing {filename} processing: {e}")

    def test_file_storage_security(self, endpoints):
        """Test file storage and access control"""
        logger.info("💾 Testing file storage security...")
        
        headers = {}
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"
        
        # Test if uploaded files are directly accessible
        test_filenames = [
            "test_access.xlsx",
            "public_access.xlsx"
        ]
        
        for endpoint, _ in endpoints:
            upload_url = f"{self.base_url}{endpoint}"
            
            for filename in test_filenames:
                try:
                    # Upload a test file
                    test_content = b"PK\x03\x04TEST_CONTENT_FOR_ACCESS_CHECK" + b"X" * 100
                    files = {"file": (filename, test_content, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")}
                    upload_response = self.session.post(upload_url, files=files, headers=headers, timeout=10)
                    
                    if upload_response.status_code == 200:
                        # Try to access the uploaded file directly
                        potential_paths = [
                            f"/uploads/{filename}",
                            f"/files/{filename}",
                            f"/static/{filename}",
                            f"/api/files/{filename}",
                            f"/download/{filename}"
                        ]
                        
                        for path in potential_paths:
                            access_url = f"{self.base_url}{path}"
                            access_response = self.session.get(access_url, timeout=5)
                            
                            if access_response.status_code == 200 and b"TEST_CONTENT_FOR_ACCESS_CHECK" in access_response.content:
                                self.add_vulnerability(
                                    endpoint, "Direct File Access", "medium",
                                    f"Uploaded files are directly accessible via {path}",
                                    f"File accessible without authentication at {path}",
                                    "Implement access controls for uploaded files"
                                )
                                break
                                
                except Exception as e:
                    logger.debug(f"Error testing file storage for {filename}: {e}")

    def test_antivirus_protection(self, endpoints):
        """Test if antivirus scanning is in place"""
        logger.info("🦠 Testing antivirus protection...")
        
        headers = {}
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"
        
        # EICAR test string - standard antivirus test
        eicar_string = b"X5O!P%@AP[4\\PZX54(P^)7CC)7}$EICAR-STANDARD-ANTIVIRUS-TEST-FILE!$H+H*"
        
        for endpoint, _ in endpoints:
            url = f"{self.base_url}{endpoint}"
            
            try:
                # Upload EICAR test file
                files = {"file": ("eicar_test.xlsx", b"PK\x03\x04" + eicar_string, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")}
                response = self.session.post(url, files=files, headers=headers, timeout=10)
                
                # Check if antivirus blocked the file
                if response.status_code == 200:
                    # Look for virus scanning rejection
                    virus_indicators = ["virus", "malware", "threat", "infected", "quarantine", "blocked"]
                    response_text = response.text.lower()
                    
                    if not any(indicator in response_text for indicator in virus_indicators):
                        self.add_vulnerability(
                            endpoint, "No Antivirus Protection", "high",
                            f"EICAR test file accepted - no antivirus scanning detected",
                            f"Known virus signature accepted without detection",
                            "Implement antivirus scanning for uploaded files"
                        )
                    else:
                        logger.info("✅ Antivirus protection detected - EICAR file rejected")
                        self.test_results["security_measures"].append("Antivirus scanning active")
                        
                elif response.status_code == 400 and "virus" in response.text.lower():
                    logger.info("✅ Antivirus protection active - malicious file blocked")
                    self.test_results["security_measures"].append("Antivirus scanning active")
                    
            except Exception as e:
                logger.debug(f"Error testing antivirus protection: {e}")

    def check_file_validation_controls(self, endpoints):
        """Test file validation and security controls"""
        logger.info("🔍 Testing file validation controls...")
        
        headers = {}
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"
        
        # Test various file validation bypasses
        validation_tests = {
            # MIME type spoofing
            "exe_as_excel.xlsx": (b"MZ\x90\x00" + b"A" * 100, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
            
            # Double extension
            "document.xlsx.exe": (b"PK\x03\x04" + b"X" * 100, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
            
            # Case sensitivity bypass
            "MALICIOUS.EXE": (b"MZ\x90\x00" + b"A" * 100, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
            
            # Null byte bypass
            "malicious\x00.txt.xlsx": (b"PK\x03\x04" + b"X" * 100, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
        }
        
        controls_detected = []
        
        for endpoint, _ in endpoints:
            url = f"{self.base_url}{endpoint}"
            
            for filename, (content, mime_type) in validation_tests.items():
                try:
                    files = {"file": (filename, content, mime_type)}
                    response = self.session.post(url, files=files, headers=headers, timeout=10)
                    
                    # Check for proper file validation
                    if response.status_code == 400:
                        validation_errors = ["invalid", "type", "extension", "format", "not allowed"]
                        if any(error in response.text.lower() for error in validation_errors):
                            controls_detected.append(f"File validation on {filename}")
                            logger.info(f"✅ File validation active - rejected {filename}")
                    elif response.status_code == 200:
                        self.add_vulnerability(
                            endpoint, "Weak File Validation", "medium",
                            f"File validation bypass successful with {filename}",
                            f"Potentially dangerous file accepted",
                            "Strengthen file type validation and content checking"
                        )
                        
                except Exception as e:
                    logger.debug(f"Error testing validation for {filename}: {e}")
        
        if controls_detected:
            self.test_results["security_measures"].extend(controls_detected)

    def generate_report(self):
        """Generate comprehensive file upload security report"""
        total_vulns = len(self.vulnerabilities)
        critical_vulns = len([v for v in self.vulnerabilities if v["severity"] == "critical"])
        high_vulns = len([v for v in self.vulnerabilities if v["severity"] == "high"])
        medium_vulns = len([v for v in self.vulnerabilities if v["severity"] == "medium"])
        low_vulns = len([v for v in self.vulnerabilities if v["severity"] == "low"])
        
        # Calculate risk score
        risk_score = (critical_vulns * 10) + (high_vulns * 7) + (medium_vulns * 4) + (low_vulns * 1)
        
        self.test_results.update({
            "summary": {
                "total_vulnerabilities": total_vulns,
                "critical_vulnerabilities": critical_vulns,
                "high_vulnerabilities": high_vulns,
                "medium_vulnerabilities": medium_vulns,
                "low_vulnerabilities": low_vulns,
                "risk_score": risk_score,
                "endpoints_tested": len(self.test_results["file_upload_endpoints"]),
                "security_controls_detected": len(self.test_results["security_measures"])
            },
            "recommendations": [
                "Implement comprehensive file type validation",
                "Add antivirus scanning for all uploaded files",
                "Validate and sanitize file names",
                "Implement file size limits",
                "Use secure file storage with access controls",
                "Disable macro execution in Excel files",
                "Implement content-based file validation",
                "Add rate limiting for file uploads"
            ]
        })
        
        # Save results
        with open("file_upload_security_results.json", "w") as f:
            json.dump(self.test_results, f, indent=2)
        
        # Print summary
        logger.info("=" * 80)
        logger.info("FILE UPLOAD SECURITY TEST RESULTS")
        logger.info("=" * 80)
        logger.info(f"Target: {self.api_base}")
        logger.info(f"Upload Endpoints Found: {len(self.test_results['file_upload_endpoints'])}")
        logger.info(f"Total Vulnerabilities: {total_vulns}")
        logger.info(f"  Critical: {critical_vulns}")
        logger.info(f"  High: {high_vulns}")
        logger.info(f"  Medium: {medium_vulns}")
        logger.info(f"  Low: {low_vulns}")
        logger.info(f"Risk Score: {risk_score}/100")
        logger.info(f"Security Controls Detected: {len(self.test_results['security_measures'])}")
        
        if self.test_results["security_measures"]:
            logger.info("Security Measures Found:")
            for measure in self.test_results["security_measures"]:
                logger.info(f"  ✅ {measure}")
        
        logger.info(f"\nDetailed results saved to: file_upload_security_results.json")
        logger.info("=" * 80)
        
        return self.test_results

    async def run_comprehensive_test(self):
        """Run all file upload security tests"""
        logger.info("🚀 Starting comprehensive file upload security test...")
        
        # Discover upload endpoints
        endpoints = self.test_file_upload_endpoints()
        
        if not endpoints:
            logger.warning("⚠️ No file upload endpoints found")
            return self.generate_report()
        
        # Run security tests
        test_methods = [
            (self.test_malicious_file_uploads, endpoints),
            (self.test_file_processing_vulnerabilities, endpoints),
            (self.test_file_storage_security, endpoints),
            (self.test_antivirus_protection, endpoints),
            (self.check_file_validation_controls, endpoints)
        ]
        
        for test_method, args in test_methods:
            try:
                test_method(args)
            except Exception as e:
                logger.error(f"Error in {test_method.__name__}: {e}")
        
        return self.generate_report()


async def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="File Upload Security Testing")
    parser.add_argument("--url", default="http://localhost:9090", help="Target URL")
    parser.add_argument("--output", default="file_upload_security_results.json", help="Output file")
    
    args = parser.parse_args()
    
    tester = FileUploadSecurityTester(args.url)
    results = await tester.run_comprehensive_test()
    
    # Return exit code based on findings
    critical_count = results["summary"]["critical_vulnerabilities"]
    high_count = results["summary"]["high_vulnerabilities"]
    
    if critical_count > 0:
        return 2
    elif high_count > 0:
        return 1
    else:
        return 0


if __name__ == "__main__":
    import asyncio
    exit_code = asyncio.run(main())
    exit(exit_code)
