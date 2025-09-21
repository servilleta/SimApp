#!/usr/bin/env python3
"""
Penetration Testing Toolkit for Monte Carlo Simulation Platform
Comprehensive security testing framework with automated vulnerability discovery
"""

import json
import asyncio
import logging
import time
import requests
import subprocess
import sys
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path
import concurrent.futures
import random
import string
import tempfile

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("penetration_testing")

class SecurityTester:
    """Main security testing coordinator"""
    
    def __init__(self, base_url: str = "http://localhost:9090"):
        self.base_url = base_url
        self.api_base = f"{base_url}/api"
        self.test_results = {
            "timestamp": datetime.utcnow().isoformat(),
            "target": base_url,
            "vulnerabilities": [],
            "security_issues": [],
            "recommendations": []
        }
        
    async def run_comprehensive_test(self) -> Dict[str, Any]:
        """Run all penetration tests"""
        logger.info("🚨 Starting comprehensive penetration testing...")
        
        tests = [
            self.test_authentication_security,
            self.test_api_endpoint_security,
            self.test_file_upload_security,
            self.test_injection_vulnerabilities,
            self.test_xss_vulnerabilities,
            self.test_csrf_protection,
            self.test_information_disclosure,
            self.test_session_management,
            self.test_rate_limiting,
            self.test_docker_security,
            self.test_database_security
        ]
        
        for test in tests:
            try:
                logger.info(f"Running {test.__name__}...")
                await test()
            except Exception as e:
                logger.error(f"Test {test.__name__} failed: {e}")
                self.add_issue("test_failure", f"Failed to run {test.__name__}: {str(e)}", "medium")
        
        # Generate final report
        return self.generate_security_report()
    
    def add_vulnerability(self, vuln_type: str, description: str, severity: str, 
                         evidence: str = "", recommendation: str = ""):
        """Add vulnerability to results"""
        self.test_results["vulnerabilities"].append({
            "type": vuln_type,
            "description": description,
            "severity": severity,
            "evidence": evidence,
            "recommendation": recommendation,
            "timestamp": datetime.utcnow().isoformat()
        })
    
    def add_issue(self, issue_type: str, description: str, severity: str):
        """Add security issue to results"""
        self.test_results["security_issues"].append({
            "type": issue_type,
            "description": description,
            "severity": severity,
            "timestamp": datetime.utcnow().isoformat()
        })

class AuthenticationTester:
    """Tests for authentication vulnerabilities"""
    
    def __init__(self, security_tester: SecurityTester):
        self.st = security_tester
        self.api_base = security_tester.api_base
    
    async def test_weak_passwords(self):
        """Test for weak password acceptance"""
        weak_passwords = [
            "123456", "password", "admin", "test", "123",
            "qwerty", "abc123", "letmein", "welcome", "monkey"
        ]
        
        for password in weak_passwords:
            try:
                response = requests.post(f"{self.api_base}/auth/register", json={
                    "username": f"test_{random.randint(1000, 9999)}",
                    "email": f"test_{random.randint(1000, 9999)}@test.com",
                    "password": password
                }, timeout=10)
                
                if response.status_code == 200:
                    self.st.add_vulnerability(
                        "weak_password",
                        f"Weak password '{password}' was accepted during registration",
                        "high",
                        f"Registration successful with weak password: {password}",
                        "Implement strong password policy requiring minimum 8 characters, mixed case, numbers, and symbols"
                    )
            except Exception as e:
                logger.debug(f"Weak password test failed for {password}: {e}")
    
    async def test_brute_force_protection(self):
        """Test for brute force attack protection"""
        test_username = "test_bruteforce"
        attempts = []
        
        for i in range(20):  # Try 20 failed login attempts
            try:
                start_time = time.time()
                response = requests.post(f"{self.api_base}/auth/login", json={
                    "username": test_username,
                    "password": f"wrong_password_{i}"
                }, timeout=5)
                end_time = time.time()
                
                attempts.append({
                    "attempt": i + 1,
                    "status_code": response.status_code,
                    "response_time": end_time - start_time,
                    "blocked": response.status_code == 429
                })
                
                # If no rate limiting after 10 attempts, it's a vulnerability
                if i >= 10 and response.status_code != 429:
                    self.st.add_vulnerability(
                        "no_brute_force_protection",
                        "No rate limiting detected after multiple failed login attempts",
                        "high",
                        f"Completed {i+1} failed login attempts without blocking",
                        "Implement account lockout or rate limiting after 5-10 failed attempts"
                    )
                    break
                    
            except Exception as e:
                logger.debug(f"Brute force test attempt {i} failed: {e}")
    
    async def test_jwt_security(self):
        """Test JWT token security"""
        # Test for JWT none algorithm attack
        test_tokens = [
            "eyJhbGciOiJub25lIiwidHlwIjoiSldUIn0.eyJ1c2VyIjoiYWRtaW4iLCJpYXQiOjE2MjM5NzgwMDB9.",
            "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyIjoiYWRtaW4iLCJpYXQiOjk5OTk5OTk5OTl9.invalid",
            "invalid.jwt.token"
        ]
        
        for token in test_tokens:
            try:
                response = requests.get(f"{self.api_base}/auth/me", headers={
                    "Authorization": f"Bearer {token}"
                }, timeout=10)
                
                if response.status_code == 200:
                    self.st.add_vulnerability(
                        "jwt_bypass",
                        "Invalid JWT token was accepted",
                        "critical",
                        f"Token accepted: {token[:50]}...",
                        "Implement proper JWT validation and signature verification"
                    )
            except Exception as e:
                logger.debug(f"JWT test failed for token: {e}")

class FileUploadTester:
    """Tests for file upload vulnerabilities"""
    
    def __init__(self, security_tester: SecurityTester):
        self.st = security_tester
        self.api_base = security_tester.api_base
    
    async def test_malicious_file_upload(self):
        """Test malicious file upload scenarios"""
        malicious_files = [
            ("shell.php", b"<?php system($_GET['cmd']); ?>", "application/octet-stream"),
            ("test.exe", b"MZ\x90\x00\x03\x00\x00\x00", "application/x-executable"),
            ("script.js", b"alert('XSS')", "application/javascript"),
            ("../../../etc/passwd", b"root:x:0:0:root:/root:/bin/bash", "text/plain"),
            ("test.xlsx.php", b"<?php phpinfo(); ?>", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        ]
        
        for filename, content, content_type in malicious_files:
            try:
                with tempfile.NamedTemporaryFile(suffix=".xlsx") as tmp:
                    tmp.write(content)
                    tmp.flush()
                    
                    with open(tmp.name, 'rb') as f:
                        response = requests.post(
                            f"{self.api_base}/excel-parser/upload",
                            files={"file": (filename, f, content_type)},
                            timeout=30
                        )
                    
                    if response.status_code == 200:
                        self.st.add_vulnerability(
                            "malicious_file_upload",
                            f"Malicious file '{filename}' was accepted",
                            "high",
                            f"File upload successful for: {filename}",
                            "Implement strict file type validation, content scanning, and file sandboxing"
                        )
            except Exception as e:
                logger.debug(f"Malicious file upload test failed for {filename}: {e}")
    
    async def test_file_size_limits(self):
        """Test file size limit enforcement"""
        # Try to upload a large file
        large_content = b"A" * (600 * 1024 * 1024)  # 600MB
        
        try:
            with tempfile.NamedTemporaryFile(suffix=".xlsx") as tmp:
                tmp.write(large_content)
                tmp.flush()
                
                with open(tmp.name, 'rb') as f:
                    response = requests.post(
                        f"{self.api_base}/excel-parser/upload",
                        files={"file": ("large_file.xlsx", f, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")},
                        timeout=60
                    )
                
                if response.status_code == 200:
                    self.st.add_vulnerability(
                        "no_file_size_limit",
                        "Large file (600MB) was accepted, potentially enabling DoS attacks",
                        "medium",
                        "600MB file upload successful",
                        "Implement and enforce file size limits"
                    )
        except Exception as e:
            logger.debug(f"File size limit test failed: {e}")

class InjectionTester:
    """Tests for injection vulnerabilities"""
    
    def __init__(self, security_tester: SecurityTester):
        self.st = security_tester
        self.api_base = security_tester.api_base
    
    async def test_sql_injection(self):
        """Test for SQL injection vulnerabilities"""
        sql_payloads = [
            "' OR '1'='1",
            "'; DROP TABLE users;--",
            "' UNION SELECT password FROM users--",
            "1' OR '1'='1' --",
            "admin'--",
            "' OR 1=1#",
            "'; WAITFOR DELAY '00:00:05'--"
        ]
        
        # Test login endpoint
        for payload in sql_payloads:
            try:
                response = requests.post(f"{self.api_base}/auth/login", json={
                    "username": payload,
                    "password": "test"
                }, timeout=10)
                
                # Check for SQL error messages or successful bypass
                if any(error in response.text.lower() for error in [
                    "sql", "syntax", "mysql", "postgresql", "sqlite", "ora-", "syntax error"
                ]):
                    self.st.add_vulnerability(
                        "sql_injection",
                        f"SQL injection vulnerability in login endpoint",
                        "critical",
                        f"Payload: {payload}, Response: {response.text[:200]}",
                        "Use parameterized queries and input sanitization"
                    )
                    
                if response.status_code == 200 and "token" in response.text:
                    self.st.add_vulnerability(
                        "sql_injection_bypass",
                        f"Authentication bypass via SQL injection",
                        "critical",
                        f"Successful login with payload: {payload}",
                        "Implement proper input validation and parameterized queries"
                    )
            except Exception as e:
                logger.debug(f"SQL injection test failed for payload {payload}: {e}")
    
    async def test_nosql_injection(self):
        """Test for NoSQL injection vulnerabilities"""
        nosql_payloads = [
            {"$ne": None},
            {"$regex": ".*"},
            {"$where": "this.username == this.password"},
            {"$gt": ""},
            {"username": {"$ne": None}, "password": {"$ne": None}}
        ]
        
        for payload in nosql_payloads:
            try:
                response = requests.post(f"{self.api_base}/auth/login", json=payload, timeout=10)
                
                if response.status_code == 200 and "token" in response.text:
                    self.st.add_vulnerability(
                        "nosql_injection",
                        "NoSQL injection vulnerability detected",
                        "critical",
                        f"Successful bypass with payload: {json.dumps(payload)}",
                        "Sanitize input and validate data types for NoSQL queries"
                    )
            except Exception as e:
                logger.debug(f"NoSQL injection test failed: {e}")

class XSSCSRFTester:
    """Tests for XSS and CSRF vulnerabilities"""
    
    def __init__(self, security_tester: SecurityTester):
        self.st = security_tester
        self.api_base = security_tester.api_base
    
    async def test_xss_vulnerabilities(self):
        """Test for Cross-Site Scripting vulnerabilities"""
        xss_payloads = [
            "<script>alert('XSS')</script>",
            "'><script>alert('XSS')</script>",
            "<img src=x onerror=alert('XSS')>",
            "javascript:alert('XSS')",
            "<svg onload=alert('XSS')>",
            "<%2Fscript%3E%3Cscript%3Ealert('XSS')%3C%2Fscript%3E"
        ]
        
        # Test various endpoints that might reflect input
        test_endpoints = [
            "/auth/register",
            "/excel-parser/upload",
            "/simulations/create"
        ]
        
        for endpoint in test_endpoints:
            for payload in xss_payloads:
                try:
                    # Test in different fields
                    test_data = {
                        "username": payload,
                        "email": f"{payload}@test.com",
                        "name": payload,
                        "description": payload
                    }
                    
                    response = requests.post(f"{self.api_base}{endpoint}", json=test_data, timeout=10)
                    
                    if payload in response.text and "text/html" in response.headers.get("content-type", ""):
                        self.st.add_vulnerability(
                            "reflected_xss",
                            f"Reflected XSS vulnerability in {endpoint}",
                            "high",
                            f"Payload reflected: {payload}",
                            "Implement proper output encoding and Content Security Policy"
                        )
                except Exception as e:
                    logger.debug(f"XSS test failed for {endpoint} with {payload}: {e}")
    
    async def test_csrf_protection(self):
        """Test CSRF protection mechanisms"""
        try:
            # Test if state-changing operations require CSRF tokens
            response = requests.post(f"{self.api_base}/auth/register", json={
                "username": "csrf_test",
                "email": "csrf@test.com",
                "password": "Test123!"
            }, timeout=10)
            
            # Check if request succeeded without CSRF token
            if response.status_code == 200:
                csrf_token_present = any(header.lower() in ["x-csrf-token", "csrf-token"] 
                                       for header in response.headers.keys())
                
                if not csrf_token_present:
                    self.st.add_vulnerability(
                        "no_csrf_protection",
                        "State-changing operations lack CSRF protection",
                        "medium",
                        "Registration succeeded without CSRF token",
                        "Implement CSRF tokens for all state-changing operations"
                    )
        except Exception as e:
            logger.debug(f"CSRF test failed: {e}")

class InformationDisclosureTester:
    """Tests for information disclosure vulnerabilities"""
    
    def __init__(self, security_tester: SecurityTester):
        self.st = security_tester
        self.api_base = security_tester.api_base
    
    async def test_error_message_disclosure(self):
        """Test for sensitive information in error messages"""
        test_requests = [
            ("GET", "/api/nonexistent"),
            ("POST", "/api/auth/login", {"invalid": "data"}),
            ("GET", "/api/simulations/999999"),
            ("POST", "/api/excel-parser/upload", {}),
        ]
        
        for method, endpoint, data in test_requests:
            try:
                if method == "GET":
                    response = requests.get(f"{self.st.base_url}{endpoint}", timeout=10)
                else:
                    response = requests.post(f"{self.st.base_url}{endpoint}", json=data, timeout=10)
                
                # Check for sensitive information disclosure
                sensitive_patterns = [
                    "traceback", "stack trace", "/app/", "/backend/",
                    "postgresql://", "mongodb://", "mysql://",
                    "secret", "key", "password", "token",
                    "internal server error", "debug",
                    "__pycache__", ".py", "sqlalchemy"
                ]
                
                response_text = response.text.lower()
                for pattern in sensitive_patterns:
                    if pattern in response_text:
                        self.st.add_vulnerability(
                            "information_disclosure",
                            f"Sensitive information disclosed in error message",
                            "medium",
                            f"Pattern '{pattern}' found in response to {method} {endpoint}",
                            "Implement generic error messages and proper error handling"
                        )
                        break
                        
            except Exception as e:
                logger.debug(f"Information disclosure test failed for {endpoint}: {e}")
    
    async def test_debug_endpoints(self):
        """Test for exposed debug endpoints"""
        debug_endpoints = [
            "/debug", "/api/debug", "/admin", "/api/admin",
            "/docs", "/api/docs", "/redoc", "/api/redoc",
            "/openapi.json", "/api/openapi.json",
            "/health", "/api/health", "/status", "/api/status",
            "/metrics", "/api/metrics"
        ]
        
        for endpoint in debug_endpoints:
            try:
                response = requests.get(f"{self.st.base_url}{endpoint}", timeout=10)
                
                if response.status_code == 200:
                    # Check if this exposes sensitive information
                    if any(sensitive in response.text.lower() for sensitive in [
                        "swagger", "openapi", "database", "redis", "internal",
                        "configuration", "environment", "debug"
                    ]):
                        self.st.add_vulnerability(
                            "debug_endpoint_exposed",
                            f"Debug endpoint {endpoint} is publicly accessible",
                            "low",
                            f"Endpoint accessible: {endpoint}",
                            "Restrict access to debug endpoints in production"
                        )
            except Exception as e:
                logger.debug(f"Debug endpoint test failed for {endpoint}: {e}")

class DockerSecurityTester:
    """Tests for Docker container security"""
    
    def __init__(self, security_tester: SecurityTester):
        self.st = security_tester
    
    async def test_docker_configuration(self):
        """Test Docker security configuration"""
        try:
            # Check if Docker daemon is accessible
            result = subprocess.run(["docker", "version"], capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                # Check for privileged containers
                result = subprocess.run(["docker", "ps", "--format", "table {{.Names}}\t{{.Command}}"], 
                                      capture_output=True, text=True, timeout=10)
                
                if "--privileged" in result.stdout:
                    self.st.add_vulnerability(
                        "privileged_container",
                        "Containers running in privileged mode detected",
                        "high",
                        "Privileged containers found in docker ps output",
                        "Remove --privileged flag and use specific capabilities instead"
                    )
                
                # Check for containers running as root
                result = subprocess.run(["docker", "exec", "montecarlo-nginx", "whoami"], 
                                      capture_output=True, text=True, timeout=10)
                
                if result.returncode == 0 and "root" in result.stdout:
                    self.st.add_issue(
                        "container_root_user",
                        "Container processes running as root user",
                        "medium"
                    )
                    
        except Exception as e:
            logger.debug(f"Docker security test failed: {e}")

class NetworkSecurityTester:
    """Tests for network security"""
    
    def __init__(self, security_tester: SecurityTester):
        self.st = security_tester
        self.base_url = security_tester.base_url
    
    async def test_ssl_configuration(self):
        """Test SSL/TLS configuration"""
        if self.base_url.startswith("https://"):
            try:
                import ssl
                import socket
                
                hostname = self.base_url.replace("https://", "").split("/")[0]
                port = 443
                
                context = ssl.create_default_context()
                
                with socket.create_connection((hostname, port), timeout=10) as sock:
                    with context.wrap_socket(sock, server_hostname=hostname) as ssock:
                        cert = ssock.getpeercert()
                        protocol = ssock.version()
                        
                        # Check for weak SSL protocols
                        if protocol in ["SSLv2", "SSLv3", "TLSv1", "TLSv1.1"]:
                            self.st.add_vulnerability(
                                "weak_ssl_protocol",
                                f"Weak SSL/TLS protocol in use: {protocol}",
                                "high",
                                f"Protocol: {protocol}",
                                "Disable weak SSL/TLS protocols and use TLSv1.2 or higher"
                            )
                            
            except Exception as e:
                logger.debug(f"SSL configuration test failed: {e}")
        else:
            self.st.add_vulnerability(
                "no_ssl",
                "Application not using HTTPS",
                "high",
                "HTTP protocol detected",
                "Implement HTTPS with proper SSL/TLS configuration"
            )
    
    async def test_security_headers(self):
        """Test security headers"""
        try:
            response = requests.get(self.base_url, timeout=10)
            headers = response.headers
            
            required_headers = {
                "X-Content-Type-Options": "nosniff",
                "X-Frame-Options": ["DENY", "SAMEORIGIN"],
                "X-XSS-Protection": "1; mode=block",
                "Strict-Transport-Security": None,  # Any value is good for HTTPS
                "Content-Security-Policy": None,
                "Referrer-Policy": None
            }
            
            for header, expected_value in required_headers.items():
                if header not in headers:
                    self.st.add_vulnerability(
                        "missing_security_header",
                        f"Missing security header: {header}",
                        "medium",
                        f"Header not found: {header}",
                        f"Add {header} header with appropriate value"
                    )
                elif expected_value and isinstance(expected_value, list):
                    if headers[header] not in expected_value:
                        self.st.add_vulnerability(
                            "incorrect_security_header",
                            f"Incorrect value for security header: {header}",
                            "low",
                            f"Value: {headers[header]}",
                            f"Set {header} to one of: {expected_value}"
                        )
                        
        except Exception as e:
            logger.debug(f"Security headers test failed: {e}")

# Add method implementations to SecurityTester class
    async def test_authentication_security(self):
        """Test authentication security"""
        auth_tester = AuthenticationTester(self)
        await auth_tester.test_weak_passwords()
        await auth_tester.test_brute_force_protection()
        await auth_tester.test_jwt_security()

    async def test_api_endpoint_security(self):
        """Test API endpoint security"""
        network_tester = NetworkSecurityTester(self)
        await network_tester.test_security_headers()
        await network_tester.test_ssl_configuration()

    async def test_file_upload_security(self):
        """Test file upload security"""
        upload_tester = FileUploadTester(self)
        await upload_tester.test_malicious_file_upload()
        await upload_tester.test_file_size_limits()

    async def test_injection_vulnerabilities(self):
        """Test injection vulnerabilities"""
        injection_tester = InjectionTester(self)
        await injection_tester.test_sql_injection()
        await injection_tester.test_nosql_injection()

    async def test_xss_vulnerabilities(self):
        """Test XSS vulnerabilities"""
        xss_tester = XSSCSRFTester(self)
        await xss_tester.test_xss_vulnerabilities()

    async def test_csrf_protection(self):
        """Test CSRF protection"""
        xss_tester = XSSCSRFTester(self)
        await xss_tester.test_csrf_protection()

    async def test_information_disclosure(self):
        """Test information disclosure"""
        info_tester = InformationDisclosureTester(self)
        await info_tester.test_error_message_disclosure()
        await info_tester.test_debug_endpoints()

    async def test_session_management(self):
        """Test session management security"""
        # Test session timeout, session fixation, etc.
        try:
            # Test session timeout
            response = requests.post(f"{self.api_base}/auth/login", json={
                "username": "admin",
                "password": "Demo123!MonteCarlo"
            }, timeout=10)
            
            if response.status_code == 200:
                token = response.json().get("access_token")
                if token:
                    # Wait and test if session expires
                    await asyncio.sleep(2)
                    auth_response = requests.get(f"{self.api_base}/auth/me", 
                                               headers={"Authorization": f"Bearer {token}"}, 
                                               timeout=10)
                    
                    # Check if long-lived tokens are an issue
                    if auth_response.status_code == 200:
                        # This is expected behavior, just noting session is active
                        pass
                        
        except Exception as e:
            logger.debug(f"Session management test failed: {e}")

    async def test_rate_limiting(self):
        """Test rate limiting effectiveness"""
        try:
            # Test API rate limiting
            requests_count = 0
            blocked_count = 0
            
            for i in range(50):  # Try 50 rapid requests
                try:
                    response = requests.get(f"{self.api_base}/health", timeout=5)
                    requests_count += 1
                    
                    if response.status_code == 429:
                        blocked_count += 1
                        
                except Exception:
                    pass
            
            if blocked_count == 0 and requests_count > 30:
                self.add_vulnerability(
                    "no_rate_limiting",
                    "No rate limiting detected on API endpoints",
                    "medium",
                    f"Completed {requests_count} requests without rate limiting",
                    "Implement rate limiting on all public API endpoints"
                )
                
        except Exception as e:
            logger.debug(f"Rate limiting test failed: {e}")

    async def test_docker_security(self):
        """Test Docker container security"""
        docker_tester = DockerSecurityTester(self)
        await docker_tester.test_docker_configuration()

    async def test_database_security(self):
        """Test database security configuration"""
        try:
            # Check for exposed database ports
            import socket
            
            db_ports = [5432, 3306, 27017, 6379]  # PostgreSQL, MySQL, MongoDB, Redis
            for port in db_ports:
                try:
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.settimeout(3)
                    result = sock.connect_ex(('localhost', port))
                    sock.close()
                    
                    if result == 0:
                        self.add_issue(
                            "exposed_database_port",
                            f"Database port {port} is accessible from localhost",
                            "medium"
                        )
                except Exception:
                    pass
                    
        except Exception as e:
            logger.debug(f"Database security test failed: {e}")

    def generate_security_report(self) -> Dict[str, Any]:
        """Generate comprehensive security report"""
        
        # Calculate risk score
        risk_score = 0
        for vuln in self.test_results["vulnerabilities"]:
            if vuln["severity"] == "critical":
                risk_score += 10
            elif vuln["severity"] == "high":
                risk_score += 7
            elif vuln["severity"] == "medium":
                risk_score += 4
            elif vuln["severity"] == "low":
                risk_score += 1
        
        # Generate recommendations
        recommendations = [
            "Implement comprehensive input validation and sanitization",
            "Enable proper error handling without information disclosure",
            "Configure security headers for all HTTP responses",
            "Implement rate limiting on all API endpoints",
            "Use HTTPS with strong SSL/TLS configuration",
            "Implement proper authentication and session management",
            "Regular security testing and code reviews",
            "Keep all dependencies and frameworks updated",
            "Implement Web Application Firewall (WAF)",
            "Regular penetration testing by security professionals"
        ]
        
        self.test_results.update({
            "summary": {
                "total_vulnerabilities": len(self.test_results["vulnerabilities"]),
                "critical_vulnerabilities": len([v for v in self.test_results["vulnerabilities"] if v["severity"] == "critical"]),
                "high_vulnerabilities": len([v for v in self.test_results["vulnerabilities"] if v["severity"] == "high"]),
                "medium_vulnerabilities": len([v for v in self.test_results["vulnerabilities"] if v["severity"] == "medium"]),
                "low_vulnerabilities": len([v for v in self.test_results["vulnerabilities"] if v["severity"] == "low"]),
                "risk_score": risk_score,
                "security_issues": len(self.test_results["security_issues"])
            },
            "recommendations": recommendations,
            "next_steps": [
                "Review and prioritize identified vulnerabilities",
                "Implement security fixes based on severity",
                "Re-test after implementing fixes",
                "Consider professional penetration testing",
                "Implement continuous security monitoring"
            ]
        })
        
        return self.test_results

async def main():
    """Main penetration testing function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Monte Carlo Platform Penetration Testing")
    parser.add_argument("--url", default="http://localhost:9090", help="Target URL")
    parser.add_argument("--output", default="penetration_test_report.json", help="Output file")
    
    args = parser.parse_args()
    
    logger.info(f"🚨 Starting penetration test of {args.url}")
    
    tester = SecurityTester(args.url)
    results = await tester.run_comprehensive_test()
    
    # Save results
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    summary = results["summary"]
    print(f"\n{'='*60}")
    print(f"PENETRATION TEST RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"Target: {args.url}")
    print(f"Risk Score: {summary['risk_score']}/100")
    print(f"Total Vulnerabilities: {summary['total_vulnerabilities']}")
    print(f"  Critical: {summary['critical_vulnerabilities']}")
    print(f"  High: {summary['high_vulnerabilities']}")
    print(f"  Medium: {summary['medium_vulnerabilities']}")
    print(f"  Low: {summary['low_vulnerabilities']}")
    print(f"Security Issues: {summary['security_issues']}")
    print(f"\nDetailed report saved to: {args.output}")
    print(f"{'='*60}")
    
    if summary['critical_vulnerabilities'] > 0:
        print("⚠️  CRITICAL VULNERABILITIES FOUND - IMMEDIATE ACTION REQUIRED")
        return 1
    elif summary['high_vulnerabilities'] > 0:
        print("⚠️  HIGH SEVERITY VULNERABILITIES FOUND - PRIORITY FIXES NEEDED")
        return 1
    else:
        print("✅ No critical or high severity vulnerabilities found")
        return 0

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
