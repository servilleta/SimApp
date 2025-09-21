#!/usr/bin/env python3
"""
API Penetration Testing Suite
Comprehensive security testing focused on API endpoints
"""

import requests
import json
import time
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
import urllib.parse
import base64
import hashlib
import random
import string

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("api_penetration_test")

class APIPenetrationTester:
    """Comprehensive API penetration testing suite"""
    
    def __init__(self, base_url: str = "http://localhost:9090"):
        self.base_url = base_url
        self.api_base = f"{base_url}/api"
        self.session = requests.Session()
        self.vulnerabilities = []
        self.test_results = {
            "timestamp": datetime.utcnow().isoformat(),
            "target_api": self.api_base,
            "vulnerabilities": [],
            "test_summary": {},
            "recommendations": []
        }
        
        # Common payloads for testing
        self.sql_payloads = [
            "' OR '1'='1",
            "\" OR \"1\"=\"1",
            "'; DROP TABLE users; --",
            "' UNION SELECT NULL,NULL,NULL --",
            "1; WAITFOR DELAY '0:0:5' --"
        ]
        
        self.xss_payloads = [
            "<script>alert('API_XSS')</script>",
            "<img src=x onerror=alert('API_XSS')>",
            "javascript:alert('API_XSS')",
            "<svg/onload=alert('API_XSS')>"
        ]
        
        self.command_injection_payloads = [
            "; ls -la",
            "| whoami",
            "$(id)",
            "`cat /etc/passwd`",
            "; cat /etc/hosts"
        ]
        
        logger.info(f"🔍 Initialized API Penetration Tester for {self.api_base}")

    def add_vulnerability(self, endpoint: str, vulnerability_type: str, severity: str, 
                         description: str, evidence: str, remediation: str):
        """Add a vulnerability finding"""
        vuln = {
            "endpoint": endpoint,
            "type": vulnerability_type,
            "severity": severity,
            "description": description,
            "evidence": evidence,
            "remediation": remediation,
            "timestamp": datetime.utcnow().isoformat()
        }
        self.vulnerabilities.append(vuln)
        self.test_results["vulnerabilities"].append(vuln)
        
        severity_emoji = {"critical": "🚨", "high": "⚠️", "medium": "🟡", "low": "ℹ️"}
        logger.warning(f"{severity_emoji.get(severity, '❔')} {severity.upper()} - {vulnerability_type} in {endpoint}: {description}")

    def test_authentication_endpoints(self):
        """Test authentication and authorization vulnerabilities"""
        logger.info("🔐 Testing Authentication & Authorization...")
        
        auth_endpoints = [
            ("/auth/login", "POST", {"username": "test", "password": "test"}),
            ("/auth/register", "POST", {"username": "test", "password": "test", "email": "test@test.com"}),
            ("/auth/reset-password", "POST", {"email": "test@test.com"}),
            ("/auth/change-password", "POST", {"old_password": "test", "new_password": "new"}),
            ("/auth/logout", "POST", {}),
            ("/auth/refresh", "POST", {"refresh_token": "fake_token"})
        ]
        
        for endpoint, method, payload in auth_endpoints:
            url = f"{self.api_base}{endpoint}"
            
            # Test 1: SQL Injection in auth fields
            for sql_payload in self.sql_payloads[:3]:  # Test first 3 payloads
                test_payload = payload.copy()
                if "username" in test_payload:
                    test_payload["username"] = f"admin{sql_payload}"
                if "password" in test_payload:
                    test_payload["password"] = f"password{sql_payload}"
                
                try:
                    response = self.session.request(method, url, json=test_payload, timeout=10)
                    
                    # Check for SQL error messages
                    error_indicators = ["SQL syntax", "mysql_fetch", "PostgreSQL", "ORA-", "sqlite3"]
                    if any(indicator in response.text.lower() for indicator in error_indicators):
                        self.add_vulnerability(
                            endpoint, "SQL Injection", "high",
                            f"SQL injection vulnerability in authentication endpoint",
                            f"Payload: {sql_payload}, Response contains SQL errors",
                            "Implement parameterized queries and input validation"
                        )
                    
                    # Check for time-based SQL injection
                    if "WAITFOR DELAY" in sql_payload or "SLEEP" in sql_payload:
                        start_time = time.time()
                        self.session.request(method, url, json=test_payload, timeout=15)
                        elapsed = time.time() - start_time
                        if elapsed > 4:
                            self.add_vulnerability(
                                endpoint, "Time-based SQL Injection", "high",
                                f"Time-based SQL injection detected in {endpoint}",
                                f"Response delayed by {elapsed:.2f} seconds",
                                "Implement parameterized queries and proper error handling"
                            )
                            
                except requests.exceptions.RequestException as e:
                    logger.debug(f"Request failed for {endpoint}: {e}")
            
            # Test 2: Authentication bypass attempts
            bypass_payloads = [
                {"username": "admin", "password": ""},
                {"username": "", "password": ""},
                {"username": "admin' --", "password": "anything"},
                {"username": "admin", "password": "admin"},
                {"username": "administrator", "password": "administrator"}
            ]
            
            for bypass_payload in bypass_payloads:
                if method == "POST" and endpoint == "/auth/login":
                    try:
                        response = self.session.post(url, json=bypass_payload, timeout=5)
                        if response.status_code == 200 and "token" in response.text.lower():
                            self.add_vulnerability(
                                endpoint, "Authentication Bypass", "critical",
                                f"Potential authentication bypass with payload: {bypass_payload}",
                                f"Status: {response.status_code}, Response contains token",
                                "Implement strong authentication validation and rate limiting"
                            )
                    except requests.exceptions.RequestException:
                        pass

    def test_api_injection_vulnerabilities(self):
        """Test for various injection vulnerabilities in API endpoints"""
        logger.info("💉 Testing Injection Vulnerabilities...")
        
        # Common API endpoints to test
        api_endpoints = [
            ("/simulations", "POST", {"name": "test", "description": "test", "file_id": "test"}),
            ("/simulations/search", "GET", {"query": "test"}),
            ("/excel-parser/upload", "POST", {"filename": "test.xlsx"}),
            ("/users/profile", "GET", {"id": "1"}),
            ("/files/list", "GET", {"filter": "excel"}),
            ("/admin/users", "GET", {"search": "test"}),
            ("/admin/analytics", "GET", {"date_range": "7d"})
        ]
        
        for endpoint, method, params in api_endpoints:
            url = f"{self.api_base}{endpoint}"
            
            # Test SQL Injection
            for param_name, param_value in params.items():
                for sql_payload in self.sql_payloads:
                    test_params = params.copy()
                    test_params[param_name] = f"{param_value}{sql_payload}"
                    
                    try:
                        if method == "GET":
                            response = self.session.get(url, params=test_params, timeout=10)
                        else:
                            response = self.session.post(url, json=test_params, timeout=10)
                        
                        # Check for SQL errors
                        sql_errors = ["SQL syntax", "mysql_fetch", "PostgreSQL error", "ORA-", "sqlite3"]
                        if any(error in response.text.lower() for error in sql_errors) or response.status_code == 500:
                            self.add_vulnerability(
                                endpoint, "SQL Injection", "high",
                                f"SQL injection in parameter '{param_name}'",
                                f"Payload: {sql_payload}, Status: {response.status_code}",
                                "Use parameterized queries and input validation"
                            )
                            
                    except requests.exceptions.RequestException:
                        pass
            
            # Test Command Injection
            for param_name, param_value in params.items():
                for cmd_payload in self.command_injection_payloads[:2]:  # Test first 2
                    test_params = params.copy()
                    test_params[param_name] = f"{param_value}{cmd_payload}"
                    
                    try:
                        if method == "GET":
                            response = self.session.get(url, params=test_params, timeout=10)
                        else:
                            response = self.session.post(url, json=test_params, timeout=10)
                        
                        # Check for command execution indicators
                        cmd_indicators = ["root:", "uid=", "gid=", "/bin/", "/etc/passwd"]
                        if any(indicator in response.text for indicator in cmd_indicators):
                            self.add_vulnerability(
                                endpoint, "Command Injection", "critical",
                                f"Command injection in parameter '{param_name}'",
                                f"Payload: {cmd_payload}, Response contains system output",
                                "Implement strict input validation and avoid system calls"
                            )
                            
                    except requests.exceptions.RequestException:
                        pass

    def test_authorization_issues(self):
        """Test for authorization and access control issues"""
        logger.info("🔒 Testing Authorization & Access Control...")
        
        # Test endpoints that should require authentication
        protected_endpoints = [
            "/admin/users",
            "/admin/simulations", 
            "/admin/analytics",
            "/users/profile",
            "/simulations",
            "/files/upload"
        ]
        
        for endpoint in protected_endpoints:
            url = f"{self.api_base}{endpoint}"
            
            # Test 1: Access without authentication
            try:
                response = self.session.get(url, timeout=5)
                if response.status_code == 200:
                    self.add_vulnerability(
                        endpoint, "Missing Authentication", "high",
                        f"Endpoint accessible without authentication",
                        f"Status: {response.status_code}, No auth required",
                        "Implement proper authentication middleware"
                    )
                elif response.status_code not in [401, 403]:
                    self.add_vulnerability(
                        endpoint, "Improper Error Handling", "medium",
                        f"Unexpected response for unauthenticated request",
                        f"Status: {response.status_code} (expected 401/403)",
                        "Return proper HTTP status codes for auth failures"
                    )
            except requests.exceptions.RequestException:
                pass
            
            # Test 2: Access with invalid token
            invalid_tokens = [
                "Bearer invalid_token",
                "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.invalid",
                "Bearer " + "A" * 100,
                "invalid_format_token"
            ]
            
            for token in invalid_tokens:
                try:
                    headers = {"Authorization": token}
                    response = self.session.get(url, headers=headers, timeout=5)
                    
                    if response.status_code == 200:
                        self.add_vulnerability(
                            endpoint, "Token Validation Bypass", "critical",
                            f"Endpoint accessible with invalid token",
                            f"Token: {token[:50]}..., Status: {response.status_code}",
                            "Implement proper JWT validation and verification"
                        )
                except requests.exceptions.RequestException:
                    pass

    def test_input_validation(self):
        """Test input validation and sanitization"""
        logger.info("🧪 Testing Input Validation...")
        
        # Test various malformed inputs
        test_endpoints = [
            ("/simulations", "POST"),
            ("/excel-parser/upload", "POST"),
            ("/users/profile", "GET")
        ]
        
        malicious_inputs = [
            # XSS payloads
            {"name": "<script>alert('xss')</script>"},
            {"description": "<img src=x onerror=alert('xss')>"},
            
            # Path traversal
            {"filename": "../../../etc/passwd"},
            {"path": "..\\..\\windows\\system32\\config\\sam"},
            
            # Large inputs (buffer overflow attempts)
            {"data": "A" * 10000},
            {"name": "B" * 5000},
            
            # Special characters
            {"input": "'; DROP TABLE users; --"},
            {"data": "\x00\x01\x02\x03\x04"},
            
            # JSON injection
            {"json": '{"admin": true}'},
            {"user": '"},"admin":true,"x":"'}
        ]
        
        for endpoint, method in test_endpoints:
            url = f"{self.api_base}{endpoint}"
            
            for malicious_input in malicious_inputs:
                try:
                    if method == "GET":
                        response = self.session.get(url, params=malicious_input, timeout=10)
                    else:
                        response = self.session.post(url, json=malicious_input, timeout=10)
                    
                    # Check for reflected XSS
                    for key, value in malicious_input.items():
                        if "<script>" in str(value) and str(value) in response.text:
                            self.add_vulnerability(
                                endpoint, "Reflected XSS", "high",
                                f"XSS payload reflected in response",
                                f"Payload: {value}, Found in response",
                                "Implement proper output encoding and CSP"
                            )
                    
                    # Check for path traversal success
                    if "root:" in response.text or "Administrator" in response.text:
                        self.add_vulnerability(
                            endpoint, "Path Traversal", "critical",
                            f"Path traversal successful",
                            f"System files accessible, Response contains sensitive data",
                            "Implement proper file path validation and sanitization"
                        )
                    
                    # Check for buffer overflow indicators
                    if response.status_code == 500 and "memory" in response.text.lower():
                        self.add_vulnerability(
                            endpoint, "Buffer Overflow", "high",
                            f"Potential buffer overflow with large input",
                            f"500 error with memory-related message",
                            "Implement input length validation and proper error handling"
                        )
                        
                except requests.exceptions.RequestException:
                    pass

    def test_api_security_headers(self):
        """Test API security headers and CORS configuration"""
        logger.info("📋 Testing API Security Headers...")
        
        test_endpoints = ["/health", "/auth/login", "/simulations"]
        
        for endpoint in test_endpoints:
            url = f"{self.api_base}{endpoint}"
            
            try:
                response = self.session.get(url, timeout=5)
                headers = response.headers
                
                # Check for missing security headers
                required_headers = {
                    "X-Content-Type-Options": "nosniff",
                    "X-Frame-Options": ["DENY", "SAMEORIGIN"],
                    "X-XSS-Protection": "1; mode=block",
                    "Strict-Transport-Security": None,  # Should be present for HTTPS
                    "Content-Security-Policy": None
                }
                
                for header, expected_value in required_headers.items():
                    if header not in headers:
                        severity = "medium" if header == "Strict-Transport-Security" else "low"
                        self.add_vulnerability(
                            endpoint, "Missing Security Header", severity,
                            f"Missing security header: {header}",
                            f"Header not present in response",
                            f"Add {header} header to all API responses"
                        )
                    elif expected_value and headers.get(header) not in expected_value:
                        self.add_vulnerability(
                            endpoint, "Weak Security Header", "low",
                            f"Weak {header} configuration",
                            f"Current: {headers.get(header)}, Expected: {expected_value}",
                            f"Configure {header} header properly"
                        )
                
                # Test CORS configuration
                cors_headers = {
                    "Origin": "https://malicious-site.com",
                    "Access-Control-Request-Method": "POST",
                    "Access-Control-Request-Headers": "Content-Type"
                }
                
                cors_response = self.session.options(url, headers=cors_headers, timeout=5)
                
                if "Access-Control-Allow-Origin" in cors_response.headers:
                    allowed_origin = cors_response.headers["Access-Control-Allow-Origin"]
                    if allowed_origin == "*":
                        self.add_vulnerability(
                            endpoint, "Permissive CORS", "medium",
                            "CORS allows all origins (*)",
                            f"Access-Control-Allow-Origin: {allowed_origin}",
                            "Configure CORS to allow only trusted domains"
                        )
                    elif "malicious-site.com" in allowed_origin:
                        self.add_vulnerability(
                            endpoint, "CORS Misconfiguration", "high",
                            "CORS allows arbitrary origins",
                            f"Malicious origin accepted: {allowed_origin}",
                            "Implement strict CORS origin validation"
                        )
                        
            except requests.exceptions.RequestException:
                pass

    def test_rate_limiting(self):
        """Test rate limiting and DoS protection"""
        logger.info("⏱️ Testing Rate Limiting...")
        
        test_endpoints = [
            "/auth/login",
            "/auth/register", 
            "/simulations"
        ]
        
        for endpoint in test_endpoints:
            url = f"{self.api_base}{endpoint}"
            
            # Send rapid requests to test rate limiting
            responses = []
            start_time = time.time()
            
            for i in range(20):  # Send 20 requests rapidly
                try:
                    if endpoint == "/auth/login":
                        payload = {"username": f"test{i}", "password": "test"}
                        response = self.session.post(url, json=payload, timeout=2)
                    else:
                        response = self.session.get(url, timeout=2)
                    
                    responses.append((response.status_code, time.time() - start_time))
                    
                    if i < 19:  # Small delay between requests
                        time.sleep(0.1)
                        
                except requests.exceptions.RequestException:
                    responses.append((0, time.time() - start_time))
            
            # Analyze responses for rate limiting
            rate_limited_responses = [r for r in responses if r[0] == 429]
            successful_responses = [r for r in responses if r[0] == 200]
            
            if len(rate_limited_responses) == 0 and len(successful_responses) > 15:
                self.add_vulnerability(
                    endpoint, "Missing Rate Limiting", "medium",
                    f"No rate limiting detected on {endpoint}",
                    f"{len(successful_responses)}/20 requests succeeded without throttling",
                    "Implement rate limiting to prevent abuse and DoS attacks"
                )
            elif len(rate_limited_responses) > 0:
                logger.info(f"✅ Rate limiting detected on {endpoint} ({len(rate_limited_responses)}/20 requests throttled)")

    def test_information_disclosure(self):
        """Test for information disclosure vulnerabilities"""
        logger.info("📤 Testing Information Disclosure...")
        
        # Test various endpoints for information leakage
        test_cases = [
            ("/admin/debug", "GET", {}),
            ("/api/debug", "GET", {}),
            ("/.env", "GET", {}),
            ("/config", "GET", {}),
            ("/status", "GET", {}),
            ("/info", "GET", {}),
            ("/version", "GET", {}),
            ("/admin/phpinfo.php", "GET", {}),  # Common info disclosure
            ("/server-status", "GET", {}),
            ("/admin/config.json", "GET", {})
        ]
        
        for endpoint, method, params in test_cases:
            url = f"{self.base_url}{endpoint}"  # Some might be at root level
            
            try:
                response = self.session.request(method, url, json=params if method == "POST" else None, 
                                             params=params if method == "GET" else None, timeout=5)
                
                if response.status_code == 200:
                    # Check for sensitive information in response
                    sensitive_patterns = [
                        "password", "secret", "token", "key", "database",
                        "config", "debug", "error", "stack trace", "version",
                        "admin", "root", "mysql", "postgresql"
                    ]
                    
                    content_lower = response.text.lower()
                    found_patterns = [pattern for pattern in sensitive_patterns if pattern in content_lower]
                    
                    if found_patterns:
                        self.add_vulnerability(
                            endpoint, "Information Disclosure", "medium",
                            f"Sensitive information exposed at {endpoint}",
                            f"Status: 200, Contains: {', '.join(found_patterns[:3])}",
                            "Remove or secure information disclosure endpoints"
                        )
                        
            except requests.exceptions.RequestException:
                pass

    def generate_report(self):
        """Generate comprehensive API penetration test report"""
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
                "security_level": "HIGH" if risk_score < 10 else "MEDIUM" if risk_score < 30 else "LOW"
            },
            "test_categories": {
                "authentication_tested": True,
                "injection_tested": True,
                "authorization_tested": True,
                "input_validation_tested": True,
                "security_headers_tested": True,
                "rate_limiting_tested": True,
                "information_disclosure_tested": True
            },
            "recommendations": [
                "Implement comprehensive input validation and sanitization",
                "Use parameterized queries to prevent SQL injection",
                "Implement proper authentication and authorization controls",
                "Configure security headers on all API responses",
                "Implement rate limiting to prevent abuse",
                "Regular security testing and code reviews",
                "Use HTTPS for all API communications",
                "Implement proper error handling without information disclosure"
            ]
        })
        
        # Save detailed results
        with open("api_penetration_test_results.json", "w") as f:
            json.dump(self.test_results, f, indent=2)
        
        # Print summary
        logger.info("=" * 80)
        logger.info("API PENETRATION TEST RESULTS")
        logger.info("=" * 80)
        logger.info(f"Target API: {self.api_base}")
        logger.info(f"Total Vulnerabilities: {total_vulns}")
        logger.info(f"  Critical: {critical_vulns}")
        logger.info(f"  High: {high_vulns}")
        logger.info(f"  Medium: {medium_vulns}")
        logger.info(f"  Low: {low_vulns}")
        logger.info(f"Risk Score: {risk_score}/100")
        logger.info(f"Security Level: {self.test_results['summary']['security_level']}")
        logger.info(f"\nDetailed results saved to: api_penetration_test_results.json")
        logger.info("=" * 80)
        
        return self.test_results

    async def run_comprehensive_test(self):
        """Run all API penetration tests"""
        logger.info("🚀 Starting comprehensive API penetration test...")
        
        test_methods = [
            self.test_authentication_endpoints,
            self.test_api_injection_vulnerabilities,
            self.test_authorization_issues,
            self.test_input_validation,
            self.test_api_security_headers,
            self.test_rate_limiting,
            self.test_information_disclosure
        ]
        
        for test_method in test_methods:
            try:
                test_method()
            except Exception as e:
                logger.error(f"Error in {test_method.__name__}: {e}")
        
        return self.generate_report()


async def main():
    """Main function to run API penetration tests"""
    import argparse
    
    parser = argparse.ArgumentParser(description="API Penetration Testing Suite")
    parser.add_argument("--url", default="http://localhost:9090", help="Target base URL")
    parser.add_argument("--output", default="api_penetration_test_results.json", help="Output file")
    
    args = parser.parse_args()
    
    tester = APIPenetrationTester(args.url)
    results = await tester.run_comprehensive_test()
    
    # Return exit code based on findings
    critical_count = results["summary"]["critical_vulnerabilities"]
    high_count = results["summary"]["high_vulnerabilities"]
    
    if critical_count > 0:
        return 2  # Critical vulnerabilities found
    elif high_count > 0:
        return 1  # High vulnerabilities found
    else:
        return 0  # No critical or high vulnerabilities


if __name__ == "__main__":
    import asyncio
    exit_code = asyncio.run(main())
    exit(exit_code)
