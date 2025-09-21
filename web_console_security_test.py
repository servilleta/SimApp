#!/usr/bin/env python3
"""
Web Console Security Testing Suite
Tests for browser console vulnerabilities, client-side code exposure, and console-based attacks
"""

import requests
import re
import json
import logging
import subprocess
from datetime import datetime
from typing import Dict, List, Any, Optional
from urllib.parse import urljoin, urlparse
import base64
import hashlib

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("web_console_security")

class WebConsoleSecurity:
    """Comprehensive web console security testing"""
    
    def __init__(self, base_url: str = "http://localhost:9090"):
        self.base_url = base_url
        self.session = requests.Session()
        self.vulnerabilities = []
        self.test_results = {
            "timestamp": datetime.utcnow().isoformat(),
            "target": self.base_url,
            "vulnerabilities": [],
            "exposed_secrets": [],
            "client_side_issues": [],
            "console_risks": []
        }
        
        logger.info(f"🌐 Initialized Web Console Security Tester for {self.base_url}")

    def add_vulnerability(self, category: str, vuln_type: str, severity: str, 
                         description: str, evidence: str, remediation: str):
        """Add a vulnerability finding"""
        vuln = {
            "category": category,
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
        logger.warning(f"{severity_emoji.get(severity, '❔')} {severity.upper()} - {category}: {description}")

    def test_frontend_source_exposure(self):
        """Test for exposed sensitive information in frontend source code"""
        logger.info("🔍 Testing frontend source code for sensitive exposure...")
        
        # Common frontend paths to check
        frontend_paths = [
            "/",
            "/index.html",
            "/static/js/main.js",
            "/static/js/app.js",
            "/static/js/bundle.js",
            "/assets/index.js",
            "/assets/main.js",
            "/js/main.js",
            "/js/app.js"
        ]
        
        # Patterns to look for sensitive data
        sensitive_patterns = {
            "api_keys": [
                r"api[_-]?key[\"\'`]?\s*[:=]\s*[\"\'`]([a-zA-Z0-9_-]{20,})[\"\'`]",
                r"apikey[\"\'`]?\s*[:=]\s*[\"\'`]([a-zA-Z0-9_-]{20,})[\"\'`]",
                r"secret[_-]?key[\"\'`]?\s*[:=]\s*[\"\'`]([a-zA-Z0-9_-]{20,})[\"\'`]"
            ],
            "tokens": [
                r"token[\"\'`]?\s*[:=]\s*[\"\'`]([a-zA-Z0-9._-]{20,})[\"\'`]",
                r"jwt[\"\'`]?\s*[:=]\s*[\"\'`]([a-zA-Z0-9._-]{20,})[\"\'`]",
                r"bearer[\"\'`]?\s*[:=]\s*[\"\'`]([a-zA-Z0-9._-]{20,})[\"\'`]"
            ],
            "passwords": [
                r"password[\"\'`]?\s*[:=]\s*[\"\'`]([^\"\'`\s]{6,})[\"\'`]",
                r"pass[\"\'`]?\s*[:=]\s*[\"\'`]([^\"\'`\s]{6,})[\"\'`]",
                r"pwd[\"\'`]?\s*[:=]\s*[\"\'`]([^\"\'`\s]{6,})[\"\'`]"
            ],
            "database_urls": [
                r"database[_-]?url[\"\'`]?\s*[:=]\s*[\"\'`]([^\"\'`\s]+)[\"\'`]",
                r"db[_-]?url[\"\'`]?\s*[:=]\s*[\"\'`]([^\"\'`\s]+)[\"\'`]",
                r"mongodb://[^\"\'`\s]+",
                r"postgresql://[^\"\'`\s]+",
                r"mysql://[^\"\'`\s]+"
            ],
            "internal_urls": [
                r"localhost:\d+",
                r"127\.0\.0\.1:\d+",
                r"192\.168\.\d+\.\d+",
                r"10\.\d+\.\d+\.\d+",
                r"172\.(1[6-9]|2[0-9]|3[0-1])\.\d+\.\d+"
            ],
            "debug_info": [
                r"console\.log\([\"\'`].*[\"\'`]\)",
                r"console\.error\([\"\'`].*[\"\'`]\)",
                r"console\.warn\([\"\'`].*[\"\'`]\)",
                r"debugger;",
                r"\.stackTrace",
                r"__REACT_DEVTOOLS_GLOBAL_HOOK__"
            ]
        }
        
        for path in frontend_paths:
            try:
                url = urljoin(self.base_url, path)
                response = self.session.get(url, timeout=10)
                
                if response.status_code == 200:
                    content = response.text
                    
                    # Check for sensitive patterns
                    for category, patterns in sensitive_patterns.items():
                        for pattern in patterns:
                            matches = re.findall(pattern, content, re.IGNORECASE)
                            
                            if matches:
                                for match in matches[:5]:  # Limit to 5 matches per pattern
                                    if category in ["api_keys", "tokens", "passwords"]:
                                        severity = "critical"
                                        evidence = f"Found in {path}: {pattern} -> {match[:10]}..."
                                        remediation = "Remove sensitive data from client-side code, use environment variables server-side"
                                    elif category == "database_urls":
                                        severity = "high"
                                        evidence = f"Found in {path}: {match}"
                                        remediation = "Remove database URLs from client-side code"
                                    elif category == "internal_urls":
                                        severity = "medium"
                                        evidence = f"Found in {path}: {match}"
                                        remediation = "Remove internal URLs from client-side code"
                                    else:  # debug_info
                                        severity = "low"
                                        evidence = f"Found in {path}: {match}"
                                        remediation = "Remove debug code from production builds"
                                    
                                    self.add_vulnerability(
                                        "Source Code Exposure", f"Exposed {category}",
                                        severity, f"Sensitive {category} found in client-side code",
                                        evidence, remediation
                                    )
                    
                    # Check for source maps
                    if "sourceMappingURL" in content:
                        self.add_vulnerability(
                            "Source Code Exposure", "Source Maps Enabled",
                            "medium", "Source maps are enabled in production",
                            f"Found sourceMappingURL in {path}",
                            "Disable source maps in production builds"
                        )
                    
                    # Check for eval() usage
                    if re.search(r"eval\s*\(", content):
                        self.add_vulnerability(
                            "Code Injection Risk", "eval() Usage",
                            "high", "eval() function usage detected",
                            f"Found eval() in {path}",
                            "Avoid eval() usage, use safer alternatives"
                        )
                        
            except Exception as e:
                logger.debug(f"Error checking {path}: {e}")

    def test_console_injection_vectors(self):
        """Test for console-based injection vulnerabilities"""
        logger.info("🎯 Testing console injection vectors...")
        
        # Test for DOM manipulation via console
        injection_payloads = [
            "document.body.innerHTML='<script>alert(\"XSS\")</script>'",
            "window.location='javascript:alert(\"XSS\")'",
            "document.cookie",
            "localStorage",
            "sessionStorage",
            "window.fetch('/api/users')",
            "XMLHttpRequest.prototype.open",
            "window.history",
            "navigator.userAgent"
        ]
        
        # These would typically require browser automation
        # For now, we'll check if the frontend has protections against console manipulation
        try:
            response = self.session.get(self.base_url)
            if response.status_code == 200:
                content = response.text
                
                # Check for Content Security Policy
                csp_header = response.headers.get('Content-Security-Policy', '')
                if not csp_header:
                    self.add_vulnerability(
                        "Console Security", "Missing CSP",
                        "medium", "No Content Security Policy to prevent console-based attacks",
                        "CSP header not found",
                        "Implement Content Security Policy with 'unsafe-eval' restrictions"
                    )
                elif "'unsafe-eval'" in csp_header:
                    self.add_vulnerability(
                        "Console Security", "Unsafe CSP",
                        "high", "CSP allows 'unsafe-eval' which enables console code execution",
                        f"CSP contains 'unsafe-eval': {csp_header}",
                        "Remove 'unsafe-eval' from Content Security Policy"
                    )
                
                # Check for global variables that could be manipulated
                global_var_patterns = [
                    r"window\.[a-zA-Z_]\w*\s*=",
                    r"globalThis\.[a-zA-Z_]\w*\s*=",
                    r"var\s+[a-zA-Z_]\w*\s*=.*window",
                ]
                
                for pattern in global_var_patterns:
                    if re.search(pattern, content):
                        self.add_vulnerability(
                            "Console Security", "Global Variable Exposure",
                            "low", "Global variables exposed that could be manipulated via console",
                            f"Pattern found: {pattern}",
                            "Minimize global variable usage and validate global state"
                        )
                        break
                        
        except Exception as e:
            logger.debug(f"Error testing console injection: {e}")

    def test_client_side_authentication(self):
        """Test for client-side authentication vulnerabilities"""
        logger.info("🔐 Testing client-side authentication security...")
        
        try:
            response = self.session.get(self.base_url)
            if response.status_code == 200:
                content = response.text
                
                # Check for client-side authentication logic
                auth_patterns = [
                    r"if\s*\([^)]*\.role\s*===?\s*[\"\'`]admin[\"\'`]",
                    r"if\s*\([^)]*\.isAdmin\s*===?\s*true",
                    r"if\s*\([^)]*\.permissions\.",
                    r"localStorage\.getItem\([\"\'`]token[\"\'`]\)",
                    r"sessionStorage\.getItem\([\"\'`]token[\"\'`]\)",
                    r"document\.cookie\.indexOf\([\"\'`]auth",
                ]
                
                for pattern in auth_patterns:
                    if re.search(pattern, content, re.IGNORECASE):
                        self.add_vulnerability(
                            "Authentication Security", "Client-Side Auth Logic",
                            "high", "Authentication logic implemented on client-side",
                            f"Pattern found: {pattern}",
                            "Move authentication logic to server-side, client-side should only display UI"
                        )
                
                # Check for hardcoded credentials
                cred_patterns = [
                    r"username[\"\'`]?\s*[:=]\s*[\"\'`]admin[\"\'`]",
                    r"password[\"\'`]?\s*[:=]\s*[\"\'`][^\"\'`]{6,}[\"\'`]",
                    r"defaultPassword",
                    r"adminPass",
                ]
                
                for pattern in cred_patterns:
                    if re.search(pattern, content, re.IGNORECASE):
                        self.add_vulnerability(
                            "Authentication Security", "Hardcoded Credentials",
                            "critical", "Hardcoded credentials found in client-side code",
                            f"Pattern found: {pattern}",
                            "Remove hardcoded credentials, use proper authentication flow"
                        )
                        
        except Exception as e:
            logger.debug(f"Error testing client-side auth: {e}")

    def test_javascript_tampering(self):
        """Test for JavaScript tampering vulnerabilities"""
        logger.info("🔧 Testing JavaScript tampering risks...")
        
        try:
            response = self.session.get(self.base_url)
            if response.status_code == 200:
                content = response.text
                
                # Check for function overriding protections
                protection_patterns = [
                    r"Object\.freeze\(",
                    r"Object\.seal\(",
                    r"Object\.preventExtensions\(",
                    r"\.hasOwnProperty\s*=",
                    r"function.*\{\s*[\"\'`]use strict[\"\'`]"
                ]
                
                has_protections = any(re.search(pattern, content) for pattern in protection_patterns)
                
                if not has_protections:
                    self.add_vulnerability(
                        "Code Tampering", "No Anti-Tampering Protections",
                        "medium", "No JavaScript anti-tampering protections detected",
                        "Missing Object.freeze, strict mode, or function protection",
                        "Implement JavaScript protection mechanisms against tampering"
                    )
                
                # Check for critical function exposure
                critical_functions = [
                    r"window\.fetch\s*=",
                    r"XMLHttpRequest\.prototype\.open\s*=",
                    r"console\.log\s*=",
                    r"JSON\.parse\s*=",
                    r"eval\s*=",
                ]
                
                for pattern in critical_functions:
                    if re.search(pattern, content):
                        self.add_vulnerability(
                            "Code Tampering", "Critical Function Override",
                            "high", "Critical JavaScript functions can be overridden",
                            f"Pattern found: {pattern}",
                            "Protect critical functions from tampering"
                        )
                        
        except Exception as e:
            logger.debug(f"Error testing JavaScript tampering: {e}")

    def test_browser_storage_security(self):
        """Test for browser storage security issues"""
        logger.info("💾 Testing browser storage security...")
        
        try:
            response = self.session.get(self.base_url)
            if response.status_code == 200:
                content = response.text
                
                # Check for sensitive data in localStorage/sessionStorage
                storage_patterns = [
                    r"localStorage\.setItem\([\"\'`][^\"\'`]*token[^\"\'`]*[\"\'`]",
                    r"localStorage\.setItem\([\"\'`][^\"\'`]*password[^\"\'`]*[\"\'`]",
                    r"localStorage\.setItem\([\"\'`][^\"\'`]*secret[^\"\'`]*[\"\'`]",
                    r"sessionStorage\.setItem\([\"\'`][^\"\'`]*token[^\"\'`]*[\"\'`]",
                    r"sessionStorage\.setItem\([\"\'`][^\"\'`]*password[^\"\'`]*[\"\'`]",
                ]
                
                for pattern in storage_patterns:
                    if re.search(pattern, content, re.IGNORECASE):
                        if "password" in pattern:
                            severity = "critical"
                        elif "token" in pattern:
                            severity = "high"
                        else:
                            severity = "medium"
                            
                        self.add_vulnerability(
                            "Storage Security", "Sensitive Data in Browser Storage",
                            severity, "Sensitive data stored in browser storage",
                            f"Pattern found: {pattern}",
                            "Avoid storing sensitive data in localStorage/sessionStorage"
                        )
                
                # Check for cookie security
                cookie_header = response.headers.get('Set-Cookie', '')
                if cookie_header:
                    if 'Secure' not in cookie_header:
                        self.add_vulnerability(
                            "Storage Security", "Insecure Cookies",
                            "medium", "Cookies missing Secure flag",
                            f"Set-Cookie: {cookie_header}",
                            "Add Secure flag to cookies"
                        )
                    
                    if 'HttpOnly' not in cookie_header:
                        self.add_vulnerability(
                            "Storage Security", "JavaScript-Accessible Cookies",
                            "medium", "Cookies missing HttpOnly flag",
                            f"Set-Cookie: {cookie_header}",
                            "Add HttpOnly flag to sensitive cookies"
                        )
                        
        except Exception as e:
            logger.debug(f"Error testing browser storage: {e}")

    def test_dev_tools_exploitation(self):
        """Test for developer tools exploitation risks"""
        logger.info("🛠️ Testing developer tools exploitation risks...")
        
        try:
            response = self.session.get(self.base_url)
            if response.status_code == 200:
                content = response.text
                
                # Check for dev tools detection/blocking
                devtools_patterns = [
                    r"devtools",
                    r"developer.*tools",
                    r"F12.*detect",
                    r"inspect.*element",
                    r"console.*detect"
                ]
                
                has_detection = any(re.search(pattern, content, re.IGNORECASE) for pattern in devtools_patterns)
                
                if not has_detection:
                    self.add_vulnerability(
                        "Developer Tools", "No DevTools Protection",
                        "low", "No developer tools detection or protection",
                        "Missing devtools detection mechanisms",
                        "Consider implementing developer tools detection for sensitive applications"
                    )
                
                # Check for debug mode indicators
                debug_patterns = [
                    r"debug\s*[:=]\s*true",
                    r"development.*mode",
                    r"\.env\.development",
                    r"process\.env\.NODE_ENV.*development"
                ]
                
                for pattern in debug_patterns:
                    if re.search(pattern, content, re.IGNORECASE):
                        self.add_vulnerability(
                            "Developer Tools", "Debug Mode Enabled",
                            "medium", "Debug mode appears to be enabled in production",
                            f"Pattern found: {pattern}",
                            "Disable debug mode in production builds"
                        )
                        
        except Exception as e:
            logger.debug(f"Error testing dev tools: {e}")

    def test_api_exposure_via_console(self):
        """Test for API endpoints that could be exploited via console"""
        logger.info("🔌 Testing API exposure via console...")
        
        try:
            response = self.session.get(self.base_url)
            if response.status_code == 200:
                content = response.text
                
                # Look for API endpoints in JavaScript
                api_patterns = [
                    r"[\"\'`]/api/[^\"\'`\s]+[\"\'`]",
                    r"fetch\([\"\'`][^\"\'`\s]+[\"\'`]",
                    r"axios\.[a-z]+\([\"\'`][^\"\'`\s]+[\"\'`]",
                    r"\.get\([\"\'`][^\"\'`\s]+[\"\'`]",
                    r"\.post\([\"\'`][^\"\'`\s]+[\"\'`]"
                ]
                
                exposed_endpoints = set()
                for pattern in api_patterns:
                    matches = re.findall(pattern, content)
                    for match in matches:
                        # Clean up the match
                        endpoint = match.strip('"\'`')
                        if endpoint.startswith('/') or endpoint.startswith('http'):
                            exposed_endpoints.add(endpoint)
                
                if exposed_endpoints:
                    self.test_results["client_side_issues"].append({
                        "type": "API Endpoints Exposed",
                        "count": len(exposed_endpoints),
                        "endpoints": list(exposed_endpoints)[:10],  # Limit to 10
                        "risk": "Endpoints can be called directly from browser console"
                    })
                    
                    if len(exposed_endpoints) > 20:
                        self.add_vulnerability(
                            "API Exposure", "Many API Endpoints Exposed",
                            "medium", f"{len(exposed_endpoints)} API endpoints exposed in client-side code",
                            f"Endpoints include: {', '.join(list(exposed_endpoints)[:5])}...",
                            "Minimize API endpoint exposure in client-side code"
                        )
                        
        except Exception as e:
            logger.debug(f"Error testing API exposure: {e}")

    def generate_console_security_report(self):
        """Generate comprehensive console security report"""
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
                "console_hackable": critical_vulns > 0 or high_vulns > 2
            },
            "recommendations": [
                "Implement Content Security Policy with strict 'unsafe-eval' restrictions",
                "Remove sensitive data from client-side code",
                "Use server-side authentication validation only",
                "Minimize API endpoint exposure in frontend code",
                "Implement JavaScript anti-tampering protections",
                "Use secure browser storage practices",
                "Disable source maps in production",
                "Remove debug code from production builds"
            ]
        })
        
        # Save results
        with open("web_console_security_results.json", "w") as f:
            json.dump(self.test_results, f, indent=2)
        
        # Print summary
        logger.info("=" * 80)
        logger.info("WEB CONSOLE SECURITY TEST RESULTS")
        logger.info("=" * 80)
        logger.info(f"Target: {self.base_url}")
        logger.info(f"Total Vulnerabilities: {total_vulns}")
        logger.info(f"  Critical: {critical_vulns}")
        logger.info(f"  High: {high_vulns}")
        logger.info(f"  Medium: {medium_vulns}")
        logger.info(f"  Low: {low_vulns}")
        logger.info(f"Risk Score: {risk_score}/100")
        
        # Console hack assessment
        if self.test_results["summary"]["console_hackable"]:
            logger.warning("⚠️  CONSOLE EXPLOITATION POSSIBLE")
        else:
            logger.info("✅ Console exploitation risk is LOW")
        
        logger.info(f"\nDetailed results saved to: web_console_security_results.json")
        logger.info("=" * 80)
        
        return self.test_results

    async def run_comprehensive_test(self):
        """Run all web console security tests"""
        logger.info("🚀 Starting comprehensive web console security test...")
        
        test_methods = [
            self.test_frontend_source_exposure,
            self.test_console_injection_vectors,
            self.test_client_side_authentication,
            self.test_javascript_tampering,
            self.test_browser_storage_security,
            self.test_dev_tools_exploitation,
            self.test_api_exposure_via_console
        ]
        
        for test_method in test_methods:
            try:
                test_method()
            except Exception as e:
                logger.error(f"Error in {test_method.__name__}: {e}")
        
        return self.generate_console_security_report()


async def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Web Console Security Testing")
    parser.add_argument("--url", default="http://localhost:9090", help="Target URL")
    parser.add_argument("--output", default="web_console_security_results.json", help="Output file")
    
    args = parser.parse_args()
    
    tester = WebConsoleSecurity(args.url)
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
