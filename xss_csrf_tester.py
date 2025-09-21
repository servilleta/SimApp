#!/usr/bin/env python3
"""
XSS and CSRF Testing Tool for Monte Carlo Platform
Comprehensive Cross-Site Scripting and Cross-Site Request Forgery testing
"""

import requests
import json
import logging
import random
import string
import urllib.parse
from datetime import datetime
from typing import Dict, List, Any, Optional
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("xss_csrf_tester")

# Optional Selenium imports - fallback if not available
try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False
    logger.info("Selenium not available - skipping browser-based tests")

class XSSCSRFTester:
    """Comprehensive XSS and CSRF testing"""
    
    def __init__(self, base_url: str = "http://localhost:9090"):
        self.base_url = base_url
        self.api_base = f"{base_url}/api"
        self.test_results = {
            "timestamp": datetime.utcnow().isoformat(),
            "target": base_url,
            "xss_vulnerabilities": [],
            "csrf_vulnerabilities": [],
            "tested_endpoints": []
        }
        
        # XSS test payloads
        self.xss_payloads = [
            # Basic XSS payloads
            "<script>alert('XSS')</script>",
            "<img src=x onerror=alert('XSS')>",
            "<svg onload=alert('XSS')>",
            "<iframe src=javascript:alert('XSS')>",
            "<body onload=alert('XSS')>",
            
            # Event handler XSS
            "'\"><script>alert('XSS')</script>",
            "'\"><img src=x onerror=alert('XSS')>",
            "javascript:alert('XSS')",
            "vbscript:alert('XSS')",
            "onmouseover=alert('XSS')",
            
            # HTML entity encoded XSS
            "&lt;script&gt;alert('XSS')&lt;/script&gt;",
            "&#60;script&#62;alert('XSS')&#60;/script&#62;",
            "%3Cscript%3Ealert('XSS')%3C/script%3E",
            
            # Filter bypass attempts
            "<ScRiPt>alert('XSS')</ScRiPt>",
            "<script>alert(String.fromCharCode(88,83,83))</script>",
            "<script>eval('al'+'ert(\"XSS\")')</script>",
            "<script>window['al'+'ert']('XSS')</script>",
            
            # Advanced XSS payloads
            "<svg><animatetransform onbegin=alert('XSS')>",
            "<math><mi//xlink:href=\"data:x,<script>alert('XSS')</script>\">",
            "<table background=\"javascript:alert('XSS')\">",
            "<object data=\"data:text/html,<script>alert('XSS')</script>\">",
            
            # DOM-based XSS
            "#<script>alert('XSS')</script>",
            "?q=<script>alert('XSS')</script>",
            "javascript:alert('XSS')//",
            
            # AngularJS/Vue.js template injection
            "{{constructor.constructor('alert(\"XSS\")')()}}",
            "{{$on.constructor('alert(\"XSS\")')()}}",
            "${alert('XSS')}",
            "#{alert('XSS')}",
            
            # React XSS
            "dangerouslySetInnerHTML={{__html: '<script>alert(\"XSS\")</script>'}}",
            
            # Content-Type confusion
            "<script>alert('XSS')</script>",
            
            # Filter evasion
            "<script>al\\x65rt('XSS')</script>",
            "<script>\\u0061lert('XSS')</script>",
            "<script>eval(unescape('%61%6c%65%72%74%28%22%58%53%53%22%29'))</script>",
            
            # Polyglot payloads
            "jaVasCript:/*-/*`/*\\`/*'/*\"/**/(/* */oNcliCk=alert('XSS') )//%0D%0A%0d%0a//</stYle/</titLe/</teXtarEa/</scRipt/--!>\\x3csVg/<sVg/oNloAd=alert('XSS')//\\x3e",
            
            # WAF bypass attempts
            "<script>alert`XSS`</script>",
            "<script>(alert)('XSS')</script>",
            "<script>setTimeout(\"alert('XSS')\",0)</script>",
            "<script>setInterval(\"alert('XSS')\",1000)</script>"
        ]
    
    def add_xss_vulnerability(self, endpoint: str, payload: str, context: str, evidence: str):
        """Add XSS vulnerability to results"""
        self.test_results["xss_vulnerabilities"].append({
            "endpoint": endpoint,
            "payload": payload,
            "context": context,
            "evidence": evidence,
            "timestamp": datetime.utcnow().isoformat()
        })
    
    def add_csrf_vulnerability(self, endpoint: str, method: str, evidence: str):
        """Add CSRF vulnerability to results"""
        self.test_results["csrf_vulnerabilities"].append({
            "endpoint": endpoint,
            "method": method,
            "evidence": evidence,
            "timestamp": datetime.utcnow().isoformat()
        })
    
    def test_reflected_xss(self, endpoint: str, parameter: str, method: str = "GET"):
        """Test for reflected XSS vulnerabilities"""
        logger.info(f"Testing reflected XSS on {method} {endpoint} parameter '{parameter}'")
        
        vulnerabilities = []
        
        for payload in self.xss_payloads:
            try:
                if method.upper() == "GET":
                    response = requests.get(f"{self.base_url}{endpoint}", 
                                          params={parameter: payload}, timeout=10)
                elif method.upper() == "POST":
                    response = requests.post(f"{self.api_base}{endpoint}", 
                                           json={parameter: payload}, timeout=10)
                else:
                    continue
                
                # Check if payload is reflected in response
                if payload in response.text:
                    # Additional check for HTML context
                    if response.headers.get('content-type', '').startswith('text/html'):
                        self.add_xss_vulnerability(
                            endpoint, payload, "reflected", 
                            f"Payload reflected in HTML response: {response.text[:200]}"
                        )
                        vulnerabilities.append(payload)
                        logger.warning(f"Reflected XSS found: {payload}")
                    elif 'application/json' in response.headers.get('content-type', ''):
                        # JSON context - check if properly encoded
                        if f'"{payload}"' not in response.text:
                            self.add_xss_vulnerability(
                                endpoint, payload, "json_reflected",
                                f"Unescaped payload in JSON: {response.text[:200]}"
                            )
                            vulnerabilities.append(payload)
                
                time.sleep(0.1)  # Rate limiting
                
            except Exception as e:
                logger.debug(f"XSS test failed for payload {payload}: {e}")
        
        return vulnerabilities
    
    def test_stored_xss_with_selenium(self):
        """Test for stored XSS using browser automation"""
        if not SELENIUM_AVAILABLE:
            logger.info("Selenium not available - skipping browser-based XSS tests")
            return
            
        logger.info("Testing stored XSS with browser automation...")
        
        # Setup Chrome options for headless testing
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        
        try:
            driver = webdriver.Chrome(options=chrome_options)
            driver.set_page_load_timeout(30)
            
            # Test stored XSS in user registration
            try:
                driver.get(f"{self.base_url}/register")
                time.sleep(2)
                
                # Try XSS in username field
                for payload in self.xss_payloads[:10]:  # Test first 10 payloads
                    try:
                        # Clear and fill form
                        username_field = driver.find_element(By.NAME, "username")
                        email_field = driver.find_element(By.NAME, "email")
                        password_field = driver.find_element(By.NAME, "password")
                        
                        username_field.clear()
                        username_field.send_keys(payload)
                        
                        email_field.clear()
                        email_field.send_keys(f"test{random.randint(1000,9999)}@test.com")
                        
                        password_field.clear()
                        password_field.send_keys("Test123!")
                        
                        # Submit form
                        submit_button = driver.find_element(By.TYPE, "submit")
                        submit_button.click()
                        
                        time.sleep(2)
                        
                        # Check for alert (XSS execution)
                        try:
                            alert = driver.switch_to.alert
                            alert_text = alert.text
                            alert.accept()
                            
                            if "XSS" in alert_text:
                                self.add_xss_vulnerability(
                                    "/register", payload, "stored_dom",
                                    f"Stored XSS triggered alert: {alert_text}"
                                )
                                logger.warning(f"Stored XSS found in registration: {payload}")
                                
                        except:
                            # No alert - check if payload is in page source
                            if payload in driver.page_source:
                                self.add_xss_vulnerability(
                                    "/register", payload, "stored_reflected",
                                    "Payload stored and reflected in page source"
                                )
                        
                    except Exception as e:
                        logger.debug(f"Selenium XSS test failed for {payload}: {e}")
                        
            except Exception as e:
                logger.debug(f"Registration XSS test failed: {e}")
            
        except Exception as e:
            logger.warning(f"Selenium setup failed: {e}")
            logger.info("Skipping browser-based XSS tests")
            return
        finally:
            try:
                driver.quit()
            except:
                pass
    
    def test_dom_xss(self):
        """Test for DOM-based XSS vulnerabilities"""
        logger.info("Testing DOM-based XSS...")
        
        # Common DOM XSS sinks to test
        dom_test_urls = [
            f"{self.base_url}/#<script>alert('XSS')</script>",
            f"{self.base_url}/?search=<script>alert('XSS')</script>",
            f"{self.base_url}/search?q=<script>alert('XSS')</script>",
            f"{self.base_url}/?redirect=javascript:alert('XSS')",
            f"{self.base_url}/?callback=<script>alert('XSS')</script>"
        ]
        
        for test_url in dom_test_urls:
            try:
                response = requests.get(test_url, timeout=10)
                
                # Check for dangerous JavaScript patterns
                dangerous_patterns = [
                    "document.write(",
                    "innerHTML",
                    "outerHTML", 
                    "eval(",
                    "setTimeout(",
                    "setInterval(",
                    "location.href",
                    "window.location"
                ]
                
                for pattern in dangerous_patterns:
                    if pattern in response.text and "<script>" in test_url:
                        self.add_xss_vulnerability(
                            test_url, "<script>alert('XSS')</script>", "dom",
                            f"Dangerous pattern '{pattern}' found with user input"
                        )
                        break
                        
            except Exception as e:
                logger.debug(f"DOM XSS test failed for {test_url}: {e}")
    
    def test_csrf_protection(self):
        """Test CSRF protection mechanisms"""
        logger.info("Testing CSRF protection...")
        
        # Test endpoints that should require CSRF protection
        csrf_test_endpoints = [
            ("/api/auth/register", "POST", {"username": "test", "email": "test@test.com", "password": "Test123!"}),
            ("/api/auth/login", "POST", {"username": "admin", "password": "password"}),
            ("/api/simulations", "POST", {"name": "test", "description": "test"}),
            ("/api/excel-parser/upload", "POST", {}),
            ("/api/auth/logout", "POST", {}),
        ]
        
        for endpoint, method, data in csrf_test_endpoints:
            try:
                # Test without any CSRF token
                if method == "POST":
                    response = requests.post(f"{self.base_url}{endpoint}", json=data, timeout=10)
                    
                    # If request succeeds, check if CSRF protection is missing
                    if response.status_code in [200, 201]:
                        # Check response headers for CSRF token requirements
                        csrf_headers = [
                            "x-csrf-token", "csrf-token", "x-xsrf-token", 
                            "xsrf-token", "_token", "authenticity_token"
                        ]
                        
                        csrf_found = any(header in response.headers for header in csrf_headers)
                        
                        if not csrf_found:
                            # Check if request body or response indicates CSRF requirement
                            csrf_in_response = any(token in response.text.lower() 
                                                 for token in ["csrf", "xsrf", "_token"])
                            
                            if not csrf_in_response:
                                self.add_csrf_vulnerability(
                                    endpoint, method,
                                    f"State-changing operation succeeded without CSRF token. Status: {response.status_code}"
                                )
                                logger.warning(f"CSRF protection missing on {endpoint}")
                
                # Test CSRF with different origins
                headers = {
                    "Origin": "http://evil.example.com",
                    "Referer": "http://evil.example.com/attack.html"
                }
                
                response = requests.post(f"{self.base_url}{endpoint}", json=data, 
                                       headers=headers, timeout=10)
                
                if response.status_code in [200, 201]:
                    self.add_csrf_vulnerability(
                        endpoint, method,
                        f"Request accepted from different origin. Status: {response.status_code}"
                    )
                    logger.warning(f"Cross-origin request accepted on {endpoint}")
                
            except Exception as e:
                logger.debug(f"CSRF test failed for {endpoint}: {e}")
    
    def test_clickjacking_protection(self):
        """Test clickjacking protection"""
        logger.info("Testing clickjacking protection...")
        
        try:
            response = requests.get(self.base_url, timeout=10)
            
            # Check for X-Frame-Options header
            x_frame_options = response.headers.get("X-Frame-Options")
            if not x_frame_options:
                self.add_xss_vulnerability(
                    "/", "iframe_embedding", "clickjacking",
                    "Missing X-Frame-Options header - page can be embedded in iframe"
                )
                logger.warning("Missing X-Frame-Options header")
            elif x_frame_options.upper() not in ["DENY", "SAMEORIGIN"]:
                self.add_xss_vulnerability(
                    "/", "iframe_embedding", "clickjacking",
                    f"Weak X-Frame-Options value: {x_frame_options}"
                )
            
            # Check for Content-Security-Policy frame-ancestors
            csp = response.headers.get("Content-Security-Policy", "")
            if "frame-ancestors" not in csp:
                self.add_xss_vulnerability(
                    "/", "iframe_embedding", "clickjacking",
                    "Missing frame-ancestors directive in CSP"
                )
                
        except Exception as e:
            logger.debug(f"Clickjacking test failed: {e}")
    
    def test_content_security_policy(self):
        """Test Content Security Policy implementation"""
        logger.info("Testing Content Security Policy...")
        
        try:
            response = requests.get(self.base_url, timeout=10)
            
            csp = response.headers.get("Content-Security-Policy")
            if not csp:
                self.add_xss_vulnerability(
                    "/", "missing_csp", "csp",
                    "Missing Content-Security-Policy header"
                )
                logger.warning("Missing CSP header")
                return
            
            # Check for unsafe CSP directives
            unsafe_patterns = [
                "'unsafe-inline'",
                "'unsafe-eval'",
                "data:",
                "'unsafe-hashes'",
                "*"  # Wildcard sources
            ]
            
            for pattern in unsafe_patterns:
                if pattern in csp:
                    self.add_xss_vulnerability(
                        "/", f"unsafe_csp_{pattern}", "csp",
                        f"Unsafe CSP directive found: {pattern}"
                    )
                    logger.warning(f"Unsafe CSP directive: {pattern}")
            
            # Check for missing important directives
            important_directives = [
                "default-src", "script-src", "style-src", "img-src",
                "connect-src", "font-src", "object-src", "media-src",
                "frame-src", "worker-src", "child-src", "form-action",
                "frame-ancestors", "base-uri"
            ]
            
            missing_directives = []
            for directive in important_directives:
                if directive not in csp:
                    missing_directives.append(directive)
            
            if missing_directives:
                self.add_xss_vulnerability(
                    "/", "incomplete_csp", "csp",
                    f"Missing CSP directives: {', '.join(missing_directives)}"
                )
                
        except Exception as e:
            logger.debug(f"CSP test failed: {e}")
    
    def generate_report(self) -> dict:
        """Generate comprehensive XSS/CSRF test report"""
        total_xss = len(self.test_results["xss_vulnerabilities"])
        total_csrf = len(self.test_results["csrf_vulnerabilities"])
        
        xss_contexts = {}
        for vuln in self.test_results["xss_vulnerabilities"]:
            context = vuln["context"]
            xss_contexts[context] = xss_contexts.get(context, 0) + 1
        
        report = {
            "summary": {
                "total_xss_vulnerabilities": total_xss,
                "total_csrf_vulnerabilities": total_csrf,
                "xss_contexts": xss_contexts,
                "endpoints_tested": len(self.test_results["tested_endpoints"])
            },
            "xss_vulnerabilities": self.test_results["xss_vulnerabilities"],
            "csrf_vulnerabilities": self.test_results["csrf_vulnerabilities"],
            "recommendations": [
                "Implement proper output encoding for all user input",
                "Use Content Security Policy (CSP) to prevent XSS execution",
                "Implement CSRF tokens for all state-changing operations",
                "Validate and sanitize all user inputs on server side",
                "Use HTTP-only and Secure flags for session cookies",
                "Implement proper input validation with allowlists",
                "Use template engines with automatic escaping",
                "Set X-Frame-Options header to prevent clickjacking",
                "Regular security code reviews focusing on input/output handling",
                "Implement proper session management"
            ]
        }
        
        return report
    
    def run_comprehensive_test(self):
        """Run comprehensive XSS and CSRF testing"""
        logger.info("🕷️ Starting comprehensive XSS and CSRF testing...")
        
        # Test reflected XSS on common endpoints
        endpoints_to_test = [
            ("/search", "q", "GET"),
            ("/api/auth/login", "username", "POST"),
            ("/api/auth/register", "username", "POST"),
            ("/api/simulations", "name", "POST"),
            ("/api/excel-parser/upload", "filename", "POST")
        ]
        
        for endpoint, param, method in endpoints_to_test:
            self.test_reflected_xss(endpoint, param, method)
        
        # Test DOM-based XSS
        self.test_dom_xss()
        
        # Test stored XSS with browser automation (if available)
        try:
            self.test_stored_xss_with_selenium()
        except Exception as e:
            logger.info(f"Skipping Selenium tests: {e}")
        
        # Test CSRF protection
        self.test_csrf_protection()
        
        # Test clickjacking protection
        self.test_clickjacking_protection()
        
        # Test Content Security Policy
        self.test_content_security_policy()
        
        # Generate report
        report = self.generate_report()
        
        return report

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="XSS and CSRF Testing Tool")
    parser.add_argument("--url", default="http://localhost:9090", help="Target URL")
    parser.add_argument("--output", default="xss_csrf_test_results.json", help="Output file")
    
    args = parser.parse_args()
    
    tester = XSSCSRFTester(args.url)
    results = tester.run_comprehensive_test()
    
    # Save results
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    summary = results["summary"]
    print(f"\n{'='*60}")
    print(f"XSS AND CSRF TEST RESULTS")
    print(f"{'='*60}")
    print(f"Target: {args.url}")
    print(f"XSS Vulnerabilities Found: {summary['total_xss_vulnerabilities']}")
    print(f"CSRF Vulnerabilities Found: {summary['total_csrf_vulnerabilities']}")
    
    if summary['xss_contexts']:
        print(f"\nXSS Contexts:")
        for context, count in summary['xss_contexts'].items():
            print(f"  {context}: {count}")
    
    print(f"\nDetailed results saved to: {args.output}")
    print(f"{'='*60}")
    
    total_vulns = summary['total_xss_vulnerabilities'] + summary['total_csrf_vulnerabilities']
    if total_vulns > 0:
        print("⚠️  XSS/CSRF VULNERABILITIES FOUND - IMMEDIATE ACTION REQUIRED")
        return 1
    else:
        print("✅ No XSS or CSRF vulnerabilities detected")
        return 0

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
