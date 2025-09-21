#!/usr/bin/env python3
"""
Browser Console Attack Testing
Tests for actual console-based exploitation attempts
"""

import requests
import json
import logging
from datetime import datetime
from typing import Dict, List

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("console_attack")

class ConsoleAttackTester:
    """Test console-based attack vectors"""
    
    def __init__(self, base_url: str = "http://localhost:9090"):
        self.base_url = base_url
        self.session = requests.Session()
        self.attack_results = []
        
        # Exposed API key found in frontend code
        self.exposed_api_key = "ak_5zno3zn8gisz5f9held6d09l6vosgft2_sk_qcK8E1nKk4RMTRB9GuWH16iWaffQnvxZuh29oK7mmaJpBRjkVFhnF9bm3Ttln9eR"
        self.exposed_password = "Demo123!MonteCarlo"
        
        logger.info(f"🎯 Console Attack Tester initialized for {base_url}")

    def test_api_key_exploitation(self):
        """Test if the exposed API key can be exploited"""
        logger.info("🔑 Testing exposed API key exploitation...")
        
        # Try to use the exposed API key
        headers = {"Authorization": f"Bearer {self.exposed_api_key}"}
        
        # Test endpoints that might be accessible
        test_endpoints = [
            "/simapp-api/health",
            "/simapp-api/models", 
            "/api/simulations",
            "/api/users/profile",
            "/api/auth/me"
        ]
        
        successful_accesses = []
        
        for endpoint in test_endpoints:
            try:
                url = f"{self.base_url}{endpoint}"
                response = self.session.get(url, headers=headers, timeout=5)
                
                if response.status_code == 200:
                    successful_accesses.append({
                        "endpoint": endpoint,
                        "status": response.status_code,
                        "response_length": len(response.text),
                        "exposed_data": response.text[:200] + "..." if len(response.text) > 200 else response.text
                    })
                    logger.warning(f"🚨 API key allows access to {endpoint}")
                    
            except Exception as e:
                logger.debug(f"Error testing {endpoint}: {e}")
        
        self.attack_results.append({
            "attack_type": "API Key Exploitation",
            "success": len(successful_accesses) > 0,
            "accessible_endpoints": successful_accesses,
            "risk_level": "CRITICAL" if successful_accesses else "LOW"
        })
        
        return successful_accesses

    def test_credential_exploitation(self):
        """Test if exposed credentials can be used"""
        logger.info("🔐 Testing credential exploitation...")
        
        # Try to login with exposed credentials
        login_endpoints = [
            "/api/auth/login",
            "/auth/login"
        ]
        
        login_data = {
            "username": "admin",
            "password": self.exposed_password
        }
        
        successful_logins = []
        
        for endpoint in login_endpoints:
            try:
                url = f"{self.base_url}{endpoint}"
                response = self.session.post(url, json=login_data, timeout=5)
                
                if response.status_code == 200:
                    response_data = response.json()
                    if "access_token" in response_data or "token" in response_data:
                        successful_logins.append({
                            "endpoint": endpoint,
                            "token_obtained": True,
                            "token_preview": str(response_data)[:100] + "..."
                        })
                        logger.warning(f"🚨 Successful login at {endpoint}")
                        
            except Exception as e:
                logger.debug(f"Error testing login at {endpoint}: {e}")
        
        self.attack_results.append({
            "attack_type": "Credential Exploitation", 
            "success": len(successful_logins) > 0,
            "successful_logins": successful_logins,
            "risk_level": "CRITICAL" if successful_logins else "LOW"
        })
        
        return successful_logins

    def test_console_based_api_calls(self):
        """Test console-based API manipulation"""
        logger.info("🌐 Testing console-based API calls...")
        
        # Simulate what an attacker could do via browser console
        console_attack_scenarios = [
            {
                "name": "Fetch User Data",
                "description": "fetch('/api/users/profile', {headers: {'Authorization': 'Bearer [EXPOSED_KEY]'}})",
                "endpoint": "/api/users/profile"
            },
            {
                "name": "Access Admin Functions", 
                "description": "fetch('/api/admin/users', {headers: {'Authorization': 'Bearer [EXPOSED_KEY]'}})",
                "endpoint": "/api/admin/users"
            },
            {
                "name": "File Upload Manipulation",
                "description": "fetch('/api/excel-parser/upload', {method: 'POST', headers: {'Authorization': 'Bearer [EXPOSED_KEY]'}})",
                "endpoint": "/api/excel-parser/upload"
            }
        ]
        
        console_successes = []
        headers = {"Authorization": f"Bearer {self.exposed_api_key}"}
        
        for scenario in console_attack_scenarios:
            try:
                url = f"{self.base_url}{scenario['endpoint']}"
                response = self.session.get(url, headers=headers, timeout=5)
                
                if response.status_code != 404:  # Endpoint exists
                    console_successes.append({
                        "scenario": scenario["name"],
                        "endpoint": scenario["endpoint"],
                        "status": response.status_code,
                        "accessible": response.status_code in [200, 401, 403]  # 401/403 means endpoint exists
                    })
                    
            except Exception as e:
                logger.debug(f"Error in console attack scenario {scenario['name']}: {e}")
        
        self.attack_results.append({
            "attack_type": "Console API Manipulation",
            "success": len(console_successes) > 0,
            "accessible_scenarios": console_successes,
            "risk_level": "HIGH" if any(s.get("accessible") for s in console_successes) else "MEDIUM"
        })
        
        return console_successes

    def test_storage_manipulation(self):
        """Test browser storage manipulation attacks"""
        logger.info("💾 Testing storage manipulation...")
        
        # Check if we can access the application without storage manipulation
        try:
            response = self.session.get(self.base_url)
            if response.status_code == 200:
                content = response.text
                
                # Look for localStorage/sessionStorage usage
                storage_indicators = [
                    "localStorage.setItem",
                    "sessionStorage.setItem", 
                    "authToken",
                    "userRole",
                    "isAdmin"
                ]
                
                found_storage_usage = []
                for indicator in storage_indicators:
                    if indicator in content:
                        found_storage_usage.append(indicator)
                
                self.attack_results.append({
                    "attack_type": "Storage Manipulation",
                    "success": len(found_storage_usage) > 0,
                    "storage_usage": found_storage_usage,
                    "risk_level": "MEDIUM" if found_storage_usage else "LOW",
                    "potential_attacks": [
                        "localStorage.setItem('authToken', 'fake_token')",
                        "localStorage.setItem('userRole', 'admin')",
                        "sessionStorage.setItem('isAdmin', 'true')"
                    ]
                })
                
        except Exception as e:
            logger.debug(f"Error testing storage manipulation: {e}")

    def generate_console_attack_report(self):
        """Generate console attack assessment report"""
        
        # Calculate overall risk
        critical_attacks = sum(1 for r in self.attack_results if r.get("risk_level") == "CRITICAL")
        high_attacks = sum(1 for r in self.attack_results if r.get("risk_level") == "HIGH") 
        successful_attacks = sum(1 for r in self.attack_results if r.get("success", False))
        
        overall_risk = "LOW"
        if critical_attacks > 0:
            overall_risk = "CRITICAL"
        elif high_attacks > 0:
            overall_risk = "HIGH"
        elif successful_attacks > 0:
            overall_risk = "MEDIUM"
        
        console_hackable = critical_attacks > 0 or (high_attacks > 0 and successful_attacks > 0)
        
        report = {
            "timestamp": datetime.utcnow().isoformat(),
            "target": self.base_url,
            "console_hackable": console_hackable,
            "overall_risk": overall_risk,
            "attack_results": self.attack_results,
            "summary": {
                "total_attack_vectors": len(self.attack_results),
                "successful_attacks": successful_attacks,
                "critical_risks": critical_attacks,
                "high_risks": high_attacks,
                "exposed_secrets": {
                    "api_key": self.exposed_api_key,
                    "password": self.exposed_password,
                    "locations": [
                        "frontend/src/pages/APITestPage.jsx:6",
                        "frontend/src/pages/APIDocumentationPage.jsx:5",
                        "frontend/src/store/authSlice.js:9"
                    ]
                }
            },
            "recommendations": [
                "CRITICAL: Remove hardcoded API key from client-side code immediately",
                "CRITICAL: Remove hardcoded credentials from client-side code",
                "Move all sensitive configuration to environment variables",
                "Implement proper secret management",
                "Add client-side code obfuscation for production",
                "Implement runtime application self-protection (RASP)",
                "Add console tampering detection"
            ]
        }
        
        # Save report
        with open("console_attack_results.json", "w") as f:
            json.dump(report, f, indent=2)
        
        # Print results
        logger.info("=" * 70)
        logger.info("CONSOLE ATTACK ASSESSMENT RESULTS")
        logger.info("=" * 70)
        logger.info(f"Target: {self.base_url}")
        logger.info(f"Console Hackable: {'YES' if console_hackable else 'NO'}")
        logger.info(f"Overall Risk: {overall_risk}")
        logger.info(f"Successful Attacks: {successful_attacks}/{len(self.attack_results)}")
        logger.info(f"Critical Risks: {critical_attacks}")
        logger.info(f"High Risks: {high_attacks}")
        
        if console_hackable:
            logger.error("🚨 PLATFORM CAN BE HACKED VIA WEB CONSOLE")
        else:
            logger.info("✅ Console exploitation risk is manageable")
            
        logger.info(f"\nDetailed results saved to: console_attack_results.json")
        logger.info("=" * 70)
        
        return report

    def run_all_tests(self):
        """Run all console attack tests"""
        logger.info("🚀 Starting console attack assessment...")
        
        self.test_api_key_exploitation()
        self.test_credential_exploitation() 
        self.test_console_based_api_calls()
        self.test_storage_manipulation()
        
        return self.generate_console_attack_report()


if __name__ == "__main__":
    tester = ConsoleAttackTester()
    results = tester.run_all_tests()
    
    # Exit with appropriate code
    if results["console_hackable"]:
        exit(2)  # Critical
    elif results["overall_risk"] in ["HIGH", "MEDIUM"]:
        exit(1)  # Warning
    else:
        exit(0)  # OK
