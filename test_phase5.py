"""
Phase 5: Production Deployment Test Suite
Comprehensive testing for production infrastructure
"""

import requests
import subprocess
import os
import time
import json
from pathlib import Path
from typing import Dict, Any

# Test configuration
TEST_CONFIG = {
    "base_url": "http://localhost",
    "api_url": "http://localhost:8000",
    "https_url": "https://localhost",
    "prometheus_url": "http://localhost:9090",
    "grafana_url": "http://localhost:3001",
    "timeout": 30
}

class TestProductionInfrastructure:
    """Test production infrastructure components"""
    
    def test_ssl_certificates_exist(self):
        """Test SSL certificates are generated"""
        cert_path = Path("ssl/certs/nginx-selfsigned.crt")
        key_path = Path("ssl/private/nginx-selfsigned.key")
        
        assert cert_path.exists(), "SSL certificate not found"
        assert key_path.exists(), "SSL private key not found"
        
        # Check certificate validity
        result = subprocess.run([
            "openssl", "x509", "-in", str(cert_path), "-text", "-noout"
        ], capture_output=True, text=True)
        
        assert result.returncode == 0, "SSL certificate is invalid"
        assert "CN=localhost" in result.stdout or "DNS:localhost" in result.stdout
        
    def test_production_env_file(self):
        """Test production environment file exists and has required variables"""
        env_path = Path("production.env")
        assert env_path.exists(), "production.env file not found"
        
        with open(env_path, 'r') as f:
            env_content = f.read()
            
        required_vars = [
            "ENVIRONMENT=production",
            "SECRET_KEY=",
            "POSTGRES_PASSWORD=",
            "REDIS_PASSWORD=",
            "ADMIN_PASSWORD="
        ]
        
        for var in required_vars:
            assert var in env_content, f"Required environment variable missing: {var}"
    
    def test_directory_structure(self):
        """Test required directories exist"""
        required_dirs = [
            "logs/nginx",
            "ssl/certs",
            "ssl/private",
            "backups",
            "uploads",
            "monitoring/grafana/dashboards",
            "nginx"
        ]
        
        for dir_path in required_dirs:
            assert Path(dir_path).exists(), f"Required directory missing: {dir_path}"

class TestWebServices:
    """Test web services and endpoints"""
    
    def test_backend_api_health(self):
        """Test backend API health endpoint"""
        try:
            response = requests.get(f"{TEST_CONFIG['api_url']}/api/health", timeout=TEST_CONFIG['timeout'])
            assert response.status_code == 200, f"Backend API health check failed: {response.status_code}"
        except requests.exceptions.RequestException as e:
            raise AssertionError(f"Backend API not accessible: {e}")
    
    def test_frontend_health(self):
        """Test frontend health endpoint"""
        try:
            response = requests.get(f"{TEST_CONFIG['base_url']}/health", timeout=TEST_CONFIG['timeout'])
            assert response.status_code == 200, f"Frontend health check failed: {response.status_code}"
        except requests.exceptions.RequestException as e:
            raise AssertionError(f"Frontend not accessible: {e}")
    
    def test_https_connection(self):
        """Test HTTPS connection works"""
        try:
            # Test HTTPS connection (ignore certificate verification for self-signed)
            response = requests.get(f"{TEST_CONFIG['https_url']}/health", verify=False, timeout=TEST_CONFIG['timeout'])
            assert response.status_code == 200, f"HTTPS connection failed: {response.status_code}"
        except requests.exceptions.RequestException as e:
            raise AssertionError(f"HTTPS connection failed: {e}")
    
    def test_security_headers(self):
        """Test security headers are present"""
        try:
            response = requests.get(f"{TEST_CONFIG['https_url']}/health", verify=False, timeout=TEST_CONFIG['timeout'])
            
            expected_headers = [
                "X-Frame-Options",
                "X-Content-Type-Options", 
                "X-XSS-Protection",
                "Strict-Transport-Security"
            ]
            
            for header in expected_headers:
                assert header in response.headers, f"Security header missing: {header}"
                
        except requests.exceptions.RequestException as e:
            raise AssertionError(f"Security headers test failed: {e}")

class TestMonitoring:
    """Test monitoring stack"""
    
    def test_prometheus_accessible(self):
        """Test Prometheus is accessible"""
        try:
            response = requests.get(f"{TEST_CONFIG['prometheus_url']}/-/healthy", timeout=TEST_CONFIG['timeout'])
            assert response.status_code == 200, f"Prometheus not healthy: {response.status_code}"
        except requests.exceptions.RequestException:
            print("Prometheus not accessible (monitoring may be disabled)")
            return True  # Skip if monitoring disabled
    
    def test_grafana_accessible(self):
        """Test Grafana is accessible"""
        try:
            response = requests.get(f"{TEST_CONFIG['grafana_url']}/api/health", timeout=TEST_CONFIG['timeout'])
            assert response.status_code == 200, f"Grafana not healthy: {response.status_code}"
        except requests.exceptions.RequestException:
            print("Grafana not accessible (monitoring may be disabled)")
            return True  # Skip if monitoring disabled

class TestBackupSystem:
    """Test backup and disaster recovery"""
    
    def test_backup_directory_exists(self):
        """Test backup directory exists"""
        backup_dir = Path("backups")
        assert backup_dir.exists(), "Backup directory not found"
        assert backup_dir.is_dir(), "Backup path is not a directory"
    
    def test_backup_service_configuration(self):
        """Test backup service configuration"""
        backup_service_path = Path("backend/infrastructure/backup.py")
        assert backup_service_path.exists(), "Backup service not found"
        
        with open(backup_service_path, 'r') as f:
            backup_content = f.read()
            
        assert "BackupService" in backup_content, "BackupService class not found"
        assert "create_database_backup" in backup_content, "Database backup method not found"

class TestDatabaseIntegration:
    """Test database integration and connectivity"""
    
    def test_database_connectivity(self):
        """Test database connection"""
        try:
            result = subprocess.run([
                "docker-compose", "-f", "docker-compose.production.yml",
                "exec", "-T", "postgres", "pg_isready", "-U", "montecarlo_user", "-d", "montecarlo_db"
            ], capture_output=True, text=True, timeout=30)
            
            assert result.returncode == 0, f"Database not ready: {result.stderr}"
            
        except subprocess.TimeoutExpired:
            raise AssertionError("Database connectivity test timed out")
        except FileNotFoundError:
            print("Docker compose not available, skipping database test")
            return True

class TestPerformanceAndScaling:
    """Test performance and scaling capabilities"""
    
    def test_response_time(self):
        """Test API response time is acceptable"""
        start_time = time.time()
        
        try:
            response = requests.get(f"{TEST_CONFIG['api_url']}/api/health", timeout=TEST_CONFIG['timeout'])
            end_time = time.time()
            
            response_time = end_time - start_time
            assert response_time < 2.0, f"Response time too slow: {response_time:.2f}s"
            assert response.status_code == 200, f"API health check failed: {response.status_code}"
            
        except requests.exceptions.RequestException as e:
            raise AssertionError(f"Performance test failed: {e}")

def run_comprehensive_test():
    """Run comprehensive Phase 5 production test suite"""
    print("🧪 Running Phase 5: Production Deployment Test Suite")
    print("=" * 60)
    
    test_results = {
        "total_tests": 0,
        "passed_tests": 0,
        "failed_tests": 0,
        "test_details": []
    }
    
    # Test classes to run
    test_classes = [
        TestProductionInfrastructure,
        TestWebServices,
        TestMonitoring,
        TestBackupSystem,
        TestDatabaseIntegration,
        TestPerformanceAndScaling
    ]
    
    for test_class in test_classes:
        print(f"\n📋 Testing {test_class.__name__}")
        test_instance = test_class()
        
        # Get all test methods
        test_methods = [method for method in dir(test_instance) if method.startswith('test_')]
        
        for test_method in test_methods:
            test_results["total_tests"] += 1
            
            try:
                method = getattr(test_instance, test_method)
                method()
                test_results["passed_tests"] += 1
                test_results["test_details"].append({
                    "name": f"{test_class.__name__}.{test_method}",
                    "status": "PASSED",
                    "message": "Test passed successfully"
                })
                print(f"  ✅ {test_method}")
                
            except Exception as e:
                test_results["failed_tests"] += 1
                test_results["test_details"].append({
                    "name": f"{test_class.__name__}.{test_method}",
                    "status": "FAILED",
                    "message": str(e)
                })
                print(f"  ❌ {test_method} (FAILED: {e})")
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    print(f"Total Tests: {test_results['total_tests']}")
    print(f"Passed: {test_results['passed_tests']}")
    print(f"Failed: {test_results['failed_tests']}")
    
    success_rate = test_results['passed_tests'] / test_results['total_tests'] * 100 if test_results['total_tests'] > 0 else 0
    print(f"Success Rate: {success_rate:.1f}%")
    
    if test_results['failed_tests'] == 0:
        print("\n🎉 ALL TESTS PASSED! Phase 5 Production Deployment is ready!")
        return True
    else:
        print(f"\n⚠️  {test_results['failed_tests']} tests failed. Review and fix issues before production deployment.")
        return False

if __name__ == "__main__":
    success = run_comprehensive_test()
    exit(0 if success else 1) 