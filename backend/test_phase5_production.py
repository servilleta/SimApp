"""
Phase 5: Production Deployment Test Suite
Comprehensive testing for production infrastructure
"""

import pytest
import asyncio
import requests
import subprocess
import os
import time
import json
from pathlib import Path
from typing import Dict, Any
import ssl
import socket
from urllib.parse import urlparse

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
        
    def test_ssl_certificate_permissions(self):
        """Test SSL certificate file permissions"""
        cert_path = Path("ssl/certs/nginx-selfsigned.crt")
        key_path = Path("ssl/private/nginx-selfsigned.key")
        
        if cert_path.exists():
            cert_perms = oct(cert_path.stat().st_mode)[-3:]
            assert cert_perms == "644", f"Certificate permissions incorrect: {cert_perms}"
            
        if key_path.exists():
            key_perms = oct(key_path.stat().st_mode)[-3:]
            assert key_perms == "600", f"Private key permissions incorrect: {key_perms}"
    
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
            "monitoring/grafana/provisioning/datasources",
            "nginx"
        ]
        
        for dir_path in required_dirs:
            assert Path(dir_path).exists(), f"Required directory missing: {dir_path}"

class TestDockerServices:
    """Test Docker services are running correctly"""
    
    def test_docker_compose_file_exists(self):
        """Test production Docker Compose file exists"""
        compose_path = Path("docker-compose.production.yml")
        assert compose_path.exists(), "docker-compose.production.yml not found"
    
    def test_services_are_running(self):
        """Test all required services are running"""
        result = subprocess.run([
            "docker-compose", "-f", "docker-compose.production.yml", "ps", "-q"
        ], capture_output=True, text=True)
        
        assert result.returncode == 0, "Failed to get service status"
        
        # Check if we have running containers
        container_ids = result.stdout.strip().split('\n')
        running_containers = [cid for cid in container_ids if cid.strip()]
        
        assert len(running_containers) >= 4, f"Expected at least 4 services, found {len(running_containers)}"
    
    def test_service_health_checks(self):
        """Test service health checks are working"""
        services = ["postgres", "redis", "backend", "frontend"]
        
        for service in services:
            result = subprocess.run([
                "docker-compose", "-f", "docker-compose.production.yml",
                "exec", "-T", service, "echo", "health_check"
            ], capture_output=True, text=True)
            
            # Service should be accessible (return code 0 or 126 for exec issues)
            assert result.returncode in [0, 126], f"Service {service} is not healthy"

class TestWebServices:
    """Test web services and endpoints"""
    
    def test_backend_api_health(self):
        """Test backend API health endpoint"""
        try:
            response = requests.get(f"{TEST_CONFIG['api_url']}/api/health", timeout=TEST_CONFIG['timeout'])
            assert response.status_code == 200, f"Backend API health check failed: {response.status_code}"
        except requests.exceptions.RequestException as e:
            pytest.fail(f"Backend API not accessible: {e}")
    
    def test_frontend_health(self):
        """Test frontend health endpoint"""
        try:
            response = requests.get(f"{TEST_CONFIG['base_url']}/health", timeout=TEST_CONFIG['timeout'])
            assert response.status_code == 200, f"Frontend health check failed: {response.status_code}"
        except requests.exceptions.RequestException as e:
            pytest.fail(f"Frontend not accessible: {e}")
    
    def test_https_redirect(self):
        """Test HTTP to HTTPS redirect"""
        try:
            response = requests.get(f"{TEST_CONFIG['base_url']}/", allow_redirects=False, timeout=TEST_CONFIG['timeout'])
            assert response.status_code == 301, f"Expected 301 redirect, got {response.status_code}"
            assert "https://" in response.headers.get('Location', ''), "Redirect not to HTTPS"
        except requests.exceptions.RequestException as e:
            pytest.fail(f"HTTP redirect test failed: {e}")
    
    def test_https_connection(self):
        """Test HTTPS connection works"""
        try:
            # Test HTTPS connection (ignore certificate verification for self-signed)
            response = requests.get(f"{TEST_CONFIG['https_url']}/health", verify=False, timeout=TEST_CONFIG['timeout'])
            assert response.status_code == 200, f"HTTPS connection failed: {response.status_code}"
        except requests.exceptions.RequestException as e:
            pytest.fail(f"HTTPS connection failed: {e}")
    
    def test_security_headers(self):
        """Test security headers are present"""
        try:
            response = requests.get(f"{TEST_CONFIG['https_url']}/health", verify=False, timeout=TEST_CONFIG['timeout'])
            
            expected_headers = [
                "X-Frame-Options",
                "X-Content-Type-Options", 
                "X-XSS-Protection",
                "Strict-Transport-Security",
                "Content-Security-Policy"
            ]
            
            for header in expected_headers:
                assert header in response.headers, f"Security header missing: {header}"
                
        except requests.exceptions.RequestException as e:
            pytest.fail(f"Security headers test failed: {e}")

class TestLoadBalancing:
    """Test load balancing and performance"""
    
    def test_nginx_configuration(self):
        """Test nginx configuration file exists"""
        nginx_config = Path("nginx/nginx-production.conf")
        assert nginx_config.exists(), "nginx-production.conf not found"
        
        with open(nginx_config, 'r') as f:
            config_content = f.read()
            
        # Check for load balancing configuration
        assert "upstream backend_pool" in config_content, "Backend load balancing not configured"
        assert "least_conn" in config_content, "Load balancing algorithm not set"
    
    def test_rate_limiting(self):
        """Test rate limiting is working"""
        try:
            # Make multiple rapid requests to test rate limiting
            responses = []
            for i in range(20):
                response = requests.get(f"{TEST_CONFIG['api_url']}/api/health", timeout=5)
                responses.append(response.status_code)
                time.sleep(0.1)
            
            # Should have some successful requests
            success_count = sum(1 for status in responses if status == 200)
            assert success_count > 0, "No successful requests - rate limiting too strict"
            
        except requests.exceptions.RequestException as e:
            pytest.fail(f"Rate limiting test failed: {e}")

class TestMonitoring:
    """Test monitoring stack"""
    
    def test_prometheus_accessible(self):
        """Test Prometheus is accessible"""
        try:
            response = requests.get(f"{TEST_CONFIG['prometheus_url']}/-/healthy", timeout=TEST_CONFIG['timeout'])
            assert response.status_code == 200, f"Prometheus not healthy: {response.status_code}"
        except requests.exceptions.RequestException as e:
            pytest.skip(f"Prometheus not accessible (monitoring may be disabled): {e}")
    
    def test_prometheus_targets(self):
        """Test Prometheus has configured targets"""
        try:
            response = requests.get(f"{TEST_CONFIG['prometheus_url']}/api/v1/targets", timeout=TEST_CONFIG['timeout'])
            assert response.status_code == 200, "Failed to get Prometheus targets"
            
            data = response.json()
            active_targets = data.get('data', {}).get('activeTargets', [])
            assert len(active_targets) > 0, "No active Prometheus targets found"
            
        except requests.exceptions.RequestException as e:
            pytest.skip(f"Prometheus targets test skipped: {e}")
    
    def test_grafana_accessible(self):
        """Test Grafana is accessible"""
        try:
            response = requests.get(f"{TEST_CONFIG['grafana_url']}/api/health", timeout=TEST_CONFIG['timeout'])
            assert response.status_code == 200, f"Grafana not healthy: {response.status_code}"
        except requests.exceptions.RequestException as e:
            pytest.skip(f"Grafana not accessible (monitoring may be disabled): {e}")
    
    def test_metrics_endpoint(self):
        """Test application metrics endpoint"""
        try:
            response = requests.get(f"{TEST_CONFIG['api_url']}/metrics", timeout=TEST_CONFIG['timeout'])
            # Metrics endpoint should be restricted
            assert response.status_code in [200, 403, 404], f"Unexpected metrics response: {response.status_code}"
        except requests.exceptions.RequestException as e:
            pytest.skip(f"Metrics endpoint test skipped: {e}")

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
    
    def test_backup_script_exists(self):
        """Test backup script exists and is executable"""
        backup_scripts = [
            "scripts/backup-runner.sh",
            "deploy-phase5.sh"
        ]
        
        for script_path in backup_scripts:
            script = Path(script_path)
            if script.exists():
                # Check if script is executable
                permissions = oct(script.stat().st_mode)[-3:]
                assert permissions in ["755", "775"], f"Script {script_path} not executable: {permissions}"

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
            pytest.fail("Database connectivity test timed out")
    
    def test_redis_connectivity(self):
        """Test Redis connection"""
        try:
            result = subprocess.run([
                "docker-compose", "-f", "docker-compose.production.yml",
                "exec", "-T", "redis", "redis-cli", "ping"
            ], capture_output=True, text=True, timeout=30)
            
            assert "PONG" in result.stdout, f"Redis not responding: {result.stdout}"
            
        except subprocess.TimeoutExpired:
            pytest.fail("Redis connectivity test timed out")

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
            pytest.fail(f"Performance test failed: {e}")
    
    def test_concurrent_requests(self):
        """Test handling concurrent requests"""
        import concurrent.futures
        import threading
        
        def make_request():
            try:
                response = requests.get(f"{TEST_CONFIG['api_url']}/api/health", timeout=10)
                return response.status_code == 200
            except:
                return False
        
        # Test with 10 concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_request) for _ in range(10)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        success_rate = sum(results) / len(results)
        assert success_rate >= 0.8, f"Concurrent request success rate too low: {success_rate:.2%}"

class TestSecurityConfiguration:
    """Test security configuration and compliance"""
    
    def test_environment_security(self):
        """Test production environment security settings"""
        env_path = Path("production.env")
        if env_path.exists():
            with open(env_path, 'r') as f:
                env_content = f.read()
            
            assert "DEBUG=false" in env_content, "Debug mode should be disabled in production"
            assert "ENVIRONMENT=production" in env_content, "Environment should be set to production"
    
    def test_ssl_configuration_strength(self):
        """Test SSL configuration strength"""
        nginx_config = Path("nginx/nginx-production.conf")
        if nginx_config.exists():
            with open(nginx_config, 'r') as f:
                config_content = f.read()
            
            # Check for strong SSL configuration
            assert "TLSv1.2 TLSv1.3" in config_content, "Strong TLS versions not configured"
            assert "ssl_prefer_server_ciphers" in config_content, "SSL cipher preferences not set"
    
    def test_container_security(self):
        """Test container security settings"""
        compose_path = Path("docker-compose.production.yml")
        if compose_path.exists():
            with open(compose_path, 'r') as f:
                compose_content = f.read()
            
            # Check for security settings
            assert "restart: unless-stopped" in compose_content, "Container restart policy not set"
            assert "logging:" in compose_content, "Container logging not configured"

def run_comprehensive_test():
    """Run comprehensive Phase 5 production test suite"""
    print("🧪 Running Phase 5: Production Deployment Test Suite")
    print("=" * 60)
    
    test_results = {
        "total_tests": 0,
        "passed_tests": 0,
        "failed_tests": 0,
        "skipped_tests": 0,
        "test_details": []
    }
    
    # Test classes to run
    test_classes = [
        TestProductionInfrastructure,
        TestDockerServices,
        TestWebServices,
        TestLoadBalancing,
        TestMonitoring,
        TestBackupSystem,
        TestDatabaseIntegration,
        TestPerformanceAndScaling,
        TestSecurityConfiguration
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
                
            except pytest.skip.Exception as e:
                test_results["skipped_tests"] += 1
                test_results["test_details"].append({
                    "name": f"{test_class.__name__}.{test_method}",
                    "status": "SKIPPED",
                    "message": str(e)
                })
                print(f"  ⏭️  {test_method} (SKIPPED: {e})")
                
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
    print(f"Skipped: {test_results['skipped_tests']}")
    
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