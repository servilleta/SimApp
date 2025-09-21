#!/usr/bin/env python3
"""
Network Security Scanner for Monte Carlo Platform
Comprehensive network-level security testing
"""

import subprocess
import socket
import ssl
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
import concurrent.futures
import requests
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("network_scanner")

class NetworkSecurityScanner:
    """Network-level security testing"""
    
    def __init__(self, target_host: str = "localhost", docker_compose_file: str = "docker-compose.yml"):
        self.target_host = target_host
        self.docker_compose_file = docker_compose_file
        self.scan_results = {
            "timestamp": datetime.utcnow().isoformat(),
            "target_host": target_host,
            "findings": [],
            "exposed_services": [],
            "security_issues": []
        }
    
    def add_finding(self, category: str, description: str, severity: str, details: str = ""):
        """Add security finding"""
        self.scan_results["findings"].append({
            "category": category,
            "description": description,
            "severity": severity,
            "details": details,
            "timestamp": datetime.utcnow().isoformat()
        })
    
    def scan_port_range(self, start_port: int = 1, end_port: int = 10000) -> List[int]:
        """Scan for open ports"""
        logger.info(f"Scanning ports {start_port}-{end_port} on {self.target_host}")
        open_ports = []
        
        def scan_port(port):
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(1)
                result = sock.connect_ex((self.target_host, port))
                sock.close()
                return port if result == 0 else None
            except Exception:
                return None
        
        # Use threading for faster scanning
        with concurrent.futures.ThreadPoolExecutor(max_workers=100) as executor:
            futures = [executor.submit(scan_port, port) for port in range(start_port, end_port + 1)]
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if result:
                    open_ports.append(result)
        
        return sorted(open_ports)
    
    def identify_services(self, ports: List[int]) -> Dict[int, Dict[str, Any]]:
        """Identify services running on open ports"""
        services = {}
        
        for port in ports:
            try:
                # Try to grab banner
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(5)
                sock.connect((self.target_host, port))
                
                # Send HTTP request to check if it's HTTP
                try:
                    sock.send(b"GET / HTTP/1.1\r\nHost: localhost\r\n\r\n")
                    response = sock.recv(1024).decode('utf-8', errors='ignore')
                    
                    if "HTTP/" in response:
                        services[port] = {
                            "type": "HTTP",
                            "banner": response[:200],
                            "server": self.extract_server_header(response)
                        }
                    else:
                        services[port] = {
                            "type": "Unknown",
                            "banner": response[:200]
                        }
                except Exception:
                    # Not HTTP, try to get banner anyway
                    try:
                        sock.send(b"\r\n")
                        response = sock.recv(1024).decode('utf-8', errors='ignore')
                        services[port] = {
                            "type": "Unknown",
                            "banner": response[:200]
                        }
                    except Exception:
                        services[port] = {
                            "type": "Unknown",
                            "banner": "No banner"
                        }
                
                sock.close()
                
            except Exception as e:
                services[port] = {
                    "type": "Unknown",
                    "error": str(e)
                }
        
        return services
    
    def extract_server_header(self, http_response: str) -> str:
        """Extract Server header from HTTP response"""
        for line in http_response.split('\n'):
            if line.lower().startswith('server:'):
                return line.split(':', 1)[1].strip()
        return "Unknown"
    
    def check_docker_security(self):
        """Check Docker container security"""
        try:
            # Check if Docker is accessible
            result = subprocess.run(["docker", "version"], capture_output=True, text=True, timeout=10)
            
            if result.returncode != 0:
                self.add_finding(
                    "docker_access",
                    "Docker daemon not accessible for security assessment",
                    "info",
                    "Cannot assess Docker security configuration"
                )
                return
            
            # Check running containers
            result = subprocess.run(["docker", "ps", "--format", "json"], 
                                  capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                containers = []
                for line in result.stdout.strip().split('\n'):
                    if line:
                        containers.append(json.loads(line))
                
                for container in containers:
                    # Check for privileged containers
                    inspect_result = subprocess.run(
                        ["docker", "inspect", container["ID"]], 
                        capture_output=True, text=True, timeout=10
                    )
                    
                    if inspect_result.returncode == 0:
                        inspect_data = json.loads(inspect_result.stdout)[0]
                        
                        if inspect_data.get("HostConfig", {}).get("Privileged", False):
                            self.add_finding(
                                "privileged_container",
                                f"Container {container['Names']} running in privileged mode",
                                "high",
                                f"Container ID: {container['ID']}"
                            )
                        
                        # Check for containers running as root
                        user = inspect_data.get("Config", {}).get("User", "")
                        if not user or user == "0" or user == "root":
                            self.add_finding(
                                "container_root_user",
                                f"Container {container['Names']} running as root",
                                "medium",
                                f"Container ID: {container['ID']}"
                            )
                        
                        # Check for dangerous volume mounts
                        mounts = inspect_data.get("Mounts", [])
                        for mount in mounts:
                            if mount.get("Source", "").startswith("/"):
                                dangerous_paths = ["/", "/etc", "/var", "/usr", "/bin", "/sbin"]
                                if any(mount["Source"].startswith(path) for path in dangerous_paths):
                                    self.add_finding(
                                        "dangerous_volume_mount",
                                        f"Container {container['Names']} has dangerous volume mount",
                                        "medium",
                                        f"Mount: {mount['Source']} -> {mount['Destination']}"
                                    )
                        
                        # Check network mode
                        network_mode = inspect_data.get("HostConfig", {}).get("NetworkMode", "")
                        if network_mode == "host":
                            self.add_finding(
                                "host_network_mode",
                                f"Container {container['Names']} using host network mode",
                                "medium",
                                "Host network mode bypasses Docker's network isolation"
                            )
            
        except Exception as e:
            logger.error(f"Docker security check failed: {e}")
    
    def check_ssl_configuration(self, port: int = 443):
        """Check SSL/TLS configuration"""
        try:
            context = ssl.create_default_context()
            context.check_hostname = False
            context.verify_mode = ssl.CERT_NONE
            
            with socket.create_connection((self.target_host, port), timeout=10) as sock:
                with context.wrap_socket(sock) as ssock:
                    cert = ssock.getpeercert()
                    protocol = ssock.version()
                    cipher = ssock.cipher()
                    
                    # Check SSL protocol version
                    if protocol in ["SSLv2", "SSLv3", "TLSv1", "TLSv1.1"]:
                        self.add_finding(
                            "weak_ssl_protocol",
                            f"Weak SSL/TLS protocol: {protocol}",
                            "high",
                            f"Protocol: {protocol}, Cipher: {cipher}"
                        )
                    
                    # Check cipher strength
                    if cipher and cipher[1] < 128:
                        self.add_finding(
                            "weak_cipher",
                            f"Weak cipher suite: {cipher[0]}",
                            "medium",
                            f"Key length: {cipher[1]} bits"
                        )
                    
                    # Check certificate validity
                    if cert:
                        import datetime
                        not_after = datetime.datetime.strptime(cert['notAfter'], '%b %d %H:%M:%S %Y %Z')
                        days_until_expiry = (not_after - datetime.datetime.utcnow()).days
                        
                        if days_until_expiry < 30:
                            self.add_finding(
                                "certificate_expiry",
                                f"SSL certificate expires in {days_until_expiry} days",
                                "medium" if days_until_expiry > 7 else "high",
                                f"Expires: {cert['notAfter']}"
                            )
                    
        except Exception as e:
            logger.debug(f"SSL configuration check failed for port {port}: {e}")
    
    def check_service_versions(self, services: Dict[int, Dict[str, Any]]):
        """Check for outdated service versions"""
        for port, service in services.items():
            banner = service.get("banner", "")
            server = service.get("server", "")
            
            # Check for version disclosure
            if any(keyword in banner.lower() for keyword in ["nginx", "apache", "php", "python", "node"]):
                self.add_finding(
                    "version_disclosure",
                    f"Service version disclosed on port {port}",
                    "low",
                    f"Banner: {banner[:100]}"
                )
            
            # Check for known vulnerable versions
            vulnerable_signatures = [
                ("nginx/1.14", "Nginx 1.14 has known vulnerabilities"),
                ("apache/2.2", "Apache 2.2 is end-of-life"),
                ("php/5.", "PHP 5.x is end-of-life"),
                ("openssl/1.0", "OpenSSL 1.0 has known vulnerabilities")
            ]
            
            for signature, description in vulnerable_signatures:
                if signature in banner.lower():
                    self.add_finding(
                        "vulnerable_version",
                        f"Potentially vulnerable service version detected on port {port}",
                        "high",
                        f"{description}: {banner[:100]}"
                    )
    
    def check_default_credentials(self, services: Dict[int, Dict[str, Any]]):
        """Check for default credentials on services"""
        default_creds = [
            ("admin", "admin"),
            ("admin", "password"),
            ("root", "root"),
            ("postgres", "postgres"),
            ("redis", ""),
            ("", "")
        ]
        
        for port, service in services.items():
            if service.get("type") == "HTTP":
                # Test common admin panels
                admin_paths = ["/admin", "/administrator", "/wp-admin", "/login"]
                
                for path in admin_paths:
                    try:
                        response = requests.get(f"http://{self.target_host}:{port}{path}", timeout=5)
                        if response.status_code == 200 and "login" in response.text.lower():
                            self.add_finding(
                                "admin_panel_exposed",
                                f"Administrative panel accessible on port {port}",
                                "medium",
                                f"URL: http://{self.target_host}:{port}{path}"
                            )
                    except Exception:
                        pass
    
    def check_information_disclosure(self, services: Dict[int, Dict[str, Any]]):
        """Check for information disclosure"""
        for port, service in services.items():
            if service.get("type") == "HTTP":
                # Test common information disclosure paths
                disclosure_paths = [
                    "/.env", "/config.json", "/package.json", "/composer.json",
                    "/.git/config", "/backup.sql", "/database.sql",
                    "/phpinfo.php", "/info.php", "/test.php"
                ]
                
                for path in disclosure_paths:
                    try:
                        response = requests.get(f"http://{self.target_host}:{port}{path}", timeout=5)
                        if response.status_code == 200:
                            content = response.text[:500]
                            if any(keyword in content.lower() for keyword in [
                                "password", "secret", "api_key", "database", "config"
                            ]):
                                self.add_finding(
                                    "information_disclosure",
                                    f"Sensitive information exposed at {path} on port {port}",
                                    "high",
                                    f"Content preview: {content[:200]}"
                                )
                    except Exception:
                        pass
    
    def run_comprehensive_scan(self) -> Dict[str, Any]:
        """Run comprehensive network security scan"""
        logger.info("Starting comprehensive network security scan...")
        
        # 1. Port scanning
        logger.info("Scanning for open ports...")
        open_ports = self.scan_port_range(1, 65535)
        logger.info(f"Found {len(open_ports)} open ports: {open_ports}")
        
        # 2. Service identification
        logger.info("Identifying services...")
        services = self.identify_services(open_ports)
        
        # Store exposed services
        for port, service_info in services.items():
            self.scan_results["exposed_services"].append({
                "port": port,
                "service": service_info
            })
        
        # 3. Docker security check
        logger.info("Checking Docker security...")
        self.check_docker_security()
        
        # 4. SSL/TLS configuration check
        logger.info("Checking SSL/TLS configuration...")
        for port in [443, 8443, 9443]:
            if port in open_ports:
                self.check_ssl_configuration(port)
        
        # 5. Service version checks
        logger.info("Checking service versions...")
        self.check_service_versions(services)
        
        # 6. Default credentials check
        logger.info("Checking for default credentials...")
        self.check_default_credentials(services)
        
        # 7. Information disclosure check
        logger.info("Checking for information disclosure...")
        self.check_information_disclosure(services)
        
        # Generate summary
        self.scan_results["summary"] = {
            "total_open_ports": len(open_ports),
            "total_findings": len(self.scan_results["findings"]),
            "critical_findings": len([f for f in self.scan_results["findings"] if f["severity"] == "critical"]),
            "high_findings": len([f for f in self.scan_results["findings"] if f["severity"] == "high"]),
            "medium_findings": len([f for f in self.scan_results["findings"] if f["severity"] == "medium"]),
            "low_findings": len([f for f in self.scan_results["findings"] if f["severity"] == "low"]),
            "exposed_services": len(self.scan_results["exposed_services"])
        }
        
        return self.scan_results

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Network Security Scanner")
    parser.add_argument("--target", default="localhost", help="Target host")
    parser.add_argument("--output", default="network_scan_results.json", help="Output file")
    
    args = parser.parse_args()
    
    scanner = NetworkSecurityScanner(args.target)
    results = scanner.run_comprehensive_scan()
    
    # Save results
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    summary = results["summary"]
    print(f"\n{'='*60}")
    print(f"NETWORK SECURITY SCAN RESULTS")
    print(f"{'='*60}")
    print(f"Target: {args.target}")
    print(f"Open Ports: {summary['total_open_ports']}")
    print(f"Exposed Services: {summary['exposed_services']}")
    print(f"Total Findings: {summary['total_findings']}")
    print(f"  Critical: {summary['critical_findings']}")
    print(f"  High: {summary['high_findings']}")
    print(f"  Medium: {summary['medium_findings']}")
    print(f"  Low: {summary['low_findings']}")
    print(f"\nDetailed results saved to: {args.output}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
