#!/usr/bin/env python3
"""
Security Hardening Implementation Script
Implements critical security fixes based on penetration testing findings
"""

import os
import json
import logging
import subprocess
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("security_hardening")

class SecurityHardeningImplementation:
    """Implement security hardening measures"""
    
    def __init__(self, project_dir: str = "/home/paperspace/PROJECT"):
        self.project_dir = project_dir
        self.implementation_log = {
            "timestamp": datetime.utcnow().isoformat(),
            "project_directory": project_dir,
            "implemented_fixes": [],
            "pending_fixes": [],
            "errors": []
        }
    
    def log_implementation(self, fix_name: str, status: str, details: str = ""):
        """Log implementation status"""
        entry = {
            "fix": fix_name,
            "status": status,
            "details": details,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        if status == "SUCCESS":
            self.implementation_log["implemented_fixes"].append(entry)
            logger.info(f"✅ {fix_name}: {details}")
        elif status == "PENDING":
            self.implementation_log["pending_fixes"].append(entry)
            logger.warning(f"⏳ {fix_name}: {details}")
        else:
            self.implementation_log["errors"].append(entry)
            logger.error(f"❌ {fix_name}: {details}")
    
    def fix_content_security_policy(self):
        """Implement secure Content Security Policy"""
        try:
            # Create improved CSP configuration
            secure_csp = {
                "default-src": ["'self'"],
                "script-src": ["'self'", "'strict-dynamic'", "'nonce-{random}'"],
                "style-src": ["'self'", "'unsafe-hashes'", "https://fonts.googleapis.com"],
                "font-src": ["'self'", "https://fonts.gstatic.com"],
                "img-src": ["'self'", "data:", "https:"],
                "connect-src": ["'self'", "https://api.stripe.com"],
                "frame-src": ["'none'"],
                "frame-ancestors": ["'none'"],
                "object-src": ["'none'"],
                "base-uri": ["'self'"],
                "form-action": ["'self'"],
                "upgrade-insecure-requests": True
            }
            
            # Write CSP configuration file
            csp_config_path = os.path.join(self.project_dir, "security", "csp_config.json")
            os.makedirs(os.path.dirname(csp_config_path), exist_ok=True)
            
            with open(csp_config_path, 'w') as f:
                json.dump(secure_csp, f, indent=2)
            
            # Create implementation guidance
            csp_implementation = '''
# Secure Content Security Policy Implementation

## Backend Implementation (FastAPI)
Add to your security middleware:

```python
from fastapi import Request, Response

def build_csp_header():
    csp_directives = [
        "default-src 'self'",
        "script-src 'self' 'strict-dynamic'",
        "style-src 'self' 'unsafe-hashes' https://fonts.googleapis.com",
        "font-src 'self' https://fonts.gstatic.com",
        "img-src 'self' data: https:",
        "connect-src 'self' https://api.stripe.com",
        "frame-src 'none'",
        "frame-ancestors 'none'",
        "object-src 'none'",
        "base-uri 'self'",
        "form-action 'self'",
        "upgrade-insecure-requests"
    ]
    return "; ".join(csp_directives)

@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["Content-Security-Policy"] = build_csp_header()
    return response
```

## Frontend Implementation
1. Remove all inline scripts and styles
2. Use nonces for dynamic scripts: <script nonce="{nonce}">
3. Hash static inline styles: sha256-{hash}
4. Move all JavaScript to external files
'''
            
            guidance_path = os.path.join(self.project_dir, "security", "CSP_IMPLEMENTATION_GUIDE.md")
            with open(guidance_path, 'w') as f:
                f.write(csp_implementation)
            
            self.log_implementation(
                "Content Security Policy", 
                "SUCCESS", 
                f"Secure CSP configuration created at {csp_config_path}"
            )
            
        except Exception as e:
            self.log_implementation("Content Security Policy", "ERROR", str(e))
    
    def fix_docker_security(self):
        """Implement Docker security improvements"""
        try:
            # Create secure Docker Compose configuration
            secure_docker_compose = '''version: '3.8'

services:
  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile
    ports:
      - "3000:3000"
    volumes:
      - ./frontend:/app:ro  # Read-only mount
      - /app/node_modules
    environment:
      - NODE_ENV=production
      - VITE_API_URL=http://localhost:9090/api
    user: "node:node"  # Run as non-root user
    read_only: true    # Read-only filesystem
    tmpfs:
      - /tmp
    networks:
      - app-network
    security_opt:
      - no-new-privileges:true

  backend:
    build:
      context: ./backend
      dockerfile: Dockerfile.secure  # Use secure Dockerfile
    ports:
      - "8000:8000"
    volumes:
      - ./uploads:/app/uploads:Z      # SELinux context
      - ./cache:/app/cache:Z
    environment:
      - PYTHONUNBUFFERED=1
      - DATABASE_URL=postgresql://postgres:postgres@postgres:5432/montecarlo
      - REDIS_URL=redis://redis:6379/0
    user: "appuser:appuser"  # Run as non-root user
    read_only: true         # Read-only filesystem
    tmpfs:
      - /tmp
      - /var/log
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_started
    networks:
      - app-network
    security_opt:
      - no-new-privileges:true
    cap_drop:
      - ALL
    cap_add:
      - CAP_NET_BIND_SERVICE  # Only if needed for port binding

  nginx:
    image: nginx:alpine
    container_name: montecarlo-nginx-secure
    ports:
      - "9090:80"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/conf.d/default.conf:ro
      - ./frontend/dist:/usr/share/nginx/html:ro
    user: "nginx:nginx"
    read_only: true
    tmpfs:
      - /var/cache/nginx
      - /var/run
    depends_on:
      - frontend
      - backend
    networks:
      - app-network
    security_opt:
      - no-new-privileges:true

  postgres:
    image: postgres:15-alpine
    environment:
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD_FILE=/run/secrets/db_password  # Use secrets
      - POSTGRES_DB=montecarlo
    volumes:
      - postgres_data:/var/lib/postgresql/data
    secrets:
      - db_password
    user: "postgres:postgres"
    read_only: true
    tmpfs:
      - /tmp
      - /var/run/postgresql
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 5s
      timeout: 5s
      retries: 5
    networks:
      - app-network
    security_opt:
      - no-new-privileges:true

  redis:
    image: redis:alpine
    user: "redis:redis"
    read_only: true
    tmpfs:
      - /tmp
    networks:
      - app-network
    security_opt:
      - no-new-privileges:true

secrets:
  db_password:
    file: ./secrets/db_password.txt

volumes:
  postgres_data:

networks:
  app-network:
    driver: bridge
    driver_opts:
      com.docker.network.enable_ipv6: "false"
'''
            
            secure_compose_path = os.path.join(self.project_dir, "docker-compose.secure.yml")
            with open(secure_compose_path, 'w') as f:
                f.write(secure_docker_compose)
            
            # Create secure Dockerfile for backend
            secure_dockerfile = '''FROM python:3.11-slim

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Install dependencies
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set proper permissions
RUN chown -R appuser:appuser /app
RUN chmod -R 755 /app

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:8000/api/health || exit 1

# Run application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
'''
            
            dockerfile_path = os.path.join(self.project_dir, "backend", "Dockerfile.secure")
            with open(dockerfile_path, 'w') as f:
                f.write(secure_dockerfile)
            
            # Create secrets directory
            secrets_dir = os.path.join(self.project_dir, "secrets")
            os.makedirs(secrets_dir, exist_ok=True)
            
            # Create example password file
            with open(os.path.join(secrets_dir, "db_password.txt"), 'w') as f:
                f.write("secure_postgres_password_change_me")
            
            os.chmod(os.path.join(secrets_dir, "db_password.txt"), 0o600)
            
            self.log_implementation(
                "Docker Security", 
                "SUCCESS", 
                f"Secure Docker configuration created at {secure_compose_path}"
            )
            
        except Exception as e:
            self.log_implementation("Docker Security", "ERROR", str(e))
    
    def implement_nginx_security_headers(self):
        """Implement secure Nginx configuration"""
        try:
            secure_nginx_config = '''server {
    listen 80;
    server_name localhost;
    
    # Security headers
    add_header X-Frame-Options "DENY" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header Referrer-Policy "strict-origin-when-cross-origin" always;
    add_header Permissions-Policy "geolocation=(), microphone=(), camera=()" always;
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains; preload" always;
    
    # Content Security Policy (will be enhanced by backend)
    add_header Content-Security-Policy "default-src 'self'; frame-ancestors 'none';" always;
    
    # Hide nginx version
    server_tokens off;
    
    # Prevent access to hidden files
    location ~ /\\. {
        deny all;
        access_log off;
        log_not_found off;
    }
    
    # Prevent access to backup files
    location ~ ~$ {
        deny all;
        access_log off;
        log_not_found off;
    }
    
    # Rate limiting
    limit_req_zone $binary_remote_addr zone=login:10m rate=5r/m;
    limit_req_zone $binary_remote_addr zone=api:10m rate=100r/m;
    
    # Frontend static files
    location / {
        root /usr/share/nginx/html;
        index index.html index.htm;
        try_files $uri $uri/ /index.html;
        
        # Cache static assets
        location ~* \\.(css|js|png|jpg|jpeg|gif|svg|ico|woff|woff2|ttf|eot)$ {
            expires 1y;
            add_header Cache-Control "public, immutable";
        }
    }
    
    # API proxy with security
    location /api/ {
        limit_req zone=api burst=20 nodelay;
        
        proxy_pass http://backend:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Security
        proxy_hide_header X-Powered-By;
        proxy_hide_header Server;
        
        # Timeouts
        proxy_connect_timeout 30s;
        proxy_send_timeout 30s;
        proxy_read_timeout 30s;
    }
    
    # Login rate limiting
    location /api/auth/login {
        limit_req zone=login burst=3 nodelay;
        proxy_pass http://backend:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
    
    # Block common attack vectors
    location ~* (wp-admin|wp-login|phpmyadmin|admin|administrator) {
        deny all;
        access_log off;
        log_not_found off;
    }
    
    # File upload restrictions
    client_max_body_size 500M;
    client_body_timeout 60s;
    client_header_timeout 60s;
}
'''
            
            nginx_config_path = os.path.join(self.project_dir, "nginx", "nginx.secure.conf")
            os.makedirs(os.path.dirname(nginx_config_path), exist_ok=True)
            
            with open(nginx_config_path, 'w') as f:
                f.write(secure_nginx_config)
            
            self.log_implementation(
                "Nginx Security Headers", 
                "SUCCESS", 
                f"Secure Nginx configuration created at {nginx_config_path}"
            )
            
        except Exception as e:
            self.log_implementation("Nginx Security Headers", "ERROR", str(e))
    
    def create_security_monitoring_setup(self):
        """Create security monitoring and logging setup"""
        try:
            # Create security monitoring Docker compose
            monitoring_compose = '''version: '3.8'

services:
  fail2ban:
    image: crazymax/fail2ban:latest
    container_name: fail2ban
    network_mode: "host"
    cap_add:
      - NET_ADMIN
      - NET_RAW
    volumes:
      - ./fail2ban:/data
      - ./logs:/var/log:ro
    environment:
      - TZ=UTC
      - F2B_LOG_LEVEL=INFO
    restart: unless-stopped

  logrotate:
    image: vegardit/traefik-logrotate:latest
    container_name: logrotate
    volumes:
      - ./logs:/logs
    environment:
      - LOGROTATE_INTERVAL=daily
      - LOGROTATE_COPIES=7
    restart: unless-stopped

  security-scanner:
    image: owasp/zap2docker-stable:latest
    container_name: security-scanner
    volumes:
      - ./security-reports:/zap/wrk/:rw
    command: zap-baseline.py -t http://nginx:80 -J security-report.json
    depends_on:
      - nginx
    profiles:
      - security-scan
'''
            
            monitoring_path = os.path.join(self.project_dir, "monitoring", "docker-compose.security.yml")
            os.makedirs(os.path.dirname(monitoring_path), exist_ok=True)
            
            with open(monitoring_path, 'w') as f:
                f.write(monitoring_compose)
            
            # Create fail2ban configuration
            fail2ban_config = '''[DEFAULT]
bantime = 3600
findtime = 600
maxretry = 5

[nginx-http-auth]
enabled = true
filter = nginx-http-auth
logpath = /var/log/nginx/error.log
maxretry = 3

[nginx-noscript]
enabled = true
filter = nginx-noscript
logpath = /var/log/nginx/access.log
maxretry = 6

[nginx-badbots]
enabled = true
filter = nginx-badbots
logpath = /var/log/nginx/access.log
maxretry = 2

[nginx-noproxy]
enabled = true
filter = nginx-noproxy
logpath = /var/log/nginx/access.log
maxretry = 2
'''
            
            fail2ban_dir = os.path.join(self.project_dir, "fail2ban")
            os.makedirs(fail2ban_dir, exist_ok=True)
            
            with open(os.path.join(fail2ban_dir, "jail.local"), 'w') as f:
                f.write(fail2ban_config)
            
            self.log_implementation(
                "Security Monitoring", 
                "SUCCESS", 
                f"Security monitoring setup created at {monitoring_path}"
            )
            
        except Exception as e:
            self.log_implementation("Security Monitoring", "ERROR", str(e))
    
    def create_security_incident_response_plan(self):
        """Create security incident response documentation"""
        try:
            incident_response_plan = '''# Security Incident Response Plan
## Monte Carlo Simulation Platform

### Incident Classification

#### Critical (P0) - Immediate Response Required
- Active security breach
- Data exfiltration
- System compromise
- Service unavailable due to security incident

#### High (P1) - Response within 2 hours
- Vulnerability exploitation attempts
- Unauthorized access attempts
- Security control bypasses
- Suspicious user activity

#### Medium (P2) - Response within 24 hours
- Security alerts from monitoring
- Failed authentication patterns
- Configuration drift
- Compliance violations

#### Low (P3) - Response within 72 hours
- Security awareness issues
- Minor policy violations
- Routine security updates

### Response Procedures

#### Immediate Actions (0-15 minutes)
1. **Assess and Classify**
   - Determine incident severity
   - Identify affected systems
   - Document initial findings

2. **Contain**
   - Isolate affected systems if necessary
   - Preserve evidence
   - Prevent further damage

3. **Communicate**
   - Notify incident response team
   - Alert stakeholders based on severity
   - Document all actions

#### Investigation Phase (15 minutes - 4 hours)
1. **Evidence Collection**
   - Collect logs and system snapshots
   - Interview affected users
   - Document timeline of events

2. **Root Cause Analysis**
   - Identify attack vectors
   - Determine scope of impact
   - Assess data confidentiality

3. **Impact Assessment**
   - Business impact analysis
   - Data breach assessment
   - Regulatory compliance impact

#### Recovery Phase (Varies by incident)
1. **System Restoration**
   - Clean and rebuild compromised systems
   - Restore from clean backups
   - Apply security patches

2. **Monitoring**
   - Enhanced monitoring for 30 days
   - Verify attack indicators are resolved
   - Monitor for reoccurrence

#### Post-Incident Phase
1. **Documentation**
   - Complete incident report
   - Lessons learned document
   - Update procedures

2. **Improvements**
   - Implement additional controls
   - Update security policies
   - Conduct training

### Contact Information

#### Internal Team
- Security Team Lead: [Contact Info]
- DevOps Team Lead: [Contact Info]
- Management: [Contact Info]

#### External Contacts
- Legal Counsel: [Contact Info]
- Cyber Insurance: [Contact Info]
- Law Enforcement: [Contact Info]
- Incident Response Consultant: [Contact Info]

### Tools and Resources

#### Incident Management
- Incident tracking system
- Secure communication channels
- Evidence collection tools
- Forensic analysis tools

#### Recovery Resources
- Clean backup systems
- Emergency contact lists
- Recovery procedures
- Communication templates

### Regular Exercises

#### Monthly
- Security alert testing
- Communication procedures
- Tool availability checks

#### Quarterly
- Tabletop exercises
- Response time testing
- Contact list updates

#### Annually
- Full-scale incident simulation
- Plan review and updates
- Training and awareness
'''
            
            incident_plan_path = os.path.join(self.project_dir, "security", "INCIDENT_RESPONSE_PLAN.md")
            os.makedirs(os.path.dirname(incident_plan_path), exist_ok=True)
            
            with open(incident_plan_path, 'w') as f:
                f.write(incident_response_plan)
            
            self.log_implementation(
                "Incident Response Plan", 
                "SUCCESS", 
                f"Incident response plan created at {incident_plan_path}"
            )
            
        except Exception as e:
            self.log_implementation("Incident Response Plan", "ERROR", str(e))
    
    def implement_all_fixes(self):
        """Implement all security hardening measures"""
        logger.info("🔧 Starting security hardening implementation...")
        
        # Fix CSP issues
        self.fix_content_security_policy()
        
        # Fix Docker security
        self.fix_docker_security()
        
        # Implement Nginx security headers
        self.implement_nginx_security_headers()
        
        # Create security monitoring
        self.create_security_monitoring_setup()
        
        # Create incident response plan
        self.create_security_incident_response_plan()
        
        # Create implementation summary
        self.create_implementation_summary()
        
        return self.implementation_log
    
    def create_implementation_summary(self):
        """Create implementation summary and next steps"""
        try:
            summary = f'''# Security Hardening Implementation Summary

## Implementation Date: {datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")} UTC

## Successfully Implemented
'''
            
            for fix in self.implementation_log["implemented_fixes"]:
                summary += f"- ✅ **{fix['fix']}**: {fix['details']}\\n"
            
            summary += '''
## Next Steps (Manual Implementation Required)

### 1. Content Security Policy Implementation
- **File**: `/security/CSP_IMPLEMENTATION_GUIDE.md`
- **Action**: Update FastAPI middleware to implement secure CSP
- **Priority**: HIGH
- **Timeline**: 1-2 days

### 2. Docker Security Migration
- **File**: `docker-compose.secure.yml`
- **Action**: Migrate to secure Docker configuration
- **Priority**: HIGH
- **Timeline**: 2-3 days
- **Steps**:
  1. Test secure configuration in development
  2. Create non-root users in containers
  3. Update deployment pipeline
  4. Schedule maintenance window for production migration

### 3. Nginx Security Headers
- **File**: `/nginx/nginx.secure.conf`
- **Action**: Replace current Nginx configuration
- **Priority**: HIGH
- **Timeline**: 1 day
- **Steps**:
  1. Test configuration with current load
  2. Update rate limiting based on actual usage
  3. Deploy during low-traffic window

### 4. Security Monitoring Setup
- **File**: `/monitoring/docker-compose.security.yml`
- **Action**: Deploy security monitoring stack
- **Priority**: MEDIUM
- **Timeline**: 1 week
- **Steps**:
  1. Configure fail2ban rules for your environment
  2. Set up alerting and notifications
  3. Test incident response procedures

### 5. Additional Recommendations

#### Immediate (1-2 weeks)
- [ ] Remove 'unsafe-inline' and 'unsafe-eval' from CSP
- [ ] Implement input validation on all user inputs
- [ ] Add rate limiting to authentication endpoints
- [ ] Enable HTTPS with proper SSL certificates
- [ ] Implement proper session management

#### Short-term (1 month)
- [ ] Regular security scanning automation
- [ ] Implement Web Application Firewall (WAF)
- [ ] Set up centralized logging and SIEM
- [ ] Conduct security awareness training
- [ ] Implement multi-factor authentication

#### Long-term (3 months)
- [ ] Regular penetration testing schedule
- [ ] Security code review process
- [ ] Disaster recovery testing
- [ ] Compliance audit preparation
- [ ] Security metrics and KPI tracking

## Verification Steps

After implementing the fixes:

1. **Re-run penetration tests**:
   ```bash
   python3 xss_csrf_tester.py --url http://localhost:9090
   python3 network_security_scanner.py --target localhost
   ```

2. **Verify security headers**:
   ```bash
   curl -I http://localhost:9090
   ```

3. **Test CSP implementation**:
   - Browser developer tools → Security tab
   - Check for CSP violations in console

4. **Validate Docker security**:
   ```bash
   docker-compose -f docker-compose.secure.yml up -d
   docker exec <container> whoami  # Should not be root
   ```

## Contact and Support

For implementation questions or issues:
- Review the created documentation in `/security/`
- Test all changes in development environment first
- Schedule maintenance windows for production changes
- Keep rollback plans ready for each change

## Success Metrics

- Zero CSP violations in browser console
- All containers running as non-root users
- Security headers present in all HTTP responses
- Successful rate limiting on authentication endpoints
- Incident response plan tested and documented

'''
            
            summary_path = os.path.join(self.project_dir, "SECURITY_HARDENING_SUMMARY.md")
            with open(summary_path, 'w') as f:
                f.write(summary)
            
            self.log_implementation(
                "Implementation Summary", 
                "SUCCESS", 
                f"Implementation summary created at {summary_path}"
            )
            
        except Exception as e:
            self.log_implementation("Implementation Summary", "ERROR", str(e))
    
    def save_implementation_log(self, filename: str = "security_hardening_log.json"):
        """Save implementation log"""
        log_path = os.path.join(self.project_dir, filename)
        
        try:
            with open(log_path, 'w') as f:
                json.dump(self.implementation_log, f, indent=2)
            logger.info(f"Implementation log saved to: {log_path}")
            return log_path
        except Exception as e:
            logger.error(f"Failed to save implementation log: {e}")
            return None

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Security Hardening Implementation")
    parser.add_argument("--project-dir", default="/home/paperspace/PROJECT", help="Project directory")
    
    args = parser.parse_args()
    
    # Implement security hardening
    hardening = SecurityHardeningImplementation(args.project_dir)
    results = hardening.implement_all_fixes()
    
    # Save implementation log
    hardening.save_implementation_log()
    
    # Print summary
    print(f"\\n{'='*80}")
    print(f"SECURITY HARDENING IMPLEMENTATION COMPLETE")
    print(f"{'='*80}")
    print(f"Successfully Implemented: {len(results['implemented_fixes'])}")
    print(f"Pending Manual Steps: {len(results['pending_fixes'])}")
    print(f"Errors: {len(results['errors'])}")
    print(f"\\nNext Steps: Review SECURITY_HARDENING_SUMMARY.md for manual implementation steps")
    print(f"{'='*80}")
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
