# Docker Security Migration Guide

## Overview
This guide walks you through migrating from the current Docker configuration to a hardened, secure setup. **Important:** Test all changes in development first!

## Current Security Issues Found
- 49 medium-severity Docker security findings
- Containers running as root users
- Excessive volume mounts with write access
- Missing security constraints

## Migration Steps

### Phase 1: Immediate Low-Risk Changes ✅ 
These changes are safe to implement immediately:

```bash
# Add security options to existing containers
# Edit docker-compose.yml to add:
security_opt:
  - no-new-privileges:true
```

### Phase 2: Create Non-Root Users (Requires Testing)
This requires modifying Dockerfiles:

1. **Backend Changes** (Test First):
   ```dockerfile
   # Add to backend/Dockerfile
   RUN groupadd -r appuser && useradd -r -g appuser appuser
   RUN chown -R appuser:appuser /app
   USER appuser
   ```

2. **Frontend Changes** (Test First):
   ```dockerfile
   # Use node user instead of root
   USER node:node
   ```

### Phase 3: Read-Only Filesystems (Advanced)
This requires comprehensive testing:

```yaml
# Add to services in docker-compose.yml
read_only: true
tmpfs:
  - /tmp
  - /var/log
```

## Quick Security Improvements (Safe to Implement Now)

### 1. Add Security Options
Edit your current `docker-compose.yml`:

```yaml
services:
  nginx:
    # ... existing config ...
    security_opt:
      - no-new-privileges:true
  
  backend:
    # ... existing config ...
    security_opt:
      - no-new-privileges:true
  
  frontend:
    # ... existing config ...
    security_opt:
      - no-new-privileges:true
```

### 2. Restrict Network Access
```yaml
networks:
  app-network:
    driver: bridge
    driver_opts:
      com.docker.network.enable_ipv6: "false"
    ipam:
      config:
        - subnet: 172.20.0.0/16
```

### 3. Resource Limits
```yaml
services:
  backend:
    # ... existing config ...
    deploy:
      resources:
        limits:
          memory: 4G
          cpus: '2.0'
        reservations:
          memory: 1G
          cpus: '0.5'
```

## Testing Strategy

### Step 1: Test Security Options Only
```bash
# Test the quick security improvements
docker-compose down
# Edit docker-compose.yml to add security_opt
docker-compose up -d
# Verify everything still works
```

### Step 2: Test Complete Secure Configuration (Staged)
```bash
# Copy your current working docker-compose.yml
cp docker-compose.yml docker-compose.backup.yml

# Test secure configuration in development
docker-compose -f docker-compose.secure.yml up --build

# If issues occur, rollback:
docker-compose -f docker-compose.backup.yml up -d
```

## Rollback Plan
Always have a rollback strategy:

```bash
# Quick rollback to working configuration
docker-compose down
docker-compose -f docker-compose.backup.yml up -d
```

## Recommendations

### Immediate (Low Risk) ✅
- Add `no-new-privileges:true` to all services
- Add resource limits
- Implement network restrictions

### Gradual (Medium Risk) ⚠️  
- Test secure Dockerfiles in development
- Gradually migrate to non-root users
- Test read-only filesystems

### Advanced (High Risk) 🚨
- Full migration to `docker-compose.secure.yml`
- Production deployment with secure containers

## Verification Commands

After making changes, verify security:

```bash
# Check if containers are running as root
docker exec <container_name> whoami

# Check filesystem permissions
docker exec <container_name> ls -la /app

# Check security options
docker inspect <container_name> | grep -A5 SecurityOpt
```

## Next Steps
1. Start with Phase 1 (security options)
2. Test each phase thoroughly
3. Only proceed to next phase after successful testing
4. Keep rollback plans ready

**Remember**: Your platform is working well now. Don't break what's working - make changes gradually and test thoroughly.
