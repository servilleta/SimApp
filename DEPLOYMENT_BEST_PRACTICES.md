# Monte Carlo Platform - Deployment Best Practices

## Overview

This document outlines the comprehensive deployment strategy for the Monte Carlo simulation platform, covering Development → Staging → Production workflows optimized for GPU-accelerated applications.

## Environment Architecture

### 1. Development Environment
- **Purpose**: Active development, feature testing, debugging
- **Branch**: `main-v2-modular` (current development)
- **Configuration**: `docker-compose.yml`
- **Database**: SQLite or lightweight PostgreSQL
- **GPU**: Development-grade testing
- **SSL**: HTTP only (no SSL required)
- **Monitoring**: Basic logging
- **Resource Requirements**: 4GB RAM, 2 CPU cores

### 2. Staging Environment  
- **Purpose**: Production-like testing, QA validation, integration testing
- **Branch**: `release/vX.X.X` branches
- **Configuration**: `docker-compose.staging.yml`
- **Database**: PostgreSQL (production-like)
- **GPU**: Full CUDA testing
- **SSL**: Self-signed certificates
- **Monitoring**: Full Prometheus/Grafana stack
- **Resource Requirements**: 6GB RAM, 4 CPU cores, 1 GPU

### 3. Production Environment
- **Purpose**: Live user-facing deployment
- **Branch**: `main-v1-production` or tagged releases
- **Configuration**: `docker-compose.production.yml`  
- **Database**: PostgreSQL with clustering/replication
- **GPU**: Full CUDA acceleration
- **SSL**: CA-issued certificates (Let's Encrypt)
- **Monitoring**: Full observability stack with alerting
- **Resource Requirements**: 8GB+ RAM, 6+ CPU cores, 1+ GPU

## Git Workflow & Branching Strategy

### Current Branch Structure
```
main (original)
├── main-v1-production (stable production)
│   ├── hotfix/security-patches
│   └── hotfix/critical-bugs
└── main-v2-modular (development)
    ├── feature/simulation-engine
    ├── feature/auth-improvements
    └── release/v2.1.0
```

### Deployment Workflow

#### Development to Staging
```bash
# Create release branch from development
git checkout development
git checkout -b release/v2.1.0

# Deploy to staging
git checkout staging
git merge release/v2.1.0
docker-compose -f docker-compose.staging.yml up -d

# Run staging tests
./scripts/test-staging.sh

# If tests pass, merge back to development
git checkout development
git merge release/v2.1.0
```

#### Staging to Production
```bash
# Tag release
git checkout production
git merge staging
git tag -a v2.1.0 -m "Release v2.1.0: Enhanced GPU performance"
git push origin v2.1.0

# Deploy to production
./deploy-production.sh v2.1.0

# Verify deployment
./scripts/test-production.sh
```

### Hotfix Workflow
```bash
# Critical production fix
git checkout production
git checkout -b hotfix/gpu-memory-leak

# Fix issue and test in staging
git checkout staging
git merge hotfix/gpu-memory-leak
docker-compose -f docker-compose.staging.yml up -d
# Test fix

# Deploy hotfix to production
git checkout production
git merge hotfix/gpu-memory-leak
git tag -a v2.0.1 -m "Hotfix: GPU memory leak"

# Deploy immediately
./deploy-production.sh v2.0.1

# Backport to development
git checkout development
git cherry-pick <hotfix-commit>
```

## Environment-Specific Configurations

### Development (.env)
```env
ENVIRONMENT=development
DEBUG=true
DATABASE_URL=sqlite:///./montecarlo_dev.db
USE_GPU=true
GPU_MEMORY_FRACTION=0.4
CORS_ORIGINS=*
LOG_LEVEL=DEBUG
```

### Staging (staging.env)
```env
ENVIRONMENT=staging
DEBUG=false
DATABASE_URL=postgresql://user:pass@postgres:5432/montecarlo_staging
USE_GPU=true
GPU_MEMORY_FRACTION=0.6
CORS_ORIGINS=https://staging.simapp.ai
LOG_LEVEL=INFO
```

### Production (production.env)
```env
ENVIRONMENT=production
DEBUG=false
DATABASE_URL=postgresql://user:pass@postgres:5432/montecarlo_production
USE_GPU=true
GPU_MEMORY_FRACTION=0.8
CORS_ORIGINS=https://simapp.ai
LOG_LEVEL=WARNING
```

## Deployment Scripts

### Development Deployment
```bash
#!/bin/bash
# deploy-development.sh

echo "🚀 Deploying Development Environment..."

# Pull latest changes
git pull origin main-v2-modular

# Build and start services
docker-compose down
docker-compose build --no-cache
docker-compose up -d

# Wait for services
./scripts/wait-for-services.sh

# Run basic health checks
./scripts/health-check.sh development

echo "✅ Development environment ready at http://localhost:3000"
```

### Staging Deployment
```bash
#!/bin/bash
# deploy-staging.sh

echo "🧪 Deploying Staging Environment..."

# Load staging environment
source staging.env

# Build staging containers
docker-compose -f docker-compose.staging.yml down
docker-compose -f docker-compose.staging.yml build --no-cache
docker-compose -f docker-compose.staging.yml up -d

# Wait for all services
./scripts/wait-for-services.sh staging

# Run comprehensive tests
./scripts/test-staging.sh

# Generate SSL certificates for staging
./scripts/generate-staging-ssl.sh

echo "✅ Staging environment ready at https://staging.simapp.ai"
```

### Production Deployment
```bash
#!/bin/bash
# deploy-production.sh

VERSION=${1:-latest}
echo "🚀 Deploying Production Environment - Version: $VERSION"

# Safety checks
./scripts/pre-deployment-checks.sh

# Backup current state
./scripts/backup-production.sh

# Load production environment
source production.env

# Deploy with zero downtime
docker-compose -f docker-compose.production.yml pull
docker-compose -f docker-compose.production.yml up -d --remove-orphans

# Health checks
./scripts/health-check.sh production

# Smoke tests
./scripts/smoke-tests.sh

# Update monitoring
./scripts/update-monitoring.sh $VERSION

echo "✅ Production deployment complete - Version: $VERSION"
```

## GPU-Specific Deployment Considerations

### Development
- **GPU Memory**: 40% allocation for testing
- **CUDA Libraries**: Mounted from host
- **Fallback**: CPU-only mode if GPU unavailable
- **Testing**: Basic GPU functionality tests

### Staging  
- **GPU Memory**: 60% allocation for realistic testing
- **CUDA Libraries**: Full production-like setup
- **Fallback**: Fail fast if GPU unavailable
- **Testing**: Complete GPU performance testing

### Production
- **GPU Memory**: 80% allocation for maximum performance
- **CUDA Libraries**: Optimized production libraries
- **Fallback**: Multi-GPU support, no CPU fallback
- **Monitoring**: GPU utilization and memory tracking

## Security Considerations by Environment

### Development
- HTTP only (no SSL)
- Permissive CORS
- Debug endpoints enabled
- Test authentication

### Staging
- Self-signed SSL certificates
- Restricted CORS to staging domain
- Production-like security headers
- Full authentication testing

### Production
- CA-issued SSL certificates (Let's Encrypt)
- Strict CORS policy
- All security headers enabled
- Rate limiting and DDoS protection

## Monitoring & Observability

### Development
- Basic console logging
- Simple health checks
- No persistent metrics

### Staging
- Full Prometheus metrics
- Grafana dashboards
- Log aggregation
- Performance profiling

### Production
- Enterprise monitoring stack
- Real-time alerting
- Log analysis and searching
- Business metrics tracking

## Database Management

### Development
- SQLite for simplicity
- No backup required
- Schema migrations testing

### Staging
- PostgreSQL with staging data
- Daily backups for testing
- Migration validation

### Production
- PostgreSQL with clustering
- Automated backups every 6 hours
- Point-in-time recovery
- Read replicas for performance

## File Storage Strategy

### Development
- Local file storage
- No cleanup (for debugging)
- Small test files

### Staging
- Local storage with cleanup
- Production-like file sizes
- 7-day retention policy

### Production
- High-performance storage
- Automated cleanup
- 30-day retention
- Optional S3 backup

## Testing Strategy

### Development Testing
```bash
# Unit tests
pytest backend/tests/unit/

# Integration tests
pytest backend/tests/integration/

# GPU tests
pytest backend/tests/gpu/
```

### Staging Testing
```bash
# Full test suite
./scripts/test-staging.sh

# Performance tests
./scripts/performance-tests.sh

# Load testing
./scripts/load-tests.sh

# GPU stress tests
./scripts/gpu-stress-tests.sh
```

### Production Testing
```bash
# Smoke tests (non-intrusive)
./scripts/smoke-tests.sh

# Health monitoring
./scripts/health-monitor.sh

# Performance monitoring
./scripts/performance-monitor.sh
```

## Rollback Procedures

### Development Rollback
```bash
# Simple git revert
git checkout main-v2-modular
git reset --hard HEAD~1
docker-compose restart
```

### Staging Rollback
```bash
# Rollback to previous release
./deploy-staging.sh $(git describe --tags --abbrev=0 HEAD~1)
```

### Production Rollback
```bash
# Emergency rollback
./scripts/emergency-rollback.sh <previous-version>

# Graceful rollback
./scripts/graceful-rollback.sh <target-version>
```

## Environment Variables Management

### Secrets Management
- **Development**: `.env` file (not committed)
- **Staging**: `staging.env` + environment injection
- **Production**: External secrets management (HashiCorp Vault, AWS Secrets Manager)

### Configuration Hierarchy
1. Environment-specific files (.env, staging.env, production.env)
2. Runtime environment variables
3. Default values in backend/config.py

## Deployment Checklist

### Pre-Deployment
- [ ] All tests passing in staging
- [ ] GPU performance validated
- [ ] Database migrations tested
- [ ] Security scan completed
- [ ] Backup strategy verified

### During Deployment
- [ ] Service health checks passing
- [ ] GPU acceleration working
- [ ] Database connectivity confirmed
- [ ] Monitoring systems active
- [ ] SSL certificates valid

### Post-Deployment
- [ ] Smoke tests completed
- [ ] Performance metrics normal
- [ ] Error rates within threshold
- [ ] GPU utilization optimal
- [ ] User acceptance testing

## Troubleshooting Common Issues

### GPU Issues
```bash
# Check GPU availability
nvidia-smi
docker run --gpus all nvidia/cuda:11.8-base nvidia-smi

# Verify CUDA libraries
docker exec -it backend-container nvidia-smi
```

### Database Connection Issues
```bash
# Check PostgreSQL status
docker exec -it postgres-container pg_isready

# Check connection strings
docker logs backend-container | grep -i database
```

### SSL Certificate Issues
```bash
# Verify certificate
openssl x509 -in ssl/certs/nginx-selfsigned.crt -text -noout

# Test SSL connection
openssl s_client -connect localhost:443
```

## Performance Optimization

### Development
- Focus on fast iteration
- Minimal resource usage
- Quick startup times

### Staging
- Production-like performance
- Comprehensive profiling
- Load testing validation

### Production
- Maximum performance optimization
- GPU utilization maximization
- Response time minimization

## Monitoring & Alerting

### Key Metrics to Monitor
- **System**: CPU, Memory, Disk, Network
- **Application**: Response times, Error rates, Throughput
- **GPU**: Utilization, Memory usage, Temperature
- **Business**: Simulations completed, User activity, File uploads

### Alert Thresholds
- **High Priority**: Service down, GPU failure, Database unavailable
- **Medium Priority**: High error rate, Slow response times, Low disk space
- **Low Priority**: High CPU usage, Memory pressure, SSL expiration warning

## Conclusion

This deployment strategy ensures:
- **Reliability**: Comprehensive testing at each stage
- **Performance**: GPU optimization maintained across environments  
- **Security**: Progressive security hardening
- **Scalability**: Environment-specific resource allocation
- **Maintainability**: Clear processes and documentation

The three-tier deployment approach (Development → Staging → Production) provides the safety and validation needed for a GPU-accelerated Monte Carlo simulation platform while maintaining development velocity and operational excellence.
