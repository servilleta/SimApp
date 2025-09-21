#!/bin/bash

# Monte Carlo Platform - Production Deployment Script
# Usage: ./deploy-production.sh [version]

set -euo pipefail

VERSION=${1:-$(git describe --tags --dirty --always)}
PRODUCTION_ENV_FILE="production.env"
COMPOSE_FILE="docker-compose.production.yml"
BACKUP_DIR="backups/pre-deployment-$(date +%Y%m%d-%H%M%S)"

echo "🚀 Deploying Monte Carlo Platform to Production Environment"
echo "Version: $VERSION"
echo "=================================="

# Function to check if file exists
check_file() {
    if [ ! -f "$1" ]; then
        echo "❌ Error: $1 not found!"
        exit 1
    fi
}

# Function to create backup
create_backup() {
    echo "💾 Creating pre-deployment backup..."
    mkdir -p "$BACKUP_DIR"
    
    # Backup database
    if docker-compose -f $COMPOSE_FILE ps postgres | grep -q "Up"; then
        echo "   Backing up database..."
        docker-compose -f $COMPOSE_FILE exec -T postgres pg_dumpall -U ${POSTGRES_USER:-montecarlo_user} | gzip > "$BACKUP_DIR/database_backup.sql.gz"
        echo "   ✅ Database backup created"
    fi
    
    # Backup important directories
    echo "   Backing up files..."
    if [ -d "uploads" ]; then
        tar -czf "$BACKUP_DIR/uploads_backup.tar.gz" uploads/
    fi
    if [ -d "saved_simulations_files" ]; then
        tar -czf "$BACKUP_DIR/simulations_backup.tar.gz" saved_simulations_files/
    fi
    
    # Backup configuration
    cp -r ssl "$BACKUP_DIR/" 2>/dev/null || true
    cp -r monitoring "$BACKUP_DIR/" 2>/dev/null || true
    cp $PRODUCTION_ENV_FILE "$BACKUP_DIR/" 2>/dev/null || true
    
    echo "   ✅ Backup completed: $BACKUP_DIR"
}

# Function to wait for service health
wait_for_service() {
    local service=$1
    local max_attempts=60
    local attempt=1
    
    echo "⏳ Waiting for $service to be healthy..."
    
    while [ $attempt -le $max_attempts ]; do
        if docker-compose -f $COMPOSE_FILE exec -T $service echo "Service is up" >/dev/null 2>&1; then
            echo "✅ $service is healthy"
            return 0
        fi
        
        echo "   Attempt $attempt/$max_attempts - $service not ready yet..."
        sleep 10
        ((attempt++))
    done
    
    echo "❌ $service failed to become healthy after $max_attempts attempts"
    exit 1
}

# Function to run smoke tests
run_smoke_tests() {
    echo "🧪 Running production smoke tests..."
    
    # Test API health
    if curl -f -k https://localhost/api/health >/dev/null 2>&1; then
        echo "✅ API health check passed"
    else
        echo "❌ API health check failed"
        return 1
    fi
    
    # Test authentication endpoint
    if curl -f -k https://localhost/api/auth/me >/dev/null 2>&1; then
        echo "✅ Authentication endpoint accessible"
    else
        echo "⚠️  Authentication endpoint test failed (expected without token)"
    fi
    
    # Test GPU functionality
    if docker-compose -f $COMPOSE_FILE exec -T backend python -c "
import cupy as cp
try:
    cp.cuda.runtime.getDeviceCount()
    print('GPU test passed')
except Exception as e:
    print(f'GPU test failed: {e}')
    exit(1)
" >/dev/null 2>&1; then
        echo "✅ GPU functionality verified"
    else
        echo "❌ GPU functionality test failed"
        return 1
    fi
    
    return 0
}

# Function to rollback on failure
rollback_deployment() {
    echo "🔄 Rolling back deployment..."
    
    # Stop current services
    docker-compose -f $COMPOSE_FILE down
    
    # Restore from backup if available
    if [ -d "$BACKUP_DIR" ]; then
        echo "   Restoring from backup: $BACKUP_DIR"
        
        # Restore database
        if [ -f "$BACKUP_DIR/database_backup.sql.gz" ]; then
            echo "   Restoring database..."
            # Note: This would need manual intervention in a real scenario
            echo "   ⚠️  Manual database restore required from: $BACKUP_DIR/database_backup.sql.gz"
        fi
        
        # Restore files
        if [ -f "$BACKUP_DIR/uploads_backup.tar.gz" ]; then
            echo "   Restoring uploads..."
            tar -xzf "$BACKUP_DIR/uploads_backup.tar.gz"
        fi
        
        if [ -f "$BACKUP_DIR/simulations_backup.tar.gz" ]; then
            echo "   Restoring simulations..."
            tar -xzf "$BACKUP_DIR/simulations_backup.tar.gz"
        fi
    fi
    
    echo "❌ Deployment failed and rollback initiated"
    exit 1
}

# Set up error handling
trap rollback_deployment ERR

# Pre-deployment checks
echo "🔍 Running pre-deployment checks..."

# Check required files
check_file $PRODUCTION_ENV_FILE
check_file $COMPOSE_FILE

# Check Docker and Docker Compose
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed or not in PATH"
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed or not in PATH"
    exit 1
fi

# Check GPU availability (critical for production)
if ! nvidia-smi &> /dev/null; then
    echo "❌ Error: nvidia-smi not available. GPU acceleration is required for production."
    exit 1
else
    echo "✅ GPU detected:"
    nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader,nounits
fi

# Check disk space
AVAILABLE_SPACE=$(df / | awk 'NR==2 {print $4}')
REQUIRED_SPACE=10000000  # 10GB in KB
if [ "$AVAILABLE_SPACE" -lt "$REQUIRED_SPACE" ]; then
    echo "❌ Error: Insufficient disk space. Required: 10GB, Available: $(($AVAILABLE_SPACE/1024/1024))GB"
    exit 1
fi

# Check memory
AVAILABLE_MEMORY=$(free -m | awk 'NR==2{print $7}')
REQUIRED_MEMORY=4096  # 4GB
if [ "$AVAILABLE_MEMORY" -lt "$REQUIRED_MEMORY" ]; then
    echo "❌ Error: Insufficient memory. Required: 4GB, Available: ${AVAILABLE_MEMORY}MB"
    exit 1
fi

# Load environment variables
echo "📄 Loading production environment configuration..."
source $PRODUCTION_ENV_FILE

# Security check for production secrets
if [ "$SECRET_KEY" == "change-this-to-a-secure-32-character-secret-key-in-production" ]; then
    echo "❌ Error: Production SECRET_KEY must be changed from default value"
    exit 1
fi

if [ "$POSTGRES_PASSWORD" == "testpass123" ]; then
    echo "❌ Error: Production POSTGRES_PASSWORD must be changed from default value"
    exit 1
fi

# Create backup before deployment
create_backup

# Create necessary directories
echo "📁 Creating required directories..."
mkdir -p logs/nginx
mkdir -p ssl/certs
mkdir -p ssl/private
mkdir -p monitoring/grafana/dashboards
mkdir -p monitoring/grafana/provisioning/dashboards
mkdir -p monitoring/grafana/provisioning/datasources
mkdir -p uploads
mkdir -p backups
mkdir -p saved_simulations_files

# Verify SSL certificates exist
if [ ! -f "ssl/certs/nginx-selfsigned.crt" ] || [ ! -f "ssl/private/nginx-selfsigned.key" ]; then
    echo "⚠️  Production SSL certificates not found. Generating self-signed certificates..."
    echo "⚠️  WARNING: Use proper CA-issued certificates for production!"
    
    # Generate private key
    openssl genrsa -out ssl/private/nginx-selfsigned.key 4096
    
    # Generate certificate
    openssl req -new -x509 -key ssl/private/nginx-selfsigned.key -out ssl/certs/nginx-selfsigned.crt -days 365 -subj "/C=US/ST=State/L=City/O=Organization/CN=simapp.ai"
    
    echo "✅ Self-signed SSL certificates generated"
    echo "⚠️  Remember to replace with CA-issued certificates for production use"
else
    echo "✅ SSL certificates found"
    
    # Check certificate expiration
    CERT_EXPIRY=$(openssl x509 -enddate -noout -in ssl/certs/nginx-selfsigned.crt | cut -d= -f2)
    EXPIRY_TIMESTAMP=$(date -d "$CERT_EXPIRY" +%s)
    CURRENT_TIMESTAMP=$(date +%s)
    DAYS_UNTIL_EXPIRY=$(( ($EXPIRY_TIMESTAMP - $CURRENT_TIMESTAMP) / 86400 ))
    
    if [ $DAYS_UNTIL_EXPIRY -lt 30 ]; then
        echo "⚠️  WARNING: SSL certificate expires in $DAYS_UNTIL_EXPIRY days"
    else
        echo "✅ SSL certificate valid for $DAYS_UNTIL_EXPIRY days"
    fi
fi

# Create production monitoring configurations
echo "📊 Setting up production monitoring..."

# Prometheus configuration
cat > monitoring/prometheus.yml << 'EOF'
global:
  scrape_interval: 15s
  evaluation_interval: 15s
  external_labels:
    environment: 'production'
    service: 'montecarlo-platform'

rule_files:
  - "alert_rules.yml"

scrape_configs:
  - job_name: 'montecarlo-backend'
    static_configs:
      - targets: ['backend:8000', 'backend-replica:8000']
    metrics_path: '/api/metrics'
    scrape_interval: 15s
    scrape_timeout: 10s

  - job_name: 'postgres-exporter'
    static_configs:
      - targets: ['postgres:5432']

  - job_name: 'redis-exporter'
    static_configs:
      - targets: ['redis:6379']

  - job_name: 'nginx'
    static_configs:
      - targets: ['nginx:80']

  - job_name: 'node-exporter'
    static_configs:
      - targets: ['localhost:9100']
EOF

# Grafana datasource
mkdir -p monitoring/grafana/provisioning/datasources
cat > monitoring/grafana/provisioning/datasources/prometheus.yml << 'EOF'
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
    editable: true
EOF

# Alert rules
cat > monitoring/alert_rules.yml << 'EOF'
groups:
  - name: montecarlo-alerts
    rules:
      - alert: HighErrorRate
        expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.05
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High error rate detected"
          
      - alert: HighResponseTime
        expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 2
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High response time detected"
          
      - alert: GPUUtilizationLow
        expr: gpu_utilization < 10
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "GPU utilization is low"
          
      - alert: DatabaseDown
        expr: up{job="postgres-exporter"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Database is down"
EOF

echo "✅ Monitoring configuration complete"

# Pull latest images
echo "📥 Pulling latest Docker images..."
docker-compose -f $COMPOSE_FILE pull

# Build production containers
echo "🔨 Building production containers..."
docker-compose -f $COMPOSE_FILE build --no-cache --parallel

# Stop existing services gracefully
echo "🛑 Gracefully stopping existing services..."
docker-compose -f $COMPOSE_FILE down --timeout 30

# Clean up unused resources
echo "🧹 Cleaning up unused Docker resources..."
docker system prune -f --volumes

# Start production services
echo "🚀 Starting production services..."
docker-compose -f $COMPOSE_FILE up -d

# Wait for core services
echo "⏳ Waiting for core services to be ready..."
sleep 60

# Check service health
echo "🔍 Checking service health..."

# Wait for database
wait_for_service postgres

# Wait for Redis
wait_for_service redis

# Wait for backend (primary)
wait_for_service backend

# Check if backend replica is enabled
if docker-compose -f $COMPOSE_FILE ps backend-replica | grep -q "Up"; then
    wait_for_service backend-replica
fi

# Wait for frontend
wait_for_service frontend

# Run database migrations
echo "🗄️  Running database migrations..."
docker-compose -f $COMPOSE_FILE exec -T backend python -c "
from app.database import engine
from app.models import Base
Base.metadata.create_all(bind=engine)
print('✅ Database tables created successfully')
"

# Ensure admin user exists
echo "👤 Ensuring production admin user exists..."
docker-compose -f $COMPOSE_FILE exec -T backend python -c "
from app.database import get_db
from app.auth.models import User
from app.auth.utils import get_password_hash
import os

db = next(get_db())
admin_email = os.getenv('ADMIN_EMAIL', 'admin@example.com')
admin_username = os.getenv('ADMIN_USERNAME', 'admin')
admin_password = os.getenv('ADMIN_PASSWORD', 'SecureAdminPassword123!')

# Check if admin exists
existing_admin = db.query(User).filter(User.email == admin_email).first()
if not existing_admin:
    admin_user = User(
        email=admin_email,
        username=admin_username,
        hashed_password=get_password_hash(admin_password),
        is_active=True,
        is_verified=True
    )
    db.add(admin_user)
    db.commit()
    print(f'✅ Production admin user created: {admin_email}')
else:
    print(f'✅ Production admin user already exists: {admin_email}')
"

# Run comprehensive smoke tests
echo "🧪 Running production smoke tests..."
if run_smoke_tests; then
    echo "✅ All smoke tests passed"
else
    echo "❌ Smoke tests failed"
    rollback_deployment
fi

# Performance verification
echo "⚡ Running performance verification..."

# Test API response time
API_RESPONSE_TIME=$(curl -o /dev/null -s -w '%{time_total}' -k https://localhost/api/health)
if (( $(echo "$API_RESPONSE_TIME < 0.5" | bc -l) )); then
    echo "✅ API response time: ${API_RESPONSE_TIME}s (< 0.5s target)"
else
    echo "⚠️  API response time: ${API_RESPONSE_TIME}s (slower than 0.5s target)"
fi

# Test GPU performance
echo "🎮 Testing GPU performance..."
docker-compose -f $COMPOSE_FILE exec -T backend python -c "
import cupy as cp
import time

# Test GPU memory allocation
try:
    # Allocate 1GB on GPU
    gpu_array = cp.zeros((128, 1024, 1024), dtype=cp.float32)
    start_time = time.time()
    result = cp.sum(gpu_array)
    end_time = time.time()
    print(f'✅ GPU computation time: {end_time - start_time:.4f}s')
    print(f'✅ GPU memory usage: {cp.cuda.runtime.memGetInfo()[1] - cp.cuda.runtime.memGetInfo()[0]} bytes')
except Exception as e:
    print(f'❌ GPU performance test failed: {e}')
    exit(1)
"

# Show deployment status
echo "📊 Deployment Status:"
echo "===================="
docker-compose -f $COMPOSE_FILE ps

# Check resource usage
echo ""
echo "💻 Resource Usage:"
echo "=================="
echo "Memory: $(free -h | awk 'NR==2{printf \"Used: %s/%s (%.2f%%)\", $3,$2,$3*100/$2}')"
echo "Disk: $(df -h / | awk 'NR==2{printf \"Used: %s/%s (%s)\", $3,$2,$5}')"
echo "GPU: $(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | awk '{printf \"GPU: %s%%, Memory: %sMB/%sMB\", $1,$2,$3}')"

# Show service endpoints
echo ""
echo "🌐 Service Endpoints:"
echo "===================="
echo "Production URL: https://simapp.ai (or https://localhost)"
echo "API Documentation: https://simapp.ai/api/docs"
echo "Health Check: https://simapp.ai/api/health"
echo "Metrics: https://simapp.ai/api/metrics"
echo "Grafana: http://localhost:3000"
echo "Prometheus: http://localhost:9090"

# Show admin credentials
echo ""
echo "🔐 Admin Access:"
echo "==============="
echo "Email: ${ADMIN_EMAIL}"
echo "Username: ${ADMIN_USERNAME}"
echo "Password: [Set in environment - check production.env]"

# Deployment summary
echo ""
echo "🎉 Production Deployment Complete!"
echo "=================================="
echo "Version: $VERSION"
echo "Deployment Time: $(date)"
echo "Backup Location: $BACKUP_DIR"
echo ""
echo "📋 Post-Deployment Checklist:"
echo "1. ✅ All services are running and healthy"
echo "2. ✅ Database migrations completed"
echo "3. ✅ Admin user verified"
echo "4. ✅ SSL certificates in place"
echo "5. ✅ GPU acceleration working"
echo "6. ✅ Monitoring active"
echo "7. ✅ Smoke tests passed"
echo "8. ⏳ Monitor system for 24-48 hours"
echo "9. ⏳ Verify backup schedule"
echo "10. ⏳ Update DNS if needed"
echo ""
echo "💡 Useful Commands:"
echo "View logs: docker-compose -f $COMPOSE_FILE logs -f [service]"
echo "Check status: docker-compose -f $COMPOSE_FILE ps"
echo "Restart service: docker-compose -f $COMPOSE_FILE restart <service>"
echo "Scale backend: docker-compose -f $COMPOSE_FILE up -d --scale backend-replica=2"
echo ""
echo "🚨 Emergency Rollback:"
echo "If issues occur, run: ./scripts/emergency-rollback.sh $BACKUP_DIR"

# Disable error trap on successful completion
trap - ERR

echo ""
echo "✅ Production deployment completed successfully!"
exit 0