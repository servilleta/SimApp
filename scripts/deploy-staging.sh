#!/bin/bash

# Monte Carlo Platform - Staging Deployment Script
# Usage: ./deploy-staging.sh [version]

set -euo pipefail

VERSION=${1:-$(git describe --tags --dirty --always)}
STAGING_ENV_FILE="staging.env"
COMPOSE_FILE="docker-compose.staging.yml"

echo "🧪 Deploying Monte Carlo Platform to Staging Environment"
echo "Version: $VERSION"
echo "=================================="

# Function to check if file exists
check_file() {
    if [ ! -f "$1" ]; then
        echo "❌ Error: $1 not found!"
        exit 1
    fi
}

# Function to wait for service health
wait_for_service() {
    local service=$1
    local max_attempts=30
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

# Pre-deployment checks
echo "🔍 Running pre-deployment checks..."

# Check required files
check_file $STAGING_ENV_FILE
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

# Check GPU availability
if ! nvidia-smi &> /dev/null; then
    echo "⚠️  Warning: nvidia-smi not available. GPU acceleration may not work."
else
    echo "✅ GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
fi

# Load environment variables
echo "📄 Loading staging environment configuration..."
source $STAGING_ENV_FILE

# Create necessary directories
echo "📁 Creating required directories..."
mkdir -p logs/nginx
mkdir -p ssl/staging
mkdir -p monitoring
mkdir -p uploads
mkdir -p backups

# Generate staging SSL certificates if they don't exist
if [ ! -f "ssl/staging/nginx-staging.crt" ] || [ ! -f "ssl/staging/nginx-staging.key" ]; then
    echo "🔐 Generating staging SSL certificates..."
    
    # Generate private key
    openssl genrsa -out ssl/staging/nginx-staging.key 2048
    
    # Generate certificate
    openssl req -new -x509 -key ssl/staging/nginx-staging.key -out ssl/staging/nginx-staging.crt -days 365 -subj "/C=US/ST=State/L=City/O=Organization/CN=staging.simapp.ai"
    
    echo "✅ Staging SSL certificates generated"
else
    echo "✅ Staging SSL certificates already exist"
fi

# Create Nginx staging configuration
echo "🌐 Creating Nginx staging configuration..."
cat > nginx/nginx-staging.conf << 'EOF'
events {
    worker_connections 1024;
}

http {
    upstream backend {
        server backend:8000;
    }
    
    upstream frontend {
        server frontend:80;
    }
    
    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    limit_req_zone $binary_remote_addr zone=auth:10m rate=1r/s;
    
    # Staging server with SSL
    server {
        listen 443 ssl http2;
        server_name staging.simapp.ai localhost;
        
        ssl_certificate /etc/ssl/certs/nginx-staging.crt;
        ssl_certificate_key /etc/ssl/certs/nginx-staging.key;
        
        ssl_protocols TLSv1.2 TLSv1.3;
        ssl_ciphers 'ECDHE+AESGCM:ECDHE+CHACHA20:DHE+AESGCM:DHE+CHACHA20:!aNULL:!MD5:!DSS';
        ssl_prefer_server_ciphers off;
        
        # Security headers
        add_header X-Frame-Options DENY;
        add_header X-Content-Type-Options nosniff;
        add_header X-XSS-Protection "1; mode=block";
        add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
        
        # API routes
        location /api/ {
            limit_req zone=api burst=20 nodelay;
            proxy_pass http://backend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
        
        # Frontend routes
        location / {
            proxy_pass http://frontend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
        
        # Health check
        location /health {
            access_log off;
            return 200 "healthy\n";
            add_header Content-Type text/plain;
        }
    }
    
    # HTTP redirect to HTTPS
    server {
        listen 80;
        server_name staging.simapp.ai localhost;
        return 301 https://$server_name$request_uri;
    }
}
EOF

# Create Prometheus staging configuration
echo "📊 Creating Prometheus staging configuration..."
cat > monitoring/prometheus-staging.yml << 'EOF'
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "alert_rules.yml"

scrape_configs:
  - job_name: 'montecarlo-backend'
    static_configs:
      - targets: ['backend:8000']
    metrics_path: '/api/metrics'
    scrape_interval: 15s

  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres:5432']

  - job_name: 'redis'
    static_configs:
      - targets: ['redis:6379']

  - job_name: 'nginx'
    static_configs:
      - targets: ['nginx:80']
EOF

# Stop existing staging services
echo "🛑 Stopping existing staging services..."
docker-compose -f $COMPOSE_FILE down --remove-orphans || true

# Pull latest images
echo "📥 Pulling latest Docker images..."
docker-compose -f $COMPOSE_FILE pull

# Build staging containers
echo "🔨 Building staging containers..."
docker-compose -f $COMPOSE_FILE build --no-cache

# Start staging services
echo "🚀 Starting staging services..."
docker-compose -f $COMPOSE_FILE up -d

# Wait for core services
echo "⏳ Waiting for core services to be ready..."
sleep 30

# Check service health
echo "🔍 Checking service health..."

# Wait for database
wait_for_service postgres

# Wait for Redis
wait_for_service redis

# Wait for backend
wait_for_service backend

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

# Create admin user
echo "👤 Creating staging admin user..."
docker-compose -f $COMPOSE_FILE exec -T backend python -c "
from app.database import get_db
from app.auth.models import User
from app.auth.utils import get_password_hash
import os

db = next(get_db())
admin_email = os.getenv('STAGING_ADMIN_EMAIL', 'admin@staging.simapp.ai')
admin_username = os.getenv('STAGING_ADMIN_USERNAME', 'staging_admin')
admin_password = os.getenv('STAGING_ADMIN_PASSWORD', 'StagingAdmin123!')

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
    print(f'✅ Staging admin user created: {admin_email}')
else:
    print(f'✅ Staging admin user already exists: {admin_email}')
"

# Run basic health checks
echo "🏥 Running health checks..."

# Check API health
if curl -f -k https://localhost:8443/api/health >/dev/null 2>&1; then
    echo "✅ API health check passed"
else
    echo "❌ API health check failed"
    exit 1
fi

# Check frontend
if curl -f -k https://localhost:8443/ >/dev/null 2>&1; then
    echo "✅ Frontend health check passed"
else
    echo "❌ Frontend health check failed"
    exit 1
fi

# Check GPU functionality
echo "🎮 Testing GPU functionality..."
docker-compose -f $COMPOSE_FILE exec -T backend python -c "
try:
    import cupy as cp
    cp.cuda.runtime.getDeviceCount()
    print('✅ GPU acceleration is working')
except Exception as e:
    print(f'⚠️  GPU test failed: {e}')
"

# Run staging tests
echo "🧪 Running staging test suite..."
if [ -f "scripts/test-staging.sh" ]; then
    ./scripts/test-staging.sh
else
    echo "⚠️  Staging test script not found, skipping tests"
fi

# Show service status
echo "📊 Service Status:"
docker-compose -f $COMPOSE_FILE ps

# Show useful information
echo ""
echo "🎉 Staging Deployment Complete!"
echo "=================================="
echo "Version: $VERSION"
echo "Staging URL: https://localhost:8443"
echo "API URL: https://localhost:8443/api"
echo "Grafana: http://localhost:3000 (admin / ${STAGING_GRAFANA_PASSWORD})"
echo "Prometheus: http://localhost:9090"
echo ""
echo "Admin Credentials:"
echo "Email: ${STAGING_ADMIN_EMAIL}"
echo "Username: ${STAGING_ADMIN_USERNAME}"
echo "Password: ${STAGING_ADMIN_PASSWORD}"
echo ""
echo "📋 Next Steps:"
echo "1. Run comprehensive testing"
echo "2. Validate GPU performance"
echo "3. Test file upload functionality"
echo "4. Verify Monte Carlo simulations"
echo "5. Check monitoring dashboards"
echo ""
echo "💡 Useful Commands:"
echo "View logs: docker-compose -f $COMPOSE_FILE logs -f"
echo "Stop staging: docker-compose -f $COMPOSE_FILE down"
echo "Restart service: docker-compose -f $COMPOSE_FILE restart <service>"

exit 0
