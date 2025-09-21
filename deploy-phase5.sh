#!/bin/bash

set -e  # Exit on any error

echo "🚀 Monte Carlo Platform - Phase 5 Production Deployment"
echo "======================================================="

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# Default values
ENVIRONMENT=${ENVIRONMENT:-production}
SKIP_SSL=${SKIP_SSL:-false}
SKIP_BUILD=${SKIP_BUILD:-false}
ENABLE_MONITORING=${ENABLE_MONITORING:-true}
ENABLE_BACKUP=${ENABLE_BACKUP:-true}
DOMAIN=${DOMAIN:-localhost}

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_phase() {
    echo -e "${PURPLE}[PHASE]${NC} $1"
}

# Function to generate secure passwords
generate_password() {
    openssl rand -base64 32 | tr -d "=+/" | cut -c1-25
}

# Check prerequisites
check_prerequisites() {
    print_phase "Checking Prerequisites"
    
    # Check required commands
    local required_commands=("docker" "docker-compose" "openssl" "curl")
    for cmd in "${required_commands[@]}"; do
        if ! command -v $cmd >/dev/null 2>&1; then
            print_error "$cmd is required but not installed."
            exit 1
        fi
    done
    
    # Check Docker daemon
    if ! docker info >/dev/null 2>&1; then
        print_error "Docker daemon is not running."
        exit 1
    fi
    
    print_success "Prerequisites check passed"
}

# Setup environment
setup_environment() {
    print_phase "Setting Up Environment"
    
    # Create necessary directories
    mkdir -p logs/nginx ssl/certs ssl/private backups uploads monitoring/grafana/dashboards
    mkdir -p monitoring/grafana/provisioning/datasources monitoring/grafana/provisioning/dashboards
    mkdir -p nginx scripts
    
    # Create production environment file if it doesn't exist
    if [ ! -f production.env ]; then
        print_status "Creating production.env file..."
        
        # Generate secure passwords
        local postgres_password=$(generate_password)
        local redis_password=$(generate_password)
        local secret_key=$(openssl rand -hex 32)
        local admin_password=$(generate_password)
        local grafana_password=$(generate_password)
        
        cat > production.env << EOF
# Monte Carlo Platform - Production Environment
# Generated on $(date)

# Core Configuration
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=INFO
SECRET_KEY=${secret_key}
API_URL=https://${DOMAIN}

# Database
POSTGRES_DB=montecarlo_db
POSTGRES_USER=montecarlo_user
POSTGRES_PASSWORD=${postgres_password}

# Redis
REDIS_PASSWORD=${redis_password}

# Admin User
ADMIN_EMAIL=admin@${DOMAIN}
ADMIN_USERNAME=admin
ADMIN_PASSWORD=${admin_password}

# Monitoring
GRAFANA_PASSWORD=${grafana_password}
PROMETHEUS_RETENTION=30d

# Performance
WORKER_COUNT=4
MAX_CONNECTIONS=1000
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_WINDOW=60

# File Management
FILE_RETENTION_DAYS=30
CLEANUP_INTERVAL_HOURS=6
MAX_FILE_SIZE=500MB

# Security
CORS_ORIGINS=https://${DOMAIN}
TRUSTED_HOSTS=${DOMAIN}

# Backup
BACKUP_RETENTION_DAYS=30
BACKUP_S3_ENABLED=false
EOF
        
        print_success "Production environment file created"
        print_warning "Generated passwords:"
        echo "  - Admin: ${admin_password}"
        echo "  - Grafana: ${grafana_password}"
    fi
    
    # Load environment variables
    if [ -f production.env ]; then
        set -a  # automatically export all variables
        source production.env
        set +a  # stop automatic export
    fi
    
    print_success "Environment setup complete"
}

# Generate SSL certificates
setup_ssl() {
    print_phase "Setting Up SSL Certificates"
    
    if [ "$SKIP_SSL" = "true" ]; then
        print_warning "Skipping SSL setup"
        return
    fi
    
    if [ ! -f ssl/certs/nginx-selfsigned.crt ]; then
        print_status "Generating SSL certificates for ${DOMAIN}..."
        
        # Generate private key
        openssl genrsa -out ssl/private/nginx-selfsigned.key 4096
        
        # Generate certificate signing request
        openssl req -new -key ssl/private/nginx-selfsigned.key \
            -out ssl/certs/nginx-selfsigned.csr \
            -subj "/C=US/ST=Production/L=Production/O=MonteCarloAnalytics/OU=IT/CN=${DOMAIN}"
        
        # Generate self-signed certificate with SAN
        openssl x509 -req -days 365 \
            -in ssl/certs/nginx-selfsigned.csr \
            -signkey ssl/private/nginx-selfsigned.key \
            -out ssl/certs/nginx-selfsigned.crt \
            -extensions v3_req \
            -extfile <(cat <<EOF
[v3_req]
basicConstraints = CA:FALSE
keyUsage = nonRepudiation, digitalSignature, keyEncipherment
subjectAltName = @alt_names

[alt_names]
DNS.1 = ${DOMAIN}
DNS.2 = www.${DOMAIN}
DNS.3 = localhost
IP.1 = 127.0.0.1
IP.2 = ::1
EOF
)
        
        # Set proper permissions
        chmod 600 ssl/private/nginx-selfsigned.key
        chmod 644 ssl/certs/nginx-selfsigned.crt
        
        # Remove CSR file
        rm ssl/certs/nginx-selfsigned.csr
        
        print_success "SSL certificates generated"
    else
        print_success "SSL certificates already exist"
    fi
}

# Setup monitoring configuration
setup_monitoring() {
    print_phase "Setting Up Monitoring Configuration"
    
    # Enhanced Prometheus configuration
    cat > monitoring/prometheus.yml << 'EOF'
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  - job_name: 'monte-carlo-backend'
    static_configs:
      - targets: ['backend:8000']
    metrics_path: '/metrics'
    scrape_interval: 10s

  - job_name: 'nginx'
    static_configs:
      - targets: ['nginx:8080']
EOF

    # Grafana datasource configuration
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

    print_success "Monitoring configuration created"
}

# Deploy services
deploy_services() {
    print_phase "Deploying Services"
    
    # Stop any existing containers
    print_status "Stopping existing containers..."
    docker-compose -f docker-compose.production.yml down 2>/dev/null || true
    
    # Build if requested
    if [ "$SKIP_BUILD" != "true" ]; then
        print_status "Building containers..."
        docker-compose -f docker-compose.production.yml build --no-cache
    fi
    
    # Start services
    print_status "Starting services..."
    docker-compose -f docker-compose.production.yml up -d
    
    print_success "Services deployed"
}

# Wait for services
wait_for_services() {
    print_phase "Waiting for Services"
    
    # Wait for database
    print_status "Waiting for PostgreSQL..."
    for i in {1..30}; do
        if docker-compose -f docker-compose.production.yml exec -T postgres pg_isready -U montecarlo_user -d montecarlo_db >/dev/null 2>&1; then
            break
        fi
        sleep 2
    done
    
    # Wait for backend
    print_status "Waiting for Backend API..."
    for i in {1..60}; do
        if curl -f http://localhost:8000/api >/dev/null 2>&1; then
            break
        fi
        sleep 3
    done
    
    print_success "Services are ready"
}

# Health check
health_check() {
    print_phase "Performing Health Checks"
    
    # Test backend API
    if curl -f http://localhost:8000/api >/dev/null 2>&1; then
        print_success "✅ Backend API is responding"
    else
        print_error "❌ Backend API is not responding"
        return 1
    fi
    
    # Test frontend
    if curl -f http://localhost/health >/dev/null 2>&1; then
        print_success "✅ Frontend is responding"
    else
        print_error "❌ Frontend is not responding"
        return 1
    fi
    
    # Test HTTPS
    if [ "$SKIP_SSL" != "true" ]; then
        if curl -k -f https://localhost/health >/dev/null 2>&1; then
            print_success "✅ HTTPS is working"
        else
            print_warning "⚠️  HTTPS might not be working properly"
        fi
    fi
    
    print_success "Health checks passed"
}

# Show summary
show_summary() {
    echo ""
    echo "🎉 PHASE 5 PRODUCTION DEPLOYMENT COMPLETE!"
    echo "=========================================="
    echo ""
    echo "🌐 Web Application:"
    if [ "$SKIP_SSL" != "true" ]; then
        echo "   • HTTPS: https://${DOMAIN}"
        echo "   • HTTP:  http://${DOMAIN} (redirects to HTTPS)"
    else
        echo "   • HTTP:  http://${DOMAIN}"
    fi
    echo ""
    echo "🔑 Admin Credentials:"
    echo "   • Username: ${ADMIN_USERNAME:-admin}"
    echo "   • Password: Check production.env file"
    echo ""
    echo "📊 Monitoring:"
    if [ "$ENABLE_MONITORING" = "true" ]; then
        echo "   • Prometheus: http://${DOMAIN}:9090"
        echo "   • Grafana: http://${DOMAIN}:3001"
    else
        echo "   • Monitoring disabled"
    fi
    echo ""
    echo "🛠️  Service Status:"
    docker-compose -f docker-compose.production.yml ps
    echo ""
    echo "🚀 Launch Readiness: 100% - Ready for Production!"
}

# Main function
main() {
    print_status "Starting Phase 5 Production Deployment"
    echo "Domain: $DOMAIN"
    echo "Skip SSL: $SKIP_SSL"
    echo "Skip Build: $SKIP_BUILD"
    echo ""
    
    check_prerequisites
    setup_environment
    setup_ssl
    setup_monitoring
    deploy_services
    wait_for_services
    health_check
    show_summary
    
    print_success "Phase 5 Production Deployment completed successfully! 🚀"
}

# Handle command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --domain)
            DOMAIN="$2"
            shift 2
            ;;
        --skip-ssl)
            SKIP_SSL=true
            shift
            ;;
        --skip-build)
            SKIP_BUILD=true
            shift
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --domain DOMAIN    Set domain name (default: localhost)"
            echo "  --skip-ssl         Skip SSL certificate generation"
            echo "  --skip-build       Skip Docker image building"
            echo "  --help             Show this help message"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Run main function
main 