#!/bin/bash

set -e  # Exit on any error

echo "🚀 Monte Carlo Simulation Platform - Production Deployment"
echo "=================================================="

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
ENVIRONMENT=${ENVIRONMENT:-production}
SKIP_SSL=${SKIP_SSL:-false}
SKIP_BUILD=${SKIP_BUILD:-false}
ENABLE_MONITORING=${ENABLE_MONITORING:-false}

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

# Check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites..."
    
    command -v docker >/dev/null 2>&1 || { print_error "Docker is required but not installed."; exit 1; }
    command -v docker-compose >/dev/null 2>&1 || { print_error "Docker Compose is required but not installed."; exit 1; }
    
    if [ "$SKIP_SSL" != "true" ]; then
        command -v openssl >/dev/null 2>&1 || { print_error "OpenSSL is required for SSL certificates."; exit 1; }
    fi
    
    print_success "Prerequisites check passed"
}

# Setup environment
setup_environment() {
    print_status "Setting up environment..."
    
    # Create necessary directories
    mkdir -p logs/nginx ssl/certs ssl/private backups uploads monitoring
    
    # Create .env file if it doesn't exist
    if [ ! -f .env ]; then
        print_status "Creating .env file from template..."
        cp .env.example .env
        print_warning "Please review and update .env file with your production values!"
        echo "Press any key to continue..."
        read -n 1
    fi
    
    # Load environment variables
    if [ -f .env ]; then
        set -a  # automatically export all variables
        source .env
        set +a  # stop automatic export
    fi
    
    print_success "Environment setup complete"
}

# Generate SSL certificates
setup_ssl() {
    print_status "Setting up SSL certificates..."
    
    if [ "$SKIP_SSL" = "true" ]; then
        print_warning "Skipping SSL setup"
        return
    fi
    
    if [ ! -f ssl/certs/nginx-selfsigned.crt ]; then
        print_status "Generating SSL certificates..."
        chmod +x scripts/generate-ssl.sh
        ./scripts/generate-ssl.sh
        print_success "SSL certificates generated"
    else
        print_success "SSL certificates already exist"
    fi
}

# Build and deploy
deploy_services() {
    print_status "Deploying services..."
    
    # Stop any existing containers
    print_status "Stopping existing containers..."
    docker-compose -f docker-compose.deploy.yml down 2>/dev/null || true
    
    if [ "$SKIP_BUILD" != "true" ]; then
        print_status "Building containers..."
        docker-compose -f docker-compose.deploy.yml build --no-cache
    fi
    
    # Start core services
    print_status "Starting core services..."
    
    if [ "$ENABLE_MONITORING" = "true" ]; then
        docker-compose -f docker-compose.deploy.yml --profile monitoring up -d
    else
        docker-compose -f docker-compose.deploy.yml up -d
    fi
    
    print_success "Services deployed"
}

# Wait for services to be healthy
wait_for_services() {
    print_status "Waiting for services to be healthy..."
    
    # Wait for PostgreSQL
    print_status "Waiting for PostgreSQL..."
    for i in {1..30}; do
        if docker-compose -f docker-compose.deploy.yml exec -T postgres pg_isready -U montecarlo_user -d montecarlo_db >/dev/null 2>&1; then
            break
        fi
        sleep 2
    done
    
    # Wait for Redis
    print_status "Waiting for Redis..."
    for i in {1..30}; do
        if docker-compose -f docker-compose.deploy.yml exec -T redis redis-cli --no-auth-warning -a "${REDIS_PASSWORD:-change-in-production}" ping >/dev/null 2>&1; then
            break
        fi
        sleep 2
    done
    
    # Wait for Backend
    print_status "Waiting for Backend API..."
    for i in {1..60}; do
        if curl -f http://localhost:8000/api >/dev/null 2>&1; then
            break
        fi
        sleep 3
    done
    
    # Wait for Frontend
    print_status "Waiting for Frontend..."
    for i in {1..30}; do
        if curl -f http://localhost/health >/dev/null 2>&1; then
            break
        fi
        sleep 2
    done
    
    print_success "All services are healthy"
}

# Run database migrations
run_migrations() {
    print_status "Running database migrations..."
    
    docker-compose -f docker-compose.deploy.yml exec backend alembic upgrade head
    
    print_success "Database migrations completed"
}

# Health check
health_check() {
    print_status "Performing health checks..."
    
    # Test API endpoint
    if curl -f http://localhost:8000/api >/dev/null 2>&1; then
        print_success "✅ Backend API is responding"
    else
        print_error "❌ Backend API is not responding"
        return 1
    fi
    
    # Test Frontend
    if curl -f http://localhost/health >/dev/null 2>&1; then
        print_success "✅ Frontend is responding"
    else
        print_error "❌ Frontend is not responding"
        return 1
    fi
    
    # Test HTTPS (if SSL is enabled)
    if [ "$SKIP_SSL" != "true" ]; then
        if curl -k -f https://localhost/health >/dev/null 2>&1; then
            print_success "✅ HTTPS is working"
        else
            print_warning "⚠️  HTTPS might not be working properly"
        fi
    fi
    
    # Test database connection
    if docker-compose -f docker-compose.deploy.yml exec -T backend python -c "
from backend.database import engine
from sqlalchemy import text
try:
    with engine.connect() as conn:
        conn.execute(text('SELECT 1'))
    print('Database connection: OK')
except Exception as e:
    print(f'Database connection: FAILED ({e})')
    exit(1)
" >/dev/null 2>&1; then
        print_success "✅ Database connection is working"
    else
        print_error "❌ Database connection failed"
        return 1
    fi
    
    print_success "Health check passed"
}

# Show deployment summary
show_summary() {
    echo ""
    echo "🎉 DEPLOYMENT COMPLETE!"
    echo "========================================"
    echo ""
    echo "🌐 Web Application:"
    if [ "$SKIP_SSL" != "true" ]; then
        echo "   • HTTPS: https://localhost (SSL enabled)"
        echo "   • HTTP:  http://localhost (redirects to HTTPS)"
    else
        echo "   • HTTP:  http://localhost"
    fi
    echo ""
    echo "🔑 Admin Credentials:"
    echo "   • Username: ${ADMIN_USERNAME:-admin}"
    echo "   • Password: ${ADMIN_PASSWORD:-Demo123!MonteCarlo}"
    echo ""
    echo "🛠️  Service Status:"
    docker-compose -f docker-compose.deploy.yml ps
    echo ""
    
    if [ "$ENABLE_MONITORING" = "true" ]; then
        echo "📊 Monitoring:"
        echo "   • Grafana: http://localhost:3001"
        echo "   • Grafana Password: ${GRAFANA_PASSWORD:-admin123}"
        echo ""
    fi
    
    echo "📋 Useful Commands:"
    echo "   • View logs: docker-compose -f docker-compose.deploy.yml logs -f"
    echo "   • Stop services: docker-compose -f docker-compose.deploy.yml down"
    echo "   • Restart services: docker-compose -f docker-compose.deploy.yml restart"
    echo ""
    echo "⚠️  Production Notes:"
    echo "   • Update passwords in .env file"
    echo "   • Replace self-signed SSL certificates with proper ones"
    echo "   • Set up regular database backups"
    echo "   • Configure firewall rules"
    echo "   • Set up log rotation"
}

# Main deployment flow
main() {
    echo ""
    print_status "Starting deployment with options:"
    echo "   Environment: $ENVIRONMENT"
    echo "   Skip SSL: $SKIP_SSL"
    echo "   Skip Build: $SKIP_BUILD"
    echo "   Enable Monitoring: $ENABLE_MONITORING"
    echo ""
    
    check_prerequisites
    setup_environment
    setup_ssl
    deploy_services
    wait_for_services
    run_migrations
    health_check
    show_summary
    
    print_success "Deployment completed successfully! 🚀"
}

# Handle command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-ssl)
            SKIP_SSL=true
            shift
            ;;
        --skip-build)
            SKIP_BUILD=true
            shift
            ;;
        --enable-monitoring)
            ENABLE_MONITORING=true
            shift
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --skip-ssl         Skip SSL certificate generation"
            echo "  --skip-build       Skip Docker image building"
            echo "  --enable-monitoring Enable Prometheus and Grafana"
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