#!/bin/bash

set -e  # Exit on any error

echo "🌐 SimApp.ai - Domain Deployment with Let's Encrypt SSL"
echo "======================================================"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
DOMAIN="simapp.ai"
WWW_DOMAIN="www.simapp.ai"
EMAIL="admin@simapp.ai"  # Change this to your email
STAGING=${STAGING:-false}

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
    
    print_success "Prerequisites check passed"
}

# Check DNS propagation
check_dns() {
    print_status "Checking DNS propagation for $DOMAIN..."
    
    # Get current server IP
    SERVER_IP=$(curl -s ifconfig.me)
    print_status "Server IP: $SERVER_IP"
    
    # Check DNS resolution
    RESOLVED_IP=$(nslookup $DOMAIN | grep -A1 "Non-authoritative answer:" | grep "Address:" | awk '{print $2}' | head -1)
    
    if [ "$RESOLVED_IP" = "$SERVER_IP" ]; then
        print_success "DNS is properly configured for $DOMAIN"
    else
        print_error "DNS mismatch!"
        print_error "Domain $DOMAIN resolves to: $RESOLVED_IP"
        print_error "Server IP is: $SERVER_IP"
        print_error ""
        print_error "Please update your DNS records in GoDaddy:"
        print_error "1. Go to GoDaddy → My Products → DNS"
        print_error "2. Set A record for @ to: $SERVER_IP"
        print_error "3. Set A record for www to: $SERVER_IP"
        print_error "4. Wait for DNS propagation (5-30 minutes)"
        print_error ""
        read -p "Continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
}

# Setup directories
setup_directories() {
    print_status "Setting up directories..."
    
    mkdir -p certbot/www certbot/conf nginx_logs backups uploads
    
    print_success "Directories created"
}

# Create temporary nginx config for certificate generation
create_temp_nginx() {
    print_status "Creating temporary nginx configuration..."
    
    cat > nginx/nginx-temp.conf << 'EOF'
events {
    worker_connections 1024;
}

http {
    server {
        listen 80;
        server_name simapp.ai www.simapp.ai;
        
        location /.well-known/acme-challenge/ {
            root /var/www/certbot;
        }
        
        location / {
            return 200 'Setting up SSL certificates...';
            add_header Content-Type text/plain;
        }
    }
}
EOF
    
    print_success "Temporary nginx configuration created"
}

# Start services for certificate generation
start_temp_services() {
    print_status "Starting temporary services for certificate generation..."
    
    # Stop any existing containers
    docker-compose -f docker-compose.domain.yml down 2>/dev/null || true
    
    # Start with temporary config
    docker run -d --name temp-nginx \
        -p 80:80 \
        -v $(pwd)/nginx/nginx-temp.conf:/etc/nginx/nginx.conf:ro \
        -v $(pwd)/certbot/www:/var/www/certbot:ro \
        nginx:1.25-alpine
    
    print_success "Temporary nginx started"
}

# Generate SSL certificates
generate_certificates() {
    print_status "Generating Let's Encrypt SSL certificates..."
    
    # Determine if using staging or production
    if [ "$STAGING" = "true" ]; then
        CERTBOT_ARGS="--staging"
        print_warning "Using Let's Encrypt STAGING environment"
    else
        CERTBOT_ARGS=""
        print_status "Using Let's Encrypt PRODUCTION environment"
    fi
    
    # Generate certificates
    docker run --rm \
        -v $(pwd)/certbot/www:/var/www/certbot \
        -v $(pwd)/certbot/conf:/etc/letsencrypt \
        certbot/certbot:latest \
        certonly --webroot \
        --webroot-path=/var/www/certbot \
        --email $EMAIL \
        --agree-tos \
        --no-eff-email \
        $CERTBOT_ARGS \
        -d $DOMAIN \
        -d $WWW_DOMAIN
    
    if [ $? -eq 0 ]; then
        print_success "SSL certificates generated successfully!"
    else
        print_error "Failed to generate SSL certificates"
        print_error "This might be due to:"
        print_error "1. DNS not properly propagated"
        print_error "2. Domain not pointing to this server"
        print_error "3. Firewall blocking port 80"
        exit 1
    fi
}

# Stop temporary services
stop_temp_services() {
    print_status "Stopping temporary services..."
    
    docker stop temp-nginx 2>/dev/null || true
    docker rm temp-nginx 2>/dev/null || true
    
    print_success "Temporary services stopped"
}

# Deploy production services
deploy_production() {
    print_status "Deploying production services with SSL..."
    
    # Build and start all services
    docker-compose -f docker-compose.domain.yml build --no-cache
    docker-compose -f docker-compose.domain.yml up -d
    
    print_success "Production services deployed"
}

# Setup certificate renewal
setup_renewal() {
    print_status "Setting up automatic certificate renewal..."
    
    # Create renewal script
    cat > scripts/renew-certificates.sh << 'EOF'
#!/bin/bash
echo "Renewing SSL certificates..."
docker-compose -f docker-compose.domain.yml run --rm certbot renew
docker-compose -f docker-compose.domain.yml restart nginx
echo "Certificate renewal completed"
EOF
    
    chmod +x scripts/renew-certificates.sh
    
    # Add to crontab (runs twice daily)
    (crontab -l 2>/dev/null; echo "0 12,0 * * * cd $(pwd) && ./scripts/renew-certificates.sh >> logs/ssl-renewal.log 2>&1") | crontab -
    
    print_success "Automatic renewal configured (runs twice daily)"
}

# Verify deployment
verify_deployment() {
    print_status "Verifying deployment..."
    
    sleep 30  # Wait for services to start
    
    # Test HTTP redirect
    HTTP_STATUS=$(curl -s -o /dev/null -w "%{http_code}" http://$DOMAIN)
    if [ "$HTTP_STATUS" = "301" ]; then
        print_success "HTTP to HTTPS redirect working"
    else
        print_warning "HTTP redirect status: $HTTP_STATUS"
    fi
    
    # Test HTTPS
    HTTPS_STATUS=$(curl -s -o /dev/null -w "%{http_code}" https://$DOMAIN)
    if [ "$HTTPS_STATUS" = "200" ]; then
        print_success "HTTPS working correctly"
    else
        print_warning "HTTPS status: $HTTPS_STATUS"
    fi
    
    # Test SSL certificate
    SSL_EXPIRY=$(echo | openssl s_client -servername $DOMAIN -connect $DOMAIN:443 2>/dev/null | openssl x509 -noout -dates | grep notAfter | cut -d= -f2)
    if [ ! -z "$SSL_EXPIRY" ]; then
        print_success "SSL certificate valid until: $SSL_EXPIRY"
    else
        print_warning "Could not verify SSL certificate"
    fi
}

# Main deployment process
main() {
    echo "Starting domain deployment process..."
    echo "Domain: $DOMAIN"
    echo "Email: $EMAIL"
    echo "Staging: $STAGING"
    echo ""
    
    check_prerequisites
    check_dns
    setup_directories
    create_temp_nginx
    start_temp_services
    
    # Wait a moment for nginx to start
    sleep 5
    
    generate_certificates
    stop_temp_services
    deploy_production
    setup_renewal
    verify_deployment
    
    echo ""
    print_success "🎉 Domain deployment completed successfully!"
    print_success "Your Monte Carlo Platform is now available at:"
    print_success "  https://$DOMAIN"
    print_success "  https://$WWW_DOMAIN"
    echo ""
    print_status "SSL certificates will auto-renew every 12 hours"
    print_status "Check logs in: logs/ssl-renewal.log"
}

# Run main function
main "$@" 