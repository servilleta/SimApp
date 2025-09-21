#!/bin/bash

echo "🌐 SimApp.ai Domain Status Check"
echo "================================="

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

SERVER_IP="209.51.170.185"
DOMAIN="simapp.ai"

echo -e "${BLUE}[INFO]${NC} Checking domain configuration..."
echo ""

# Check DNS resolution
echo "1. DNS Resolution:"
echo "   Main domain: $DOMAIN"
RESOLVED_IPS=$(dig +short $DOMAIN)
echo "   Resolved IPs:"
for ip in $RESOLVED_IPS; do
    if [ "$ip" = "$SERVER_IP" ]; then
        echo -e "   ✅ $ip (Your server)"
    else
        echo -e "   ❌ $ip (GoDaddy/Other)"
    fi
done

echo ""
echo "   WWW domain: www.$DOMAIN"
WWW_RESOLVED=$(dig +short www.$DOMAIN)
echo "   Resolved IPs:"
for ip in $WWW_RESOLVED; do
    if [ "$ip" = "$SERVER_IP" ]; then
        echo -e "   ✅ $ip (Your server)"
    else
        echo -e "   ❌ $ip (GoDaddy/Other)"
    fi
done

echo ""
echo "2. Server Response Test:"

# Test direct IP
echo "   Testing direct IP ($SERVER_IP):"
HTTP_STATUS=$(curl -s -o /dev/null -w "%{http_code}" http://$SERVER_IP)
if [ "$HTTP_STATUS" = "301" ]; then
    echo -e "   ✅ HTTP redirect working ($HTTP_STATUS)"
else
    echo -e "   ❌ HTTP status: $HTTP_STATUS"
fi

HTTPS_STATUS=$(curl -s -o /dev/null -w "%{http_code}" https://$SERVER_IP -k)
if [ "$HTTPS_STATUS" = "200" ]; then
    echo -e "   ✅ HTTPS working ($HTTPS_STATUS)"
else
    echo -e "   ❌ HTTPS status: $HTTPS_STATUS"
fi

# Test domain
echo ""
echo "   Testing domain ($DOMAIN):"
DOMAIN_HTTP=$(curl -s -o /dev/null -w "%{http_code}" http://$DOMAIN)
echo -e "   HTTP status: $DOMAIN_HTTP"

DOMAIN_HTTPS=$(curl -s -o /dev/null -w "%{http_code}" https://$DOMAIN)
echo -e "   HTTPS status: $DOMAIN_HTTPS"

# SSL Certificate check
echo ""
echo "3. SSL Certificate:"
SSL_INFO=$(echo | openssl s_client -servername $DOMAIN -connect $SERVER_IP:443 2>/dev/null | openssl x509 -noout -dates 2>/dev/null)
if [ ! -z "$SSL_INFO" ]; then
    echo -e "   ✅ SSL certificate valid"
    echo "   $SSL_INFO"
else
    echo -e "   ❌ SSL certificate check failed"
fi

echo ""
echo "4. Container Status:"
docker-compose -f docker-compose.domain.yml ps --format "table {{.Name}}\t{{.Status}}\t{{.Ports}}"

echo ""
echo "🎯 Summary:"
if echo "$RESOLVED_IPS" | grep -q "$SERVER_IP"; then
    echo -e "✅ Your server IP is in DNS resolution"
    if [ "$DOMAIN_HTTPS" = "200" ]; then
        echo -e "✅ Domain is accessible via HTTPS"
        echo -e "${GREEN}🎉 SimApp.ai is working!${NC}"
        echo ""
        echo "Access your site at:"
        echo "  🌐 https://simapp.ai"
        echo "  🌐 https://www.simapp.ai"
    else
        echo -e "⚠️  Domain HTTPS needs attention"
        echo -e "${YELLOW}The SSL certificates are ready, but DNS routing needs time to propagate${NC}"
    fi
else
    echo -e "❌ Your server IP not found in DNS"
    echo -e "${RED}Please check GoDaddy DNS settings${NC}"
fi

echo ""
echo "🔧 If domain isn't working:"
echo "1. Wait for DNS propagation (can take up to 24 hours)"
echo "2. Check GoDaddy DNS settings"
echo "3. Ensure @ record points to: $SERVER_IP"
echo "4. Clear DNS cache: sudo systemctl flush-dns (if available)" 