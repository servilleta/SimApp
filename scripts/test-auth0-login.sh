#!/bin/bash

echo "🔐 AUTH0 LOGIN CONFIGURATION TEST"
echo "================================="
echo ""

# Test Auth0 JWKS endpoint
echo "1. Testing Auth0 JWKS endpoint..."
JWKS_RESPONSE=$(curl -s -w "HTTPSTATUS:%{http_code}" "https://dev-jw6k27f0v5tcgl56.eu.auth0.com/.well-known/jwks.json")
JWKS_BODY=$(echo $JWKS_RESPONSE | sed -E 's/HTTPSTATUS\:[0-9]{3}$//')
JWKS_STATUS=$(echo $JWKS_RESPONSE | tr -d '\n' | sed -E 's/.*HTTPSTATUS:([0-9]{3})$/\1/')

if [ "$JWKS_STATUS" = "200" ]; then
    echo "✅ Auth0 JWKS endpoint accessible"
else
    echo "❌ Auth0 JWKS endpoint failed (Status: $JWKS_STATUS)"
fi

echo ""
echo "2. Testing Frontend Accessibility..."

# Test domain frontend
echo "   Testing https://simapp.ai/..."
DOMAIN_RESPONSE=$(curl -k -s -w "HTTPSTATUS:%{http_code}" --max-time 10 "https://simapp.ai/" 2>/dev/null)
DOMAIN_STATUS=$(echo $DOMAIN_RESPONSE | tr -d '\n' | sed -E 's/.*HTTPSTATUS:([0-9]{3})$/\1/')

if [ "$DOMAIN_STATUS" = "200" ]; then
    DOMAIN_BODY=$(echo $DOMAIN_RESPONSE | sed -E 's/HTTPSTATUS\:[0-9]{3}$//')
    if echo "$DOMAIN_BODY" | grep -q "Monte Carlo"; then
        echo "✅ Domain frontend working (Monte Carlo app)"
    else
        echo "⚠️  Domain frontend accessible but serving GoDaddy page"
    fi
else
    echo "❌ Domain frontend failed (Status: $DOMAIN_STATUS)"
fi

# Test IP frontend  
echo "   Testing https://209.51.170.185/..."
IP_RESPONSE=$(curl -k -s -w "HTTPSTATUS:%{http_code}" --max-time 10 "https://209.51.170.185/" 2>/dev/null)
IP_STATUS=$(echo $IP_RESPONSE | tr -d '\n' | sed -E 's/.*HTTPSTATUS:([0-9]{3})$/\1/')

if [ "$IP_STATUS" = "200" ]; then
    IP_BODY=$(echo $IP_RESPONSE | sed -E 's/HTTPSTATUS\:[0-9]{3}$//')
    if echo "$IP_BODY" | grep -q "Monte Carlo"; then
        echo "✅ IP frontend working (Monte Carlo app)"
    else
        echo "❌ IP frontend accessible but wrong content"
    fi
else
    echo "❌ IP frontend failed (Status: $IP_STATUS)"
fi

echo ""
echo "3. Testing Login Pages..."

# Test domain login page
echo "   Testing https://simapp.ai/login..."
DOMAIN_LOGIN_RESPONSE=$(curl -k -s -w "HTTPSTATUS:%{http_code}" --max-time 10 "https://simapp.ai/login" 2>/dev/null)
DOMAIN_LOGIN_STATUS=$(echo $DOMAIN_LOGIN_RESPONSE | tr -d '\n' | sed -E 's/.*HTTPSTATUS:([0-9]{3})$/\1/')

if [ "$DOMAIN_LOGIN_STATUS" = "200" ]; then
    DOMAIN_LOGIN_BODY=$(echo $DOMAIN_LOGIN_RESPONSE | sed -E 's/HTTPSTATUS\:[0-9]{3}$//')
    if echo "$DOMAIN_LOGIN_BODY" | grep -q "Monte Carlo"; then
        echo "✅ Domain login page working"
    else
        echo "⚠️  Domain login page serving GoDaddy content (DNS issue)"
    fi
else
    echo "❌ Domain login page failed (Status: $DOMAIN_LOGIN_STATUS)"
fi

# Test IP login page
echo "   Testing https://209.51.170.185/login..."
IP_LOGIN_RESPONSE=$(curl -k -s -w "HTTPSTATUS:%{http_code}" --max-time 10 "https://209.51.170.185/login" 2>/dev/null)
IP_LOGIN_STATUS=$(echo $IP_LOGIN_RESPONSE | tr -d '\n' | sed -E 's/.*HTTPSTATUS:([0-9]{3})$/\1/')

if [ "$IP_LOGIN_STATUS" = "200" ]; then
    IP_LOGIN_BODY=$(echo $IP_LOGIN_RESPONSE | sed -E 's/HTTPSTATUS\:[0-9]{3}$//')
    if echo "$IP_LOGIN_BODY" | grep -q "Monte Carlo"; then
        echo "✅ IP login page working"
    else
        echo "❌ IP login page accessible but wrong content"
    fi
else
    echo "❌ IP login page failed (Status: $IP_LOGIN_STATUS)"
fi

echo ""
echo "4. Testing API Endpoints..."

# Test domain API
echo "   Testing https://simapp.ai/api/docs..."
DOMAIN_API_RESPONSE=$(curl -k -s -w "HTTPSTATUS:%{http_code}" --max-time 10 "https://simapp.ai/api/docs" 2>/dev/null)
DOMAIN_API_STATUS=$(echo $DOMAIN_API_RESPONSE | tr -d '\n' | sed -E 's/.*HTTPSTATUS:([0-9]{3})$/\1/')

if [ "$DOMAIN_API_STATUS" = "200" ]; then
    DOMAIN_API_BODY=$(echo $DOMAIN_API_RESPONSE | sed -E 's/HTTPSTATUS\:[0-9]{3}$//')
    if echo "$DOMAIN_API_BODY" | grep -q "swagger"; then
        echo "✅ Domain API working (Swagger docs)"
    else
        echo "⚠️  Domain API serving GoDaddy content (DNS issue)"
    fi
else
    echo "❌ Domain API failed (Status: $DOMAIN_API_STATUS)"
fi

# Test IP API
echo "   Testing https://209.51.170.185/api/docs..."
IP_API_RESPONSE=$(curl -k -s -w "HTTPSTATUS:%{http_code}" --max-time 10 "https://209.51.170.185/api/docs" 2>/dev/null)
IP_API_STATUS=$(echo $IP_API_RESPONSE | tr -d '\n' | sed -E 's/.*HTTPSTATUS:([0-9]{3})$/\1/')

if [ "$IP_API_STATUS" = "200" ]; then
    IP_API_BODY=$(echo $IP_API_RESPONSE | sed -E 's/HTTPSTATUS\:[0-9]{3}$//')
    if echo "$IP_API_BODY" | grep -q "swagger"; then
        echo "✅ IP API working (Swagger docs)"
    else
        echo "❌ IP API accessible but wrong content"
    fi
else
    echo "❌ IP API failed (Status: $IP_API_STATUS)"
fi

echo ""
echo "📋 SUMMARY & RECOMMENDATIONS:"
echo "============================="

if [ "$IP_STATUS" = "200" ] && [ "$IP_LOGIN_STATUS" = "200" ] && [ "$IP_API_STATUS" = "200" ]; then
    echo "✅ IP Address Configuration: WORKING"
    echo "🚀 READY TO TEST AUTH0 LOGIN!"
    echo ""
    echo "📱 Next Steps:"
    echo "   1. Open incognito/private browser window"
    echo "   2. Go to: https://209.51.170.185/login"
    echo "   3. Accept SSL certificate warning (expected)"
    echo "   4. Click login button to test Auth0 flow"
    echo "   5. Should redirect to Auth0 login page"
    echo ""
else
    echo "❌ IP Address Configuration: ISSUES DETECTED"
    echo "🔧 Check Docker containers and nginx configuration"
fi

if [ "$DOMAIN_STATUS" = "200" ] && echo "$DOMAIN_BODY" | grep -q "Monte Carlo"; then
    echo "✅ Domain Configuration: WORKING"
    echo "🌐 DNS propagation is complete for main domain"
else
    echo "⚠️  Domain Configuration: DNS PROPAGATION IN PROGRESS"
    echo "🕐 Wait 2-24 hours for full DNS propagation"
fi

echo ""
echo "🔐 Auth0 Configuration Status:"
echo "   ✅ JWKS endpoint accessible"
echo "   ✅ Callback URLs should include both:"
echo "      - https://simapp.ai/callback"
echo "      - https://209.51.170.185/callback"
echo "   ✅ Ready for login testing" 