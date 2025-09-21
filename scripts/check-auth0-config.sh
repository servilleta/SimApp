#!/bin/bash

echo "🔐 AUTH0 CONFIGURATION CHECKER"
echo "================================"
echo ""

echo "📋 Current Backend Configuration:"
echo "  Domain: dev-jw6k27f0v5tcgl56.eu.auth0.com"
echo "  Client ID: lFdzAAYNXsdNaAVfirZK44xRT2MvMqKr"
echo "  Callback URL: https://simapp.ai/callback"
echo "  Logout URL: https://simapp.ai"
echo ""

echo "🚨 REQUIRED ACTION: Update Auth0 Dashboard"
echo "==========================================="
echo ""
echo "1. Go to: https://manage.auth0.com/"
echo "2. Navigate to: Applications → Your Application"
echo "3. Update these settings:"
echo ""
echo "   📝 Allowed Callback URLs:"
echo "   https://simapp.ai/callback, https://www.simapp.ai/callback"
echo ""
echo "   📝 Allowed Logout URLs:"
echo "   https://simapp.ai, https://www.simapp.ai"
echo ""
echo "   📝 Allowed Web Origins:"
echo "   https://simapp.ai, https://www.simapp.ai"
echo ""
echo "   📝 Allowed Origins (CORS):"
echo "   https://simapp.ai, https://www.simapp.ai"
echo ""
echo "4. Click 'Save Changes'"
echo ""

echo "🧪 Testing Auth0 Configuration..."
echo ""

# Test Auth0 well-known endpoint
echo "Testing Auth0 JWKS endpoint..."
curl -s "https://dev-jw6k27f0v5tcgl56.eu.auth0.com/.well-known/jwks.json" > /dev/null
if [ $? -eq 0 ]; then
    echo "✅ Auth0 JWKS endpoint is accessible"
else
    echo "❌ Auth0 JWKS endpoint is not accessible"
fi

# Test backend API
echo "Testing backend API..."
curl -s -o /dev/null -w "%{http_code}" "https://simapp.ai/api/docs" | grep -q "200"
if [ $? -eq 0 ]; then
    echo "✅ Backend API is accessible"
else
    echo "❌ Backend API is not accessible (might be DNS propagation issue)"
fi

# Test frontend
echo "Testing frontend..."
curl -s -o /dev/null -w "%{http_code}" "https://simapp.ai/" | grep -q "200"
if [ $? -eq 0 ]; then
    echo "✅ Frontend is accessible"
else
    echo "❌ Frontend is not accessible"
fi

echo ""
echo "🔧 After updating Auth0 Dashboard settings:"
echo "1. Wait 1-2 minutes for changes to propagate"
echo "2. Try logging in again at: https://simapp.ai/login"
echo "3. If issues persist, run: docker-compose restart"
echo "" 