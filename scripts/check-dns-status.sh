#!/bin/bash

echo "🌐 DNS PROPAGATION CHECKER FOR SIMAPP.AI"
echo "========================================"
echo ""

echo "📍 Current DNS Resolution:"
echo "-------------------------"

# Check A record for simapp.ai
echo -n "simapp.ai A record: "
dig +short simapp.ai A | head -1

# Check A record for www.simapp.ai
echo -n "www.simapp.ai A record: "
dig +short www.simapp.ai A | head -1

echo ""
echo "🎯 Expected IP: 209.51.170.185"
echo ""

# Check if simapp.ai resolves to correct IP
CURRENT_IP=$(dig +short simapp.ai A | head -1)
if [ "$CURRENT_IP" = "209.51.170.185" ]; then
    echo "✅ simapp.ai resolves correctly"
else
    echo "❌ simapp.ai resolves to: $CURRENT_IP (should be 209.51.170.185)"
fi

# Check if www.simapp.ai resolves to correct IP
WWW_IP=$(dig +short www.simapp.ai A | head -1)
if [ "$WWW_IP" = "209.51.170.185" ]; then
    echo "✅ www.simapp.ai resolves correctly"
else
    echo "❌ www.simapp.ai resolves to: $WWW_IP (should be 209.51.170.185)"
fi

echo ""
echo "🔍 Testing HTTP/HTTPS Connectivity:"
echo "-----------------------------------"

# Test HTTP redirect
echo -n "HTTP redirect test: "
HTTP_STATUS=$(curl -s -o /dev/null -w "%{http_code}" -L --max-time 10 "http://simapp.ai/" 2>/dev/null)
if [ "$HTTP_STATUS" = "200" ]; then
    echo "✅ Working (Status: $HTTP_STATUS)"
else
    echo "❌ Failed (Status: $HTTP_STATUS)"
fi

# Test HTTPS
echo -n "HTTPS connection test: "
HTTPS_STATUS=$(curl -s -o /dev/null -w "%{http_code}" --max-time 10 "https://simapp.ai/" 2>/dev/null)
if [ "$HTTPS_STATUS" = "200" ]; then
    echo "✅ Working (Status: $HTTPS_STATUS)"
else
    echo "❌ Failed (Status: $HTTPS_STATUS)"
fi

# Test API
echo -n "API connection test: "
API_STATUS=$(curl -s -o /dev/null -w "%{http_code}" --max-time 10 "https://simapp.ai/api/docs" 2>/dev/null)
if [ "$API_STATUS" = "200" ]; then
    echo "✅ Working (Status: $API_STATUS)"
else
    echo "❌ Failed (Status: $API_STATUS)"
fi

echo ""
echo "💡 RECOMMENDATIONS:"
echo "==================="

if [ "$CURRENT_IP" != "209.51.170.185" ] || [ "$WWW_IP" != "209.51.170.185" ]; then
    echo "🔧 DNS Issue Detected:"
    echo "   1. Check your GoDaddy DNS settings"
    echo "   2. Ensure A records point to 209.51.170.185"
    echo "   3. DNS propagation can take up to 24 hours"
    echo "   4. Use https://209.51.170.185/login temporarily"
    echo ""
fi

if [ "$HTTPS_STATUS" != "200" ] || [ "$API_STATUS" != "200" ]; then
    echo "🔧 Connection Issue Detected:"
    echo "   1. Some requests may still go to GoDaddy servers"
    echo "   2. This is normal during DNS propagation"
    echo "   3. Try clearing your browser cache"
    echo "   4. Use incognito/private browsing mode"
    echo ""
fi

echo "✅ Auth0 Configuration Updated:"
echo "   - Backend callback URL: https://simapp.ai/callback"
echo "   - Frontend uses dynamic origin"
echo "   - Ready for login once DNS propagates"
echo ""

echo "🚀 Next Steps:"
echo "   1. Update Auth0 Dashboard with simapp.ai URLs"
echo "   2. Wait for DNS propagation (check every hour)"
echo "   3. Test login at https://simapp.ai/login"
echo "   4. If urgent, use https://209.51.170.185/login temporarily" 