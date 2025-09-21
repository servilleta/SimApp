#!/bin/bash

# Setup Secure Environment Variables
echo "🔧 Setting up secure environment variables..."

cd frontend

# Create .env file with secure values
cat > .env << 'EOF'
# API Configuration
VITE_API_URL=http://localhost:8000/api
VITE_DEMO_API_KEY=ak_f957cbf46d0f2cbe93dc0db510b574e6_sk_4e49ee331757b1e80a76df40e0185846b9ba94f0c8c4894baa54155fcc23fb9d

# Demo Authentication (for development only)
VITE_DEMO_USERNAME=admin
VITE_DEMO_PASSWORD=NewSecurePassword123!

# Auth0 Configuration (if using Auth0)
VITE_AUTH0_DOMAIN=your-auth0-domain.auth0.com
VITE_AUTH0_CLIENT_ID=your_auth0_client_id
VITE_AUTH0_AUDIENCE=your_api_identifier

# External Services
VITE_STRIPE_PUBLISHABLE_KEY=pk_test_your_stripe_key

# Security Settings
VITE_ENABLE_CONSOLE_PROTECTION=true
VITE_ENABLE_DEVTOOLS_DETECTION=true

# Development Settings
VITE_DEBUG_MODE=false
VITE_LOG_LEVEL=error
EOF

echo "✅ Created .env file with secure values"
echo ""
echo "🔑 NEW API KEY: ak_f957cbf46d0f2cbe93dc0db510b574e6_sk_4e49ee331757b1e80a76df40e0185846b9ba94f0c8c4894baa54155fcc23fb9d"
echo "🔒 NEW ADMIN PASSWORD: NewSecurePassword123!"
echo ""
echo "⚠️  IMPORTANT NEXT STEPS:"
echo "1. Change your admin password in the backend to: NewSecurePassword123!"
echo "2. Revoke the old compromised API key"
echo "3. Restart your frontend: npm run dev"
echo ""
echo "📁 Environment file created at: frontend/.env"
