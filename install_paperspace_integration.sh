#!/bin/bash
# Paperspace API Integration Installation Script
# ==============================================

set -e  # Exit on any error

echo "🚀 Installing Paperspace API Integration for SimApp"
echo "===================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Functions
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

# Check if we're in the right directory
if [ ! -f "paperspace_api_manager.py" ]; then
    print_error "Please run this script from the SimApp root directory"
    exit 1
fi

print_status "Step 1: Installing Python dependencies..."

# Check if pip is available
if ! command -v pip3 &> /dev/null; then
    print_error "pip3 is not installed. Please install Python 3 and pip first."
    exit 1
fi

# Install Paperspace-specific dependencies
pip3 install -r requirements-paperspace.txt || {
    print_warning "Some dependencies may have failed to install. Continuing with core dependencies..."
    pip3 install paperspace requests psutil
}

print_success "Python dependencies installed"

print_status "Step 2: Setting up directories and permissions..."

# Create logs directory
mkdir -p logs
mkdir -p kubernetes/logs

# Set executable permissions
chmod +x paperspace_api_manager.py
chmod +x kubernetes/deploy-simapp.sh
chmod +x kubernetes/setup-worker-node.sh
chmod +x worker-node-complete-setup.sh
chmod +x install_paperspace_integration.sh

print_success "Directories and permissions configured"

print_status "Step 3: Checking environment configuration..."

# Check if PAPERSPACE_API_KEY is set
if [ -z "$PAPERSPACE_API_KEY" ]; then
    print_warning "PAPERSPACE_API_KEY environment variable is not set"
    echo ""
    echo "To complete the setup, you need to:"
    echo "1. Get your API key from https://console.paperspace.com"
    echo "2. Set the environment variable:"
    echo "   export PAPERSPACE_API_KEY=\"your_api_key_here\""
    echo "3. Add it to your shell profile (e.g., ~/.bashrc)"
    echo ""
else
    print_success "PAPERSPACE_API_KEY is configured"
fi

print_status "Step 4: Testing API connectivity..."

# Test basic Python import
python3 -c "import requests; print('✅ Requests module available')" || {
    print_error "Failed to import requests module"
    exit 1
}

# Test Paperspace module if API key is available
if [ ! -z "$PAPERSPACE_API_KEY" ]; then
    python3 -c "
import sys
sys.path.append('.')
try:
    from paperspace_api_manager import PaperspaceAPIManager
    print('✅ Paperspace API manager imports successfully')
except Exception as e:
    print(f'⚠️ API manager import error: {e}')
" || print_warning "Paperspace API manager could not be imported"
else
    print_warning "Skipping API connectivity test (no API key)"
fi

print_status "Step 5: Creating service integration..."

# Update Docker Compose if it exists
if [ -f "docker-compose.yml" ]; then
    # Add environment variable to backend service if not already present
    if ! grep -q "PAPERSPACE_API_KEY" docker-compose.yml; then
        print_status "Adding Paperspace API key to Docker Compose..."
        # This is a basic check - in production you'd want more sophisticated YAML manipulation
        print_warning "Please manually add PAPERSPACE_API_KEY to your docker-compose.yml backend service environment"
    fi
fi

print_status "Step 6: Setting up log rotation..."

# Create logrotate configuration
sudo tee /etc/logrotate.d/simapp-paperspace > /dev/null << 'EOF'
/home/paperspace/SimApp/logs/paperspace_api.log {
    daily
    missingok
    rotate 7
    compress
    delaycompress
    notifempty
    copytruncate
}
EOF

print_success "Log rotation configured"

print_status "Step 7: Installation verification..."

# Test script execution
./paperspace_api_manager.py --help > /dev/null 2>&1 || {
    print_warning "API manager script might have issues - check permissions and dependencies"
}

print_success "Basic verification completed"

echo ""
echo "🎉 Paperspace API Integration Installation Complete!"
echo "=================================================="
echo ""
echo "📋 Next Steps:"
echo "1. Set your Paperspace API key: export PAPERSPACE_API_KEY=\"your_key\""
echo "2. Test the integration: ./paperspace_api_manager.py list"
echo "3. Restart your backend service to load the new endpoints"
echo "4. Access the new API endpoints at /api/infrastructure/*"
echo ""
echo "📚 Documentation:"
echo "- Setup Guide: ./paperspace_setup_guide.md"
echo "- Multi-server Documentation: ./multiserver.txt"
echo ""
echo "🔗 API Endpoints:"
echo "- GET /api/infrastructure/status - Cluster status"
echo "- POST /api/infrastructure/scale - Manual scaling"
echo "- GET /api/infrastructure/cost-optimization - Cost analysis"
echo "- GET /api/infrastructure/machines - List all machines"
echo ""
echo "✅ Ready for on-demand multi-server scaling!"

# Optional: Test API if key is available
if [ ! -z "$PAPERSPACE_API_KEY" ]; then
    echo ""
    print_status "Testing API connectivity..."
    if ./paperspace_api_manager.py cluster-status > /dev/null 2>&1; then
        print_success "API connectivity test passed!"
    else
        print_warning "API connectivity test failed - check your API key and network"
    fi
fi

echo ""
print_success "Installation script completed successfully!"
