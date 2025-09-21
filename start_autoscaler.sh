#!/bin/bash
# Paperspace Auto-Scaler Startup Script
# This script starts the auto-scaling system for Monte Carlo platform

set -e

echo "🚀 Starting Paperspace Auto-Scaler for Monte Carlo Platform..."

# Check if configuration exists
if [ ! -f ".env.autoscaler" ]; then
    echo "❌ Configuration file .env.autoscaler not found!"
    echo "Please create it with your Paperspace API credentials:"
    echo ""
    echo "PAPERSPACE_API_KEY=your_api_key_here"
    echo "PRIMARY_MACHINE_ID=your_primary_machine_id"
    echo "SECONDARY_MACHINE_ID=your_secondary_machine_id"
    echo ""
    exit 1
fi

# Install required Python packages
echo "📦 Installing required packages..."
pip install aiohttp psutil redis

# Check if nginx is installed
if ! command -v nginx &> /dev/null; then
    echo "⚠️  Nginx not found. Installing nginx for load balancing..."
    sudo apt-get update
    sudo apt-get install -y nginx
fi

# Backup existing nginx config
if [ -f "/etc/nginx/sites-enabled/default" ]; then
    echo "💾 Backing up existing nginx configuration..."
    sudo cp /etc/nginx/sites-enabled/default /etc/nginx/sites-enabled/default.backup
fi

# Install our load balancer config
echo "🔧 Installing load balancer configuration..."
sudo cp nginx_autoscale.conf /etc/nginx/sites-available/monte-carlo-autoscale
sudo ln -sf /etc/nginx/sites-available/monte-carlo-autoscale /etc/nginx/sites-enabled/default

# Test nginx configuration
echo "🧪 Testing nginx configuration..."
sudo nginx -t

if [ $? -eq 0 ]; then
    echo "✅ Nginx configuration is valid"
    sudo systemctl reload nginx
else
    echo "❌ Nginx configuration error. Please check nginx_autoscale.conf"
    exit 1
fi

# Create systemd service for auto-scaler
echo "🔧 Creating systemd service for auto-scaler..."
sudo tee /etc/systemd/system/monte-carlo-autoscaler.service > /dev/null <<EOF
[Unit]
Description=Monte Carlo Platform Auto-Scaler
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$(pwd)
Environment=PATH=/usr/bin:/usr/local/bin:$(dirname $(which python3))
ExecStart=$(which python3) paperspace_autoscaler.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Reload systemd and start service
sudo systemctl daemon-reload
sudo systemctl enable monte-carlo-autoscaler

echo "🎯 Auto-scaler setup complete!"
echo ""
echo "📋 Next steps:"
echo "1. Fill in your .env.autoscaler with Paperspace credentials"
echo "2. Start the auto-scaler: sudo systemctl start monte-carlo-autoscaler"
echo "3. Check status: sudo systemctl status monte-carlo-autoscaler"
echo "4. View logs: sudo journalctl -u monte-carlo-autoscaler -f"
echo ""
echo "💡 The auto-scaler will:"
echo "   - Monitor your platform metrics every 2 minutes"
echo "   - Start secondary instance when you have 6+ users"
echo "   - Stop secondary instance when load drops below 4 users"
echo "   - Maintain 10-minute cooldown between scaling actions"
echo ""
echo "💰 Expected costs:"
echo "   - Primary instance (always on): ~$367/month"
echo "   - Secondary instance (auto-scaled): ~$0-367/month depending on usage"
echo "   - Total range: $367-734/month for 6-16 concurrent users"







