#!/bin/bash

echo "🚀 Setting up Chrome-Cursor Debugging Integration"
echo "================================================"

# Install dependencies for browser debugger
echo "📦 Installing dependencies..."
cd scripts && npm install && cd ..

# Create a convenient launcher script
cat > launch-debug-chrome.sh << 'EOF'
#!/bin/bash
echo "🌐 Launching Chrome in debug mode..."
echo "Port: 9222"
echo ""

# Kill any existing Chrome debug instances
pkill -f "remote-debugging-port=9222" 2>/dev/null

# Launch Chrome with debugging
google-chrome \
  --remote-debugging-port=9222 \
  --user-data-dir=/tmp/chrome-debug \
  --auto-open-devtools-for-tabs \
  http://localhost:8000 &

# Wait for Chrome to start
sleep 2

# Start the console log streamer
echo "📡 Starting console log streamer..."
cd scripts && npm start
EOF

chmod +x launch-debug-chrome.sh

echo ""
echo "✅ Setup complete! Here's how to use it:"
echo ""
echo "1. BUILT-IN DEBUGGER (Recommended for breakpoints):"
echo "   - Press F5 in Cursor and select 'Launch Chrome against localhost'"
echo "   - Set breakpoints directly in your code"
echo ""
echo "2. CONSOLE LOG STREAMING:"
echo "   - Run: ./launch-debug-chrome.sh"
echo "   - Console logs will stream in terminal"
echo ""
echo "3. MCP BROWSER TOOLS (AI-powered):"
echo "   - Restart Cursor after MCP config"
echo "   - Use @browsertools in chat to capture screenshots/logs"
echo ""
echo "📝 Pro Tips:"
echo "   - Use Ctrl+Shift+P → 'Debug: Open Link' for quick debugging"
echo "   - Console logs appear in Cursor's Debug Console when using F5"
echo "   - For React apps, install React Developer Tools extension" 