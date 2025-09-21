#!/bin/bash

echo "🤖 MCP BrowserTools Setup for Cursor AI"
echo "======================================="
echo ""

# Function to install Chrome
install_chrome() {
    echo "📦 Installing Google Chrome..."
    wget -q -O - https://dl-ssl.google.com/linux/linux_signing_key.pub | sudo apt-key add -
    echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" | sudo tee /etc/apt/sources.list.d/google-chrome.list
    sudo apt update
    sudo apt install -y google-chrome-stable
}

# Function to install Chromium
install_chromium() {
    echo "📦 Installing Chromium..."
    sudo apt update
    sudo apt install -y chromium-browser
}

# Check for existing browsers
BROWSER_PATH=""
if command -v google-chrome &> /dev/null; then
    BROWSER_PATH="google-chrome"
    echo "✅ Google Chrome is already installed"
elif command -v google-chrome-stable &> /dev/null; then
    BROWSER_PATH="google-chrome-stable"
    echo "✅ Google Chrome Stable is already installed"
elif command -v chromium-browser &> /dev/null; then
    BROWSER_PATH="chromium-browser"
    echo "✅ Chromium is already installed"
elif command -v chromium &> /dev/null; then
    BROWSER_PATH="chromium"
    echo "✅ Chromium is already installed"
else
    echo "❌ No Chrome/Chromium browser found"
    echo ""
    echo "Which browser would you like to install?"
    echo "1) Google Chrome (recommended)"
    echo "2) Chromium (open-source alternative)"
    echo "3) Skip browser installation (I'll install manually)"
    echo ""
    read -p "Enter your choice (1-3): " choice

    case $choice in
        1)
            if install_chrome; then
                BROWSER_PATH="google-chrome-stable"
            else
                echo "❌ Failed to install Chrome"
                exit 1
            fi
            ;;
        2)
            if install_chromium; then
                BROWSER_PATH="chromium-browser"
            else
                echo "❌ Failed to install Chromium"
                exit 1
            fi
            ;;
        3)
            echo "⚠️  Skipping browser installation"
            echo "   You'll need to install Chrome/Chromium manually and update ~/.cursor/mcp.json"
            BROWSER_PATH="google-chrome"  # Default for config
            ;;
        *)
            echo "Invalid choice"
            exit 1
            ;;
    esac
fi

# Ensure Node.js is installed
if ! command -v npx &> /dev/null; then
    echo ""
    echo "📦 Installing Node.js..."
    curl -fsSL https://deb.nodesource.com/setup_lts.x | sudo -E bash -
    sudo apt-get install -y nodejs
fi

# Create MCP configuration directory
echo ""
echo "📁 Setting up MCP configuration..."
mkdir -p ~/.cursor

# Create MCP configuration with detected browser
cat > ~/.cursor/mcp.json << EOF
{
  "mcpServers": {
    "browsertools": {
      "command": "npx",
      "args": ["-y", "browsertools-mcp@latest"],
      "env": {
        "BROWSER_PATH": "${BROWSER_PATH}",
        "HEADLESS": "false",
        "DEBUG": "true"
      }
    }
  }
}
EOF

echo "✅ Created MCP configuration at ~/.cursor/mcp.json"

# Pre-download browsertools-mcp
echo ""
echo "📦 Pre-downloading BrowserTools MCP server..."
npx -y browsertools-mcp@latest --help > /dev/null 2>&1 || true

# Create test files in the current directory
echo ""
echo "🧪 Creating test files..."

# Create test HTML
cat > test-browsertools.html << 'EOF'
<!DOCTYPE html>
<html>
<head>
    <title>BrowserTools MCP Test</title>
    <style>
        body { font-family: Arial; padding: 20px; background: #f0f0f0; }
        .test-area { background: white; padding: 20px; border-radius: 8px; }
        button { margin: 5px; padding: 10px 20px; background: #007bff; color: white; border: none; border-radius: 4px; cursor: pointer; }
        button:hover { background: #0056b3; }
        #log { background: #f8f8f8; padding: 10px; margin-top: 20px; border: 1px solid #ddd; font-family: monospace; }
    </style>
</head>
<body>
    <div class="test-area">
        <h1>BrowserTools MCP Test Page</h1>
        <p>This page generates console logs for testing MCP BrowserTools integration.</p>
        
        <button onclick="generateLog()">Generate Log</button>
        <button onclick="generateError()">Generate Error</button>
        <button onclick="generateWarning()">Generate Warning</button>
        
        <div id="log">Console activity will appear here...</div>
    </div>

    <script>
        console.log('🚀 Test page loaded at', new Date().toISOString());
        
        function generateLog() {
            const msg = 'Test log message at ' + new Date().toLocaleTimeString();
            console.log('✅', msg);
            addToLog('LOG: ' + msg);
        }
        
        function generateError() {
            const msg = 'Test error at ' + new Date().toLocaleTimeString();
            console.error('❌', msg);
            addToLog('ERROR: ' + msg);
        }
        
        function generateWarning() {
            const msg = 'Test warning at ' + new Date().toLocaleTimeString();
            console.warn('⚠️', msg);
            addToLog('WARN: ' + msg);
        }
        
        function addToLog(msg) {
            const log = document.getElementById('log');
            log.innerHTML += msg + '<br>';
        }
        
        // Auto-generate some logs
        setInterval(() => {
            console.log('Heartbeat:', new Date().toLocaleTimeString());
        }, 5000);
    </script>
</body>
</html>
EOF

# Create quick start guide
cat > MCP_BROWSERTOOLS_QUICKSTART.md << 'EOF'
# MCP BrowserTools Quick Start Guide

## ✅ Setup Complete!

### 🚀 Next Steps:

1. **RESTART CURSOR** (Required!)
   - Close all Cursor windows
   - Reopen Cursor
   - MCP servers only activate after restart

2. **Test the Integration**
   After restarting Cursor, in the chat (Agent mode), type:
   ```
   @browsertools take a screenshot of https://example.com
   ```

3. **Test with Local Page**
   ```bash
   # Start local server
   python3 -m http.server 8000
   
   # In Cursor chat:
   @browsertools navigate to http://localhost:8000/test-browsertools.html and capture console logs
   ```

### 📝 Example Commands:

**Screenshot a page:**
```
@browsertools screenshot http://localhost:8000
```

**Capture console logs:**
```
@browsertools go to http://localhost:8000 and show me all console output
```

**Debug errors:**
```
@browsertools visit the page and tell me about any JavaScript errors
```

**Interactive actions:**
```
@browsertools click the "Generate Error" button and show what happens
```

**Wait for content:**
```
@browsertools wait for the page to fully load, then take a screenshot
```

### 🔧 Troubleshooting:

**@browsertools not appearing?**
- Make sure you restarted Cursor
- Check that ~/.cursor/mcp.json exists
- Try: `cat ~/.cursor/mcp.json`

**Browser issues?**
- Test browser: `${BROWSER_PATH} --version`
- Update BROWSER_PATH in ~/.cursor/mcp.json if needed

**Need help?**
- MCP logs: Check Cursor's output panel
- Test npx: `npx -y browsertools-mcp@latest --version`

### 💡 Pro Tips:
- Be specific about what you want (screenshot, logs, errors)
- The browser stays open between commands for efficiency
- Combine with code generation for automated testing
- Use "wait for" commands for dynamic content
EOF

echo ""
echo "✅ MCP BrowserTools setup complete!"
echo ""
echo "📋 IMPORTANT NEXT STEPS:"
echo "1. ⚠️  RESTART CURSOR NOW (close and reopen)"
echo "2. 📖 Read MCP_BROWSERTOOLS_QUICKSTART.md"
echo "3. 🧪 Test with: @browsertools screenshot https://example.com"
echo ""
echo "🔧 Configuration saved to: ~/.cursor/mcp.json"
echo "🌐 Browser configured: ${BROWSER_PATH}"
echo ""
echo "After restarting Cursor, you'll see @browsertools in the chat tools!" 