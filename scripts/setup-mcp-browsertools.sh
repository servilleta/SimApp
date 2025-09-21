#!/bin/bash

echo "🤖 Setting up MCP BrowserTools for Cursor AI Integration"
echo "========================================================"
echo ""

# Check if Cursor is installed
if ! command -v cursor &> /dev/null; then
    echo "❌ Cursor is not installed or not in PATH"
    echo "   Please install Cursor first: https://cursor.sh"
    exit 1
fi

# Check if Chrome is installed
if ! command -v google-chrome &> /dev/null; then
    echo "❌ Google Chrome is not installed"
    echo "   Installing Chrome is required for BrowserTools"
    echo "   Run: sudo apt update && sudo apt install google-chrome-stable"
    exit 1
fi

# Ensure MCP config directory exists
echo "📁 Creating MCP configuration directory..."
mkdir -p ~/.cursor

# Create or update MCP configuration
echo "📝 Configuring MCP BrowserTools..."
cat > ~/.cursor/mcp.json << 'EOF'
{
  "mcpServers": {
    "browsertools": {
      "command": "npx",
      "args": ["-y", "browsertools-mcp@latest"],
      "env": {
        "BROWSER_PATH": "google-chrome",
        "HEADLESS": "false",
        "DEBUG": "true"
      }
    }
  }
}
EOF

# Test if npx is available
if ! command -v npx &> /dev/null; then
    echo "❌ npx is not installed. Installing Node.js..."
    curl -fsSL https://deb.nodesource.com/setup_lts.x | sudo -E bash -
    sudo apt-get install -y nodejs
fi

# Pre-download browsertools-mcp to ensure it's cached
echo "📦 Pre-downloading BrowserTools MCP server..."
npx -y browsertools-mcp@latest --help > /dev/null 2>&1 || true

# Create test HTML file
echo "🧪 Creating test page..."
cat > test-browsertools.html << 'EOF'
<!DOCTYPE html>
<html>
<head>
    <title>BrowserTools MCP Test Page</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background: #f0f0f0;
        }
        .console-test {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin: 20px 0;
        }
        button {
            background: #0066cc;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 4px;
            cursor: pointer;
            margin: 5px;
        }
        button:hover {
            background: #0052a3;
        }
        #output {
            background: #f8f8f8;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 4px;
            margin-top: 20px;
            font-family: monospace;
            white-space: pre-wrap;
        }
    </style>
</head>
<body>
    <h1>BrowserTools MCP Test Page</h1>
    
    <div class="console-test">
        <h2>Console Log Tests</h2>
        <button onclick="testConsoleLog()">Test console.log</button>
        <button onclick="testConsoleError()">Test console.error</button>
        <button onclick="testConsoleWarn()">Test console.warn</button>
        <button onclick="testMultipleLogs()">Test Multiple Logs</button>
        <button onclick="throwError()">Throw Error</button>
        
        <div id="output">Click buttons to generate console output...</div>
    </div>

    <script>
        // Initial log
        console.log('🚀 BrowserTools Test Page Loaded!');
        console.log('Timestamp:', new Date().toISOString());
        
        function testConsoleLog() {
            console.log('✅ This is a test console.log message');
            console.log('Current time:', new Date().toLocaleTimeString());
            updateOutput('Sent console.log message');
        }
        
        function testConsoleError() {
            console.error('❌ This is a test error message');
            console.error('Error details:', { code: 500, message: 'Test error' });
            updateOutput('Sent console.error message');
        }
        
        function testConsoleWarn() {
            console.warn('⚠️ This is a test warning');
            console.warn('Warning: Performance might be impacted');
            updateOutput('Sent console.warn message');
        }
        
        function testMultipleLogs() {
            console.group('Multiple Logs Test');
            console.log('Log 1: Starting process...');
            console.log('Log 2: Processing data...');
            console.warn('Log 3: Warning - slow operation');
            console.log('Log 4: Process complete!');
            console.groupEnd();
            updateOutput('Sent multiple grouped logs');
        }
        
        function throwError() {
            updateOutput('Throwing an error...');
            setTimeout(() => {
                throw new Error('This is a test error thrown from the page');
            }, 100);
        }
        
        function updateOutput(message) {
            const output = document.getElementById('output');
            const timestamp = new Date().toLocaleTimeString();
            output.textContent += `[${timestamp}] ${message}\n`;
        }
        
        // Periodic console activity
        let counter = 0;
        setInterval(() => {
            counter++;
            console.log(`Periodic log #${counter} - Everything is working fine`);
        }, 5000);
    </script>
</body>
</html>
EOF

# Create a usage guide
echo "📚 Creating usage guide..."
cat > MCP_BROWSERTOOLS_GUIDE.md << 'EOF'
# MCP BrowserTools Usage Guide

## Setup Complete! 🎉

MCP BrowserTools is now configured for Cursor. Here's how to use it:

## 1. Restart Cursor
**Important**: You must restart Cursor for MCP changes to take effect.
- Close all Cursor windows
- Reopen Cursor

## 2. Verify MCP is Active
In Cursor's chat (Agent mode), you should see @browsertools available in the tools list.

## 3. Basic Usage Examples

### Capture Screenshot
```
@browsertools please take a screenshot of http://localhost:8000
```

### Get Console Logs
```
@browsertools navigate to http://localhost:8000 and capture all console logs
```

### Debug Errors
```
@browsertools go to http://localhost:8000, wait for the page to load, and tell me about any JavaScript errors
```

### Full Page Analysis
```
@browsertools analyze http://localhost:8000 - take a screenshot, capture console logs, and check for any errors
```

### Interactive Debugging
```
@browsertools navigate to http://localhost:8000, click the "Submit" button, and capture what happens in the console
```

## 4. Advanced Features

### Wait for Elements
```
@browsertools go to the page, wait for the element with class "loaded" to appear, then take a screenshot
```

### Extract Page Data
```
@browsertools extract all the text content from the main article on the page
```

### Monitor Network
```
@browsertools monitor network requests while loading the page and report any failed requests
```

## 5. Troubleshooting

### MCP Not Working?
1. Ensure Cursor was restarted after configuration
2. Check ~/.cursor/mcp.json exists
3. Try: `npx -y browsertools-mcp@latest --version` in terminal

### Browser Not Opening?
- The browser runs in the background by default
- Set HEADLESS: "false" in mcp.json to see the browser

### No Console Logs?
- Make sure the page has console.log statements
- Ask specifically for "console logs" or "console output"

## 6. Test Page
Open the test page to verify everything works:
```bash
python3 -m http.server 8080
# Then open http://localhost:8080/test-browsertools.html
```

## 7. Tips
- Be specific about what you want (screenshot, console, errors)
- The browser session persists, so you can do multiple actions
- Use "wait for" commands for dynamic content
- Combine with code generation for automated testing
EOF

echo ""
echo "✅ MCP BrowserTools setup complete!"
echo ""
echo "📋 Next Steps:"
echo "1. ⚠️  RESTART CURSOR (required for MCP to activate)"
echo "2. 📖 Read MCP_BROWSERTOOLS_GUIDE.md for usage examples"
echo "3. 🧪 Test with: python3 -m http.server 8080"
echo "   Then ask Cursor: '@browsertools screenshot localhost:8080/test-browsertools.html'"
echo ""
echo "💡 Quick Test After Restart:"
echo "   In Cursor chat: '@browsertools take a screenshot of https://example.com'"
echo ""
echo "🔧 Config location: ~/.cursor/mcp.json" 