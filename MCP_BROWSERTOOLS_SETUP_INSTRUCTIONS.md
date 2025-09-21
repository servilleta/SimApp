# MCP BrowserTools Setup Instructions

## What is MCP BrowserTools?

MCP (Model Context Protocol) BrowserTools allows Cursor's AI assistant to:
- 📸 Take screenshots of web pages
- 📝 Capture console logs automatically
- 🐛 Debug JavaScript errors
- 🤖 Interact with web pages (click buttons, fill forms)
- 🔍 Extract content from pages

This means you can ask Cursor to "debug my localhost:8000 page" and it will automatically capture screenshots, console logs, and errors!

## Setup Steps

### Step 1: Run the Setup Script

```bash
./scripts/install-browser-and-setup-mcp.sh
```

This script will:
- Check if Chrome/Chromium is installed (and offer to install if not)
- Configure MCP BrowserTools for Cursor
- Create test files for verification

### Step 2: Choose Browser Installation (if needed)

If no browser is found, you'll see:
```
Which browser would you like to install?
1) Google Chrome (recommended)
2) Chromium (open-source alternative)
3) Skip browser installation (I'll install manually)
```

Choose option 1 or 2 for automatic installation.

### Step 3: Restart Cursor (IMPORTANT!)

**⚠️ This step is critical - MCP won't work without restarting!**

1. Close ALL Cursor windows
2. Wait a few seconds
3. Reopen Cursor

### Step 4: Verify MCP is Active

After restarting Cursor:
1. Open any project
2. Go to the chat (Composer/Agent mode)
3. Type `@` - you should see `@browsertools` in the suggestions

### Step 5: Test It Out!

Try these commands in Cursor chat:

**Simple screenshot:**
```
@browsertools take a screenshot of https://example.com
```

**Debug your local app:**
```
@browsertools navigate to http://localhost:8000 and capture all console logs
```

**Test with the included test page:**
```bash
# In terminal:
python3 -m http.server 8000

# In Cursor chat:
@browsertools go to http://localhost:8000/test-browsertools.html, click the "Generate Error" button, and show me what happens
```

## Example Use Cases

### 1. Debug Console Errors
```
@browsertools visit http://localhost:3000, wait for it to load, and tell me about any JavaScript errors or console warnings
```

### 2. Visual Regression Testing
```
@browsertools take a screenshot of the homepage before and after I deploy these changes
```

### 3. Form Testing
```
@browsertools fill out the contact form with test data and show me what happens when I submit it
```

### 4. Performance Monitoring
```
@browsertools monitor the network tab while loading the page and tell me about slow requests
```

## Troubleshooting

### @browsertools not showing up?
1. Make sure you restarted Cursor completely
2. Check config exists: `cat ~/.cursor/mcp.json`
3. Try refreshing Cursor's window (Ctrl+R)

### Browser not opening?
- The browser runs in background by default
- To see it, edit `~/.cursor/mcp.json` and set `"HEADLESS": "false"`

### Getting errors?
- Check MCP is properly configured: `npx -y browsertools-mcp@latest --version`
- Ensure your browser path is correct in `~/.cursor/mcp.json`

## Advanced Tips

1. **Combine with code generation:**
   ```
   @browsertools analyze my React app at localhost:3000 and generate Playwright tests based on what you see
   ```

2. **Debug specific elements:**
   ```
   @browsertools inspect the element with class "error-message" and tell me why it's appearing
   ```

3. **Monitor real-time changes:**
   ```
   @browsertools watch the console while I click through the checkout flow
   ```

## Next Steps

- Read `MCP_BROWSERTOOLS_QUICKSTART.md` for more examples
- Explore other MCP servers at https://github.com/modelcontextprotocol
- Join the MCP community for tips and tricks

Happy debugging! 🚀 