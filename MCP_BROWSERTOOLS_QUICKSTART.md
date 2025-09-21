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
