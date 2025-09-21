const CDP = require('chrome-remote-interface');

async function connectToChrome() {
    let client;
    try {
        // Connect to Chrome DevTools Protocol
        client = await CDP({ port: 9222 });
        
        const { Console, Runtime } = client;
        
        // Enable console and runtime
        await Console.enable();
        await Runtime.enable();
        
        // Listen for console messages
        Console.on('messageAdded', (params) => {
            const { level, text, url, line, column } = params.message;
            console.log(`[${level}] ${text}`);
            if (url) {
                console.log(`  at ${url}:${line}:${column}`);
            }
        });
        
        // Listen for runtime exceptions
        Runtime.on('exceptionThrown', (params) => {
            console.error('Exception:', params.exceptionDetails);
        });
        
        console.log('Connected to Chrome DevTools. Console logs will stream here...');
        
    } catch (err) {
        console.error('Error connecting to Chrome:', err);
    }
}

// Start Chrome with: google-chrome --remote-debugging-port=9222
connectToChrome(); 