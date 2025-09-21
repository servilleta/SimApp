/**
 * Console Protection and Anti-Tampering Module
 * Detects and mitigates console-based attacks
 */

import { securityConfig } from './securityConfig';

class ConsoleProtection {
  constructor() {
    this.devtoolsOpen = false;
    this.originalConsole = { ...console };
    this.protectionEnabled = securityConfig.enableConsoleProtection;
    this.detectionEnabled = securityConfig.enableDevtoolsDetection;
    
    if (this.protectionEnabled) {
      this.initializeProtection();
    }
  }

  initializeProtection() {
    this.detectDevTools();
    this.protectConsoleFunctions();
    this.detectConsoleModification();
    this.addTamperingDetection();
    
    console.log('🛡️ Console protection initialized');
  }

  detectDevTools() {
    if (!this.detectionEnabled) return;

    let devtools = { open: false, orientation: null };
    const threshold = 160;

    const check = () => {
      const heightDiff = window.outerHeight - window.innerHeight;
      const widthDiff = window.outerWidth - window.innerWidth;
      
      if (heightDiff > threshold || widthDiff > threshold) {
        if (!devtools.open) {
          devtools.open = true;
          this.devtoolsOpen = true;
          this.onDevToolsDetected();
        }
      } else {
        if (devtools.open) {
          devtools.open = false;
          this.devtoolsOpen = false;
        }
      }
    };

    // Check every 500ms
    setInterval(check, 500);

    // Additional detection method using console.profile
    let profileCount = 0;
    const profileDetection = () => {
      console.profile();
      console.profileEnd();
      profileCount++;
      
      if (profileCount > 2) {
        this.onDevToolsDetected();
      }
    };

    // Trigger profile detection
    setTimeout(profileDetection, 100);
  }

  protectConsoleFunctions() {
    // Freeze important global objects
    try {
      Object.freeze(window.fetch);
      Object.freeze(XMLHttpRequest.prototype);
      Object.freeze(localStorage);
      Object.freeze(sessionStorage);
    } catch (e) {
      console.warn('Could not freeze some global objects');
    }

    // Override console methods to detect tampering
    const protectedMethods = ['log', 'warn', 'error', 'info', 'debug'];
    
    protectedMethods.forEach(method => {
      const original = console[method];
      console[method] = (...args) => {
        // Check if console method has been tampered with
        if (console[method].toString().indexOf('[native code]') === -1) {
          this.onTamperingDetected('console', method);
        }
        
        // Filter sensitive information from logs
        const filteredArgs = args.map(arg => 
          typeof arg === 'string' ? this.filterSensitiveInfo(arg) : arg
        );
        
        return original.apply(console, filteredArgs);
      };
    });
  }

  detectConsoleModification() {
    // Check if console has been modified
    const checkConsole = () => {
      if (typeof console.log !== 'function' || 
          console.log.toString().indexOf('[native code]') === -1) {
        this.onTamperingDetected('console', 'modification');
      }
    };

    // Check every 2 seconds
    setInterval(checkConsole, 2000);
  }

  addTamperingDetection() {
    // Detect if important functions have been overridden
    const importantGlobals = [
      'fetch',
      'XMLHttpRequest',
      'eval',
      'Function'
    ];

    const originalGlobals = {};
    importantGlobals.forEach(name => {
      originalGlobals[name] = window[name];
    });

    const checkTampering = () => {
      importantGlobals.forEach(name => {
        if (window[name] !== originalGlobals[name]) {
          this.onTamperingDetected('global', name);
        }
      });
    };

    // Check every 3 seconds
    setInterval(checkTampering, 3000);
  }

  filterSensitiveInfo(text) {
    // Remove potential secrets from logs
    return text
      .replace(/ak_[a-z0-9_]{50,}/gi, '[API_KEY_REDACTED]')
      .replace(/sk_[a-z0-9_]{50,}/gi, '[SECRET_KEY_REDACTED]')
      .replace(/Demo123!MonteCarlo/gi, '[PASSWORD_REDACTED]')
      .replace(/Bearer\s+[a-zA-Z0-9._-]{20,}/gi, 'Bearer [TOKEN_REDACTED]');
  }

  onDevToolsDetected() {
    if (!securityConfig.debugMode) {
      console.warn('🔍 Developer tools detected. Some features may be restricted.');
      
      // In production, you might want to:
      // 1. Redirect to a warning page
      // 2. Disable sensitive functionality  
      // 3. Log the event for security monitoring
      
      this.logSecurityEvent('devtools_detected', {
        userAgent: navigator.userAgent,
        timestamp: new Date().toISOString(),
        url: window.location.href
      });
    }
  }

  onTamperingDetected(type, details) {
    console.error(`🚨 Security Alert: ${type} tampering detected - ${details}`);
    
    this.logSecurityEvent('tampering_detected', {
      type,
      details,
      timestamp: new Date().toISOString(),
      url: window.location.href
    });

    // In production, you might want to:
    // 1. Disable the application
    // 2. Clear sensitive data
    // 3. Redirect to security warning
  }

  logSecurityEvent(eventType, data) {
    // Store security events (in production, send to security monitoring)
    const securityEvents = JSON.parse(
      localStorage.getItem('security_events') || '[]'
    );
    
    securityEvents.push({
      event: eventType,
      data,
      timestamp: new Date().toISOString()
    });
    
    // Keep only last 50 events
    if (securityEvents.length > 50) {
      securityEvents.splice(0, securityEvents.length - 50);
    }
    
    localStorage.setItem('security_events', JSON.stringify(securityEvents));
  }

  // Anti-debugging measures
  enableAntiDebugging() {
    // Detect debugger statements
    setInterval(() => {
      debugger;
    }, 1000);

    // Disable right-click context menu
    document.addEventListener('contextmenu', (e) => {
      e.preventDefault();
      this.onTamperingDetected('ui', 'context_menu');
    });

    // Disable F12, Ctrl+Shift+I, etc.
    document.addEventListener('keydown', (e) => {
      if (e.key === 'F12' || 
          (e.ctrlKey && e.shiftKey && e.key === 'I') ||
          (e.ctrlKey && e.shiftKey && e.key === 'J') ||
          (e.ctrlKey && e.key === 'U')) {
        e.preventDefault();
        this.onTamperingDetected('ui', 'keyboard_shortcut');
      }
    });
  }

  disable() {
    this.protectionEnabled = false;
    console.log('🔓 Console protection disabled');
  }
}

// Initialize console protection
let consoleProtection = null;

export const initializeConsoleProtection = () => {
  if (!consoleProtection && securityConfig.enableConsoleProtection) {
    consoleProtection = new ConsoleProtection();
  }
  return consoleProtection;
};

export const disableConsoleProtection = () => {
  if (consoleProtection) {
    consoleProtection.disable();
  }
};

export default ConsoleProtection;
