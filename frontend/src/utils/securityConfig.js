/**
 * Security Configuration Module
 * Centralized security settings and validation
 */

// Security configuration from environment variables
export const securityConfig = {
  // Console protection
  enableConsoleProtection: import.meta.env.VITE_ENABLE_CONSOLE_PROTECTION === 'true',
  enableDevtoolsDetection: import.meta.env.VITE_ENABLE_DEVTOOLS_DETECTION === 'true',
  
  // Debug settings
  debugMode: import.meta.env.VITE_DEBUG_MODE === 'true',
  logLevel: import.meta.env.VITE_LOG_LEVEL || 'error',
  
  // API settings
  apiUrl: import.meta.env.VITE_API_URL || 'http://localhost:8000/api',
  
  // Validate that required environment variables are present
  isConfigValid() {
    const required = [
      'VITE_API_URL'
    ];
    
    const missing = required.filter(key => !import.meta.env[key]);
    
    if (missing.length > 0) {
      console.warn('Missing required environment variables:', missing);
      return false;
    }
    
    return true;
  }
};

// Validate configuration on import
if (!securityConfig.isConfigValid()) {
  console.error('Security configuration validation failed');
}

// Environment variable validation helper
export const getSecureEnvVar = (key, defaultValue = '') => {
  const value = import.meta.env[key];
  
  if (!value && defaultValue === '') {
    console.warn(`Environment variable ${key} is not set`);
  }
  
  return value || defaultValue;
};

// Secret validation (ensure no hardcoded secrets)
export const validateNoHardcodedSecrets = (value) => {
  // Check for common secret patterns
  const secretPatterns = [
    /ak_[a-z0-9_]{50,}/i,  // API key pattern
    /sk_[a-z0-9_]{50,}/i,  // Secret key pattern
    /Demo123!MonteCarlo/i,  // Our specific demo password
    /password.*['"]/i,      // Password assignments
  ];
  
  for (const pattern of secretPatterns) {
    if (pattern.test(value)) {
      console.error('SECURITY VIOLATION: Hardcoded secret detected');
      return false;
    }
  }
  
  return true;
};

export default securityConfig;
