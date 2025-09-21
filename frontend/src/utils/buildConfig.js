/**
 * Build Configuration for Security
 * Handles production-specific security settings
 */

// Detect if we're in production build
export const isProduction = import.meta.env.PROD;
export const isDevelopment = import.meta.env.DEV;

// Production security configuration
export const buildConfig = {
  // Security features enabled in production
  enableObfuscation: isProduction,
  enableConsoleProtection: isProduction || import.meta.env.VITE_ENABLE_CONSOLE_PROTECTION === 'true',
  enableDevtoolsDetection: isProduction || import.meta.env.VITE_ENABLE_DEVTOOLS_DETECTION === 'true',
  
  // Debug features (disabled in production)
  enableDebugLogs: isDevelopment && import.meta.env.VITE_DEBUG_MODE === 'true',
  enableSourceMaps: isDevelopment,
  enableDevtools: isDevelopment,
  
  // Build optimizations
  enableMinification: isProduction,
  enableTreeShaking: isProduction,
  enableChunkSplitting: isProduction,
  
  // Security headers (should be set by server)
  securityHeaders: {
    'X-Content-Type-Options': 'nosniff',
    'X-Frame-Options': 'DENY',
    'X-XSS-Protection': '1; mode=block',
    'Referrer-Policy': 'strict-origin-when-cross-origin',
  }
};

// Validation function to ensure secure build
export const validateBuildSecurity = () => {
  const issues = [];
  
  if (isProduction) {
    // Check for development artifacts in production
    if (import.meta.env.VITE_DEBUG_MODE === 'true') {
      issues.push('Debug mode should be disabled in production');
    }
    
    if (import.meta.env.VITE_LOG_LEVEL === 'debug') {
      issues.push('Log level should not be debug in production');
    }
    
    // Check for exposed secrets (this should never happen after our fixes)
    const exposedPatterns = [
      'ak_5zno3zn8gisz5f9held6d09l6vosgft2',
      'Demo123!MonteCarlo',
      'localhost:8000'
    ];
    
    const currentScript = document.documentElement.innerHTML;
    for (const pattern of exposedPatterns) {
      if (currentScript.includes(pattern)) {
        issues.push(`Potential secret exposure detected: ${pattern.substring(0, 10)}...`);
      }
    }
  }
  
  if (issues.length > 0) {
    console.error('🚨 BUILD SECURITY ISSUES:', issues);
    return false;
  }
  
  return true;
};

// Initialize build validation on import
if (isProduction) {
  setTimeout(() => {
    const isSecure = validateBuildSecurity();
    if (!isSecure) {
      console.error('🚨 SECURITY VALIDATION FAILED - Contact support');
    } else {
      console.log('✅ Production build security validation passed');
    }
  }, 1000);
}

export default buildConfig;
