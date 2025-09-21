import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate, Trend } from 'k6/metrics';

// Custom metrics
const successRate = new Rate('success_rate');
const responseTime = new Trend('response_time');

// Test configuration
export const options = {
  stages: [
    // Quick ramp up for testing
    { duration: '30s', target: 5 },
    { duration: '1m', target: 5 },
    { duration: '30s', target: 0 },
  ],
  thresholds: {
    http_req_duration: ['p(95)<3000'], // 95% of requests should be below 3s
    http_req_failed: ['rate<0.1'],     // Error rate should be below 10%
    success_rate: ['rate>0.8'],        // Success rate should be above 80%
  },
};

// Main test function
export default function() {
  // Test 1: Health Check
  const healthCheck = http.get('https://localhost/health', {
    headers: {
      'Accept': 'text/plain'
    }
  });
  
  const healthSuccess = check(healthCheck, {
    'health check status is 200': (r) => r.status === 200,
  });
  successRate.add(healthSuccess);
  responseTime.add(healthCheck.timings.duration);
  
  // Test 2: Frontend Homepage
  const frontendCheck = http.get('https://localhost/', {
    headers: {
      'Accept': 'text/html'
    }
  });
  
  const frontendSuccess = check(frontendCheck, {
    'frontend status is 200': (r) => r.status === 200,
    'frontend has content': (r) => r.body.length > 500 && r.body.includes('Monte Carlo'),
  });
  successRate.add(frontendSuccess);
  responseTime.add(frontendCheck.timings.duration);
  
  // Test 3: API Documentation
  const apiDocsCheck = http.get('https://localhost/api/docs', {
    headers: {
      'Accept': 'text/html'
    }
  });
  
  const apiDocsSuccess = check(apiDocsCheck, {
    'api docs status is 200': (r) => r.status === 200,
    'api docs has content': (r) => r.body.length > 500 && r.body.includes('swagger-ui'),
  });
  successRate.add(apiDocsSuccess);
  responseTime.add(apiDocsCheck.timings.duration);
  
  // Test 4: API OpenAPI Spec
  const openApiCheck = http.get('https://localhost/api/openapi.json', {
    headers: {
      'Accept': 'application/json'
    }
  });
  
  const openApiSuccess = check(openApiCheck, {
    'openapi status is 200': (r) => r.status === 200,
    'openapi is valid json': (r) => {
      try {
        JSON.parse(r.body);
        return true;
      } catch (e) {
        return false;
      }
    },
  });
  successRate.add(openApiSuccess);
  responseTime.add(openApiCheck.timings.duration);
  
  // Test 5: Static Assets (if they exist)
  const staticAssets = [
    '/assets/index-D2XFJw4u.js',
    '/assets/vendor-react-DLlxo1H4.js',
    '/favicon.ico'
  ];
  
  const asset = staticAssets[Math.floor(Math.random() * staticAssets.length)];
  const assetCheck = http.get(`https://localhost${asset}`, {
    headers: {
      'Accept': '*/*'
    }
  });
  
  const assetSuccess = check(assetCheck, {
    'asset status is 200 or 404': (r) => r.status === 200 || r.status === 404,
  });
  successRate.add(assetSuccess);
  responseTime.add(assetCheck.timings.duration);
  
  // Random sleep between requests (1 to 3 seconds)
  sleep(Math.random() * 2 + 1);
} 