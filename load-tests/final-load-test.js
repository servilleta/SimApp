import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate, Trend } from 'k6/metrics';

// Custom metrics
const authSuccessRate = new Rate('auth_success');
const apiSuccessRate = new Rate('api_success');
const responseTime = new Trend('response_time');

// Test configuration
export const options = {
  stages: [
    // Warm up
    { duration: '30s', target: 2 },
    // Ramp up to 10 users
    { duration: '1m', target: 10 },
    // Stay at 10 users for 3 minutes
    { duration: '3m', target: 10 },
    // Ramp up to 20 users
    { duration: '1m', target: 20 },
    // Stay at 20 users for 5 minutes
    { duration: '5m', target: 20 },
    // Ramp down
    { duration: '1m', target: 0 },
  ],
  thresholds: {
    http_req_duration: ['p(95)<3000'], // 95% of requests should be below 3s
    http_req_failed: ['rate<0.05'],    // Error rate should be below 5%
    auth_success: ['rate>0.95'],       // Auth success rate should be above 95%
    api_success: ['rate>0.9'],         // API success rate should be above 90%
  },
};

// Test data
const testUsers = [
  { username: 'loadtest1', password: 'testpass123' },
  { username: 'loadtest2', password: 'testpass123' },
  { username: 'loadtest3', password: 'testpass123' },
  { username: 'loadtest4', password: 'testpass123' },
  { username: 'loadtest5', password: 'testpass123' },
];

// Main test function
export default function() {
  // Random user selection
  const user = testUsers[Math.floor(Math.random() * testUsers.length)];
  let authToken = null;
  
  // Test 1: Health Check (100% of requests)
  const healthCheck = http.get('https://localhost/health', {
    headers: {
      'Accept': 'text/plain'
    }
  });
  
  check(healthCheck, {
    'health check status is 200': (r) => r.status === 200,
  });
  responseTime.add(healthCheck.timings.duration);
  
  // Test 2: Frontend Homepage (80% of requests)
  if (Math.random() < 0.8) {
    const frontendCheck = http.get('https://localhost/', {
      headers: {
        'Accept': 'text/html'
      }
    });
    
    check(frontendCheck, {
      'frontend status is 200': (r) => r.status === 200,
    });
    responseTime.add(frontendCheck.timings.duration);
  }
  
  // Test 3: API Documentation (60% of requests)
  if (Math.random() < 0.6) {
    const apiDocsCheck = http.get('https://localhost/api/docs', {
      headers: {
        'Accept': 'text/html'
      }
    });
    
    check(apiDocsCheck, {
      'api docs status is 200': (r) => r.status === 200,
    });
    responseTime.add(apiDocsCheck.timings.duration);
  }
  
  // Test 4: Authentication (40% of requests)
  if (Math.random() < 0.4) {
    const loginRes = http.post('https://localhost/api/auth/token', {
      username: user.username,
      password: user.password
    }, {
      headers: {
        'Content-Type': 'application/x-www-form-urlencoded'
      }
    });
    
    const authCheck = check(loginRes, {
      'auth status is 200': (r) => r.status === 200,
      'auth response has token': (r) => {
        try {
          const data = JSON.parse(r.body);
          return data.access_token !== undefined;
        } catch (e) {
          return false;
        }
      },
    });
    
    authSuccessRate.add(authCheck);
    responseTime.add(loginRes.timings.duration);
    
    if (loginRes.status === 200) {
      try {
        const tokenData = JSON.parse(loginRes.body);
        authToken = tokenData.access_token;
      } catch (e) {
        console.log('Failed to parse auth response');
      }
    }
  }
  
  // Test 5: API Endpoints (30% of requests with auth)
  if (Math.random() < 0.3 && authToken) {
    const apiEndpoints = [
      '/api/auth/profile',
      '/api/simulations/list',
      '/api/excel-parser/files'
    ];
    
    const endpoint = apiEndpoints[Math.floor(Math.random() * apiEndpoints.length)];
    const apiRes = http.get(`https://localhost${endpoint}`, {
      headers: {
        'Authorization': `Bearer ${authToken}`,
        'Accept': 'application/json'
      }
    });
    
    const apiCheck = check(apiRes, {
      'api endpoint accessible': (r) => r.status === 200 || r.status === 401 || r.status === 404,
    });
    
    apiSuccessRate.add(apiCheck);
    responseTime.add(apiRes.timings.duration);
  }
  
  // Test 6: Static Assets (20% of requests)
  if (Math.random() < 0.2) {
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
    
    check(assetCheck, {
      'asset status is 200 or 404': (r) => r.status === 200 || r.status === 404,
    });
    responseTime.add(assetCheck.timings.duration);
  }
  
  // Test 7: OpenAPI Spec (15% of requests)
  if (Math.random() < 0.15) {
    const openApiCheck = http.get('https://localhost/api/openapi.json', {
      headers: {
        'Accept': 'application/json'
      }
    });
    
    check(openApiCheck, {
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
    responseTime.add(openApiCheck.timings.duration);
  }
  
  // Random sleep between requests (1 to 3 seconds)
  sleep(Math.random() * 2 + 1);
} 