import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate, Trend } from 'k6/metrics';

// Custom metrics
const pageLoadSuccessRate = new Rate('page_load_success');
const frontendResponseTime = new Trend('frontend_response_time');

// Test configuration
export const options = {
  stages: [
    // Ramp up to 20 users over 2 minutes
    { duration: '2m', target: 20 },
    // Stay at 20 users for 5 minutes
    { duration: '5m', target: 20 },
    // Ramp up to 100 users over 3 minutes
    { duration: '3m', target: 100 },
    // Stay at 100 users for 10 minutes
    { duration: '10m', target: 100 },
    // Ramp down to 0 users over 2 minutes
    { duration: '2m', target: 0 },
  ],
  thresholds: {
    http_req_duration: ['p(95)<3000'], // 95% of requests should be below 3s
    http_req_failed: ['rate<0.05'],    // Error rate should be below 5%
    page_load_success: ['rate>0.95'],  // Page load success rate should be above 95%
  },
};

// Test pages
const testPages = [
  '/',
  '/login',
  '/register',
  '/dashboard',
  '/simulations',
  '/upload',
  '/results',
  '/profile',
  '/help',
  '/about'
];

// Main test function
export default function() {
  // Test 1: Homepage (30% of requests)
  if (Math.random() < 0.3) {
    const homeRes = http.get('http://localhost/');
    const homeCheck = check(homeRes, {
      'homepage status is 200': (r) => r.status === 200,
      'homepage loads quickly': (r) => r.timings.duration < 2000,
      'homepage has content': (r) => r.body.length > 1000,
    });
    pageLoadSuccessRate.add(homeCheck);
    frontendResponseTime.add(homeRes.timings.duration);
  }
  
  // Test 2: Random page access (40% of requests)
  if (Math.random() < 0.4) {
    const randomPage = testPages[Math.floor(Math.random() * testPages.length)];
    const pageRes = http.get(`http://localhost${randomPage}`);
    const pageCheck = check(pageRes, {
      'page status is 200': (r) => r.status === 200,
      'page loads quickly': (r) => r.timings.duration < 3000,
      'page has content': (r) => r.body.length > 500,
    });
    pageLoadSuccessRate.add(pageCheck);
    frontendResponseTime.add(pageRes.timings.duration);
  }
  
  // Test 3: Static assets (20% of requests)
  if (Math.random() < 0.2) {
    const staticAssets = [
      '/assets/index.js',
      '/assets/index.css',
      '/favicon.ico',
      '/logo.png'
    ];
    
    const asset = staticAssets[Math.floor(Math.random() * staticAssets.length)];
    const assetRes = http.get(`http://localhost${asset}`);
    
    check(assetRes, {
      'asset status is 200': (r) => r.status === 200,
      'asset loads quickly': (r) => r.timings.duration < 1000,
    });
  }
  
  // Test 4: API endpoints through frontend (10% of requests)
  if (Math.random() < 0.1) {
    const apiEndpoints = [
      '/api/health',
      '/api/auth/profile',
      '/api/simulations/list'
    ];
    
    const endpoint = apiEndpoints[Math.floor(Math.random() * apiEndpoints.length)];
    const apiRes = http.get(`http://localhost${endpoint}`);
    
    check(apiRes, {
      'api endpoint accessible': (r) => r.status === 200 || r.status === 401, // 401 is expected for unauthenticated
    });
  }
  
  // Random sleep between requests (1 to 3 seconds)
  sleep(Math.random() * 2 + 1);
} 