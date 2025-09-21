import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate, Trend, Counter } from 'k6/metrics';

// Custom metrics
const errorRate = new Rate('errors');
const authSuccessRate = new Rate('auth_success');
const simulationSuccessRate = new Rate('simulation_success');
const uploadSuccessRate = new Rate('upload_success');
const responseTime = new Trend('response_time');

// Test configuration
export const options = {
  stages: [
    // Ramp up to 10 users over 2 minutes
    { duration: '2m', target: 10 },
    // Stay at 10 users for 5 minutes
    { duration: '5m', target: 10 },
    // Ramp up to 50 users over 3 minutes
    { duration: '3m', target: 50 },
    // Stay at 50 users for 10 minutes
    { duration: '10m', target: 50 },
    // Ramp down to 0 users over 2 minutes
    { duration: '2m', target: 0 },
  ],
  thresholds: {
    http_req_duration: ['p(95)<2000'], // 95% of requests should be below 2s
    http_req_failed: ['rate<0.1'],     // Error rate should be below 10%
    errors: ['rate<0.1'],              // Custom error rate
    auth_success: ['rate>0.9'],        // Auth success rate should be above 90%
    simulation_success: ['rate>0.8'],  // Simulation success rate should be above 80%
  },
};

// Test data
const testUsers = [
  { username: 'testuser1', password: 'testpass123' },
  { username: 'testuser2', password: 'testpass123' },
  { username: 'testuser3', password: 'testpass123' },
  { username: 'testuser4', password: 'testpass123' },
  { username: 'testuser5', password: 'testpass123' },
];

// Global variables
let authToken = null;
let testFileId = null;
let testFileData = null;

// Init function - runs once at the beginning
export function setup() {
  console.log('Setting up load test...');
  
  // Load test file data
  testFileData = open('./test-data/simple-monte-carlo.xlsx', 'b');
  
  // Create test user if needed
  const createUserRes = http.post('https://localhost/api/auth/register', {
    username: 'loadtestuser',
    email: 'loadtest@example.com',
    password: 'loadtestpass123',
    full_name: 'Load Test User'
  }, {
    headers: {
      'Content-Type': 'application/json'
    }
  });
  
  if (createUserRes.status === 201) {
    console.log('Created test user');
  }
  
  // Login to get auth token
  const loginRes = http.post('https://localhost/api/auth/token', {
    username: 'loadtestuser',
    password: 'loadtestpass123'
  }, {
    headers: {
      'Content-Type': 'application/x-www-form-urlencoded'
    }
  });
  
  if (loginRes.status === 200) {
    const tokenData = JSON.parse(loginRes.body);
    authToken = tokenData.access_token;
    console.log('Got auth token');
  }
  
  // Upload a test Excel file
  const uploadRes = http.post('https://localhost/api/excel-parser/upload', {
    file: http.file(testFileData, 'test.xlsx', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
  }, {
    headers: {
      'Authorization': `Bearer ${authToken}`,
      'Content-Type': 'multipart/form-data'
    }
  });
  
  if (uploadRes.status === 200) {
    const uploadData = JSON.parse(uploadRes.body);
    testFileId = uploadData.file_id;
    console.log('Uploaded test file:', testFileId);
  }
  
  return { authToken, testFileId, testFileData };
}

// Main test function
export default function(data) {
  const { authToken, testFileId, testFileData } = data;
  
  // Random user selection
  const user = testUsers[Math.floor(Math.random() * testUsers.length)];
  
  // Test 1: Health Check
  const healthCheck = http.get('https://localhost/health');
  check(healthCheck, {
    'health check status is 200': (r) => r.status === 200,
  });
  
  // Test 2: API Documentation
  const apiDocs = http.get('https://localhost/api/docs');
  check(apiDocs, {
    'api docs status is 200': (r) => r.status === 200,
  });
  
  // Test 3: Authentication (30% of requests)
  if (Math.random() < 0.3) {
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
      'auth response has token': (r) => JSON.parse(r.body).access_token !== undefined,
    });
    
    authSuccessRate.add(authCheck);
    
    if (loginRes.status === 200) {
      const tokenData = JSON.parse(loginRes.body);
      authToken = tokenData.access_token;
    }
  }
  
  // Test 4: File Upload (20% of requests)
  if (Math.random() < 0.2 && authToken) {
    const uploadRes = http.post('https://localhost/api/excel-parser/upload', {
      file: http.file(testFileData, 'test.xlsx', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
    }, {
      headers: {
        'Authorization': `Bearer ${authToken}`,
        'Content-Type': 'multipart/form-data'
      }
    });
    
    const uploadCheck = check(uploadRes, {
      'upload status is 200': (r) => r.status === 200,
      'upload response has file_id': (r) => JSON.parse(r.body).file_id !== undefined,
    });
    
    uploadSuccessRate.add(uploadCheck);
  }
  
  // Test 5: File Parsing (25% of requests)
  if (Math.random() < 0.25 && authToken && testFileId) {
    const parseRes = http.get(`https://localhost/api/excel-parser/parse/${testFileId}`, {
      headers: {
        'Authorization': `Bearer ${authToken}`
      }
    });
    
    check(parseRes, {
      'parse status is 200': (r) => r.status === 200,
      'parse response has sheets': (r) => JSON.parse(r.body).sheets !== undefined,
    });
  }
  
  // Test 6: Simulation Creation (15% of requests)
  if (Math.random() < 0.15 && authToken && testFileId) {
    const simulationData = {
      file_id: testFileId,
      engine: ['enhanced', 'arrow', 'power', 'super', 'big'][Math.floor(Math.random() * 5)],
      iterations: [100, 500, 1000][Math.floor(Math.random() * 3)],
      target_cells: ['A1', 'B1', 'C1'],
      variables: {
        'A1': { distribution: 'normal', mean: 100, std_dev: 10 },
        'B1': { distribution: 'uniform', min: 50, max: 150 }
      }
    };
    
    const createSimRes = http.post('https://localhost/api/simulations/create', JSON.stringify(simulationData), {
      headers: {
        'Authorization': `Bearer ${authToken}`,
        'Content-Type': 'application/json'
      }
    });
    
    const simCheck = check(createSimRes, {
      'simulation creation status is 200': (r) => r.status === 200,
      'simulation response has simulation_id': (r) => JSON.parse(r.body).simulation_id !== undefined,
    });
    
    simulationSuccessRate.add(simCheck);
    
    // If simulation created successfully, test status checking
    if (createSimRes.status === 200) {
      const simData = JSON.parse(createSimRes.body);
      const simId = simData.simulation_id;
      
      // Check simulation status
      const statusRes = http.get(`https://localhost/api/simulations/status/${simId}`, {
        headers: {
          'Authorization': `Bearer ${authToken}`
        }
      });
      
      check(statusRes, {
        'status check is 200': (r) => r.status === 200,
      });
      
      // Wait a bit and check results if simulation is complete
      sleep(2);
      
      const resultsRes = http.get(`https://localhost/api/simulations/results/${simId}`, {
        headers: {
          'Authorization': `Bearer ${authToken}`
        }
      });
      
      check(resultsRes, {
        'results check is 200': (r) => r.status === 200,
      });
    }
  }
  
  // Test 7: User Profile (10% of requests)
  if (Math.random() < 0.1 && authToken) {
    const profileRes = http.get('https://localhost/api/auth/profile', {
      headers: {
        'Authorization': `Bearer ${authToken}`
      }
    });
    
    check(profileRes, {
      'profile status is 200': (r) => r.status === 200,
    });
  }
  
  // Record response time
  responseTime.add(Date.now());
  
  // Random sleep between requests (0.5 to 2 seconds)
  sleep(Math.random() * 1.5 + 0.5);
}

// Teardown function - runs once after the test
export function teardown(data) {
  console.log('Cleaning up load test...');
  
  // Clean up test data if needed
  if (data.authToken) {
    http.delete('https://localhost/api/auth/profile', {
      headers: {
        'Authorization': `Bearer ${data.authToken}`
      }
    });
  }
} 