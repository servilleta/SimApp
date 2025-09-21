import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate, Trend, Counter } from 'k6/metrics';

// Custom metrics
const authSuccessRate = new Rate('auth_success');
const uploadSuccessRate = new Rate('upload_success');
const simulationSuccessRate = new Rate('simulation_success');
const responseTime = new Trend('response_time');
const simulationDuration = new Trend('simulation_duration');

// Test configuration
export const options = {
  stages: [
    // Warm up
    { duration: '1m', target: 2 },
    // Ramp up to 10 users
    { duration: '2m', target: 10 },
    // Stay at 10 users for 5 minutes
    { duration: '5m', target: 10 },
    // Ramp up to 25 users
    { duration: '3m', target: 25 },
    // Stay at 25 users for 10 minutes
    { duration: '10m', target: 25 },
    // Ramp down
    { duration: '2m', target: 0 },
  ],
  thresholds: {
    http_req_duration: ['p(95)<5000'], // 95% of requests should be below 5s
    http_req_failed: ['rate<0.05'],    // Error rate should be below 5%
    auth_success: ['rate>0.95'],       // Auth success rate should be above 95%
    upload_success: ['rate>0.9'],      // Upload success rate should be above 90%
    simulation_success: ['rate>0.8'],  // Simulation success rate should be above 80%
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

// Global variables
let testFileData = null;

// Setup function - runs once before the test
export function setup() {
  console.log('Setting up comprehensive load test...');
  
  // Load test file data
  testFileData = open('./test-data/simple-monte-carlo.xlsx', 'b');
  
  // Create test users
  for (let i = 0; i < testUsers.length; i++) {
    const user = testUsers[i];
    const createUserRes = http.post('https://localhost/api/auth/register', {
      username: user.username,
      email: `${user.username}@example.com`,
      password: user.password,
      full_name: `Load Test User ${i + 1}`
    }, {
      headers: {
        'Content-Type': 'application/json'
      }
    });
    
    if (createUserRes.status === 201) {
      console.log(`Created test user: ${user.username}`);
    }
  }
  
  return { testFileData };
}

// Main test function
export default function(data) {
  const { testFileData } = data;
  
  // Random user selection
  const user = testUsers[Math.floor(Math.random() * testUsers.length)];
  let authToken = null;
  let fileId = null;
  let simulationId = null;
  
  // Step 1: Authentication (100% of users)
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
  responseTime.add(loginRes.timings.duration);
  
  if (loginRes.status === 200) {
    const tokenData = JSON.parse(loginRes.body);
    authToken = tokenData.access_token;
  } else {
    console.log(`Auth failed for user ${user.username}: ${loginRes.status}`);
    sleep(1);
    return;
  }
  
  // Step 2: File Upload (80% of authenticated users)
  if (Math.random() < 0.8 && authToken) {
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
    responseTime.add(uploadRes.timings.duration);
    
    if (uploadRes.status === 200) {
      const uploadData = JSON.parse(uploadRes.body);
      fileId = uploadData.file_id;
    }
  }
  
  // Step 3: File Parsing (70% of users with files)
  if (Math.random() < 0.7 && authToken && fileId) {
    const parseRes = http.get(`https://localhost/api/excel-parser/parse/${fileId}`, {
      headers: {
        'Authorization': `Bearer ${authToken}`
      }
    });
    
    check(parseRes, {
      'parse status is 200': (r) => r.status === 200,
      'parse response has sheets': (r) => JSON.parse(r.body).sheets !== undefined,
    });
    
    responseTime.add(parseRes.timings.duration);
  }
  
  // Step 4: Simulation Creation (60% of users with files)
  if (Math.random() < 0.6 && authToken && fileId) {
    const engines = ['enhanced', 'arrow', 'power', 'super', 'big'];
    const engine = engines[Math.floor(Math.random() * engines.length)];
    const iterations = [100, 500, 1000][Math.floor(Math.random() * 3)];
    
    const simulationData = {
      file_id: fileId,
      engine: engine,
      iterations: iterations,
      target_cells: ['A1', 'B1', 'C1'],
      variables: {
        'A1': { distribution: 'normal', mean: 100, std_dev: 10 },
        'B1': { distribution: 'uniform', min: 50, max: 150 },
        'C1': { distribution: 'triangular', min: 80, mode: 100, max: 120 }
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
    responseTime.add(createSimRes.timings.duration);
    
    if (createSimRes.status === 200) {
      const simData = JSON.parse(createSimRes.body);
      simulationId = simData.simulation_id;
      
      // Step 5: Monitor Simulation Progress (for created simulations)
      let attempts = 0;
      const maxAttempts = 30; // 30 seconds max
      
      while (attempts < maxAttempts && simulationId) {
        const statusRes = http.get(`https://localhost/api/simulations/status/${simulationId}`, {
          headers: {
            'Authorization': `Bearer ${authToken}`
          }
        });
        
        if (statusRes.status === 200) {
          const statusData = JSON.parse(statusRes.body);
          
          check(statusRes, {
            'status check is 200': (r) => r.status === 200,
          });
          
          // If simulation is complete, get results
          if (statusData.status === 'completed') {
            const resultsRes = http.get(`https://localhost/api/simulations/results/${simulationId}`, {
              headers: {
                'Authorization': `Bearer ${authToken}`
              }
            });
            
            check(resultsRes, {
              'results check is 200': (r) => r.status === 200,
              'results have data': (r) => JSON.parse(r.body).results !== undefined,
            });
            
            simulationDuration.add(attempts * 1000); // Convert to milliseconds
            break;
          } else if (statusData.status === 'failed') {
            console.log(`Simulation ${simulationId} failed`);
            break;
          }
        }
        
        attempts++;
        sleep(1); // Wait 1 second before next check
      }
    }
  }
  
  // Step 6: User Profile Check (20% of users)
  if (Math.random() < 0.2 && authToken) {
    const profileRes = http.get('https://localhost/api/auth/profile', {
      headers: {
        'Authorization': `Bearer ${authToken}`
      }
    });
    
    check(profileRes, {
      'profile status is 200': (r) => r.status === 200,
    });
    
    responseTime.add(profileRes.timings.duration);
  }
  
  // Random sleep between iterations (2 to 5 seconds)
  sleep(Math.random() * 3 + 2);
}

// Teardown function - runs once after the test
export function teardown(data) {
  console.log('Cleaning up comprehensive load test...');
  
  // Clean up test users if needed
  for (const user of testUsers) {
    // Note: In a real scenario, you might want to clean up test data
    // For now, we'll leave the test users for potential reuse
  }
} 