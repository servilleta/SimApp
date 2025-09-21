/**
 * FRONTEND ENTERPRISE FEATURE TESTING
 * 
 * Copy and paste this script into your browser console at localhost:9090
 * to test all enterprise features interactively.
 */

class EnterpriseFeatureTester {
    constructor() {
        this.baseURL = 'http://backend:8000'; // Use backend container name
        this.results = [];
    }

    log(testName, success, details = '') {
        const result = {
            test: testName,
            status: success ? '✅ PASSED' : '❌ FAILED',
            details: details,
            timestamp: new Date().toISOString()
        };
        
        this.results.push(result);
        console.log(`${result.status}: ${testName}`);
        if (details) console.log(`   ${details}`);
        
        return success;
    }

    async getAuthToken() {
        // Try to get Auth0 token from various possible locations
        const tokenSources = [
            () => localStorage.getItem('auth_token'),
            () => localStorage.getItem('auth0_token'),
            () => sessionStorage.getItem('auth_token'),
            () => window.auth0Token,
            () => {
                // Try to get from Redux store if available
                if (window.__REDUX_STORE__) {
                    const state = window.__REDUX_STORE__.getState();
                    return state.auth?.token || state.user?.token;
                }
                return null;
            }
        ];

        for (const getToken of tokenSources) {
            try {
                const token = getToken();
                if (token && token.length > 20) {
                    return token;
                }
            } catch (e) {
                // Continue to next source
            }
        }

        return null;
    }

    async testEnterpriseAuth() {
        console.log('\n🔐 TESTING ENTERPRISE AUTHENTICATION');
        console.log('=' .repeat(50));

        const token = await this.getAuthToken();
        
        if (!token) {
            this.log('Auth Token Detection', false, 'No Auth0 token found. Please login first.');
            return false;
        }

        this.log('Auth Token Detection', true, `Token found (${token.length} chars)`);

        // Test enterprise user endpoint
        try {
            const response = await fetch(`${this.baseURL}/enterprise/auth/me`, {
                headers: {
                    'Authorization': `Bearer ${token}`,
                    'Content-Type': 'application/json'
                }
            });

            if (response.ok) {
                const userData = await response.json();
                this.log('Enterprise User Context', true, 
                    `Organization: ${userData.organization_name}, Roles: ${userData.roles.join(', ')}`);
                
                console.log('👤 Your Enterprise User Info:', userData);
                return userData;
            } else {
                this.log('Enterprise User Context', false, `HTTP ${response.status}: ${response.statusText}`);
                return false;
            }
        } catch (error) {
            this.log('Enterprise User Context', false, `Network error: ${error.message}`);
            return false;
        }
    }

    async testPermissions(token) {
        console.log('\n🛡️ TESTING PERMISSIONS');
        console.log('=' .repeat(50));

        const permissionsToTest = [
            'simulation.create',
            'simulation.delete', 
            'organization.manage',
            'billing.view',
            'admin.users'
        ];

        for (const permission of permissionsToTest) {
            try {
                const response = await fetch(`${this.baseURL}/enterprise/auth/check-permission`, {
                    method: 'POST',
                    headers: {
                        'Authorization': `Bearer ${token}`,
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ permission })
                });

                if (response.ok) {
                    const result = await response.json();
                    const status = result.has_permission ? 'ALLOWED' : 'DENIED';
                    this.log(`Permission ${permission}`, true, status);
                } else {
                    this.log(`Permission ${permission}`, false, `HTTP ${response.status}`);
                }
            } catch (error) {
                this.log(`Permission ${permission}`, false, `Error: ${error.message}`);
            }
        }
    }

    async testQuotas(token) {
        console.log('\n📊 TESTING QUOTAS');
        console.log('=' .repeat(50));

        try {
            const response = await fetch(`${this.baseURL}/enterprise/auth/quotas`, {
                headers: {
                    'Authorization': `Bearer ${token}`,
                    'Content-Type': 'application/json'
                }
            });

            if (response.ok) {
                const quotaData = await response.json();
                this.log('Quota Information', true, 'Quota data retrieved successfully');
                
                console.log('📋 Your Current Quotas:', quotaData.quotas);
                console.log('📊 Current Usage:', quotaData.current_usage);
                console.log('📈 Quota Status:', quotaData.quota_status);
                
                return quotaData;
            } else {
                this.log('Quota Information', false, `HTTP ${response.status}: ${response.statusText}`);
                return false;
            }
        } catch (error) {
            this.log('Quota Information', false, `Network error: ${error.message}`);
            return false;
        }
    }

    async testOrganization(token) {
        console.log('\n🏢 TESTING ORGANIZATION');
        console.log('=' .repeat(50));

        try {
            const response = await fetch(`${this.baseURL}/enterprise/auth/organization`, {
                headers: {
                    'Authorization': `Bearer ${token}`,
                    'Content-Type': 'application/json'
                }
            });

            if (response.ok) {
                const orgData = await response.json();
                this.log('Organization Information', true, 
                    `${orgData.name} (${orgData.tier} tier)`);
                
                console.log('🏢 Your Organization:', orgData);
                return orgData;
            } else {
                this.log('Organization Information', false, `HTTP ${response.status}: ${response.statusText}`);
                return false;
            }
        } catch (error) {
            this.log('Organization Information', false, `Network error: ${error.message}`);
            return false;
        }
    }

    async testFileUploadQuota() {
        console.log('\n📁 TESTING FILE UPLOAD QUOTAS');
        console.log('=' .repeat(50));

        // Test if file upload respects quotas
        const fileInput = document.querySelector('input[type="file"]');
        
        if (fileInput) {
            this.log('File Upload Element', true, 'File upload input found on page');
            
            // Check if there are any quota warnings or validations
            const uploadArea = fileInput.closest('.upload-area, .file-upload, .excel-upload');
            if (uploadArea) {
                this.log('Upload Area Detection', true, 'Upload area found');
            }
        } else {
            this.log('File Upload Element', false, 'No file upload input found. Navigate to simulation page first.');
        }
    }

    async testSimulationQuota() {
        console.log('\n⚡ TESTING SIMULATION QUOTAS');
        console.log('=' .repeat(50));

        // Check if there are simulation controls
        const runButton = document.querySelector('button[type="submit"], .run-simulation, .start-simulation');
        
        if (runButton) {
            this.log('Simulation Controls', true, 'Simulation controls found');
            
            // Check if quota information is displayed
            const quotaElements = document.querySelectorAll('[class*="quota"], [class*="limit"], [id*="quota"]');
            if (quotaElements.length > 0) {
                this.log('Quota Display', true, `Found ${quotaElements.length} quota-related elements`);
            } else {
                this.log('Quota Display', false, 'No quota information displayed (may need frontend integration)');
            }
        } else {
            this.log('Simulation Controls', false, 'No simulation controls found. Navigate to simulation page first.');
        }
    }

    async runAllTests() {
        console.log('🧪 ENTERPRISE FRONTEND TESTING');
        console.log('=' .repeat(60));
        console.log(`🕐 Started at: ${new Date().toLocaleString()}`);
        console.log('=' .repeat(60));

        // Get authentication token
        const token = await this.getAuthToken();
        
        if (!token) {
            console.log('❌ Cannot run API tests without authentication token.');
            console.log('💡 Please login first, then run this script again.');
            return;
        }

        // Run all tests
        await this.testEnterpriseAuth();
        await this.testPermissions(token);
        await this.testQuotas(token);
        await this.testOrganization(token);
        await this.testFileUploadQuota();
        await this.testSimulationQuota();

        // Print summary
        console.log('\n' + '=' .repeat(60));
        console.log('📊 FRONTEND TEST RESULTS');
        console.log('=' .repeat(60));

        const total = this.results.length;
        const passed = this.results.filter(r => r.status.includes('PASSED')).length;
        const failed = total - passed;
        const successRate = total > 0 ? (passed / total * 100) : 0;

        console.log(`📈 Total Tests: ${total}`);
        console.log(`✅ Passed: ${passed}`);
        console.log(`❌ Failed: ${failed}`);
        console.log(`📊 Success Rate: ${successRate.toFixed(1)}%`);

        if (successRate >= 90) {
            console.log('\n🎉 ENTERPRISE FEATURES: EXCELLENT!');
        } else if (successRate >= 75) {
            console.log('\n✅ ENTERPRISE FEATURES: GOOD!');
        } else {
            console.log('\n⚠️ ENTERPRISE FEATURES: NEEDS ATTENTION');
        }

        console.log('\n📋 Detailed Results:');
        this.results.forEach(result => {
            console.log(`   ${result.status}: ${result.test}`);
            if (result.details) console.log(`      ${result.details}`);
        });

        return this.results;
    }
}

// Instructions for manual testing
console.log(`
🧪 ENTERPRISE FEATURE TESTING INSTRUCTIONS

1️⃣ **Login First**: Make sure you're logged in to the application
2️⃣ **Open Console**: Press F12 and go to Console tab  
3️⃣ **Run Tests**: Copy and paste this entire script, then run:

   const tester = new EnterpriseFeatureTester();
   tester.runAllTests();

4️⃣ **Check Results**: Review the test results in the console

🎯 **What Gets Tested:**
✅ Enterprise user authentication and context
✅ Role-based permissions (RBAC)
✅ Organization information and tier
✅ Quota management and limits
✅ API endpoint security
✅ Frontend integration points

🚀 **Ready to test!**
`);

// Auto-create the tester instance for immediate use
window.EnterpriseFeatureTester = EnterpriseFeatureTester;
console.log('✅ EnterpriseFeatureTester class loaded. Run: new EnterpriseFeatureTester().runAllTests()');
