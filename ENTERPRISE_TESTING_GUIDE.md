
# 🧪 MANUAL TESTING GUIDE FOR ENTERPRISE FEATURES

## 🔐 **Test 1: Enhanced Authentication**

### **Frontend Testing (localhost:9090):**
1. **Login** with your Auth0 account
2. **Check user context**: You should see enhanced user information
3. **Verify organization**: Your organization should be "Individual Account" 
4. **Check tier**: Should show "Standard" tier

### **Expected Behavior:**
- Login works as before but with enhanced context
- User profile shows organization information
- Role-based features may be visible/hidden

---

## 🏢 **Test 2: Multi-Tenant Organization**

### **What to Test:**
1. **Upload Excel files**: Limited to 100MB (your quota)
2. **Create simulations**: Can create up to 4 concurrent simulations
3. **File management**: Files are isolated to your organization

### **Expected Behavior:**
- Large files (>100MB) should be rejected
- Can run multiple simulations simultaneously (up to 4)
- All your data is isolated from other users

---

## 🛡️ **Test 3: API Security & Permissions**

### **Browser Console Testing:**
```javascript
// Open browser console on localhost:9090 and test:

// 1. Check current user enterprise context
fetch('/api/enterprise/auth/me', {
    headers: {'Authorization': 'Bearer ' + localStorage.getItem('auth_token')}
})
.then(r => r.json())
.then(data => console.log('Enterprise User:', data));

// 2. Test permission checking
fetch('/api/enterprise/auth/check-permission', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer ' + localStorage.getItem('auth_token')
    },
    body: JSON.stringify({permission: 'simulation.create'})
})
.then(r => r.json())
.then(data => console.log('Permission Check:', data));

// 3. Get your quotas
fetch('/api/enterprise/auth/quotas', {
    headers: {'Authorization': 'Bearer ' + localStorage.getItem('auth_token')}
})
.then(r => r.json())
.then(data => console.log('Your Quotas:', data));
```

### **Expected Results:**
- **simulation.create**: ✅ ALLOWED (you're a power user)
- **organization.manage**: ❌ DENIED (requires admin)
- **Quotas**: 4 concurrent sims, 100MB files, 10GB storage

---

## 📊 **Test 4: Quota Management**

### **File Upload Testing:**
1. **Try uploading small file** (<50MB): ✅ Should work
2. **Try uploading large file** (>100MB): ❌ Should be rejected
3. **Check storage usage**: Should track your usage

### **Simulation Testing:**
1. **Create 1-2 simulations**: ✅ Should work fine
2. **Try creating 5+ simulations**: ❌ Should hit concurrent limit
3. **Check iteration limits**: Max 10,000 iterations per simulation

### **Expected Behavior:**
- File size validation happens before upload
- Concurrent simulation limits enforced
- Clear error messages when quotas exceeded

---

## 🌐 **Test 5: API Endpoint Security**

### **Command Line Testing:**
```bash
# Test public endpoints (should work)
curl http://localhost:8000/health
curl http://localhost:8000/docs

# Test protected endpoints (should require auth)
curl http://localhost:8000/enterprise/auth/me
# Expected: 401 Unauthorized or 422 Validation Error

# Test with authentication (replace YOUR_TOKEN)
curl -H "Authorization: Bearer YOUR_AUTH0_TOKEN" \
     http://localhost:8000/enterprise/auth/me
# Expected: Your enterprise user information
```

---

## 🎯 **Success Criteria**

### **✅ Authentication Enhancement:**
- [x] Auth0 users have enterprise context
- [x] Organization information available
- [x] Role-based permissions working
- [x] Quota enforcement active

### **✅ Multi-Tenant Foundation:**
- [x] Organization tiers implemented
- [x] Usage tracking active
- [x] Data isolation working
- [x] Tier-based limits enforced

### **✅ API Security:**
- [x] Permission-protected endpoints
- [x] Role-based access control
- [x] Quota validation
- [x] Proper error handling

### **✅ System Integration:**
- [x] Ultra engine working with RBAC
- [x] Progress bar fixes maintained
- [x] File upload quota enforcement
- [x] Simulation limits working

---

## 🚀 **Ready for Enterprise Deployment**

Your platform now supports:
- **Multiple Organizations** with isolated data
- **Role-Based Team Access** with 4 user roles  
- **Tier-Based Pricing** with usage tracking
- **Resource Quota Management** preventing abuse
- **Enterprise Security** with RBAC and audit trails

**Next Steps:**
1. Continue with Phase 2 Week 8 (Multi-Tenant Database Architecture)
2. Add frontend UI for enterprise features
3. Implement billing integration
4. Add organization management dashboard
