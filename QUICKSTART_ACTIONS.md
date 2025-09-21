# QUICKSTART ACTIONS - Monte Carlo Platform Launch
## Immediate Actions for Phase 1 (Next 2 Weeks)

### Week 1 Priority Actions

#### 🔒 Security (CRITICAL)
```bash
# 1. Install security dependencies
cd backend
pip install python-jose[cryptography] slowapi email-validator pyotp qrcode
pip install python-multipart aiofiles

# 2. Add virus scanning
pip install pyclamd  # For ClamAV integration

# 3. Security headers middleware
# Add to backend/main.py
```

```python
# backend/security/middleware.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from starlette.middleware.sessions import SessionMiddleware
import secrets

def add_security_headers(app: FastAPI):
    @app.middleware("http")
    async def security_headers(request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Content-Security-Policy"] = "default-src 'self'"
        return response
```

#### 📊 User Limits Implementation
```python
# backend/auth/limits.py
from datetime import datetime, timedelta
from typing import Dict

FREE_TIER_LIMITS = {
    "simulations_per_month": 100,
    "iterations_per_simulation": 1000,
    "concurrent_simulations": 3,
    "result_retention_days": 30,
    "file_size_mb": 10,
    "gpu_access": False,
    "engines": ["power", "arrow", "enhanced"]  # No GPU/Super engines
}

async def check_user_limits(user_id: int, db) -> Dict:
    # Implementation for checking user limits
    pass
```

#### 🚀 Quick Docker Security Updates
```yaml
# docker-compose.yml updates
services:
  backend:
    environment:
      - SECURE_COOKIES=True
      - HTTPS_REDIRECT=True
      - MAX_UPLOAD_SIZE=10485760  # 10MB
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: '2'
```

### Week 2 Priority Actions

#### 🎨 Frontend Security & Limits
```javascript
// frontend/src/utils/security.js
export const validateFileUpload = (file) => {
  const maxSize = 10 * 1024 * 1024; // 10MB
  const allowedTypes = [
    'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
    'application/vnd.ms-excel.sheet.macroEnabled.12'
  ];
  
  if (file.size > maxSize) {
    throw new Error('File size exceeds 10MB limit');
  }
  
  if (!allowedTypes.includes(file.type)) {
    throw new Error('Only .xlsx and .xlsm files are allowed');
  }
  
  return true;
};
```

#### 📝 Legal Documents Update
1. Update `legal/TERMS_OF_SERVICE.md` for public use
2. Add data retention policy to `legal/PRIVACY_POLICY.md`
3. Create `legal/FAIR_USE_POLICY.md` for free tier

#### 🔧 Monitoring Setup
```bash
# Install monitoring tools
pip install prometheus-client sentry-sdk[fastapi]

# Frontend monitoring
npm install @sentry/react @sentry/tracing
```

### Immediate Infrastructure Changes

#### 1. Enable HTTPS (Today!)
```bash
# Use docker-compose.prod.yml with SSL
./scripts/generate-ssl.sh
docker-compose -f docker-compose.prod.yml up -d
```

#### 2. Database Backup (Tomorrow)
```bash
# Add to crontab
0 2 * * * docker exec backend python -c "from database import backup_db; backup_db()"
```

#### 3. Rate Limiting (This Week)
```python
# backend/main.py
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Add to routes
@app.post("/api/simulations/run")
@limiter.limit("10/minute")  # Free tier limit
async def run_simulation(request: Request):
    pass
```

### Testing Checklist Before Launch

- [ ] Upload malicious Excel file → Should be rejected
- [ ] Upload 15MB file → Should be rejected  
- [ ] Run 4 concurrent simulations → 4th should be queued
- [ ] Run 101st simulation in month → Should show upgrade prompt
- [ ] SQL injection in inputs → Should be sanitized
- [ ] XSS in file names → Should be escaped
- [ ] Rate limit testing → Should block after limit
- [ ] Memory exhaustion test → Should handle gracefully

### Marketing Prep (Start Now!)

1. **Landing Page Copy**
   - "Free Monte Carlo Simulations for Excel"
   - "No Credit Card Required"
   - "100 Free Simulations Monthly"

2. **Social Media Accounts**
   - Twitter/X: @MonteCarloExcel
   - LinkedIn: Company page
   - Reddit: Prepare for r/excel, r/finance

3. **Content Creation**
   - Blog post: "What is Monte Carlo Simulation?"
   - Video: 2-minute demo
   - Case study: Financial modeling example

### Emergency Contacts Setup

- Security incidents: security@yourdomain.com
- Bug bounty: bounty@yourdomain.com  
- Support: support@yourdomain.com
- Legal: legal@yourdomain.com

### Go/No-Go Checklist (Week 12)

**Technical**
- [ ] Security scan passed (0 critical issues)
- [ ] Load test passed (1000 users)
- [ ] Backup/restore tested
- [ ] SSL certificate active
- [ ] Monitoring active

**Legal**
- [ ] Terms of Service reviewed by lawyer
- [ ] Privacy Policy GDPR compliant
- [ ] Copyright assignments clear
- [ ] Open source licenses documented

**Business**
- [ ] Support system ready
- [ ] Payment system tested (for Phase 2)
- [ ] Analytics tracking active
- [ ] Beta feedback incorporated

---

**Remember**: It's better to launch with fewer features that are rock-solid than many features that are buggy. Focus on security and reliability first! 