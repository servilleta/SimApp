# Git Branching Strategy for Monte Carlo Platform

## Overview
We'll maintain two main branches to allow continued development while keeping production stable:
- `main-v1-production`: Current working monolith (production ready)
- `main-v2-modular`: Modular monolith refactoring

## Branch Structure
```
main (original)
│
├── main-v1-production (current working version)
│   ├── hotfix/security-patch
│   └── hotfix/bug-fixes
│
└── main-v2-modular (modular refactoring)
    ├── feature/auth-module
    ├── feature/simulation-module
    └── feature/service-container
```

## Setup Instructions

### Step 1: Commit Current Work
```bash
# First, let's commit all the planning documents
git add ARCHITECTURE_EVOLUTION.md EXECUTIVE_BRIEF.md LAUNCH_SUMMARY.md MASTERPLAN.txt
git add MICROSERVICES_DECISION.md PHASE1_SECURITY_IMPLEMENTATION.md QUICKSTART_ACTIONS.md
git add backend/modular_architecture_example.py backend/modular_refactor_plan.md
git add GIT_BRANCHING_STRATEGY.md

git commit -m "feat: Add comprehensive launch planning and modular architecture docs"
```

### Step 2: Create V1 Production Branch
```bash
# Create and push V1 production branch
git checkout -b main-v1-production
git push -u origin main-v1-production

# Tag the current stable version
git tag -a v1.0.0 -m "V1 Production: Working monolith with 5 engines"
git push origin v1.0.0
```

### Step 3: Create V2 Modular Branch
```bash
# Create V2 branch from current state
git checkout prototype-deployment
git checkout -b main-v2-modular
git push -u origin main-v2-modular
```

### Step 4: Protect Branches (GitHub/GitLab)
Configure branch protection rules:
- `main-v1-production`: Require PR reviews, no force push
- `main-v2-modular`: Require PR reviews for merges

## Development Workflow

### V1 Production Hotfixes
```bash
# For urgent production fixes
git checkout main-v1-production
git checkout -b hotfix/fix-description
# Make fixes
git push -u origin hotfix/fix-description
# Create PR to main-v1-production
```

### V2 Modular Development
```bash
# For new modular features
git checkout main-v2-modular
git checkout -b feature/module-name
# Develop feature
git push -u origin feature/module-name
# Create PR to main-v2-modular
```

### Syncing V1 Fixes to V2
```bash
# Regularly merge V1 fixes into V2
git checkout main-v2-modular
git merge main-v1-production
# Resolve conflicts if any
git push
```

## Deployment Strategy

### V1 Production (Immediate)
```bash
# Deploy from main-v1-production
git checkout main-v1-production
docker-compose -f docker-compose.prod.yml up -d
```

### V2 Testing (Development)
```bash
# Test V2 in staging
git checkout main-v2-modular
docker-compose -f docker-compose.staging.yml up -d
```

## Migration Plan
1. **Weeks 1-3**: Continue V1 production deployment
2. **Weeks 1-2**: Develop V2 modular structure in parallel
3. **Week 3**: Test V2 in staging environment
4. **Week 4**: Feature flag V2 for beta users
5. **Month 2**: Gradual rollout of V2
6. **Month 3**: V2 becomes primary, V1 maintained for rollback 