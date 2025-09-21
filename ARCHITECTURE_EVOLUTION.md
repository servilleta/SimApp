# Monte Carlo Platform - Architecture Evolution

## Current Architecture (Monolith)
```
┌─────────────────────────────────────────────────────────┐
│                    Frontend (React)                      │
│                   localhost:80 (nginx)                   │
└────────────────────────┬────────────────────────────────┘
                         │ /api/*
┌────────────────────────▼────────────────────────────────┐
│                   Backend (FastAPI)                      │
│                   localhost:8000                         │
│  ┌─────────────────────────────────────────────────┐   │
│  │  Routes: auth, excel-parser, simulations, etc   │   │
│  ├─────────────────────────────────────────────────┤   │
│  │  Services: All in one process                   │   │
│  │  - Authentication (JWT)                         │   │
│  │  - Excel Parsing                               │   │
│  │  - Simulation Engines (Power/Arrow/GPU/etc)    │   │
│  │  - Results Storage                             │   │
│  │  - WebSocket connections                       │   │
│  └─────────────────────────────────────────────────┘   │
└────────────────────────┬────────────────────────────────┘
                         │
         ┌───────────────┴───────────────┐
         │                               │
    ┌────▼─────┐                   ┌────▼─────┐
    │  SQLite  │                   │  Redis   │
    │    DB    │                   │  Cache   │
    └──────────┘                   └──────────┘
```

## Phase 1: Free MVP Architecture (Minimal Changes)
```
┌─────────────────────────────────────────────────────────┐
│                    Frontend (React)                      │
│                 app.yourdomain.com                       │
│                    (Cloudflare)                          │
└────────────────────────┬────────────────────────────────┘
                         │ HTTPS
┌────────────────────────▼────────────────────────────────┐
│                 Load Balancer + WAF                      │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│              Backend (FastAPI) - Multiple Instances      │
│  ┌─────────────────────────────────────────────────┐   │
│  │  Enhanced Monolith with:                        │   │
│  │  + Rate Limiting (per user/IP)                  │   │
│  │  + File Scanning (ClamAV)                       │   │
│  │  + User Quotas                                  │   │
│  │  + Security Headers                             │   │
│  │  + Audit Logging                                │   │
│  └─────────────────────────────────────────────────┘   │
└────────────────────────┬────────────────────────────────┘
                         │
    ┌────────────────────┼────────────────────┐
    │                    │                    │
┌───▼────┐          ┌────▼─────┐      ┌──────▼──────┐
│PostgreSQL         │  Redis    │      │  S3/GCS     │
│(Primary)          │ (Cache)   │      │(File Store) │
└─────────┘         └──────────┘      └─────────────┘
```

## Phase 2: MVP Pay Architecture (Service Separation)
```
                    ┌─────────────────────┐
                    │   CDN (Cloudflare)  │
                    └──────────┬──────────┘
                               │
┌──────────────────────────────▼──────────────────────────┐
│                     API Gateway (Kong)                   │
│              Rate Limiting | Auth | Routing              │
└─────┬──────────┬──────────┬──────────┬─────────────────┘
      │          │          │          │
┌─────▼─────┐ ┌─▼──────┐ ┌─▼──────┐ ┌▼──────────────┐
│   Auth    │ │Payment │ │Core API│ │ Admin API     │
│  Service  │ │Service │ │Service │ │  Service      │
│           │ │        │ │        │ │               │
│ • JWT     │ │• Stripe│ │• Files │ │• User Mgmt    │
│ • OAuth   │ │• Plans │ │• Sims  │ │• Analytics    │
│ • Users   │ │• Billing│ │• Results│ │• Monitoring  │
└───────────┘ └────────┘ └────────┘ └───────────────┘
      │            │          │             │
      └────────────┴──────────┴─────────────┘
                         │
              ┌──────────┴───────────┐
              │   Message Queue      │
              │   (RabbitMQ/Kafka)   │
              └──────────┬───────────┘
                         │
    ┌────────────────────┼──────────────────────┐
    │                    │                      │
┌───▼──────────┐ ┌──────▼───────┐ ┌───────────▼───┐
│ Simulation   │ │ Notification │ │  Background   │
│  Workers     │ │   Service    │ │   Workers     │
│              │ │              │ │               │
│ • Power Eng  │ │ • Email      │ │ • Cleanup     │
│ • Arrow Eng  │ │ • Webhooks   │ │ • Reports     │
│ • GPU Eng    │ │ • SMS        │ │ • Exports     │
└──────────────┘ └──────────────┘ └───────────────┘
```

## Phase 3: Enterprise Microservices Architecture
```
┌─────────────────────────────────────────────────────────┐
│                  Global Load Balancer                    │
│                    (Multi-Region)                        │
└─────────────────────┬───────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────┐
│                   Apigee API Gateway                     │
│          OAuth | Rate Limiting | Analytics | Portal      │
└─────┬───────┬───────┬───────┬───────┬─────────────────┘
      │       │       │       │       │
  ┌───▼──┐ ┌─▼───┐ ┌─▼───┐ ┌▼───┐ ┌▼────────────┐
  │ Auth │ │Excel│ │ Sim │ │Bill│ │   Admin     │
  │ Mesh │ │Parse│ │ API │ │API │ │    API      │
  └───┬──┘ └──┬──┘ └──┬──┘ └─┬──┘ └──────┬──────┘
      │       │       │      │            │
┌─────▼───────▼───────▼──────▼────────────▼──────┐
│           Kubernetes Cluster (GKE)              │
│  ┌─────────────────────────────────────────┐   │
│  │         Service Mesh (Istio)            │   │
│  └─────────────────────────────────────────┘   │
│                                                 │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐      │
│  │  Auth    │ │  Excel   │ │Simulation│      │
│  │ Service  │ │  Parser  │ │ Service  │      │
│  │          │ │  Service │ │          │      │
│  │ • SAML   │ │ • Parse  │ │ • Queue  │      │
│  │ • AD     │ │ • Formula│ │ • Engine │      │
│  │ • MFA    │ │ • Valid  │ │ • Scale  │      │
│  └──────────┘ └──────────┘ └──────────┘      │
│                                                 │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐      │
│  │Analytics │ │ Billing  │ │ Storage  │      │
│  │ Service  │ │ Service  │ │ Service  │      │
│  │          │ │          │ │          │      │
│  │ • Metrics│ │ • Stripe │ │ • Files  │      │
│  │ • ML     │ │ • Invoice│ │ • Results│      │
│  │ • Reports│ │ • Usage  │ │ • Cache  │      │
│  └──────────┘ └──────────┘ └──────────┘      │
│                                                 │
│  ┌─────────────────────────────────────────┐   │
│  │     Horizontal Pod Autoscalers (HPA)    │   │
│  └─────────────────────────────────────────┘   │
└─────────────────────────────────────────────────┘
                         │
    ┌────────────────────┼──────────────────────┐
    │                    │                      │
┌───▼───────┐    ┌──────▼────────┐    ┌───────▼────┐
│Cloud SQL  │    │  Cloud Pub/Sub │    │   GCS/S3   │
│(PostgreSQL)    │  (Message Queue)│    │(Object Store)
│• Multi-AZ │    │  • Topics      │    │• Versioning│
│• Replicas │    │  • Fanout      │    │• Lifecycle │
└───────────┘    └───────────────┘    └────────────┘
```

## Service Definitions

### Core Services

#### 1. Authentication Service
```yaml
name: auth-service
responsibilities:
  - User authentication (JWT, OAuth, SAML)
  - Authorization and permissions
  - Session management
  - MFA/2FA
  - API key management
apis:
  - POST /auth/login
  - POST /auth/logout
  - POST /auth/refresh
  - GET /auth/verify
  - POST /auth/mfa/setup
  - POST /auth/apikeys
```

#### 2. Excel Parser Service
```yaml
name: excel-parser-service
responsibilities:
  - Excel file parsing
  - Formula extraction
  - Dependency analysis
  - Validation
  - Security scanning
apis:
  - POST /parse/upload
  - GET /parse/status/{id}
  - GET /parse/formulas/{id}
  - GET /parse/dependencies/{id}
```

#### 3. Simulation Service
```yaml
name: simulation-service
responsibilities:
  - Simulation orchestration
  - Engine selection
  - Queue management
  - Result aggregation
  - Progress tracking
apis:
  - POST /simulations/create
  - GET /simulations/{id}
  - POST /simulations/{id}/run
  - GET /simulations/{id}/progress
  - GET /simulations/{id}/results
```

#### 4. Billing Service
```yaml
name: billing-service
responsibilities:
  - Subscription management
  - Payment processing
  - Usage tracking
  - Invoice generation
  - Quota enforcement
apis:
  - POST /billing/subscribe
  - PUT /billing/subscription/{id}
  - GET /billing/usage
  - GET /billing/invoices
  - POST /billing/payment-method
```

### Migration Strategy

#### Step 1: Database Migration (Month 7, Week 1)
```sql
-- Migrate from SQLite to PostgreSQL
-- Create service-specific schemas
CREATE SCHEMA auth;
CREATE SCHEMA billing;
CREATE SCHEMA simulations;
CREATE SCHEMA analytics;

-- Migrate tables to appropriate schemas
ALTER TABLE users SET SCHEMA auth;
ALTER TABLE subscriptions SET SCHEMA billing;
ALTER TABLE simulation_runs SET SCHEMA simulations;
```

#### Step 2: API Gateway Introduction (Month 7, Week 2-3)
```javascript
// Apigee proxy configuration
const proxyConfig = {
  "/api/auth/*": "http://auth-service:8001",
  "/api/parse/*": "http://parser-service:8002",
  "/api/simulations/*": "http://simulation-service:8003",
  "/api/billing/*": "http://billing-service:8004"
};
```

#### Step 3: Service Extraction (Month 7-8)
```python
# Example: Extract Authentication Service
# From: backend/auth/router.py (monolith)
# To: services/auth-service/main.py (microservice)

from fastapi import FastAPI
from auth.routes import router as auth_router

app = FastAPI(title="Auth Service")
app.include_router(auth_router)

# Service-specific configuration
SERVICE_CONFIG = {
    "name": "auth-service",
    "version": "1.0.0",
    "port": 8001,
    "dependencies": ["database", "redis", "messaging"]
}
```

## DevOps Evolution

### Current: Docker Compose
```yaml
version: '3.8'
services:
  backend:
    build: ./backend
    ports:
      - "8000:8000"
  frontend:
    build: ./frontend
    ports:
      - "80:80"
```

### Phase 2: Docker Swarm/ECS
```yaml
version: '3.8'
services:
  auth:
    image: montecarlo/auth:latest
    deploy:
      replicas: 3
  simulation:
    image: montecarlo/simulation:latest
    deploy:
      replicas: 5
```

### Phase 3: Kubernetes
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: auth-service
spec:
  replicas: 3
  selector:
    matchLabels:
      app: auth-service
  template:
    metadata:
      labels:
        app: auth-service
    spec:
      containers:
      - name: auth
        image: gcr.io/montecarlo/auth:v1.0.0
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
```

## Security Evolution

### Phase 1: Basic Security
- HTTPS everywhere
- JWT authentication
- Input validation
- Rate limiting

### Phase 2: Enhanced Security
- OAuth2/OIDC
- API keys with scopes
- Audit logging
- WAF protection

### Phase 3: Enterprise Security
- SAML/SSO
- mTLS between services
- Zero-trust networking
- Compliance certifications

## Monitoring Evolution

### Phase 1: Basic Monitoring
```python
# Simple health checks
@app.get("/health")
def health_check():
    return {"status": "healthy"}
```

### Phase 2: APM Integration
```python
# Sentry + Prometheus
import sentry_sdk
from prometheus_client import Counter, Histogram

request_count = Counter('requests_total', 'Total requests')
request_duration = Histogram('request_duration_seconds', 'Request duration')
```

### Phase 3: Full Observability
```yaml
# Istio service mesh provides:
- Distributed tracing (Jaeger)
- Metrics (Prometheus/Grafana)
- Logging (ELK stack)
- Service topology
- Circuit breaking
```

## Cost Optimization

### Infrastructure Costs by Phase
```
Phase 1 (Free MVP):
- 2x t3.medium instances: $60/month
- RDS PostgreSQL: $50/month
- S3 storage: $20/month
- Total: ~$130/month

Phase 2 (MVP Pay):
- 4x t3.large instances: $240/month
- RDS with replica: $150/month
- ElastiCache: $50/month
- Total: ~$440/month

Phase 3 (Enterprise):
- GKE cluster: $500/month
- Cloud SQL HA: $300/month
- Pub/Sub + Storage: $200/month
- Apigee: $500/month
- Total: ~$1,500/month base
```

## Success Criteria

### Phase 1
- Handle 1,000 concurrent users
- 99.9% uptime
- <2s simulation start time

### Phase 2
- Handle 10,000 concurrent users
- 99.95% uptime
- <500ms API response time

### Phase 3
- Handle 100,000 concurrent users
- 99.99% uptime
- <200ms API response time
- Multi-region deployment 