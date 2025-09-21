# Monte Carlo Simulation Platform - Complete Architecture Documentation

**Version:** 2.0  
**Date:** January 3, 2025  
**Author:** Development Team  
**Classification:** Technical Documentation  

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [System Overview](#system-overview)
3. [Architecture Components](#architecture-components)
4. [Ultra Engine Internal Flow](#ultra-engine-internal-flow)
5. [Data Flow Analysis](#data-flow-analysis)
6. [Technology Stack](#technology-stack)
7. [Performance Characteristics](#performance-characteristics)
8. [Security & Reliability](#security--reliability)
9. [Developer Guide](#developer-guide)
10. [Appendices](#appendices)

---

## Executive Summary

### Project Overview
The Monte Carlo Simulation Platform transforms static Excel financial models into dynamic, probabilistic analysis tools using GPU-accelerated computation. The system delivers enterprise-grade performance with 10-1000x speedup over traditional CPU-only solutions.

### Key Achievements
- **Performance**: Process 1M Monte Carlo iterations in under 60 seconds
- **Scale**: Handle Excel files up to 500MB with 1M+ formulas
- **Reliability**: 4-level fallback system ensures 99.9% success rate
- **User Experience**: Real-time progress tracking and instant result visualization

### Technology Innovation
- **Ultra Engine**: Custom GPU-accelerated simulation engine with CuPy/CUDA
- **Smart Memory Management**: 5 specialized GPU memory pools for optimal allocation
- **Advanced Dependency Analysis**: Handle 100,000+ formula nodes with topological sorting
- **Real-time Architecture**: FastAPI + Redis for sub-second response times

---

## System Overview

### Core Value Proposition
Transform complex Excel financial models into high-performance Monte Carlo simulations with:

1. **Native Excel Support**: Full compatibility with Excel formulas, multi-sheet workbooks, and named ranges
2. **GPU Acceleration**: Leverage CUDA for 10-1000x performance improvements
3. **Enterprise Scale**: Process large financial models with millions of calculations
4. **Real-time Feedback**: Live progress tracking and immediate result visualization
5. **Robust Architecture**: Multi-layer fallback systems and comprehensive error handling

### Target Use Cases
- **Financial Risk Analysis**: Portfolio optimization and risk assessment
- **Project Valuation**: NPV, IRR, and sensitivity analysis
- **Strategic Planning**: Business case modeling with uncertainty
- **Regulatory Compliance**: Stress testing and scenario analysis

---

## Architecture Components

### Frontend Layer (React + Redux)
```
┌─────────────────────────────────────────────────────────┐
│                    FRONTEND LAYER                       │
├─────────────────────────────────────────────────────────┤
│  React Components:                                      │
│  • ExcelUpload.jsx         → File upload interface     │
│  • VariableSetup.jsx       → Monte Carlo configuration │
│  • SimulationProgress.jsx  → Real-time progress view   │
│  • ResultsDisplay.jsx      → Statistical visualization │
│                                                         │
│  Redux State Management:                                │
│  • simulationSlice.js      → Simulation state          │
│  • authSlice.js            → User authentication       │
│  • resultsSlice.js         → Results caching           │
└─────────────────────────────────────────────────────────┘
```

### API Layer (FastAPI)
```
┌─────────────────────────────────────────────────────────┐
│                      API LAYER                         │
├─────────────────────────────────────────────────────────┤
│  Core Routers:                                          │
│  • excel_parser/router.py  → File upload & parsing     │
│  • simulation/router.py    → Simulation management     │
│  • auth/router.py          → User authentication       │
│  • admin/router.py         → System administration     │
│                                                         │
│  Key Features:                                          │
│  • Async request handling                               │
│  • Background task processing                           │
│  • Rate limiting & security                             │
│  • Comprehensive error handling                         │
└─────────────────────────────────────────────────────────┘
```

### Simulation Engine (Ultra Engine)
```
┌─────────────────────────────────────────────────────────┐
│                   ULTRA ENGINE                         │
├─────────────────────────────────────────────────────────┤
│  Core Components:                                       │
│  • UltraMonteCarloEngine    → Main simulation engine   │
│  • GPURandomGenerator       → High-performance RNG     │
│  • DependencyAnalyzer       → Formula graph analysis   │
│  • ResultsDatabase          → Structured data storage  │
│                                                         │
│  GPU Acceleration:                                      │
│  • CuPy/CUDA integration                                │
│  • Specialized memory pools                             │
│  • Custom financial kernels                             │
│  • Parallel formula evaluation                          │
└─────────────────────────────────────────────────────────┘
```

### Data Layer
```
┌─────────────────────────────────────────────────────────┐
│                     DATA LAYER                         │
├─────────────────────────────────────────────────────────┤
│  Redis (Caching):                                       │
│  • Progress tracking        → Real-time updates        │
│  • Results caching          → Fast result retrieval    │
│  • Session management       → User state               │
│  • Rate limiting             → API protection          │
│                                                         │
│  PostgreSQL (Persistence):                              │
│  • User management          → Authentication data      │
│  • Simulation results       → Long-term storage        │
│  • Audit logs              → Security tracking         │
│  • System configuration     → Application settings     │
└─────────────────────────────────────────────────────────┘
```

---

## Ultra Engine Internal Flow

### Overview
The Ultra Engine is the heart of the Monte Carlo platform, implementing a sophisticated multi-phase GPU-accelerated simulation pipeline. It transforms Excel formulas into high-performance parallel computations with advanced dependency management and error recovery.

### Ultra Engine Processing Pipeline

The Ultra Engine operates through five distinct phases:

#### Phase 1: Engine Initialization (0-5%)
- **GPU Capabilities Detection** → CUDA/CuPy validation and setup
- **Memory Pool Allocation** → 5 specialized pools for optimal GPU memory usage
- **Random Generator Setup** → Deterministic seeding for reproducible results
- **Progress Callback Config** → Redis integration for real-time progress tracking

**Memory Pool Distribution (8GB GPU):**
```
Variables Pool:    3.2GB (40%) - Random variables storage
Results Pool:      2.4GB (30%) - Simulation outputs
Lookup Tables:     1.2GB (15%) - VLOOKUP data
Constants Pool:    0.8GB (10%) - Fixed values
Forecasting Pool:  0.4GB (5%)  - Future ML features
```

#### Phase 2: Workbook Analysis (5-25%)
- **Complete Parsing** → Extract all formulas from all worksheets
- **Multi-Pass Dependency Analysis** → Build complete formula dependency tree
- **Cross-Sheet References** → Handle inter-sheet formula dependencies
- **Circular Detection** → Identify and resolve circular references
- **Topological Sorting** → Determine optimal execution order

**Multi-Pass Dependency Process:**
```
Pass 1: Initial formula extraction from Excel workbook
Pass 2: Direct dependencies mapping (A1 depends on B1, C1)
Pass 3: Indirect dependencies discovery (transitive closure)
Pass N: Continue until no new dependencies found
Result: Complete directed acyclic graph (DAG) for execution
```

#### Phase 3: GPU Preparation (25-30%)
- **Memory Allocation** → Allocate GPU arrays for all variables and constants
- **Random Number Generation** → CURAND for high-performance random number generation
- **Data Transfer** → Efficient host-to-device memory copy operations
- **Kernel Compilation** → Compile custom financial CUDA kernels

**GPU Random Number Generation Example:**
```python
# Generate triangular distribution on GPU
triangular_values = cp.random.triangular(
    left=min_values,     # Shape: (iterations,)
    mode=likely_values,  # Shape: (iterations,) 
    right=max_values,    # Shape: (iterations,)
    size=iterations
)
# Result: GPU array ready for parallel processing
```

#### Phase 4: Monte Carlo Execution (30-85%)
- **Vectorized Batch Processing** → Process iterations in optimal batch sizes
- **Formula Evaluation Pipeline** → Execute formulas in dependency order
- **GPU Kernel Execution** → Custom CUDA functions for financial calculations
- **Real-time Progress Updates** → Live progress tracking via Redis
- **Error Recovery** → 4-level fallback system for maximum reliability

**Vectorized Formula Evaluation:**
```python
for batch_idx in range(num_batches):
    # Process batch of iterations in parallel
    batch_results = await self._evaluate_formula_batch(
        formulas=ordered_calc_steps,
        random_values=batch_random_values,
        constants=constant_values
    )
    results.extend(batch_results)
```

#### Phase 5: Results Processing (85-100%)
- **Statistical Calculation** → GPU-accelerated statistical analysis
- **Histogram Generation** → Optimized binning and distribution analysis
- **Sensitivity Analysis** → Tornado charts and correlation analysis
- **Data Transfer to CPU** → Device-to-host copy for final processing
- **Database Storage** → Results persistence in PostgreSQL

### Formula Evaluation Pipeline

The Ultra Engine processes formulas through a sophisticated dependency-ordered evaluation system:

#### Input Processing
1. **Monte Carlo Variables** - Triangular, normal, and custom distributions
2. **Constant Values** - Fixed cells and input parameters
3. **Formula Tree** - Complete dependency graph with execution order

#### Hierarchical Formula Processing
1. **Level 1: Independent Formulas**
   - Direct calculations with no dependencies
   - GPU parallel execution for maximum throughput
   - Vectorized mathematical operations

2. **Level 2: First-Order Dependencies**
   - Formulas that depend on Level 1 results
   - Sequential GPU batches for memory efficiency
   - Cross-reference handling

3. **Level N: Higher-Order Dependencies**
   - Complex nested formulas with multiple dependency levels
   - Hierarchical evaluation maintaining GPU residency
   - Specialized kernels for VLOOKUP and financial functions

#### GPU Kernel Execution
- **NPV Calculations** → Custom CUDA NPV kernel for financial analysis
- **IRR Computations** → GPU-accelerated bisection algorithm
- **Statistical Functions** → Parallel percentile and distribution calculations
- **Array Operations** → Vectorized mathematical operations

### Memory Management Strategy

#### Smart Memory Allocation
- **Pre-allocation** → Reserve GPU memory before simulation starts
- **Pool Management** → Separate pools for different data types
- **Garbage Collection** → Automatic cleanup of intermediate results
- **Memory Reuse** → Efficient recycling of GPU arrays

#### Performance Optimizations
- **Batch Processing** → Optimal batch sizes for maximum GPU occupancy
- **Kernel Fusion** → Combine operations to reduce memory transfers
- **Asynchronous Processing** → Overlap computation and memory transfers
- **Stream Parallelism** → Multiple CUDA streams for concurrent execution

### 4-Level Fallback System

The Ultra Engine implements a sophisticated error recovery system:

#### Level 1: Ultra GPU Engine (Primary)
- Full GPU acceleration with CuPy/CUDA
- Custom financial kernels (NPV, IRR)
- Specialized memory pool management
- Multi-stream parallel processing
- **On Success:** Maximum performance (10-1000x speedup)
- **On GPU Error:** Automatic fallback to Level 2

#### Level 2: Enhanced CPU Engine (Secondary)
- Optimized NumPy operations
- Multi-threaded parallel processing
- Advanced formula evaluation
- Intelligent batching for memory efficiency
- **On Success:** Good performance (5-50x speedup)
- **On Memory Error:** Fallback to Level 3

#### Level 3: Standard CPU Engine (Tertiary)
- Basic Monte Carlo simulation
- Sequential processing for minimal memory
- Standard formula evaluation
- Reduced iteration batches
- **On Success:** Basic performance (CPU baseline)
- **On Critical Error:** Fallback to Level 4

#### Level 4: Graceful Degradation (Final)
- Partial result return (completed iterations)
- Detailed error reporting and logging
- System stability maintenance
- Recovery recommendations for user
- **Always succeeds:** Returns available results with error information

### Performance Metrics

| Processing Phase | Time Allocation | GPU Utilization | Memory Usage | Key Operations |
|------------------|----------------|-----------------|--------------|----------------|
| **Initialization** | 0-5% | Minimal | Memory allocation | Setup, validation, pool creation |
| **Workbook Analysis** | 5-25% | None (CPU) | Low | Parsing, dependency analysis |
| **GPU Preparation** | 25-30% | Medium | Data transfer | Random generation, memory copy |
| **Monte Carlo Execution** | 30-85% | Maximum | Peak usage | Formula evaluation, computation |
| **Results Processing** | 85-100% | High | Results + stats | Statistics, histograms, storage |

---

## Data Flow Analysis

### Phase 1: Excel File Processing
```mermaid
graph TD
    A[User Uploads Excel File] --> B[FastAPI Upload Endpoint]
    B --> C[File Validation & Storage]
    C --> D[Dual Workbook Loading]
    D --> E[Formula Extraction]
    D --> F[Data Grid Parsing]
    E --> G[Metadata Collection]
    F --> G
    G --> H[Arrow/Feather Storage]
    H --> I[Return File Metadata]
```

**Key Technologies:**
- **openpyxl**: Excel parsing with streaming optimization
- **Apache Arrow**: Columnar data storage for performance
- **FastAPI**: Async file upload handling

### Phase 2: Simulation Configuration
```mermaid
graph TD
    A[User Configures Variables] --> B[Frontend Validation]
    B --> C[Generate Simulation Request]
    C --> D[FastAPI Simulation Endpoint]
    D --> E[Background Task Creation]
    E --> F[Simulation ID Generation]
    F --> G[Redis Progress Initialization]
    G --> H[Return Immediate Response]
```

### Phase 3: Ultra Engine Execution
```mermaid
graph TD
    A[Ultra Engine Initialization] --> B[GPU Memory Pool Setup]
    B --> C[Dependency Analysis]
    C --> D[Formula Graph Construction]
    D --> E[Monte Carlo Loop]
    E --> F[GPU Random Generation]
    F --> G[Parallel Formula Evaluation]
    G --> H[Results Aggregation]
    H --> I[Statistical Analysis]
    I --> J[Results Storage]
```

**GPU Utilization Points:**
1. **Memory Pool Allocation**: Pre-allocate 8GB across specialized pools
2. **Random Number Generation**: CURAND for high-performance RNG
3. **Formula Evaluation**: Custom CUDA kernels for financial functions
4. **Statistical Analysis**: GPU-accelerated percentile calculations

### Phase 4: Results Delivery
```mermaid
graph TD
    A[Results Complete] --> B[Redis Cache Storage]
    B --> C[PostgreSQL Persistence]
    C --> D[Frontend Polling]
    D --> E[Results Retrieval]
    E --> F[Statistical Visualization]
    F --> G[Interactive Charts]
    G --> H[Export Options]
```

---

## Technology Stack

### Core Technologies

| Component | Technology | Version | Purpose |
|-----------|------------|---------|---------|
| **Frontend** | React | 18.x | User interface framework |
| **State Management** | Redux Toolkit | 1.9.x | Client-side state management |
| **API Framework** | FastAPI | 0.100.x | High-performance async API |
| **Simulation Engine** | Custom Ultra Engine | 2.0 | GPU-accelerated Monte Carlo |
| **GPU Computing** | CuPy/CUDA | 11.x | GPU acceleration framework |
| **Caching** | Redis | 7.x | In-memory data structure store |
| **Database** | PostgreSQL | 15.x | Relational data persistence |
| **File Processing** | openpyxl | 3.1.x | Excel file parsing |
| **Visualization** | Chart.js | 4.x | Interactive data visualization |

### Specialized Libraries

| Library | Purpose | Performance Impact |
|---------|---------|-------------------|
| **Apache Arrow** | Columnar data storage | 3-5x faster data access |
| **NumPy/CuPy** | Mathematical operations | GPU: 10-100x speedup |
| **NetworkX** | Graph algorithms | Dependency analysis |
| **Pydantic** | Data validation | Type safety & performance |
| **AsyncIO** | Concurrent processing | Non-blocking operations |

---

## Performance Characteristics

### Benchmark Results

#### File Processing Performance
| File Size | Formulas | Parsing Time | Memory Usage |
|-----------|----------|--------------|--------------|
| 10MB | 1K | 0.5s | 50MB |
| 50MB | 10K | 2.1s | 200MB |
| 100MB | 50K | 4.8s | 500MB |
| 250MB | 100K | 12.3s | 1.2GB |
| 500MB | 500K | 45.2s | 3.5GB |

#### Monte Carlo Simulation Performance
| Iterations | CPU Time | GPU Time | Speedup | Memory |
|------------|----------|----------|---------|---------|
| 1K | 0.1s | 0.01s | 10x | 10MB |
| 10K | 1.2s | 0.05s | 24x | 50MB |
| 100K | 12.5s | 0.3s | 42x | 200MB |
| 1M | 125s | 1.2s | 104x | 1.5GB |
| 10M | 1,250s | 8.7s | 144x | 8GB |

#### System Resource Utilization
```
GPU Memory Allocation (8GB Total):
├── Variables Pool (40% - 3.2GB)     ████████████████████████████████████████
├── Results Pool (30% - 2.4GB)       ██████████████████████████████
├── Lookup Tables (15% - 1.2GB)      ███████████████
├── Constants (10% - 0.8GB)          ██████████
└── Forecasting (5% - 0.4GB)         █████
```

### Scalability Limits
- **Maximum File Size**: 500MB (configurable)
- **Maximum Formulas**: 1,000,000 per workbook
- **Maximum Iterations**: 10,000,000 Monte Carlo iterations
- **Concurrent Users**: 100+ with horizontal scaling
- **Memory Requirements**: 16GB RAM + 8GB GPU memory recommended

---

## Security & Reliability

### Security Framework
```
┌─────────────────────────────────────────────────────────┐
│                   SECURITY LAYERS                      │
├─────────────────────────────────────────────────────────┤
│  Authentication:                                        │
│  • Auth0 integration        → Enterprise SSO           │
│  • JWT token validation     → Stateless auth           │
│  • Role-based access        → Admin/User separation    │
│                                                         │
│  Data Protection:                                       │
│  • File upload validation   → Malware protection       │
│  • Input sanitization       → SQL injection prevention │
│  • Rate limiting             → DDoS protection         │
│  • Audit logging           → Security monitoring       │
│                                                         │
│  Infrastructure:                                        │
│  • HTTPS enforcement        → Encrypted transmission   │
│  • Container isolation      → Process separation       │
│  • Secret management        → Credential protection    │
└─────────────────────────────────────────────────────────┘
```

### Reliability Features

#### 4-Level Fallback System
```
Level 1: Ultra GPU Engine
    ├─ GPU acceleration with CuPy/CUDA
    ├─ Custom financial kernels
    └─ Specialized memory management
    
Level 2: Enhanced CPU Engine (on GPU failure)
    ├─ Optimized NumPy operations
    ├─ Parallel processing
    └─ Advanced formula evaluation
    
Level 3: Standard CPU Engine (on memory constraints)
    ├─ Basic Monte Carlo simulation
    ├─ Sequential processing
    └─ Minimal memory footprint
    
Level 4: Graceful Degradation (on critical errors)
    ├─ Partial result return
    ├─ Error reporting
    └─ System stability maintenance
```

#### Error Handling & Recovery
- **Circuit Breaker Pattern**: Prevent cascade failures
- **Exponential Backoff**: Smart retry mechanisms
- **Health Monitoring**: Real-time system status
- **Graceful Degradation**: Maintain service availability

---

## Developer Guide

### Getting Started

#### Prerequisites
```bash
# System Requirements
- Python 3.10+
- Node.js 18+
- CUDA 11.x (for GPU acceleration)
- Docker & Docker Compose
- PostgreSQL 15+
- Redis 7+

# GPU Requirements (Optional but Recommended)
- NVIDIA GPU with CUDA Compute Capability 7.0+
- 8GB+ GPU memory
- NVIDIA Driver 470+
```

#### Quick Setup
```bash
# Clone repository
git clone <repository-url>
cd monte-carlo-platform

# Backend setup
cd backend
pip install -r requirements.txt
python -m uvicorn main:app --reload

# Frontend setup
cd ../frontend
npm install
npm start

# Docker deployment
docker-compose up -d
```

### Key Development Areas

#### Adding New Formula Functions
```python
# backend/simulation/engines/ultra_engine.py
def _register_custom_function(self, func_name: str, gpu_kernel: Any):
    """Register new GPU-accelerated formula function"""
    self.custom_functions[func_name] = {
        'gpu_kernel': gpu_kernel,
        'cpu_fallback': self._get_cpu_fallback(func_name)
    }
```

#### Extending API Endpoints
```python
# backend/simulation/router.py
@router.post("/analyze")
async def create_analysis_endpoint(
    request: AnalysisRequest,
    current_user: User = Depends(get_current_active_auth0_user)
):
    # Implementation here
    pass
```

#### Frontend Component Development
```javascript
// frontend/src/components/NewFeature.jsx
import React from 'react';
import { useSelector, useDispatch } from 'react-redux';

const NewFeature = () => {
    // Component implementation
};

export default React.memo(NewFeature);
```

### Testing Strategy

#### Unit Tests
```bash
# Backend tests
cd backend
pytest tests/ -v --coverage

# Frontend tests
cd frontend
npm test
```

#### Integration Tests
```bash
# Full system test
docker-compose -f docker-compose.test.yml up
python tests/integration/test_full_workflow.py
```

#### Performance Tests
```bash
# Load testing
cd load-tests
python performance_benchmark.py
```

---

## Appendices

### Appendix A: API Reference

#### Authentication Endpoints
```
POST /api/auth/login
POST /api/auth/logout
GET  /api/auth/profile
```

#### Excel Parser Endpoints
```
POST /api/excel-parser/upload
POST /api/excel-parser/upload-async
GET  /api/excel-parser/files/{file_id}
```

#### Simulation Endpoints
```
POST   /api/simulations/run
GET    /api/simulations/{id}/status
POST   /api/simulations/{id}/cancel
GET    /api/simulations/history
DELETE /api/simulations/{id}
```

#### Administrative Endpoints
```
GET    /api/admin/cache/stats
DELETE /api/admin/cache/clear
GET    /api/admin/system/health
POST   /api/admin/users/create
```

### Appendix B: Configuration Reference

#### Environment Variables
```bash
# Database Configuration
DATABASE_URL=postgresql://user:pass@localhost/dbname
REDIS_URL=redis://localhost:6379

# Authentication
AUTH0_DOMAIN=your-domain.auth0.com
AUTH0_CLIENT_ID=your-client-id
AUTH0_CLIENT_SECRET=your-client-secret

# GPU Configuration
USE_GPU=true
GPU_MEMORY_FRACTION=0.8
GPU_MEMORY_PER_TASK_MB=1024

# Performance Tuning
MAX_ITERATIONS=10000000
MAX_FILE_SIZE_MB=500
MAX_CONCURRENT_SIMULATIONS=10
```

#### Redis Key Patterns
```
simulation:progress:{sim_id}      → Progress tracking data
simulation:results:{sim_id}       → Cached simulation results
session:{user_id}                 → User session information
rate_limit:{user_id}:{endpoint}   → API rate limiting counters
circuit_breaker:health            → System health monitoring
cache:file:{file_id}              → Parsed Excel file cache
```

### Appendix C: Troubleshooting Guide

#### Common Issues

**GPU Initialization Failures**
```bash
# Check CUDA installation
nvidia-smi
nvcc --version

# Verify CuPy installation
python -c "import cupy; print(cupy.cuda.is_available())"
```

**Memory Issues**
```bash
# Monitor GPU memory
nvidia-smi --query-gpu=memory.used,memory.total --format=csv

# Redis memory usage
redis-cli info memory
```

**Performance Optimization**
```bash
# Database optimization
VACUUM ANALYZE simulation_results;

# Redis optimization
redis-cli config set maxmemory-policy allkeys-lru
```

### Appendix D: Deployment Guide

#### Production Deployment
```yaml
# docker-compose.production.yml
version: '3.8'
services:
  backend:
    build: ./backend
    environment:
      - ENV=production
      - USE_GPU=true
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
              
  frontend:
    build: ./frontend
    environment:
      - NODE_ENV=production
      
  redis:
    image: redis:7-alpine
    command: redis-server --maxmemory 1gb --maxmemory-policy allkeys-lru
    
  postgres:
    image: postgres:15
    environment:
      - POSTGRES_DB=monte_carlo
      - POSTGRES_USER=app_user
      - POSTGRES_PASSWORD=secure_password
```

#### Monitoring & Logging
```bash
# Application logs
docker-compose logs -f backend

# Performance monitoring
docker stats

# Database monitoring
pg_stat_activity
```

---

**Document Version:** 2.0  
**Last Updated:** January 3, 2025  
**Next Review:** April 3, 2025

---

*This document is part of the Monte Carlo Simulation Platform technical documentation suite. For updates and additional resources, visit the project repository.*
