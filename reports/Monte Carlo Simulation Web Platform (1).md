# Monte Carlo Simulation Web Platform
# Backend Developer Guide

## Table of Contents
1. [Introduction](#introduction)
2. [Architecture Overview](#architecture-overview)
3. [Project Structure](#project-structure)
4. [Development Environment Setup](#development-environment-setup)
5. [FastAPI Implementation](#fastapi-implementation)
6. [Excel Parsing](#excel-parsing)
7. [Simulation Engine](#simulation-engine)
8. [GPU Integration](#gpu-integration)
9. [API Endpoints](#api-endpoints)
10. [Testing](#testing)
11. [Performance Optimization](#performance-optimization)
12. [Containerization](#containerization)
13. [References](#references)

## Introduction

This document provides comprehensive guidelines for backend development of the Monte Carlo Simulation Web Platform. The backend is responsible for handling file uploads, parsing Excel data, running simulations with triangular probability distributions, and providing results to the frontend for visualization.

### System Requirements

- Python 3.11+
- FastAPI for API development
- pandas and openpyxl for Excel parsing
- numpy and scipy for simulation calculations
- GPU acceleration for computation-intensive tasks
- Docker for containerization

## Architecture Overview

The backend follows a microservice architecture with the following components:

1. **API Layer**: FastAPI application handling HTTP requests
2. **Excel Parser**: Module for extracting data and formulas from uploaded Excel files
3. **Simulation Engine**: Core module for running Monte Carlo simulations
4. **Results Processor**: Module for aggregating and formatting simulation results
5. **GPU Orchestrator**: Module for managing GPU resources and workloads

### Data Flow

1. User uploads Excel file through frontend
2. Backend parses Excel file and extracts data/formulas
3. User configures variables with triangular distributions
4. Backend runs Monte Carlo simulation using configured parameters
5. Results are processed and returned to frontend for visualization

## Project Structure

Follow this domain-driven structure for better organization and scalability:

```
backend/
├── main.py                 # FastAPI application entry point
├── config.py               # Global configuration
├── models.py               # Shared database models
├── exceptions.py           # Custom exception classes
├── auth/                   # Authentication module
│   ├── router.py           # Auth endpoints
│   ├── schemas.py          # Pydantic models
│   ├── dependencies.py     # Auth dependencies
│   └── service.py          # Auth business logic
├── excel_parser/           # Excel parsing module
│   ├── router.py           # Excel upload endpoints
│   ├── schemas.py          # Pydantic models
│   ├── service.py          # Excel parsing logic
│   └── utils.py            # Helper functions
├── simulation/             # Simulation module
│   ├── router.py           # Simulation endpoints
│   ├── schemas.py          # Pydantic models
│   ├── service.py          # Simulation business logic
│   ├── engine.py           # Core simulation engine
│   └── utils.py            # Helper functions
├── results/                # Results processing module
│   ├── router.py           # Results endpoints
│   ├── schemas.py          # Pydantic models
│   ├── service.py          # Results processing logic
│   └── utils.py            # Helper functions
└── gpu/                    # GPU orchestration module
    ├── manager.py          # GPU resource management
    ├── scheduler.py        # Task scheduling
    └── utils.py            # Helper functions
```

## Development Environment Setup

### Prerequisites

- Python 3.11+
- CUDA Toolkit (for GPU support)
- Docker and Docker Compose
- Git

### Setup Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/your-org/monte-carlo-app.git
   cd monte-carlo-app/backend
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

5. Run the development server:
   ```bash
   uvicorn main:app --reload
   ```

### requirements.txt

```
fastapi==0.110.0
uvicorn==0.27.1
pydantic==2.6.1
python-multipart==0.0.9
pandas==2.2.0
openpyxl==3.1.2
numpy==1.26.3
scipy==1.12.0
joblib==1.3.2
pynvml==11.5.0
httpx==0.26.0
pytest==7.4.3
pytest-asyncio==0.23.2
```

## FastAPI Implementation

### Main Application

Create a `main.py` file as the entry point:

```python
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware

from config import settings
from excel_parser.router import router as excel_router
from simulation.router import router as simulation_router
from results.router import router as results_router

app = FastAPI(
    title="Monte Carlo Simulation API",
    description="API for running Monte Carlo simulations on Excel data",
    version="1.0.0",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(excel_router, prefix="/api/excel", tags=["Excel"])
app.include_router(simulation_router, prefix="/api/simulate", tags=["Simulation"])
app.include_router(results_router, prefix="/api/results", tags=["Results"])

@app.get("/api/health")
async def health_check():
    return {"status": "healthy"}
```

### Configuration

Create a `config.py` file for application settings:

```python
from pydantic_settings import BaseSettings
from typing import List

class Settings(BaseSettings):
    # API settings
    API_PREFIX: str = "/api"
    DEBUG: bool = False
    
    # CORS settings
    CORS_ORIGINS: List[str] = ["http://localhost:3000"]
    
    # File upload settings
    UPLOAD_DIR: str = "uploads"
    MAX_UPLOAD_SIZE: int = 10 * 1024 * 1024  # 10MB
    
    # Simulation settings
    DEFAULT_ITERATIONS: int = 1000
    MAX_ITERATIONS: int = 100000
    
    # GPU settings
    USE_GPU: bool = True
    GPU_MEMORY_FRACTION: float = 0.8
    
    class Config:
        env_file = ".env"

settings = Settings()
```

### Async Route Handling

Follow these best practices for route handling:

- Use **async** routes for I/O-bound operations (file operations, network calls)
- Use **sync** routes for CPU-bound operations (calculations, data processing)
- Avoid blocking operations in async routes
- Use thread pools for CPU-intensive tasks

Example router implementation:

```python
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
from typing import List

from .schemas import ExcelFileResponse
from .service import parse_excel_file

router = APIRouter()

@router.post("/upload", response_model=ExcelFileResponse)
async def upload_excel(file: UploadFile = File(...)):
    """
    Upload and parse an Excel file.
    This is an async route because file operations are I/O-bound.
    """
    if not file.filename.endswith((".xlsx", ".xls")):
        raise HTTPException(status_code=400, detail="File must be an Excel file")
    
    # Parse the Excel file (I/O-bound operation)
    result = await parse_excel_file(file)
    return result

@router.get("/variables/{file_id}", response_model=List[str])
def get_variables(file_id: str):
    """
    Get available variables from a parsed Excel file.
    This is a sync route because it's primarily data retrieval.
    """
    # Get variables (CPU-bound operation)
    variables = get_file_variables(file_id)
    return variables
```

## Excel Parsing

### Excel Parser Implementation

Create an Excel parsing service in `excel_parser/service.py`:

```python
import pandas as pd
import openpyxl
from fastapi import UploadFile
import os
from uuid import uuid4

from config import settings

async def parse_excel_file(file: UploadFile):
    """Parse an Excel file and extract data and formulas."""
    # Generate a unique ID for the file
    file_id = str(uuid4())
    
    # Create upload directory if it doesn't exist
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    
    # Save the file
    file_path = os.path.join(settings.UPLOAD_DIR, f"{file_id}.xlsx")
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)
    
    # Parse with pandas for data
    df = pd.read_excel(file_path, engine="openpyxl")
    
    # Parse with openpyxl for formulas
    workbook = openpyxl.load_workbook(file_path, data_only=False)
    sheet = workbook.active
    
    # Extract formulas
    formulas = {}
    for row in sheet.iter_rows():
        for cell in row:
            if cell.formula:
                formulas[cell.coordinate] = cell.formula
    
    # Extract sheet names
    sheet_names = workbook.sheetnames
    
    return {
        "file_id": file_id,
        "filename": file.filename,
        "sheet_names": sheet_names,
        "columns": df.columns.tolist(),
        "row_count": len(df),
        "formulas_count": len(formulas),
        "preview": df.head(5).to_dict(orient="records")
    }
```

### Best Practices for Excel Parsing

1. **Memory Optimization**: Use `read_only=True` for large files:
   ```python
   workbook = openpyxl.load_workbook(file_path, read_only=True, data_only=False)
   ```

2. **Error Handling**: Implement robust error handling for formula parsing:
   ```python
   try:
       workbook = openpyxl.load_workbook(file_path, data_only=False)
   except Exception as e:
       raise HTTPException(status_code=400, detail=f"Failed to parse Excel file: {str(e)}")
   ```

3. **Chunking**: Process large files in chunks:
   ```python
   # For large files, read in chunks
   chunks = pd.read_excel(file_path, engine="openpyxl", chunksize=1000)
   data = []
   for chunk in chunks:
       data.append(chunk)
   df = pd.concat(data)
   ```

4. **Validation**: Validate Excel structure before processing:
   ```python
   def validate_excel_structure(df):
       """Validate that the Excel file has the required structure."""
       required_columns = ["Variable", "Value"]
       missing_columns = [col for col in required_columns if col not in df.columns]
       if missing_columns:
           raise ValueError(f"Missing required columns: {missing_columns}")
   ```

## Simulation Engine

### Core Simulation Engine

Create a simulation engine in `simulation/engine.py`:

```python
import numpy as np
from typing import Dict, List, Tuple
import asyncio
from concurrent.futures import ThreadPoolExecutor

from config import settings

class MonteCarloSimulation:
    def __init__(self, iterations: int = settings.DEFAULT_ITERATIONS):
        self.iterations = min(iterations, settings.MAX_ITERATIONS)
        self.executor = ThreadPoolExecutor()
    
    async def run_simulation(self, variables: Dict[str, Tuple[float, float, float]], formula: str):
        """
        Run a Monte Carlo simulation with triangular distributions.
        
        Args:
            variables: Dict mapping variable names to (min, mode, max) values
            formula: String representation of the formula to evaluate
        
        Returns:
            Dict containing simulation results
        """
        # Run the CPU-intensive simulation in a thread pool
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            self.executor,
            self._run_simulation_sync,
            variables,
            formula
        )
        return results
    
    def _run_simulation_sync(self, variables: Dict[str, Tuple[float, float, float]], formula: str):
        """Synchronous implementation of the simulation."""
        # Generate random values for each variable
        random_values = {}
        for var_name, (min_val, mode_val, max_val) in variables.items():
            random_values[var_name] = np.random.triangular(
                min_val, mode_val, max_val, size=self.iterations
            )
        
        # Evaluate the formula for each iteration
        results = np.zeros(self.iterations)
        for i in range(self.iterations):
            # Create a dictionary of variable values for this iteration
            iter_values = {var: values[i] for var, values in random_values.items()}
            
            # Evaluate the formula using the variable values
            # Note: In a real implementation, you would need a safe way to evaluate the formula
            # This is a simplified example
            formula_with_values = formula
            for var, value in iter_values.items():
                formula_with_values = formula_with_values.replace(var, str(value))
            
            results[i] = eval(formula_with_values)  # Warning: eval is unsafe for production
        
        # Calculate statistics
        return {
            "mean": float(np.mean(results)),
            "median": float(np.median(results)),
            "std_dev": float(np.std(results)),
            "min": float(np.min(results)),
            "max": float(np.max(results)),
            "percentiles": {
                "10": float(np.percentile(results, 10)),
                "25": float(np.percentile(results, 25)),
                "50": float(np.percentile(results, 50)),
                "75": float(np.percentile(results, 75)),
                "90": float(np.percentile(results, 90)),
            },
            "histogram": self._create_histogram(results),
            "iterations": self.iterations
        }
    
    def _create_histogram(self, results: np.ndarray) -> Dict:
        """Create a histogram of simulation results."""
        hist, bin_edges = np.histogram(results, bins=30)
        return {
            "counts": hist.tolist(),
            "bin_edges": bin_edges.tolist()
        }
```

### Simulation Service

Create a simulation service in `simulation/service.py`:

```python
from typing import Dict, List, Tuple
from .engine import MonteCarloSimulation
from .schemas import SimulationRequest, SimulationResponse

async def run_monte_carlo_simulation(request: SimulationRequest) -> SimulationResponse:
    """Run a Monte Carlo simulation based on the request parameters."""
    # Create simulation engine
    simulation = MonteCarloSimulation(iterations=request.iterations)
    
    # Prepare variables with triangular distributions
    variables = {
        var.name: (var.min_value, var.most_likely, var.max_value)
        for var in request.variables
    }
    
    # Run simulation
    results = await simulation.run_simulation(variables, request.formula)
    
    return SimulationResponse(
        simulation_id=request.simulation_id,
        results=results
    )
```

### Simulation Router

Create a simulation router in `simulation/router.py`:

```python
from fastapi import APIRouter, Depends, HTTPException
from .schemas import SimulationRequest, SimulationResponse
from .service import run_monte_carlo_simulation

router = APIRouter()

@router.post("/run", response_model=SimulationResponse)
async def run_simulation(request: SimulationRequest):
    """Run a Monte Carlo simulation with the provided parameters."""
    try:
        response = await run_monte_carlo_simulation(request)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Simulation failed: {str(e)}")
```

## GPU Integration

### GPU Resource Management

Create a GPU resource manager in `gpu/manager.py`:

```python
import pynvml
import asyncio
from typing import Optional

class GPUManager:
    def __init__(self, memory_fraction: float = 0.8):
        """
        Initialize the GPU manager.
        
        Args:
            memory_fraction: Fraction of GPU memory to use (0.0 to 1.0)
        """
        self.memory_fraction = memory_fraction
        self.initialized = False
        self.semaphore = None
        self.max_concurrent_tasks = 1
    
    async def initialize(self):
        """Initialize NVIDIA Management Library and set up resources."""
        if self.initialized:
            return
        
        try:
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            
            if device_count == 0:
                print("No GPU devices found. Running in CPU-only mode.")
                self.max_concurrent_tasks = 1
            else:
                # Use the first GPU for simplicity
                # In a multi-GPU setup, you would implement more sophisticated allocation
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                
                # Calculate available memory in MB
                total_memory_mb = mem_info.total / (1024 * 1024)
                available_memory_mb = total_memory_mb * self.memory_fraction
                
                # Estimate memory required per simulation task (adjust based on your workload)
                memory_per_task_mb = 2048  # 2GB per task
                
                # Calculate max concurrent tasks
                self.max_concurrent_tasks = max(1, int(available_memory_mb / memory_per_task_mb))
                
                print(f"GPU initialized with {self.max_concurrent_tasks} concurrent task slots")
            
            # Create semaphore to limit concurrent GPU tasks
            self.semaphore = asyncio.Semaphore(self.max_concurrent_tasks)
            self.initialized = True
            
        except Exception as e:
            print(f"Failed to initialize GPU: {str(e)}. Running in CPU-only mode.")
            self.max_concurrent_tasks = 1
            self.semaphore = asyncio.Semaphore(self.max_concurrent_tasks)
            self.initialized = True
    
    async def run_task(self, task_func, *args, **kwargs):
        """
        Run a task with GPU resource management.
        
        Args:
            task_func: Async function to run
            *args, **kwargs: Arguments to pass to the task function
        
        Returns:
            Result of the task function
        """
        if not self.initialized:
            await self.initialize()
        
        async with self.semaphore:
            return await task_func(*args, **kwargs)
    
    def shutdown(self):
        """Clean up GPU resources."""
        if self.initialized:
            try:
                pynvml.nvmlShutdown()
            except:
                pass
            self.initialized = False

# Create a singleton instance
gpu_manager = GPUManager()
```

### GPU-Accelerated Simulation

Modify the simulation engine to use GPU acceleration:

```python
import numpy as np
import cupy as cp  # GPU-accelerated array library
from typing import Dict, List, Tuple
import asyncio

from ..gpu.manager import gpu_manager
from config import settings

class GPUMonteCarloSimulation:
    def __init__(self, iterations: int = settings.DEFAULT_ITERATIONS):
        self.iterations = min(iterations, settings.MAX_ITERATIONS)
    
    async def run_simulation(self, variables: Dict[str, Tuple[float, float, float]], formula: str):
        """Run a Monte Carlo simulation with GPU acceleration."""
        # Use the GPU manager to run the simulation
        return await gpu_manager.run_task(
            self._run_simulation_gpu,
            variables,
            formula
        )
    
    async def _run_simulation_gpu(self, variables: Dict[str, Tuple[float, float, float]], formula: str):
        """GPU-accelerated implementation of the simulation."""
        # Generate random values for each variable using cupy
        random_values = {}
        for var_name, (min_val, mode_val, max_val) in variables.items():
            # Use cupy for GPU-accelerated random number generation
            random_values[var_name] = cp.random.triangular(
                min_val, mode_val, max_val, size=self.iterations
            )
        
        # Implement formula evaluation on GPU
        # This is a simplified example - in practice, you would need to compile
        # the formula into a GPU kernel or use a library that supports this
        
        # For demonstration, we'll transfer back to CPU for the formula evaluation
        cpu_values = {var: cp.asnumpy(values) for var, values in random_values.items()}
        
        # Evaluate the formula for each iteration (on CPU)
        results = np.zeros(self.iterations)
        for i in range(self.iterations):
            iter_values = {var: values[i] for var, values in cpu_values.items()}
            
            formula_with_values = formula
            for var, value in iter_values.items():
                formula_with_values = formula_with_values.replace(var, str(value))
            
            results[i] = eval(formula_with_values)  # Warning: eval is unsafe for production
        
        # Transfer results back to GPU for statistics calculation
        gpu_results = cp.array(results)
        
        # Calculate statistics on GPU
        return {
            "mean": float(cp.mean(gpu_results).get()),
            "median": float(cp.median(gpu_results).get()),
            "std_dev": float(cp.std(gpu_results).get()),
            "min": float(cp.min(gpu_results).get()),
            "max": float(cp.max(gpu_results).get()),
            "percentiles": {
                "10": float(cp.percentile(gpu_results, 10).get()),
                "25": float(cp.percentile(gpu_results, 25).get()),
                "50": float(cp.percentile(gpu_results, 50).get()),
                "75": float(cp.percentile(gpu_results, 75).get()),
                "90": float(cp.percentile(gpu_results, 90).get()),
            },
            "histogram": self._create_histogram(cp.asnumpy(gpu_results)),
            "iterations": self.iterations
        }
    
    def _create_histogram(self, results: np.ndarray) -> Dict:
        """Create a histogram of simulation results."""
        hist, bin_edges = np.histogram(results, bins=30)
        return {
            "counts": hist.tolist(),
            "bin_edges": bin_edges.tolist()
        }
```

## API Endpoints

### API Schema Definitions

Create Pydantic models for request/response schemas:

```python
# excel_parser/schemas.py
from pydantic import BaseModel
from typing import List, Dict, Any

class ExcelFileResponse(BaseModel):
    file_id: str
    filename: str
    sheet_names: List[str]
    columns: List[str]
    row_count: int
    formulas_count: int
    preview: List[Dict[str, Any]]

# simulation/schemas.py
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from uuid import uuid4

class VariableConfig(BaseModel):
    name: str
    min_value: float
    most_likely: float
    max_value: float

class SimulationRequest(BaseModel):
    simulation_id: str = Field(default_factory=lambda: str(uuid4()))
    file_id: Optional[str] = None
    formula: str
    variables: List[VariableConfig]
    iterations: int = 1000

class SimulationResult(BaseModel):
    mean: float
    median: float
    std_dev: float
    min: float
    max: float
    percentiles: Dict[str, float]
    histogram: Dict[str, List[float]]
    iterations: int

class SimulationResponse(BaseModel):
    simulation_id: str
    results: SimulationResult
```

### Complete API Endpoints

Implement all required API endpoints:

```python
# excel_parser/router.py
from fastapi import APIRouter, UploadFile, File, HTTPException
from .schemas import ExcelFileResponse
from .service import parse_excel_file, get_file_variables

router = APIRouter()

@router.post("/upload", response_model=ExcelFileResponse)
async def upload_excel(file: UploadFile = File(...)):
    """Upload and parse an Excel file."""
    if not file.filename.endswith((".xlsx", ".xls")):
        raise HTTPException(status_code=400, detail="File must be an Excel file")
    
    result = await parse_excel_file(file)
    return result

@router.get("/files/{file_id}")
async def get_file_info(file_id: str):
    """Get information about a previously uploaded Excel file."""
    file_info = await get_file_info(file_id)
    if not file_info:
        raise HTTPException(status_code=404, detail="File not found")
    return file_info

# simulation/router.py
from fastapi import APIRouter, HTTPException
from .schemas import SimulationRequest, SimulationResponse
from .service import run_monte_carlo_simulation

router = APIRouter()

@router.post("/run", response_model=SimulationResponse)
async def run_simulation(request: SimulationRequest):
    """Run a Monte Carlo simulation with the provided parameters."""
    try:
        response = await run_monte_carlo_simulation(request)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Simulation failed: {str(e)}")

@router.get("/{simulation_id}", response_model=SimulationResponse)
async def get_simulation_results(simulation_id: str):
    """Get the results of a previously run simulation."""
    results = await get_simulation_results(simulation_id)
    if not results:
        raise HTTPException(status_code=404, detail="Simulation not found")
    return results
```

## Testing

### Unit Testing

Create unit tests for the simulation engine:

```python
# tests/simulation/test_engine.py
import pytest
import numpy as np
from simulation.engine import MonteCarloSimulation

def test_monte_carlo_simulation():
    """Test that the Monte Carlo simulation produces expected results."""
    # Create a simulation with a fixed number of iterations
    simulation = MonteCarloSimulation(iterations=10000)
    
    # Define variables with triangular distributions
    variables = {
        "x": (10, 20, 30),  # min, mode, max
        "y": (5, 15, 25)
    }
    
    # Define a simple formula
    formula = "x + y"
    
    # Run the simulation
    results = simulation._run_simulation_sync(variables, formula)
    
    # Check that the results have the expected structure
    assert "mean" in results
    assert "median" in results
    assert "std_dev" in results
    assert "min" in results
    assert "max" in results
    assert "percentiles" in results
    assert "histogram" in results
    
    # Check that the mean is approximately as expected
    # For triangular distributions:
    # E[X] = (min + mode + max) / 3
    expected_mean_x = (10 + 20 + 30) / 3  # = 20
    expected_mean_y = (5 + 15 + 25) / 3   # = 15
    expected_mean = expected_mean_x + expected_mean_y  # = 35
    
    assert abs(results["mean"] - expected_mean) < 1.0  # Allow for some random variation
```

### Integration Testing

Create integration tests for the API endpoints:

```python
# tests/api/test_excel_api.py
import pytest
from fastapi.testclient import TestClient
from main import app
import os

client = TestClient(app)

def test_upload_excel():
    """Test uploading an Excel file."""
    # Create a test Excel file
    test_file_path = "test_data.xlsx"
    
    # Open the file in binary mode
    with open(test_file_path, "rb") as f:
        # Send a POST request to the upload endpoint
        response = client.post(
            "/api/excel/upload",
            files={"file": ("test_data.xlsx", f, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")}
        )
    
    # Check that the response is successful
    assert response.status_code == 200
    
    # Check that the response has the expected structure
    data = response.json()
    assert "file_id" in data
    assert "filename" in data
    assert "sheet_names" in data
    assert "columns" in data
    assert "row_count" in data
    assert "formulas_count" in data
    assert "preview" in data
```

## Performance Optimization

### Vectorization

Use numpy vectorization for better performance:

```python
# Instead of looping through each iteration
for i in range(self.iterations):
    result[i] = x[i] + y[i]

# Use vectorized operations
result = x + y
```

### Parallel Processing

Use joblib for parallel processing:

```python
from joblib import Parallel, delayed

def _process_chunk(chunk_variables, formula, chunk_size):
    """Process a chunk of the simulation."""
    results = np.zeros(chunk_size)
    for i in range(chunk_size):
        # Process each iteration in the chunk
        # ...
    return results

def run_parallel_simulation(variables, formula, iterations, n_jobs=-1):
    """Run a simulation in parallel using joblib."""
    # Split the iterations into chunks
    chunk_size = iterations // n_jobs
    chunks = []
    
    for i in range(n_jobs):
        start = i * chunk_size
        end = (i + 1) * chunk_size if i < n_jobs - 1 else iterations
        chunk_size = end - start
        
        # Create chunk variables
        chunk_variables = {}
        for var_name, (min_val, mode_val, max_val) in variables.items():
            chunk_variables[var_name] = np.random.triangular(
                min_val, mode_val, max_val, size=chunk_size
            )
        
        chunks.append((chunk_variables, formula, chunk_size))
    
    # Process chunks in parallel
    results = Parallel(n_jobs=n_jobs)(
        delayed(_process_chunk)(chunk_vars, formula, chunk_size)
        for chunk_vars, formula, chunk_size in chunks
    )
    
    # Combine results
    combined_results = np.concatenate(results)
    return combined_results
```

### Caching

Implement caching for frequently accessed data:

```python
from functools import lru_cache

@lru_cache(maxsize=100)
def get_file_info(file_id: str):
    """Get information about a file with caching."""
    # Implementation...
    return file_info
```

## Containerization

### Dockerfile

Create a Dockerfile for the backend:

```dockerfile
# Build stage
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Runtime stage
FROM python:3.11-slim

WORKDIR /app

# Copy installed packages from builder stage
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY . .

# Create upload directory
RUN mkdir -p uploads && chmod 777 uploads

# Create non-root user
RUN useradd -m appuser
USER appuser

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Docker Compose

Create a docker-compose.yml file for local development:

```yaml
version: '3.8'

services:
  backend:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    volumes:
      - ./uploads:/app/uploads
    environment:
      - DEBUG=true
      - CORS_ORIGINS=http://localhost:3000
      - USE_GPU=false  # Set to true if GPU is available
```

## References

1. FastAPI Documentation: https://fastapi.tiangolo.com/
2. Pandas Documentation: https://pandas.pydata.org/docs/
3. OpenPyXL Documentation: https://openpyxl.readthedocs.io/
4. NumPy Documentation: https://numpy.org/doc/
5. CuPy Documentation: https://docs.cupy.dev/
6. NVIDIA NVML Documentation: https://docs.nvidia.com/deploy/nvml-api/
7. Docker Documentation: https://docs.docker.com/
8. Kubernetes Documentation: https://kubernetes.io/docs/
