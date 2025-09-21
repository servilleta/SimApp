# Monte Carlo Simulation Web Platform
# Deployment Guide for Paperspace

## Table of Contents
1. [Introduction](#introduction)
2. [Prerequisites](#prerequisites)
3. [Environment Setup](#environment-setup)
4. [Containerization](#containerization)
5. [Deployment Process](#deployment-process)
6. [GPU Configuration](#gpu-configuration)
7. [Scaling and Performance](#scaling-and-performance)
8. [Security Considerations](#security-considerations)
9. [Monitoring and Maintenance](#monitoring-and-maintenance)
10. [Troubleshooting](#troubleshooting)
11. [References](#references)

## Introduction

This document provides comprehensive guidelines for deploying the Monte Carlo Simulation Web Platform on Paperspace. The platform allows users to upload Excel files, configure variables with triangular probability distributions, run simulations, and visualize results. This deployment guide focuses on leveraging Paperspace's GPU capabilities for optimal performance.

### System Architecture Overview

The Monte Carlo Simulation platform consists of:
- **Frontend**: React-based UI with Plotly.js for visualization
- **Backend**: FastAPI application for handling requests and running simulations
- **Simulation Engine**: Python-based engine using numpy for calculations
- **Container Orchestration**: Docker and Kubernetes for deployment

## Prerequisites

Before beginning deployment, ensure you have:

- A Paperspace account with team privileges
- Docker and Docker Compose installed locally
- Python 3.11+ installed locally
- Node.js installed locally
- Git for version control
- Access to the project repository

## Environment Setup

### 1. Paperspace Account Configuration

1. Create a Paperspace account at [paperspace.com](https://www.paperspace.com/)
2. Request team account privileges by contacting support@paperspace.com
3. Create a private network in your preferred region
   - Note: All resources must be in the same region as your private network

### 2. Setting Up Master Node (CPU)

1. Create a Paperspace C3 instance with Ubuntu 22.04
   ```
   Machine Type: C3
   OS: Ubuntu 22.04
   Network: Your private network
   ```

2. Once provisioned, connect to the machine and disable the firewall for testing:
   ```bash
   sudo ufw disable
   ```
   Note: Re-enable the firewall in production with proper rules

3. Install Kubernetes on the master node:
   ```bash
   wget https://raw.githubusercontent.com/Paperspace/GPU-Kubernetes-Guide/master/scripts/init-master.sh
   chmod +x init-master.sh
   sudo ./init-master.sh
   ```

4. Save the join command output for worker nodes:
   ```
   kubeadm join <IP>:6443 --token <TOKEN> --discovery-token-ca-cert-hash <HASH>
   ```

### 3. Setting Up Worker Node (GPU)

1. Create a Paperspace GPU instance (recommended: A4000 or A5000 for simulation workloads)
   ```
   Machine Type: A4000 or A5000
   OS: Ubuntu 22.04
   Network: Your private network
   ```

2. Install NVIDIA drivers and CUDA:
   ```bash
   wget https://raw.githubusercontent.com/Paperspace/GPU-Kubernetes-Guide/master/scripts/init-worker.sh
   chmod +x init-worker.sh
   sudo ./init-worker.sh <IP:PORT> <TOKEN> <CA_CERT_HASH>
   ```

3. Reboot the worker node to complete NVIDIA driver installation:
   ```bash
   sudo reboot
   ```

4. Verify the node has joined the cluster (from master node):
   ```bash
   kubectl get nodes
   ```

## Containerization

### 1. Dockerfile for Backend

Create a multi-stage Dockerfile for the backend:

```dockerfile
# Build stage
FROM python:3.11-slim AS builder

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Runtime stage
FROM python:3.11-slim

WORKDIR /app

COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

COPY ./backend /app/backend
COPY ./main.py /app/

# Run as non-root user for security
RUN useradd -m appuser
USER appuser

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 2. Docker Compose Configuration

Create a `docker-compose.yml` file for local testing:

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
      - ./data:/app/data
    environment:
      - ENVIRONMENT=development
      - LOG_LEVEL=debug

  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile
    ports:
      - "3000:3000"
    depends_on:
      - backend
    environment:
      - REACT_APP_API_URL=http://backend:8000
```

### 3. Building and Testing Locally

1. Build the Docker images:
   ```bash
   docker-compose build
   ```

2. Run the containers locally:
   ```bash
   docker-compose up
   ```

3. Verify the application is working by accessing:
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:8000/docs

## Deployment Process

### 1. Kubernetes Deployment

Create Kubernetes deployment files for the application:

1. Backend Deployment (`backend-deployment.yaml`):
   ```yaml
   apiVersion: apps/v1
   kind: Deployment
   metadata:
     name: monte-carlo-backend
   spec:
     replicas: 2
     selector:
       matchLabels:
         app: monte-carlo-backend
     template:
       metadata:
         labels:
           app: monte-carlo-backend
       spec:
         containers:
         - name: backend
           image: your-registry/monte-carlo-backend:latest
           ports:
           - containerPort: 8000
           resources:
             limits:
               nvidia.com/gpu: 1
             requests:
               memory: "1Gi"
               cpu: "500m"
           env:
           - name: ENVIRONMENT
             value: "production"
           - name: LOG_LEVEL
             value: "info"
   ```

2. Frontend Deployment (`frontend-deployment.yaml`):
   ```yaml
   apiVersion: apps/v1
   kind: Deployment
   metadata:
     name: monte-carlo-frontend
   spec:
     replicas: 2
     selector:
       matchLabels:
         app: monte-carlo-frontend
     template:
       metadata:
         labels:
           app: monte-carlo-frontend
       spec:
         containers:
         - name: frontend
           image: your-registry/monte-carlo-frontend:latest
           ports:
           - containerPort: 80
           resources:
             requests:
               memory: "512Mi"
               cpu: "200m"
   ```

3. Service Configuration (`services.yaml`):
   ```yaml
   apiVersion: v1
   kind: Service
   metadata:
     name: monte-carlo-backend-service
   spec:
     selector:
       app: monte-carlo-backend
     ports:
     - port: 8000
       targetPort: 8000
     type: ClusterIP
   ---
   apiVersion: v1
   kind: Service
   metadata:
     name: monte-carlo-frontend-service
   spec:
     selector:
       app: monte-carlo-frontend
     ports:
     - port: 80
       targetPort: 80
     type: NodePort
   ```

4. Ingress Configuration (`ingress.yaml`):
   ```yaml
   apiVersion: networking.k8s.io/v1
   kind: Ingress
   metadata:
     name: monte-carlo-ingress
     annotations:
       kubernetes.io/ingress.class: nginx
       nginx.ingress.kubernetes.io/ssl-redirect: "true"
   spec:
     rules:
     - host: monte-carlo.example.com
       http:
         paths:
         - path: /api
           pathType: Prefix
           backend:
             service:
               name: monte-carlo-backend-service
               port:
                 number: 8000
         - path: /
           pathType: Prefix
           backend:
             service:
               name: monte-carlo-frontend-service
               port:
                 number: 80
   ```

### 2. Deploying to Kubernetes

1. Apply the Kubernetes configurations:
   ```bash
   kubectl apply -f backend-deployment.yaml
   kubectl apply -f frontend-deployment.yaml
   kubectl apply -f services.yaml
   kubectl apply -f ingress.yaml
   ```

2. Verify the deployments:
   ```bash
   kubectl get deployments
   kubectl get pods
   kubectl get services
   kubectl get ingress
   ```

### 3. Container Registry Setup

1. Set up a container registry on Paperspace:
   - Navigate to team settings
   - Go to the Containers tab
   - Add a new container registry

2. Push Docker images to the registry:
   ```bash
   docker tag monte-carlo-backend:latest your-registry/monte-carlo-backend:latest
   docker tag monte-carlo-frontend:latest your-registry/monte-carlo-frontend:latest
   docker push your-registry/monte-carlo-backend:latest
   docker push your-registry/monte-carlo-frontend:latest
   ```

## GPU Configuration

### 1. NVIDIA GPU Operator

Install the NVIDIA GPU Operator for Kubernetes:

```bash
helm repo add nvidia https://helm.ngc.nvidia.com/nvidia
helm repo update
helm install --wait --generate-name \
     -n gpu-operator --create-namespace \
     nvidia/gpu-operator
```

### 2. GPU Resource Allocation

Configure GPU resource allocation in your deployment:

```yaml
resources:
  limits:
    nvidia.com/gpu: 1  # Allocate 1 GPU per pod
  requests:
    memory: "2Gi"
    cpu: "1000m"
```

### 3. VRAM Management

For optimal performance with Monte Carlo simulations:

1. Monitor VRAM usage with NVIDIA's `pynvml` library
2. Implement dynamic batch sizing based on available VRAM
3. Use a semaphore to control concurrent GPU operations:

```python
import asyncio
import pynvml

# Initialize NVIDIA Management Library
pynvml.nvmlInit()
gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
mem_info = pynvml.nvmlDeviceGetMemoryInfo(gpu_handle)

# Calculate available VRAM in MB
available_vram_mb = mem_info.free // (1024 * 1024)
vram_buffer_mb = 1024  # 1GB buffer
usable_vram_mb = max(available_vram_mb - vram_buffer_mb, 0)

# Calculate max concurrent simulations
vram_per_simulation_mb = 2048  # Adjust based on your model
max_concurrent = max(usable_vram_mb // vram_per_simulation_mb, 1)

# Create semaphore to limit concurrent GPU operations
semaphore = asyncio.Semaphore(max_concurrent)

async def run_simulation():
    async with semaphore:
        # Run GPU-intensive simulation here
        pass
```

## Scaling and Performance

### 1. Horizontal Pod Autoscaling

Configure Horizontal Pod Autoscaling (HPA) for your backend:

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: monte-carlo-backend-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: monte-carlo-backend
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

### 2. Vertical Pod Autoscaling

Consider implementing Vertical Pod Autoscaling (VPA) for resource optimization:

```bash
kubectl apply -f https://raw.githubusercontent.com/kubernetes/autoscaler/master/vertical-pod-autoscaler/deploy/vpa-v0.8.0.yaml
```

### 3. Performance Monitoring

Set up monitoring for your GPU workloads:

1. Install Prometheus and Grafana:
   ```bash
   helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
   helm repo update
   helm install prometheus prometheus-community/kube-prometheus-stack
   ```

2. Configure NVIDIA DCGM exporter:
   ```bash
   helm install --wait --generate-name \
        -n gpu-operator --create-namespace \
        nvidia/dcgm-exporter
   ```

## Security Considerations

### 1. Network Security

1. Enable and configure the firewall:
   ```bash
   sudo ufw allow 22/tcp
   sudo ufw allow 6443/tcp
   sudo ufw allow 80/tcp
   sudo ufw allow 443/tcp
   sudo ufw enable
   ```

2. Implement network policies:
   ```yaml
   apiVersion: networking.k8s.io/v1
   kind: NetworkPolicy
   metadata:
     name: monte-carlo-network-policy
   spec:
     podSelector:
       matchLabels:
         app: monte-carlo-backend
     policyTypes:
     - Ingress
     - Egress
     ingress:
     - from:
       - podSelector:
           matchLabels:
             app: monte-carlo-frontend
       ports:
       - protocol: TCP
         port: 8000
     egress:
     - to:
       - podSelector:
           matchLabels:
             app: monte-carlo-database
       ports:
       - protocol: TCP
         port: 5432
   ```

### 2. Data Security

1. Implement file upload limits:
   ```python
   @app.post("/upload/")
   async def upload_file(file: UploadFile = File(...)):
       # Check file size
       content = await file.read(1024 * 1024 * 10)  # 10MB limit
       if len(content) == 1024 * 1024 * 10:
           return {"error": "File too large"}
       
       # Process file
       # ...
   ```

2. Configure secure storage for uploaded files:
   ```yaml
   apiVersion: v1
   kind: PersistentVolumeClaim
   metadata:
     name: monte-carlo-storage
   spec:
     accessModes:
       - ReadWriteOnce
     resources:
       requests:
         storage: 10Gi
   ```

### 3. Authentication

Implement JWT authentication for API endpoints:

```python
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

@app.post("/simulate/")
async def run_simulation(data: SimulationInput, token: str = Depends(oauth2_scheme)):
    # Verify token
    if not verify_token(token):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Run simulation
    # ...
```

## Monitoring and Maintenance

### 1. Logging

Configure centralized logging with Elasticsearch, Fluentd, and Kibana (EFK stack):

```bash
helm repo add elastic https://helm.elastic.co
helm repo update
helm install elasticsearch elastic/elasticsearch
helm install kibana elastic/kibana
kubectl apply -f https://raw.githubusercontent.com/fluent/fluentd-kubernetes-daemonset/master/fluentd-daemonset-elasticsearch.yaml
```

### 2. Alerting

Set up alerting with Prometheus Alertmanager:

```yaml
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: monte-carlo-alerts
  namespace: monitoring
spec:
  groups:
  - name: monte-carlo
    rules:
    - alert: HighGPUUsage
      expr: nvidia_gpu_utilization > 90
      for: 10m
      labels:
        severity: warning
      annotations:
        summary: "High GPU usage detected"
        description: "GPU usage is above 90% for more than 10 minutes"
```

### 3. Backup Strategy

Implement a backup strategy for your application:

1. Database backups:
   ```bash
   kubectl create cronjob db-backup --image=postgres:13 --schedule="0 1 * * *" -- pg_dump -h db-service -U postgres -d monte_carlo > /backup/monte_carlo_$(date +%Y%m%d).sql
   ```

2. Configuration backups:
   ```bash
   kubectl create cronjob config-backup --image=bitnami/kubectl --schedule="0 2 * * *" -- kubectl get all -o yaml > /backup/k8s_config_$(date +%Y%m%d).yaml
   ```

## Troubleshooting

### Common Issues and Solutions

1. **GPU Not Detected**
   - Check NVIDIA driver installation: `nvidia-smi`
   - Verify GPU operator status: `kubectl get pods -n gpu-operator`
   - Solution: Reinstall NVIDIA drivers or GPU operator

2. **Container Crashes**
   - Check logs: `kubectl logs <pod-name>`
   - Check resource limits: `kubectl describe pod <pod-name>`
   - Solution: Adjust resource limits or fix application errors

3. **Performance Issues**
   - Monitor GPU usage: `kubectl exec -it <pod-name> -- nvidia-smi dmon`
   - Check network performance: `kubectl exec -it <pod-name> -- iperf3 -c <service-name>`
   - Solution: Optimize code, adjust resource allocation, or scale horizontally

### Support Resources

- Paperspace Support: support@paperspace.com
- Kubernetes Documentation: https://kubernetes.io/docs/
- NVIDIA GPU Cloud Documentation: https://docs.nvidia.com/ngc/

## References

1. Paperspace Documentation: https://docs.paperspace.com/
2. Kubernetes GPU Guide: https://github.com/Paperspace/GPU-Kubernetes-Guide
3. NVIDIA GPU Operator: https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/overview.html
4. FastAPI Deployment: https://fastapi.tiangolo.com/deployment/
5. Docker Documentation: https://docs.docker.com/
6. Kubernetes Documentation: https://kubernetes.io/docs/
