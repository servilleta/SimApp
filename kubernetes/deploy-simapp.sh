#!/bin/bash

echo "🚀 Deploying SimApp to Kubernetes Cluster"
echo "=========================================="

# Apply namespaces
echo "📁 Creating namespaces..."
kubectl apply -f kubernetes/namespaces/simapp-production.yaml

# Apply storage
echo "💾 Setting up storage..."
kubectl apply -f kubernetes/storage/postgresql-statefulset.yaml

# Wait for PostgreSQL to be ready
echo "⏳ Waiting for PostgreSQL to be ready..."
kubectl wait --for=condition=ready pod -l app=postgresql -n simapp-production --timeout=300s

# Apply Redis
echo "🔴 Deploying Redis..."
kubectl apply -f kubernetes/deployments/redis.yaml

# Wait for Redis to be ready
echo "⏳ Waiting for Redis to be ready..."
kubectl wait --for=condition=ready pod -l app=redis -n simapp-production --timeout=120s

# Apply backend
echo "🔧 Deploying Backend..."
kubectl apply -f kubernetes/deployments/backend.yaml

# Apply simulation engine
echo "🎲 Deploying Simulation Engine..."
kubectl apply -f kubernetes/deployments/simulation-engine.yaml

# Apply frontend
echo "🌐 Deploying Frontend..."
kubectl apply -f kubernetes/deployments/frontend.yaml

# Apply ingress
echo "🌍 Setting up Ingress..."
kubectl apply -f kubernetes/ingress/nginx-ingress.yaml

# Install metrics server for HPA
echo "📊 Installing metrics server..."
kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml

# Patch metrics server for development (insecure TLS)
kubectl patch deployment metrics-server -n kube-system --type='json' -p='[{"op": "add", "path": "/spec/template/spec/containers/0/args/-", "value": "--kubelet-insecure-tls"}]'

# Wait a bit for metrics server
echo "⏳ Waiting for metrics server..."
sleep 30

# Apply HPA
echo "📈 Setting up Auto-scaling..."
kubectl apply -f kubernetes/scaling/hpa-configs.yaml

echo ""
echo "✅ Deployment completed!"
echo ""
echo "📋 Checking status..."
kubectl get pods -n simapp-production
echo ""
echo "🌐 Services:"
kubectl get services -n simapp-production
echo ""
echo "📈 HPA Status:"
kubectl get hpa -n simapp-production
echo ""
echo "🎯 Access SimApp at: http://$(hostname -I | awk '{print $1}'):30090"
echo "📊 Monitor with: kubectl get pods -n simapp-production -w"
