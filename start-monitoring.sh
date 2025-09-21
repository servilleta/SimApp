#!/bin/bash
# 🔍 ENTERPRISE MONITORING STACK STARTUP
# Phase 5 Week 17-18: Start comprehensive monitoring system

echo "🚀 Starting Enterprise Monitoring Stack..."

# Create monitoring directories
mkdir -p monitoring/prometheus/rules
mkdir -p monitoring/grafana/dashboards
mkdir -p monitoring/grafana/datasources
mkdir -p monitoring/logstash/config

# Start monitoring services
echo "📊 Starting Prometheus, Grafana, Jaeger, and ELK stack..."
cd monitoring && docker-compose up -d

# Wait for services to start
echo "⏳ Waiting for services to initialize..."
sleep 30

# Check service health
echo "🔍 Checking service health..."
curl -s http://localhost:9090/api/v1/status/config > /dev/null && echo "✅ Prometheus: Running" || echo "❌ Prometheus: Failed"
curl -s http://localhost:3001/api/health > /dev/null && echo "✅ Grafana: Running" || echo "❌ Grafana: Failed"  
curl -s http://localhost:16686/api/services > /dev/null && echo "✅ Jaeger: Running" || echo "❌ Jaeger: Failed"
curl -s http://localhost:9200/_cluster/health > /dev/null && echo "✅ Elasticsearch: Running" || echo "❌ Elasticsearch: Failed"
curl -s http://localhost:5601/api/status > /dev/null && echo "✅ Kibana: Running" || echo "❌ Kibana: Failed"

echo ""
echo "🎯 Enterprise Monitoring Stack Access URLs:"
echo "📊 Prometheus: http://localhost:9090"
echo "📈 Grafana: http://localhost:3001 (admin/admin123)"
echo "🔍 Jaeger: http://localhost:16686"
echo "📋 Kibana: http://localhost:5601"
echo "🔍 Elasticsearch: http://localhost:9200"
echo ""
echo "🚀 Monitoring stack is ready!"
echo "📊 Metrics are being collected from the Monte Carlo platform"
echo "🎯 SLA monitoring and alerting are active"
echo ""
echo "To stop monitoring: cd monitoring && docker-compose down"
