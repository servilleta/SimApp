# 📋 PRODUCTION OPERATIONS RUNBOOK
## Monte Carlo Platform - Enterprise Operations Guide

---

## **🎯 OVERVIEW**

This runbook provides standardized procedures for operating the Monte Carlo Platform in production, including incident response, monitoring, maintenance, and escalation procedures.

**Scope**: Multi-instance production environment with enterprise customers
**Audience**: DevOps, SRE, Support, and Engineering teams
**SLA Targets**: 99.9% uptime, <2s response time, <4h critical issue resolution

---

## **🚨 INCIDENT RESPONSE PROCEDURES**

### **Incident Classification**

| **Severity** | **Definition** | **Response Time** | **Examples** |
|--------------|----------------|-------------------|--------------|
| **P0 - Critical** | Service completely down | **15 minutes** | API gateway down, database outage |
| **P1 - High** | Major feature impacted | **1 hour** | Simulation engine failure, auth issues |
| **P2 - Medium** | Minor feature impacted | **4 hours** | Slow responses, non-critical features |
| **P3 - Low** | Cosmetic or minor issues | **24 hours** | UI glitches, documentation errors |

### **🔥 P0 - CRITICAL INCIDENT RESPONSE**

#### **Immediate Actions (0-15 minutes)**
```bash
# 1. Acknowledge and assess
□ Update status page: https://status.montecarloanalytics.com
□ Create incident channel: #incident-YYYY-MM-DD-HH-MM
□ Page on-call engineer via PagerDuty
□ Begin incident log documentation

# 2. Initial diagnosis
□ Check system health dashboard: http://localhost:3001/dashboard/overview
□ Verify infrastructure status: kubectl get pods --all-namespaces
□ Check external dependencies: Redis, PostgreSQL, monitoring stack
□ Review recent deployments and changes
```

#### **Detailed Response Actions (15-60 minutes)**
```bash
# 3. Technical investigation
□ Analyze logs: ELK stack http://localhost:5601
□ Check metrics: Prometheus http://localhost:9090
□ Review traces: Jaeger http://localhost:16686
□ Validate database connectivity and performance

# 4. Communication
□ Update status page with initial findings
□ Notify customer success team for enterprise customers
□ Brief executive team for extended outages (>30 minutes)
□ Post updates every 15 minutes during active incident
```

#### **Resolution and Recovery (60+ minutes)**
```bash
# 5. Resolution implementation
□ Apply fixes (with change control approval for production)
□ Verify fix in staging environment first (if possible)
□ Monitor system stability post-fix
□ Validate all core functionality restored

# 6. Post-incident activities
□ Update status page with "Monitoring" status
□ Schedule post-mortem within 24 hours
□ Document lessons learned
□ Implement preventive measures
```

### **🚨 ESCALATION MATRIX**

```bash
# Primary On-Call (24/7)
├── L1 Support Engineer (0-15 min response)
├── L2 Platform Engineer (15-60 min response)
├── L3 Senior Engineer (1-4 hour response)
└── Engineering Manager (4+ hour or customer escalation)

# Executive Escalation (for major outages)
├── CTO (>1 hour P0 incident)
├── CEO (>4 hour P0 incident or major customer impact)
└── Legal/PR (security incidents or regulatory issues)

# Customer Communication
├── Customer Success (all P0/P1 incidents affecting enterprise customers)
├── Sales Engineering (prospects during trial periods)
└── Executive Sponsor (for strategic accounts)
```

---

## **📊 MONITORING & ALERTING**

### **Critical System Metrics**

#### **Application Health Monitoring**
```bash
# API Response Time Alerts
Alert: API response time > 2 seconds (95th percentile)
Action: Page on-call engineer
Dashboard: Grafana - API Performance

Alert: API error rate > 1%
Action: Create incident ticket
Dashboard: Grafana - Error Rates

Alert: Simulation queue backlog > 100 jobs
Action: Scale simulation workers
Dashboard: Grafana - Queue Metrics
```

#### **Infrastructure Monitoring**
```bash
# Kubernetes Cluster Health
Alert: Pod restart count > 5 in 1 hour
Action: Investigate and potentially scale
Dashboard: Grafana - Kubernetes Overview

Alert: Memory usage > 80%
Action: Scale horizontally or investigate memory leaks
Dashboard: Grafana - Resource Utilization

Alert: Disk usage > 85%
Action: Immediate storage expansion
Dashboard: Grafana - Storage Metrics
```

#### **Database Monitoring**
```bash
# PostgreSQL Performance
Alert: Database connection pool > 80% utilized
Action: Scale connection pool or investigate long-running queries
Query: SELECT * FROM pg_stat_activity WHERE state = 'active';

Alert: Slow query detected (>5 seconds)
Action: Investigate query performance
Tool: pg_stat_statements analysis

Alert: Database replication lag > 30 seconds
Action: Check replication health immediately
Dashboard: Grafana - Database Replication
```

### **Business Metrics Monitoring**
```bash
# Customer Experience Metrics
Alert: Simulation failure rate > 5%
Action: Investigate Ultra engine and GPU allocation
Dashboard: Grafana - Simulation Success Rate

Alert: User signup conversion < 15%
Action: Notify product team for UX investigation
Dashboard: Grafana - Conversion Metrics

Alert: Enterprise customer billing processing failure
Action: Immediate investigation and customer notification
Dashboard: Stripe Dashboard + Internal Analytics
```

---

## **🔧 MAINTENANCE PROCEDURES**

### **Weekly Maintenance Tasks**

#### **Security Maintenance**
```bash
# Every Monday 6 AM UTC
□ Review security logs for anomalies
□ Update dependency vulnerability reports
□ Rotate non-critical API keys
□ Review user access permissions

# Security checklist
audit_security_logs.py --last-week
check_dependency_vulnerabilities.sh
rotate_api_keys.py --non-critical
review_user_permissions.py --inactive-users
```

#### **Performance Optimization**
```bash
# Every Wednesday 6 AM UTC
□ Analyze slow query reports
□ Review Redis memory usage and eviction policies
□ Optimize database indexes based on query patterns
□ Clear unnecessary log files and temporary data

# Performance checklist
analyze_slow_queries.py --last-week
optimize_redis_memory.py
update_database_indexes.py --analyze-only
cleanup_logs.py --older-than-30-days
```

### **Monthly Maintenance Tasks**

#### **Disaster Recovery Testing**
```bash
# First Saturday of each month
□ Test database backup restoration
□ Validate cross-region failover procedures
□ Test incident response communication channels
□ Review and update runbook procedures

# DR testing checklist
test_database_restore.py --to-staging
test_regional_failover.py --dry-run
test_incident_channels.py
update_runbook_docs.py
```

#### **Capacity Planning Review**
```bash
# Monthly capacity assessment
□ Analyze growth trends and resource utilization
□ Project capacity needs for next 3 months
□ Review auto-scaling configuration
□ Plan infrastructure scaling activities

# Capacity planning tools
capacity_analysis.py --last-month
generate_growth_projections.py
review_autoscaling_metrics.py
plan_infrastructure_scaling.py
```

---

## **💾 BACKUP & DISASTER RECOVERY**

### **Backup Procedures**

#### **Database Backups**
```bash
# Automated daily backups (2 AM UTC)
pg_dump -h $DB_HOST -U $DB_USER -d monte_carlo_prod > backup_$(date +%Y%m%d).sql
aws s3 cp backup_$(date +%Y%m%d).sql s3://mc-backups/database/

# Backup retention policy
├── Daily backups: Retained for 30 days
├── Weekly backups: Retained for 12 weeks  
├── Monthly backups: Retained for 12 months
└── Yearly backups: Retained for 7 years

# Backup validation (weekly)
test_restore_backup.py --latest-backup --to-staging
validate_backup_integrity.py --all-recent-backups
```

#### **File Storage Backups**
```bash
# User file backups (continuous replication)
aws s3 sync s3://mc-user-files s3://mc-user-files-backup --delete

# Configuration backups (daily)
kubectl get all --all-namespaces -o yaml > k8s_config_$(date +%Y%m%d).yaml
tar -czf config_backup_$(date +%Y%m%d).tar.gz k8s_config_$(date +%Y%m%d).yaml
```

### **Disaster Recovery Procedures**

#### **RTO/RPO Targets**
- **Recovery Time Objective (RTO)**: 4 hours for complete service restoration
- **Recovery Point Objective (RPO)**: 1 hour maximum data loss
- **Service Degradation Threshold**: 2 hours for partial service restoration

#### **Regional Failover Procedure**
```bash
# 1. Assess regional outage scope
□ Verify primary region unavailability
□ Check secondary region readiness
□ Notify customers of potential service disruption

# 2. Execute failover
□ Update DNS records to point to secondary region
□ Restore latest database backup to secondary region
□ Start application services in secondary region
□ Validate service functionality

# 3. Monitor and communicate
□ Monitor service health in secondary region
□ Update status page with recovery progress
□ Notify customers when service is fully restored
□ Begin planning for primary region recovery
```

---

## **🎯 CUSTOMER SUPPORT PROCEDURES**

### **Support Tier Definitions**

| **Tier** | **Response Time** | **Scope** | **Escalation** |
|----------|------------------|-----------|----------------|
| **T1 - Basic** | 2 hours | Account issues, basic usage | T2 after 4 hours |
| **T2 - Technical** | 1 hour | API issues, simulation problems | T3 after 2 hours |
| **T3 - Expert** | 30 minutes | Complex technical issues | Engineering after 1 hour |

### **Common Issue Resolution**

#### **Authentication Problems**
```bash
# User cannot log in
□ Check user account status: SELECT * FROM users WHERE email = ?
□ Verify OAuth provider connectivity
□ Check for account lockout: SELECT failed_login_attempts FROM user_auth WHERE user_id = ?
□ Reset user password if needed: reset_user_password.py --user-id X

# SSO integration issues
□ Validate SAML/OAuth configuration
□ Check enterprise SSO provider status
□ Verify user exists in enterprise directory
□ Test SSO configuration: test_sso_config.py --org-id X
```

#### **Simulation Performance Issues**
```bash
# Slow simulation execution
□ Check GPU utilization: nvidia-smi
□ Verify simulation queue status: redis-cli LLEN simulation_queue
□ Check for resource constraints: kubectl top pods
□ Review simulation complexity: analyze_simulation.py --sim-id X

# Failed simulations
□ Check simulation logs: kubectl logs simulation-worker-X
□ Verify input file integrity: validate_excel_file.py --file-id X
□ Check Monte Carlo parameters: validate_simulation_config.py --config X
□ Test with simplified parameters: test_simulation.py --basic-config
```

#### **Billing and Subscription Issues**
```bash
# Payment failures
□ Check Stripe payment status: stripe payments list --customer X
□ Verify subscription status: SELECT * FROM subscriptions WHERE user_id = ?
□ Check for expired payment methods
□ Process manual payment if needed: process_manual_payment.py --user-id X

# Usage limit exceeded
□ Check current usage: get_user_usage.py --user-id X --month current
□ Verify subscription tier limits: get_subscription_limits.py --user-id X
□ Offer upgrade or temporary limit increase
□ Process subscription upgrade: upgrade_subscription.py --user-id X --tier pro
```

---

## **🔄 DEPLOYMENT PROCEDURES**

### **Production Deployment Checklist**

#### **Pre-Deployment Validation**
```bash
# Code quality checks
□ All tests passing: pytest backend/tests/ --cov=80
□ Security scan completed: bandit -r backend/
□ Dependency vulnerability check: safety check
□ Load testing passed: k6 run --vus 1000 --duration 5m load-test.js

# Staging validation
□ Full functionality test in staging
□ Performance benchmarks meet SLA requirements
□ Database migration tested (if applicable)
□ Rollback procedure verified
```

#### **Deployment Execution**
```bash
# Blue-green deployment process
□ Deploy to green environment
□ Run smoke tests on green environment
□ Switch traffic gradually (10%, 50%, 100%)
□ Monitor error rates and performance metrics
□ Keep blue environment ready for immediate rollback

# Database migrations (if required)
□ Create database backup before migration
□ Execute migration in maintenance window
□ Validate data integrity post-migration
□ Update application configuration
```

### **Emergency Rollback Procedure**
```bash
# Immediate rollback triggers
- Error rate > 5% for more than 5 minutes
- Response time > 5 seconds for more than 2 minutes  
- Critical feature completely non-functional
- Security vulnerability discovered in new release

# Rollback execution
□ Switch traffic back to previous version
□ Rollback database changes (if safe)
□ Notify engineering team of rollback
□ Schedule immediate post-mortem
□ Communicate with affected customers
```

---

## **📞 CONTACT INFORMATION**

### **Emergency Contacts**
```bash
# Primary On-Call (24/7)
├── Phone: +1-XXX-XXX-XXXX
├── Email: oncall@montecarloanalytics.com
├── Slack: #on-call-primary
└── PagerDuty: https://montecarloanalytics.pagerduty.com

# Escalation Contacts
├── Engineering Manager: +1-XXX-XXX-XXXX
├── CTO: +1-XXX-XXX-XXXX
├── CEO: +1-XXX-XXX-XXXX
└── Customer Success: +1-XXX-XXX-XXXX

# External Vendors
├── Cloud Provider Support: [Cloud provider emergency number]
├── Security Vendor: [Security vendor 24/7 number]
├── Database Vendor: [Database vendor support]
└── Monitoring Vendor: [Monitoring vendor support]
```

### **Internal Communication Channels**
```bash
# Slack Channels
├── #incidents - Active incident coordination
├── #alerts - Automated monitoring alerts
├── #deployments - Deployment notifications
├── #customer-escalations - Customer issue escalations
└── #executive-notifications - Executive updates

# Documentation Links
├── System Architecture: http://docs.internal/architecture
├── API Documentation: http://localhost:8080/docs
├── Monitoring Dashboards: http://localhost:3001
└── Status Page: https://status.montecarloanalytics.com
```

---

## **📚 APPENDIX: USEFUL COMMANDS**

### **System Health Checks**
```bash
# Quick system overview
kubectl get pods --all-namespaces | grep -v Running
docker ps --filter "status=exited"
curl -s http://localhost:8000/health | jq .

# Database health
psql -h $DB_HOST -U $DB_USER -c "SELECT version();"
redis-cli ping
curl -s http://localhost:9200/_cluster/health | jq .

# Performance metrics
kubectl top nodes
kubectl top pods --all-namespaces
iostat -x 1 3
```

### **Log Analysis**
```bash
# Application logs
kubectl logs -f deployment/monte-carlo-backend --tail=100
journalctl -u docker -f
tail -f /var/log/nginx/access.log

# Error investigation
grep -i error /var/log/monte-carlo/*.log | tail -20
elasticsearch_query.py --query "level:ERROR AND timestamp:>now-1h"
kibana_dashboard.py --dashboard "Error Analysis"
```

---

**🎯 REMEMBER**: This runbook is a living document. Update procedures based on lessons learned from incidents and operational experience. Review and update monthly during team meetings.**

---

*Last Updated: September 18, 2025*
*Next Review: October 18, 2025*
*Owner: DevOps Team*
