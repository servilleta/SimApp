# 🚀 Paperspace Operations Guide
## Cost-Optimized Multi-Server Management

### 📊 **YOUR CURRENT SETUP**

#### **Server 1 (Primary) - ALWAYS ON**
- **ID**: `psotdtcda5ap`
- **IP**: `64.71.146.187`
- **Specs**: Quadro P4000, 8 CPUs, 32GB RAM
- **Cost**: ~$0.45/hour ($324/month)
- **Purpose**: Main SimApp, web interface, database

#### **Server 2 (On-Demand) - AUTO START/STOP**
- **ID**: `pso1zne8qfxx`
- **IP**: `72.52.107.230`
- **Specs**: **Ampere A4000, 8 CPUs, 48GB RAM** 🚀
- **Cost**: ~$0.76/hour ($547/month if always on)
- **Purpose**: Heavy simulations, GPU acceleration

---

## 💰 **COST OPTIMIZATION STRATEGY**

### **Current Cost (Both Always On)**
```
Server 1: $324/month (always on)
Server 2: $547/month (if always on)
Total:    $871/month
```

### **Optimized Cost (Smart Usage)**
```
Server 1: $324/month (always on)
Server 2: $76/month (5 hours/day average)
Total:    $400/month
SAVINGS:  $471/month (54% reduction!)
```

---

## 🎯 **OPERATIONAL WORKFLOWS**

### **🟢 NORMAL OPERATIONS (Server 1 Only)**

#### What runs on Server 1:
- ✅ SimApp web interface (`http://localhost:9090`)
- ✅ Regular Monte Carlo simulations (1-1,000 iterations)
- ✅ User dashboard and file management
- ✅ Database (PostgreSQL) and cache (Redis)
- ✅ Auth0 authentication
- ✅ Basic Excel file processing

#### When to use Server 1 only:
- Daily business simulations
- User testing and development
- File uploads and management
- Regular reports and dashboards

---

### **🔥 HIGH-PERFORMANCE OPERATIONS (Both Servers)**

#### When to start Server 2:
1. **Large simulations** (10,000+ iterations)
2. **Complex Excel models** with heavy calculations
3. **Multiple concurrent users** (10+ simultaneous)
4. **GPU-accelerated computations**
5. **Batch processing** of multiple files
6. **Performance testing** under load

#### Manual Server 2 control:
```bash
# Start Server 2 for heavy work
cd /home/paperspace/SimApp
./paperspace_api_manager.py start --machine-id pso1zne8qfxx

# Check status
./paperspace_api_manager.py status --machine-id pso1zne8qfxx

# Stop Server 2 when done (save money!)
./paperspace_api_manager.py stop --machine-id pso1zne8qfxx
```

---

## 🤖 **AUTOMATIC SCALING**

### **Intelligent Monitoring System**

#### Run automatic scaling:
```bash
# Start intelligent monitoring (runs continuously)
cd /home/paperspace/SimApp
./intelligent_scaling.py

# Or run as background service
nohup ./intelligent_scaling.py > scaling.log 2>&1 &
```

#### Scaling triggers (Server 2 AUTO-START):
- ✅ CPU usage > 80% for 2+ minutes
- ✅ Memory usage > 85%
- ✅ More than 5 active simulations
- ✅ More than 3 queued simulations
- ✅ More than 10 concurrent users
- ✅ Response time > 5 seconds

#### Scaling triggers (Server 2 AUTO-STOP):
- ✅ CPU usage < 30% for 10+ minutes
- ✅ Memory usage < 40%
- ✅ Less than 2 active simulations
- ✅ No queued simulations
- ✅ Less than 3 concurrent users
- ✅ Response time < 2 seconds

---

## 📋 **DAILY OPERATION CHECKLIST**

### **🌅 MORNING ROUTINE**
1. Check Server 1 status: `./paperspace_api_manager.py status --machine-id psotdtcda5ap`
2. Verify SimApp is accessible: `curl http://localhost:9090`
3. Check if Server 2 is needed today (heavy simulations planned?)
4. Start intelligent monitoring: `./intelligent_scaling.py &`

### **🌇 EVENING ROUTINE**
1. Check current cluster status: `./paperspace_api_manager.py cluster-status`
2. If Server 2 is running but not needed: `./paperspace_api_manager.py stop --machine-id pso1zne8qfxx`
3. Review cost for the day: `./paperspace_api_manager.py cost-report`

---

## 📊 **MONITORING & ALERTS**

### **System Status Commands**
```bash
# Quick cluster overview
./paperspace_api_manager.py cluster-status

# Detailed machine info
./paperspace_api_manager.py list

# Cost tracking
./paperspace_api_manager.py cost-report

# Real-time monitoring
./intelligent_scaling.py --once
```

### **Performance Metrics**
```bash
# System resources on Server 1
htop
df -h
free -h

# SimApp performance
curl http://localhost:9090/api/health
```

---

## 🎛️ **ADVANCED SCENARIOS**

### **💼 BUSINESS SCENARIOS**

#### **Demo/Presentation Day**
```bash
# Start Server 2 proactively
./paperspace_api_manager.py start --machine-id pso1zne8qfxx
# Ensure maximum performance for demos
```

#### **Month-End Reporting**
```bash
# Heavy batch processing expected
./paperspace_api_manager.py start --machine-id pso1zne8qfxx
# Run intelligent scaling with aggressive thresholds
```

#### **Development/Testing**
```bash
# Server 1 usually sufficient
# Only start Server 2 for performance testing
```

#### **Weekend/Off-Hours**
```bash
# Ensure Server 2 is stopped to save costs
./paperspace_api_manager.py stop --machine-id pso1zne8qfxx
```

---

## 🚨 **TROUBLESHOOTING**

### **Common Issues**

#### **Server 2 won't start**
```bash
# Check machine status
./paperspace_api_manager.py status --machine-id pso1zne8qfxx

# Check Paperspace account limits
./paperspace_api_manager.py list
```

#### **High costs**
```bash
# Check if Server 2 is running unnecessarily
./paperspace_api_manager.py cluster-status

# Stop Server 2 if not needed
./paperspace_api_manager.py stop --machine-id pso1zne8qfxx
```

#### **Performance issues**
```bash
# Check system resources
htop
df -h

# Consider starting Server 2
./paperspace_api_manager.py start --machine-id pso1zne8qfxx
```

---

## 📞 **QUICK REFERENCE**

### **Key Commands**
```bash
# Status check
./paperspace_api_manager.py cluster-status

# Start high-performance mode
./paperspace_api_manager.py start --machine-id pso1zne8qfxx

# Stop and save money
./paperspace_api_manager.py stop --machine-id pso1zne8qfxx

# Auto-scaling
./intelligent_scaling.py
```

### **Key Machine IDs**
- **Server 1 (Primary)**: `psotdtcda5ap`
- **Server 2 (Performance)**: `pso1zne8qfxx`
- **GPU+ (Reserve)**: `psb2pc82mxop`

---

## 🎯 **OPTIMIZATION TIPS**

1. **Start your day** with Server 1 only
2. **Monitor workload** throughout the day
3. **Start Server 2** only when needed
4. **Stop Server 2** as soon as heavy work is done
5. **Use intelligent scaling** for hands-off management
6. **Review costs weekly** to optimize usage patterns

Remember: **Server 2 costs $0.76/hour** - every hour saved is money in your pocket! 💰
