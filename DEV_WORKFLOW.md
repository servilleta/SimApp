# 🔧 Development Mode Workflow
## Smart Auto-Scaling for Development

Perfect for your development needs where you want cost efficiency but automatic scaling when needed!

---

## 🎯 **YOUR DEVELOPMENT SETUP**

### **💡 Smart Development Strategy**
- **Both servers OFF** by default (save money! 💰)
- **Start Server 1** when you want to work
- **Server 2 auto-starts** when Server 1 gets stressed
- **Auto-shutdown** when you're done working

### **💰 Development Cost Savings**
```
Traditional: Both always on = $871/month
Development: Only when working = $50-150/month
SAVINGS: $720+ per month! 🎉
```

---

## 🚀 **DAILY DEVELOPMENT WORKFLOW**

### **🌅 Starting Your Dev Session**

#### **Step 1: Start Server 1 (Manual)**
```bash
# Via Paperspace Console or API
cd /home/paperspace/SimApp
./paperspace_api_manager.py start --machine-id psotdtcda5ap
```

#### **Step 2: Start Development Monitoring**
```bash
# Start the intelligent development scaler
./development_scaling.py
```

#### **Step 3: Begin Your Work**
```bash
# Access your SimApp
# Server 2 will automatically start if you stress Server 1
```

### **🌇 Ending Your Dev Session**
```bash
# Stop the monitoring (Ctrl+C)
# Server 2 will auto-stop to save money
# Optionally stop Server 1 via console
```

---

## 🤖 **AUTOMATIC SCALING BEHAVIOR**

### **🚨 Server 2 AUTO-STARTS When:**
- ✅ **CPU > 70%** on Server 1 (stressed)
- ✅ **Memory > 75%** on Server 1
- ✅ **API response > 3 seconds** (slow performance)
- ✅ **3+ active simulations** running
- ✅ **High disk I/O** activity

### **⬇️ Server 2 AUTO-STOPS When:**
- ✅ **CPU < 20%** for 2+ minutes
- ✅ **Memory < 30%**
- ✅ **API response < 1 second**
- ✅ **No active simulations**
- ✅ **Load stays low** after 1-minute confirmation

### **💤 IDLE AUTO-SHUTDOWN:**
- **Server 2 stops** after 30 minutes of low activity
- **Saves money** during breaks, lunch, meetings
- **You control Server 1** manually

---

## 📊 **DEVELOPMENT SCENARIOS**

### **🧪 Light Development Work**
```
Server 1: Running (code editing, small tests)
Server 2: OFF
Cost: ~$0.45/hour
```

### **🔥 Heavy Development Work**
```
Server 1: Running (main work)
Server 2: AUTO-STARTED (big simulations, performance testing)
Cost: ~$1.21/hour (only while both needed)
```

### **☕ Break Time**
```
Server 1: Running (keeps your session)
Server 2: AUTO-STOPPED after 30min idle
Cost: ~$0.45/hour
```

### **🏠 End of Day**
```
Server 1: Manually stopped via console
Server 2: Already auto-stopped
Cost: $0/hour
```

---

## 🎮 **DEVELOPMENT COMMANDS**

### **Quick Start Development Session**
```bash
# 1. Start Server 1 (manually via console)
# 2. Start intelligent scaling
cd /home/paperspace/SimApp
./development_scaling.py

# Monitor in real-time
tail -f development_scaling.log
```

### **Check Development Status**
```bash
# Quick status check
./paperspace_api_manager.py cluster-status

# Detailed monitoring (run once)
./development_scaling.py --once

# Custom monitoring interval
./development_scaling.py --interval 60  # Check every minute
```

### **Override Controls**
```bash
# Force start Server 2 for testing
./paperspace_api_manager.py start --machine-id pso1zne8qfxx

# Force stop Server 2 to save money
./paperspace_api_manager.py stop --machine-id pso1zne8qfxx

# Check what's running
./paperspace_api_manager.py list
```

---

## 💡 **DEVELOPMENT TIPS**

### **🎯 Optimize Your Dev Costs**
1. **Always start** with Server 1 only
2. **Let auto-scaling** handle Server 2
3. **Take breaks** and let Server 2 auto-stop
4. **Stop Server 1** when done for the day
5. **Monitor usage** to understand patterns

### **🧪 Testing Heavy Simulations**
```bash
# Before big simulation tests:
./paperspace_api_manager.py start --machine-id pso1zne8qfxx

# Your heavy test will run on powerful A4000 GPU
# Let auto-scaling stop it afterward
```

### **👥 Team Development**
```bash
# Keep Server 1 running during team hours
# Server 2 auto-scales based on team load
# Saves money during low-activity periods
```

---

## 📊 **MONITORING YOUR DEV SESSION**

### **Real-Time Status**
The development monitor shows:
```
📊 DEV STATUS: Server1:🟢 Server2:💤 CPU:25.1% Memory:45.2% API:890ms Sims:1
🚨 SERVER 1 STRESSED: CPU:78.5% Memory:82.1% Response:3200ms Sims:4
🚀 DEV SCALING UP: Starting Server 2 (A4000) to help Server 1
✅ Server 2 started - High performance mode active
⬇️ DEV SCALING DOWN: Stopping Server 2 (load reduced)
💤 SYSTEM IDLE for 30 minutes - Auto-shutdown initiated
```

### **Log Analysis**
```bash
# Review your scaling patterns
grep "SCALING" development_scaling.log

# Check cost-saving events
grep "Auto-shutdown\|stopped" development_scaling.log

# Monitor stress triggers
grep "STRESSED" development_scaling.log
```

---

## 🚨 **TROUBLESHOOTING DEVELOPMENT**

### **Server 2 Won't Start**
```bash
# Check Server 1 status first
./paperspace_api_manager.py status --machine-id psotdtcda5ap

# Manually start if needed
./paperspace_api_manager.py start --machine-id pso1zne8qfxx
```

### **Too Aggressive Scaling**
```bash
# Increase stress thresholds in development_scaling.py
# Or use longer intervals
./development_scaling.py --interval 120  # Check every 2 minutes
```

### **Costs Too High**
```bash
# Check what's running
./paperspace_api_manager.py cluster-status

# Force stop Server 2
./paperspace_api_manager.py stop --machine-id pso1zne8qfxx

# Reduce idle timeout
./development_scaling.py --idle-shutdown 15  # 15 minutes instead of 30
```

---

## 🎯 **PERFECT FOR DEVELOPMENT!**

Your setup now gives you:

✅ **Cost Efficiency**: Only pay for what you use  
✅ **Auto-Scaling**: Performance when needed  
✅ **Zero Management**: Set and forget  
✅ **Development Speed**: Fast iteration cycles  
✅ **Production Ready**: Same system scales to production  

**Start your development session and let the system optimize itself!** 🚀
