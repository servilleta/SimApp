# Paperspace API Integration Setup Guide
=============================================

This guide walks you through setting up the Paperspace API integration for SimApp's multi-server architecture.

## 🔑 Step 1: Get Your Paperspace API Key

### Method 1: Via Paperspace Console
1. Log in to your [Paperspace account](https://console.paperspace.com)
2. Navigate to **Account Settings** → **API**
3. Click **"Generate new API key"**
4. Copy the generated API key (it starts with `ps_...`)
5. **Important**: Save this key securely - it won't be shown again!

### Method 2: Via CLI (if you have paperspace-cli installed)
```bash
# Install Paperspace CLI
npm install -g paperspace-node

# Login and get API key
paperspace login
paperspace machines list  # This will show your API key in use
```

## 🛠️ Step 2: Install Required Dependencies

### On Primary Server
```bash
cd /home/paperspace/SimApp

# Install Python dependencies
pip install paperspace requests

# Create logs directory
mkdir -p logs

# Make scripts executable
chmod +x paperspace_api_manager.py
```

### Environment Variable Setup
```bash
# Add to your shell profile (~/.bashrc or ~/.zshrc)
export PAPERSPACE_API_KEY="your_api_key_here"

# Or create a .env file
echo "PAPERSPACE_API_KEY=your_api_key_here" >> .env

# Reload environment
source ~/.bashrc
```

## 🔍 Step 3: Test the Integration

### Test API Connectivity
```bash
# Test basic API connection
python3 paperspace_api_manager.py list

# Get cluster status
python3 paperspace_api_manager.py cluster-status
```

### Expected Output
```
📋 Found 2 machines:
  - Primary Server (ps_abc123): running @ 64.71.146.187
  - Secondary Server (ps_def456): stopped @ 72.52.107.230

🏗️ SimApp Cluster Status:
{
  "timestamp": "2025-09-21 14:30:00 UTC",
  "primary_server": {
    "ip": "64.71.146.187",
    "status": "running",
    "machine_id": "ps_abc123"
  },
  "secondary_server": {
    "ip": "72.52.107.230", 
    "status": "stopped",
    "machine_id": "ps_def456"
  }
}
```

## 🚀 Step 4: Manual Server Control

### Start Secondary Server
```bash
# Using the Python script
python3 paperspace_api_manager.py scale-up

# Or direct machine control
python3 paperspace_api_manager.py start --machine-id ps_def456
```

### Stop Secondary Server
```bash
# Using the Python script
python3 paperspace_api_manager.py scale-down

# Or direct machine control
python3 paperspace_api_manager.py stop --machine-id ps_def456
```

## 🌐 Step 5: API Endpoint Usage

### Via FastAPI Backend
Once the backend is updated, you can use these endpoints:

```bash
# Get cluster status
curl -H "Authorization: Bearer YOUR_JWT_TOKEN" \
     http://localhost:9090/api/infrastructure/status

# Manual scaling
curl -X POST \
     -H "Authorization: Bearer YOUR_JWT_TOKEN" \
     -H "Content-Type: application/json" \
     -d '{"action": "scale-up", "reason": "high_load"}' \
     http://localhost:9090/api/infrastructure/scale

# Cost optimization report
curl -H "Authorization: Bearer YOUR_JWT_TOKEN" \
     http://localhost:9090/api/infrastructure/cost-optimization
```

### Frontend Integration
```javascript
// Example React component usage
const handleScaleUp = async () => {
  try {
    const response = await fetch('/api/infrastructure/scale', {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${token}`,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        action: 'scale-up',
        reason: 'user_request'
      })
    });
    
    const result = await response.json();
    console.log('Scaling result:', result);
  } catch (error) {
    console.error('Scaling failed:', error);
  }
};
```

## ⚙️ Step 6: Automatic Scaling Configuration

### Configure Auto-Scaling Thresholds
Edit `/home/paperspace/SimApp/backend/services/paperspace_service.py`:

```python
# Scaling thresholds
self.scale_up_threshold = 5    # Scale up if >5 pending simulations
self.scale_down_threshold = 2  # Scale down if <2 pending simulations
self.scaling_cooldown = 300    # 5 minutes between actions

# Cost settings
self.secondary_server_cost_per_hour = 0.40  # Adjust based on actual cost
```

### Enable Background Auto-Scaling
```bash
# Create a cron job for automatic scaling checks
crontab -e

# Add this line to check every 5 minutes
*/5 * * * * cd /home/paperspace/SimApp && python3 -c "
import asyncio
from backend.services.paperspace_service import paperspace_service
asyncio.run(paperspace_service.auto_scale_cluster())
"
```

## 🔐 Security Best Practices

### API Key Security
1. **Never commit API keys to Git**
2. **Use environment variables only**
3. **Rotate keys regularly**
4. **Restrict key permissions if possible**

### Network Security
```bash
# Ensure API traffic is encrypted
export PAPERSPACE_API_URL="https://api.paperspace.io"

# Monitor API usage
tail -f /home/paperspace/SimApp/logs/paperspace_api.log
```

## 📊 Step 7: Monitoring and Logging

### View Logs
```bash
# Real-time log monitoring
tail -f /home/paperspace/SimApp/logs/paperspace_api.log

# Search for specific events
grep "SCALE UP" /home/paperspace/SimApp/logs/paperspace_api.log
grep "❌" /home/paperspace/SimApp/logs/paperspace_api.log
```

### Monitor Costs
```bash
# Daily cost report
python3 paperspace_api_manager.py cluster-status | jq '.cost_analysis'

# Check secondary server uptime
python3 paperspace_api_manager.py status --machine-id ps_def456
```

## 🚨 Troubleshooting

### Common Issues

#### 1. "Authorization Required" Error
```bash
# Check if API key is set
echo $PAPERSPACE_API_KEY

# Test API key validity
curl -H "Authorization: Bearer $PAPERSPACE_API_KEY" \
     https://api.paperspace.io/machines/getMachines
```

#### 2. "Machine Not Found" Error
```bash
# List all machines to find correct ID
python3 paperspace_api_manager.py list

# Update machine ID in script if needed
```

#### 3. Network Connectivity Issues
```bash
# Test Paperspace API connectivity
ping api.paperspace.io

# Check firewall rules
sudo ufw status
```

#### 4. Scaling Cooldown Messages
```bash
# Force scaling action (bypass cooldown)
curl -X POST \
     -H "Authorization: Bearer YOUR_JWT_TOKEN" \
     -H "Content-Type: application/json" \
     -d '{"action": "scale-up", "force": true}' \
     http://localhost:9090/api/infrastructure/scale
```

## 💡 Advanced Usage

### Blue-Green Deployment
```bash
# Stop all services on secondary
ssh paperspace3 "sudo docker-compose down"

# Start new version on secondary
ssh paperspace3 "git pull && sudo docker-compose up -d"

# Switch traffic to secondary
# (Configure load balancer or DNS)

# Verify and rollback if needed
```

### Custom Scaling Logic
```python
# Example: Scale based on user count
async def custom_scaling_check():
    active_users = await get_active_user_count()
    if active_users > 50:
        await paperspace_service.scale_up_secondary_server()
    elif active_users < 10:
        await paperspace_service.scale_down_secondary_server()
```

## 📈 Performance Optimization

### Caching Machine Status
```python
# Cache machine status for 30 seconds to reduce API calls
from functools import lru_cache
import time

@lru_cache(maxsize=128)
def cached_machine_status(machine_id, timestamp):
    return api_manager.get_machine_status(machine_id)

# Use with current timestamp rounded to 30-second intervals
status = cached_machine_status(machine_id, int(time.time() // 30))
```

### Batch Operations
```python
# Start multiple machines at once
machines_to_start = ["ps_def456", "ps_ghi789"]
for machine_id in machines_to_start:
    api_manager.start_machine(machine_id)
```

## 🎯 Next Steps

1. **Test the integration** with your actual Paperspace machines
2. **Monitor costs** during the first week of usage
3. **Fine-tune scaling thresholds** based on your workload patterns
4. **Set up alerts** for scaling events and failures
5. **Consider implementing** Blue-Green deployment automation
6. **Plan for disaster recovery** scenarios

## 📞 Support

- **Paperspace API Docs**: https://paperspace.gitbook.io/paperspace-core-api
- **SimApp Issues**: Check logs in `/home/paperspace/SimApp/logs/`
- **Community Support**: Paperspace Community Forums

---
**Generated**: September 21, 2025  
**Version**: 1.0.0  
**Author**: SimApp DevOps Team
