# Monte Carlo Platform - Debugging Guide

## Quick Debugging Commands

### 1. Monitor Logs in Real-time
```bash
# Backend logs (most important for simulations)
docker-compose logs -f backend

# Frontend logs
docker-compose logs -f frontend

# All services at once
docker-compose logs -f

# Filter for specific log levels
docker-compose logs -f backend | grep -E "(DEBUG|INFO|ERROR|WARNING)"

# Search for specific issues (e.g., user/filename tracking)
docker-compose logs backend | grep -A5 -B5 "initiate_simulation"
```

### 2. Check Container Status
```bash
# See all containers
docker-compose ps

# Check resource usage
docker stats

# Inspect a specific container
docker inspect project-backend-1
```

### 3. Redis Debugging (Simulation Data)
```bash
# Access Redis CLI
docker exec -it project-redis-1 redis-cli

# In Redis CLI:
# List all keys
KEYS *

# Check simulation progress
GET simulation:progress:<simulation_id>

# Check simulation results
GET simulation:results:<simulation_id>

# Exit Redis
exit
```

### 4. Database Debugging (PostgreSQL)
```bash
# Access PostgreSQL
docker exec -it montecarlo-postgres psql -U postgres -d montecarlo

# Common queries:
# List all users
SELECT * FROM users;

# Check recent simulations
SELECT * FROM simulations ORDER BY created_at DESC LIMIT 10;

# Exit PostgreSQL
\q
```

### 5. Python Interactive Debugging
```bash
# Access Python shell in backend
docker exec -it project-backend-1 python

# Test imports and functions
>>> from backend.simulation.service import SimulationService
>>> from backend.excel_parser.service import ExcelParserService
>>> # Test your code here
>>> exit()
```

### 6. API Testing
```bash
# Test backend health
curl http://localhost:8001/api/health

# Test with authentication
TOKEN="your-jwt-token"
curl -H "Authorization: Bearer $TOKEN" http://localhost:8001/api/user/dashboard/stats

# View API docs
# Open browser: http://localhost:8001/api/docs
```

### 7. File System Debugging
```bash
# Check uploaded files
docker exec -it project-backend-1 ls -la /app/uploads/

# Check parsed Excel data
docker exec -it project-backend-1 ls -la /app/excel_parser_results/

# View a specific file
docker exec -it project-backend-1 cat /app/excel_parser_results/<file_id>_formulas.json
```

### 8. Restart Services (if needed)
```bash
# Restart specific service
docker-compose restart backend

# Full restart
docker-compose down && docker-compose up -d

# Rebuild with cache clear (for code changes)
docker-compose down
docker system prune -f
docker-compose build --no-cache
docker-compose up -d
```

## Debugging the Admin Log KPI Issue

### Step 1: Run a New Simulation
1. Login to the platform
2. Upload an Excel file
3. Run a simulation
4. Note the simulation ID from the URL or logs

### Step 2: Check Backend Logs
```bash
# Watch for the new simulation
docker-compose logs -f backend | grep -E "(initiate_simulation|DEBUG.*user|DEBUG.*filename)"
```

### Step 3: Verify Redis Data
```bash
docker exec -it project-redis-1 redis-cli
GET simulation:progress:<your-simulation-id>
GET simulation:results:<your-simulation-id>
exit
```

### Step 4: Check Admin Dashboard
- Go to http://localhost:8080/admin/simulations
- Check if the new simulation shows correct user/filename

## Common Issues and Solutions

### 1. "User: n/a, Filename: Unknown"
- **Cause**: User/filename data not preserved during simulation updates
- **Fix**: Already implemented in code, needs new simulations to test
- **Debug**: Check `initiate_simulation` logs for DEBUG messages

### 2. Frontend Not Updating
- **Fix**: Hard refresh (Ctrl+Shift+R) or clear browser cache
- **Or**: Rebuild frontend: `docker-compose restart frontend`

### 3. Simulation Stuck
- **Check**: Backend logs for errors
- **Fix**: Clear Redis cache if needed:
  ```bash
  docker exec -it project-redis-1 redis-cli FLUSHDB
  ```

### 4. Authentication Issues
- **Check**: JWT token expiration
- **Fix**: Re-login to get new token

## Useful Log Patterns to Search

```bash
# Find all simulation initiations
docker-compose logs backend | grep "initiate_simulation"

# Find authentication issues
docker-compose logs backend | grep -i "auth"

# Find Excel parsing errors
docker-compose logs backend | grep -i "excel.*error"

# Find GPU/engine issues
docker-compose logs backend | grep -E "(GPU|engine|cuda)"
```

## Environment Variables
Check current environment:
```bash
docker exec -it project-backend-1 env | grep -E "(ENV|DEBUG|AUTH0)"
```

## Performance Monitoring
```bash
# CPU and Memory usage
docker stats

# GPU usage (if available)
docker exec -it project-backend-1 nvidia-smi
``` 