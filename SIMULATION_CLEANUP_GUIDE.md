# Simulation Data Cleanup Guide

This guide provides multiple methods to safely delete all simulation data from your Monte Carlo simulation platform.

## ⚠️ Important Warning

**ALL METHODS WILL PERMANENTLY DELETE ALL SIMULATION DATA FOR ALL USERS**

This includes:
- All simulation results from the database
- All saved simulation configurations  
- All Redis cache data
- All temporary simulation files
- All Ultra Engine database files

**This action cannot be undone!**

## Method 1: Command Line Script (Recommended)

The easiest method is to use the interactive shell script:

```bash
./clear_simulations.sh
```

This script will:
1. Show a warning about the destructive action
2. Ask for confirmation (you must type "DELETE ALL")
3. Execute the cleanup inside the Docker containers
4. Show the results

## Method 2: Direct Docker Exec

If you want to run the cleanup directly without the interactive script:

```bash
docker-compose exec -T backend python3 admin_scripts/clear_all_simulations.py
```

## Method 3: Admin API Endpoint

For admin users with API access, you can call the admin endpoint:

```bash
curl -X POST http://localhost:8000/api/admin/clear-all-simulations \
  -H "Authorization: Bearer YOUR_ADMIN_TOKEN" \
  -H "Content-Type: application/json"
```

## Method 4: Web Interface

Open `clear_simulations.html` in your browser and follow the instructions. You'll need to:
1. Update the API URL to match your backend
2. Add proper authentication headers
3. Type "DELETE ALL" to confirm
4. Click the cleanup button

## What Gets Cleaned

### Database Tables
- `simulation_results` - Main simulation data
- `saved_simulations` - Saved simulation configurations

### Ultra Engine Database
- SQLite database tables (if using Ultra Engine):
  - `simulations`
  - `target_cells`
  - `histogram_data`
  - `tornado_data`
  - `dependency_tree`

### Redis Cache
- All keys matching patterns:
  - `progress:*`
  - `simulation:*`
  - `sim:*`
  - `*simulation*`

### Temporary Files
- `/tmp/simulations/`
- `/home/paperspace/PROJECT/backend/temp_files/`
- `/home/paperspace/PROJECT/backend/uploads/`

## Verification

After cleanup, you can verify the system is clean by:

1. **Check the frontend**: Visit the application and verify the "Recent" simulations list is empty
2. **Check database**: `docker-compose exec -T backend python3 -c "from database import SessionLocal; from models import SimulationResult; db = SessionLocal(); print(f'Simulations remaining: {db.query(SimulationResult).count()}'); db.close()"`
3. **Check Redis**: `docker-compose exec redis redis-cli KEYS "*simulation*"`

## Safety Features

- **Confirmation required**: All interactive methods require explicit confirmation
- **Admin-only API**: The API endpoint requires admin privileges
- **Comprehensive logging**: All cleanup operations are logged
- **Error handling**: Script continues even if some cleanup steps fail
- **Transaction safety**: Database operations use transactions

## Files Created

This solution includes the following files:

1. **`backend/admin_scripts/clear_all_simulations.py`** - Main cleanup script
2. **`clear_simulations.sh`** - Interactive shell script
3. **`clear_simulations.html`** - Web interface for cleanup
4. **`backend/admin/router.py`** - Updated with admin API endpoint

## Troubleshooting

### Redis Connection Issues
If Redis is not accessible, the script will skip Redis cleanup with a warning. This is normal if Redis is not running or not accessible.

### Database Connection Issues
Make sure the backend container is running and the database is accessible.

### Permission Issues
Ensure you have admin privileges for API-based cleanup methods.

### TTY Issues with Docker
Use the `-T` flag with docker-compose exec to avoid TTY issues in non-interactive environments.

## Recovery

After cleanup, the system is ready for fresh simulations. To start using the system again:

1. Upload a new Excel file
2. Configure your simulation parameters
3. Run a new simulation
4. The new simulation will appear in the "Recent" list

## Support

If you encounter issues during cleanup:

1. Check the Docker container logs: `docker-compose logs backend`
2. Verify all containers are running: `docker-compose ps`
3. Check database connectivity
4. Ensure proper permissions for admin operations
