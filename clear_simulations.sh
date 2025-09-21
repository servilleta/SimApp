#!/bin/bash

echo "🚨 DANGER: This will permanently delete ALL simulation data!"
echo "This includes:"
echo "  - All simulation results from database"
echo "  - All saved simulations"
echo "  - All Redis cache data"
echo "  - All temporary files"
echo ""

read -p "Are you sure you want to continue? Type 'DELETE ALL' to confirm: " confirm

if [ "$confirm" != "DELETE ALL" ]; then
    echo "❌ Operation cancelled"
    exit 1
fi

echo ""
echo "🗑️ Starting cleanup process..."

# Execute cleanup script inside the backend container
echo "Running cleanup script..."
docker-compose exec -T backend python3 admin_scripts/clear_all_simulations.py

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 SUCCESS: All simulation data has been cleared!"
    echo "The system is now ready for fresh simulations."
else
    echo ""
    echo "❌ FAILED: Error occurred during cleanup"
    exit 1
fi
