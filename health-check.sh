#!/bin/bash

echo "=== Monte Carlo Platform Health Check ==="
echo "Checking all services..."

# Check Docker containers
echo -e "\n1. Docker Containers:"
docker-compose ps

# Check nginx
echo -e "\n2. Nginx Health:"
curl -s -o /dev/null -w "HTTP Status: %{http_code}, Response Time: %{time_total}s\n" http://localhost/health

# Check API health
echo -e "\n3. API Health:"
curl -s -o /dev/null -w "HTTP Status: %{http_code}, Response Time: %{time_total}s\n" http://localhost/api/health

# Check frontend
echo -e "\n4. Frontend Health:"
curl -s -o /dev/null -w "HTTP Status: %{http_code}, Response Time: %{time_total}s\n" http://localhost/

# Check PostgreSQL
echo -e "\n5. PostgreSQL Health:"
docker exec montecarlo-postgres pg_isready -U montecarlo_user -d montecarlo_db

# Check Redis
echo -e "\n6. Redis Health:"
docker exec project-redis-1 redis-cli ping

# Check system resources
echo -e "\n7. System Resources:"
echo "CPU Usage:"
top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1
echo "Memory Usage:"
free -h | grep Mem | awk '{print $3"/"$2}'
echo "Disk Usage:"
df -h / | tail -1 | awk '{print $5}'

echo -e "\n=== Health Check Complete ===" 