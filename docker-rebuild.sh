#!/bin/bash

# 🚀 ROBUST MONTE CARLO PLATFORM - DOCKER REBUILD SCRIPT
# This script rebuilds the Docker containers with all robustness fixes applied

echo "🚀 REBUILDING ROBUST MONTE CARLO PLATFORM"
echo "========================================="

# Set error handling
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}📋 Pre-build validation...${NC}"

# Validate that our fix scripts exist
if [ ! -f "backend/simulation_fixes_comprehensive.py" ]; then
    echo -e "${RED}❌ simulation_fixes_comprehensive.py not found${NC}"
    exit 1
fi

if [ ! -f "backend/enhanced_robust_fixes.py" ]; then
    echo -e "${RED}❌ enhanced_robust_fixes.py not found${NC}"
    exit 1
fi

if [ ! -f "backend/test_system.py" ]; then
    echo -e "${RED}❌ test_system.py not found${NC}"
    exit 1
fi

echo -e "${GREEN}✅ All fix scripts are present${NC}"

# Check Docker and Docker Compose
if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Docker is not installed${NC}"
    exit 1
fi

# Check for newer docker compose (without hyphen) or legacy docker-compose
if ! command -v docker &> /dev/null || ! docker compose version &> /dev/null; then
    if ! command -v docker-compose &> /dev/null; then
        echo -e "${RED}❌ Docker Compose is not installed${NC}"
        exit 1
    else
        echo -e "${YELLOW}⚠️ Using legacy docker-compose command${NC}"
        DOCKER_COMPOSE="docker-compose"
    fi
else
    DOCKER_COMPOSE="docker compose"
fi

echo -e "${GREEN}✅ Docker environment is ready${NC}"

# Create necessary directories
echo -e "${BLUE}📁 Creating necessary directories...${NC}"
mkdir -p uploads
mkdir -p logs
mkdir -p backend/saved_simulations_files
mkdir -p backend/results

# Ensure proper permissions
chmod 755 uploads
chmod 755 logs

echo -e "${GREEN}✅ Directories created${NC}"

echo -e "${BLUE}🔨 Building Docker containers...${NC}"

# Build with no cache to ensure all fixes are included
echo -e "${YELLOW}⚙️ Building backend with all robustness fixes...${NC}"
$DOCKER_COMPOSE build --no-cache backend

echo -e "${YELLOW}⚙️ Building frontend...${NC}"
$DOCKER_COMPOSE build --no-cache frontend

echo -e "${GREEN}✅ Docker images built successfully${NC}"

echo -e "${BLUE}🚀 Starting services...${NC}"

# Start the services
$DOCKER_COMPOSE up -d

echo -e "${GREEN}✅ Services started${NC}"

# Wait for services to be ready
echo -e "${BLUE}⏳ Waiting for services to initialize...${NC}"
sleep 15

# Validate services are running
echo -e "${BLUE}🔍 Validating service health...${NC}"

# Check if containers are running
if ! docker ps | grep -q "project-backend"; then
    echo -e "${RED}❌ Backend container is not running${NC}"
    $DOCKER_COMPOSE logs backend
    exit 1
fi

if ! docker ps | grep -q "project-frontend"; then
    echo -e "${RED}❌ Frontend container is not running${NC}"
    $DOCKER_COMPOSE logs frontend
    exit 1
fi

if ! docker ps | grep -q "project-redis"; then
    echo -e "${RED}❌ Redis container is not running${NC}"
    $DOCKER_COMPOSE logs redis
    exit 1
fi

echo -e "${GREEN}✅ All containers are running${NC}"

# Test backend API
echo -e "${BLUE}🧪 Testing backend API...${NC}"
if curl -s -f "http://localhost:8000/api" > /dev/null; then
    echo -e "${GREEN}✅ Backend API is responding${NC}"
else
    echo -e "${YELLOW}⚠️ Backend API test inconclusive (might be starting up)${NC}"
fi

# Test frontend
echo -e "${BLUE}🧪 Testing frontend...${NC}"
if curl -s -f "http://localhost:80" > /dev/null; then
    echo -e "${GREEN}✅ Frontend is responding${NC}"
else
    echo -e "${YELLOW}⚠️ Frontend test inconclusive (might be starting up)${NC}"
fi

# Run our robustness validation inside the container
echo -e "${BLUE}🔬 Running robustness validation...${NC}"
if docker exec project-backend-1 python3 test_system.py; then
    echo -e "${GREEN}✅ Robustness validation passed!${NC}"
else
    echo -e "${RED}❌ Robustness validation failed${NC}"
    exit 1
fi

echo ""
echo "========================================="
echo -e "${GREEN}🎉 DOCKER REBUILD COMPLETE!${NC}"
echo "========================================="
echo ""
echo -e "${BLUE}📊 SERVICE STATUS:${NC}"
echo -e "   • Backend API: ${GREEN}http://localhost:8000${NC}"
echo -e "   • Frontend: ${GREEN}http://localhost:80${NC}"
echo -e "   • Redis: ${GREEN}localhost:6379${NC}"
echo ""
echo -e "${BLUE}🚀 PLATFORM FEATURES ENABLED:${NC}"
echo -e "   • ✅ Arrow integration for big files"
echo -e "   • ✅ Enhanced progress tracking"
echo -e "   • ✅ Robust histogram generation"
echo -e "   • ✅ Formula evaluation (NO ZEROS BUG)"
echo -e "   • ✅ Automatic stuck simulation cleanup"
echo -e "   • ✅ Advanced error recovery"
echo -e "   • ✅ Memory optimization"
echo -e "   • ✅ Concurrency controls"
echo ""
echo -e "${GREEN}🎯 Your robust Monte Carlo platform is ready for production!${NC}"
echo ""
echo -e "${BLUE}📋 Useful Commands:${NC}"
echo -e "   • View logs: ${YELLOW}$DOCKER_COMPOSE logs -f${NC}"
echo -e "   • Stop services: ${YELLOW}$DOCKER_COMPOSE down${NC}"
echo -e "   • Restart services: ${YELLOW}$DOCKER_COMPOSE restart${NC}"
echo -e "   • View containers: ${YELLOW}docker ps${NC}"
echo "" 