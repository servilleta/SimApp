"""
Main FastAPI application for Monte Carlo simulation platform.
"""
import os
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Import routers
from auth.router import router as auth_router
from excel_parser.router import router as excel_parser_router
from simulation.router import router as simulation_router
from simulation.phases_router import router as phases_router  # New phases router
from limits.router import router as limits_router

# Import new Stripe billing routers
from api.billing import router as billing_router
from api.webhooks import router as webhooks_router

# Import B2B API router
from api.v1.router import router as api_v1_router

# Import services
from database.config import init_db
from simulation.service import SimulationService
from gpu.manager import GPUManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Suppress verbose logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    logger.info("🚀 Starting Monte Carlo Simulation Platform...")
    
    try:
        # Initialize database
        logger.info("📊 Initializing database...")
        init_db()
        
        # Initialize GPU manager
        logger.info("🎮 Initializing GPU manager...")
        gpu_manager = GPUManager()
        app.state.gpu_manager = gpu_manager
        
        # Initialize simulation service
        logger.info("🎲 Initializing simulation service...")
        simulation_service = SimulationService()
        app.state.simulation_service = simulation_service
        
        # Create necessary directories
        uploads_dir = Path("uploads")
        cache_dir = Path("cache")
        uploads_dir.mkdir(exist_ok=True)
        cache_dir.mkdir(exist_ok=True)
        
        logger.info("✅ Application started successfully!")
        
        yield
        
    except Exception as e:
        logger.error(f"❌ Failed to start application: {str(e)}")
        raise
    finally:
        logger.info("👋 Shutting down application...")
        # Cleanup resources if needed
        if hasattr(app.state, 'simulation_service'):
            # Any cleanup needed for simulation service
            pass
        logger.info("✅ Application shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="Monte Carlo Simulation Platform",
    description="Enterprise-grade Monte Carlo simulation platform with GPU acceleration",
    version="2.0.0",
    lifespan=lifespan,
    servers=[
        {"url": "http://209.51.170.185:8000", "description": "Production server"},
        {"url": "http://localhost:8000", "description": "Local development server"}
    ]
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(auth_router, prefix="/api/auth", tags=["Authentication"])
app.include_router(excel_parser_router, prefix="/api/excel-parser", tags=["Excel Parser"])
app.include_router(simulation_router, prefix="/api/simulations", tags=["Simulations"])
app.include_router(phases_router, prefix="/api/phases", tags=["Simulation Phases"])
app.include_router(limits_router, prefix="/api/limits", tags=["Limits"])

# Register Stripe billing and webhook routers
app.include_router(billing_router, prefix="/api", tags=["Billing & Subscriptions"])
app.include_router(webhooks_router, prefix="/api", tags=["Webhooks"])

# Register B2B API v1 router
app.include_router(api_v1_router, tags=["B2B API v1"])

# Mount static files (if needed)
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )

# Health check endpoint
@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "version": "2.0.0",
        "gpu_available": hasattr(app.state, 'gpu_manager') and app.state.gpu_manager.gpu_available
    }

# Root endpoint
@app.get("/api")
async def root():
    return {
        "message": "Monte Carlo Simulation Platform API",
        "version": "2.0.0",
        "docs": "/api/docs"
    } 