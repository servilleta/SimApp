# 🚀 Ultra Engine Monte Carlo Simulation Platform - Backend
# Use Python base image with CUDA runtime support
FROM python:3.10-slim

# Set environment variables for CUDA
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

WORKDIR /app

# Install system dependencies and CUDA runtime
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    libmagic1 \
    libmagic-dev \
    wget \
    gnupg2 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY backend/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install Playwright browsers for PDF export functionality
RUN playwright install --with-deps chromium

# Copy the entire backend
COPY backend/ .

# Create necessary directories
RUN mkdir -p uploads enterprise-storage

# Set Python path
ENV PYTHONPATH=/app

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
