# Simple Python-based Dockerfile for Monte Carlo Backend with CUDA support
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    curl \
    libmagic1 \
    libmagic-dev \
    wget \
    gnupg \
    && rm -rf /var/lib/apt/lists/*

# SKIP CUDA installation at build time - will handle at runtime
RUN echo "🔧 Setting up CUDA library paths for runtime installation..." \
    && mkdir -p /usr/local/cuda-11.2/lib64 \
    && mkdir -p /usr/local/cuda-11.2/targets/x86_64-linux/lib \
    && echo '/usr/local/cuda-11.2/lib64' >> /etc/ld.so.conf.d/cuda.conf \
    && echo '/usr/local/cuda-11.2/targets/x86_64-linux/lib' >> /etc/ld.so.conf.d/cuda.conf \
    && echo '/usr/lib/x86_64-linux-gnu' >> /etc/ld.so.conf.d/cuda.conf \
    && ldconfig \
    && echo "🔧 CUDA paths configured - NVRTC will be installed at runtime if needed"

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel
RUN python3 -m pip install --no-cache-dir -r requirements.txt

# Copy application code first (needed for user creation)
COPY . .

# Create user first
RUN useradd -m appuser

# Install Playwright system dependencies as root
RUN python3 -m playwright install-deps chromium

# Install Playwright browsers as the appuser
USER appuser
RUN python3 -m playwright install chromium
USER root

# Make NVRTC installer executable
RUN chmod +x install_nvrtc.py

# User already created above

# No startup script - NVRTC installation will be handled by main.py

# Create directories and set permissions
RUN mkdir -p uploads cache \
    && chown -R appuser:appuser uploads cache \
    && chmod 755 uploads cache \
    && chmod 664 *.db || true \
    && chown appuser:appuser *.db || true \
    && chown -R appuser:appuser /app

# Set GPU environment variables and CUDA library paths
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}"
ENV CUDA_HOME="/usr/local/cuda"
ENV PATH="/usr/local/cuda/bin:${PATH}"

USER appuser

# Expose port
EXPOSE 8000

# Run the application
CMD ["python3", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]