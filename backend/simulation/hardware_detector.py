"""
GPU Hardware Detection Utility
"""
import logging
from typing import Optional

try:
    import cupy as cp
except ImportError:
    cp = None

logger = logging.getLogger(__name__)

def get_gpu_compute_capability() -> Optional[float]:
    """
    Detects the compute capability of the available NVIDIA GPU using CuPy.

    Returns:
        A float representing the compute capability (e.g., 7.5)
        or None if no GPU is found or an error occurs.
    """
    if not cp:
        logger.info("CuPy not installed. No GPU capability detection possible.")
        return None

    try:
        device = cp.cuda.Device(0)
        props = cp.cuda.runtime.getDeviceProperties(device.id)
        major = props['major']
        minor = props['minor']
        compute_capability = float(f"{major}.{minor}")
        logger.info(f"Detected GPU with Compute Capability: {compute_capability}")
        return compute_capability
    except cp.cuda.runtime.CUDARuntimeError as e:
        logger.warning(f"Could not get GPU properties: {e}")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred during GPU detection: {e}")
        return None