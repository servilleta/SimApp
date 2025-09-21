"""
Shared data sanitization utilities for JSON serialization safety.
This module provides comprehensive data cleaning to prevent NaN/inf serialization errors.
"""

import math
import logging
from typing import Any, Dict, List, Union

logger = logging.getLogger(__name__)

def sanitize_data_structure(data: Any) -> Any:
    """
    Recursively sanitize any data structure to remove NaN/inf values.
    This is a comprehensive catch-all to prevent JSON serialization errors.
    
    Args:
        data: Any data structure (dict, list, primitive types)
        
    Returns:
        Sanitized data structure safe for JSON serialization
    """
    if data is None:
        return data
    elif isinstance(data, (int, str, bool)):
        return data
    elif isinstance(data, float):
        if math.isnan(data) or math.isinf(data):
            logger.debug(f"Sanitized invalid float value: {data} -> 0.0")
            return 0.0
        return data
    elif isinstance(data, dict):
        return {key: sanitize_data_structure(value) for key, value in data.items()}
    elif isinstance(data, (list, tuple)):
        return [sanitize_data_structure(item) for item in data]
    else:
        # For any other types, try to convert to string as fallback
        try:
            return str(data)
        except Exception as e:
            logger.warning(f"Failed to sanitize data type {type(data)}: {e}")
            return None

def sanitize_simulation_response(response_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Specialized sanitization for simulation response data.
    Handles common simulation-specific fields with extra care.
    
    Args:
        response_data: Simulation response dictionary
        
    Returns:
        Sanitized simulation response
    """
    if not isinstance(response_data, dict):
        return sanitize_data_structure(response_data)
    
    # Start with general sanitization
    sanitized = sanitize_data_structure(response_data)
    
    # Ensure critical fields are properly typed
    if 'progress_percentage' in sanitized:
        try:
            sanitized['progress_percentage'] = float(sanitized['progress_percentage'] or 0.0)
            sanitized['progress_percentage'] = max(0.0, min(100.0, sanitized['progress_percentage']))
        except (ValueError, TypeError):
            sanitized['progress_percentage'] = 0.0
    
    if 'current_iteration' in sanitized:
        try:
            sanitized['current_iteration'] = int(sanitized['current_iteration'] or 0)
        except (ValueError, TypeError):
            sanitized['current_iteration'] = 0
    
    if 'total_iterations' in sanitized:
        try:
            sanitized['total_iterations'] = int(sanitized['total_iterations'] or 0)
        except (ValueError, TypeError):
            sanitized['total_iterations'] = 0
    
    if 'timestamp' in sanitized:
        try:
            # Ensure timestamp is a valid number
            sanitized['timestamp'] = float(sanitized['timestamp'] or 0.0)
        except (ValueError, TypeError):
            import time
            sanitized['timestamp'] = time.time()
    
    return sanitized

def sanitize_statistics_data(stats_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Specialized sanitization for statistical data (means, std dev, etc.).
    
    Args:
        stats_data: Statistics dictionary
        
    Returns:
        Sanitized statistics data
    """
    if not isinstance(stats_data, dict):
        return sanitize_data_structure(stats_data)
    
    sanitized = {}
    
    for key, value in stats_data.items():
        if isinstance(value, (int, float)):
            if math.isnan(value) or math.isinf(value):
                # For statistical fields, use None instead of 0 to indicate missing data
                sanitized[key] = None
                logger.debug(f"Sanitized invalid statistic {key}: {value} -> None")
            else:
                sanitized[key] = value
        else:
            sanitized[key] = sanitize_data_structure(value)
    
    return sanitized

def validate_json_serializable(data: Any) -> bool:
    """
    Test if data is JSON serializable.
    
    Args:
        data: Data to test
        
    Returns:
        True if serializable, False otherwise
    """
    try:
        import json
        json.dumps(data)
        return True
    except (TypeError, ValueError) as e:
        logger.warning(f"Data not JSON serializable: {e}")
        return False



