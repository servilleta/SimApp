"""
HISTOGRAM SERVICE MODULE
Provides histogram generation utilities for simulation results
"""

import numpy as np
import logging
from typing import Dict, Any, List, Union, Optional

logger = logging.getLogger(__name__)

def generate_histogram_statistics(
    results_array: Union[np.ndarray, List[float]], 
    bins: int = 50,
    bin_method: str = "auto"
) -> Dict[str, Any]:
    """
    Generate histogram statistics from simulation results array
    
    Args:
        results_array: Array or list of simulation results
        bins: Number of histogram bins (default: 50)
        bin_method: Method for bin calculation ("auto", "fixed", "adaptive")
        
    Returns:
        Dictionary containing histogram data with keys:
        - bins: bin edges
        - values: bin counts  
        - bin_edges: bin edges (duplicate for compatibility)
        - counts: bin counts (duplicate for compatibility)
    """
    try:
        # Convert to numpy array if needed
        if isinstance(results_array, list):
            results_array = np.array(results_array)
        
        # Handle empty or invalid arrays
        if results_array is None or len(results_array) == 0:
            logger.warning("Empty results array provided to histogram generation")
            return {
                "bins": [],
                "values": [],
                "bin_edges": [],
                "counts": []
            }
        
        # Remove any NaN or infinite values
        finite_mask = np.isfinite(results_array)
        if np.sum(finite_mask) == 0:
            logger.warning("No finite values in results array")
            return {
                "bins": [],
                "values": [],
                "bin_edges": [],
                "counts": []
            }
        
        finite_results = results_array[finite_mask]
        
        # Determine optimal number of bins
        if bin_method == "adaptive":
            # Use adaptive binning based on data size
            data_size = len(finite_results)
            if data_size < 100:
                bins = min(20, max(10, data_size // 5))
            elif data_size < 1000:
                bins = min(30, max(15, data_size // 20))
            else:
                bins = min(50, max(25, data_size // 50))
        elif bin_method == "auto":
            # Use numpy's auto binning but constrain to reasonable range
            bins = min(50, max(15, int(np.sqrt(len(finite_results)))))
        
        # Generate histogram
        hist_counts, hist_edges = np.histogram(finite_results, bins=bins)
        
        # Log histogram generation info
        data_range = np.max(finite_results) - np.min(finite_results)
        logger.info(f"Generated histogram with {len(hist_counts)} bins for {len(finite_results)} values")
        logger.info(f"Data range: {data_range:.2e}, bin width: {(hist_edges[1] - hist_edges[0]):.2e}")
        
        # Return histogram data in expected format
        histogram_data = {
            "bins": hist_edges.tolist(),
            "values": hist_counts.tolist(),
            "bin_edges": hist_edges.tolist(),
            "counts": hist_counts.tolist()
        }
        
        return histogram_data
        
    except Exception as e:
        logger.error(f"Error generating histogram statistics: {e}")
        # Return empty histogram on error
        return {
            "bins": [],
            "values": [],
            "bin_edges": [],
            "counts": []
        }

def generate_enhanced_histogram_statistics(
    results_array: Union[np.ndarray, List[float]], 
    bins: int = 50,
    include_percentiles: bool = True,
    include_statistics: bool = True
) -> Dict[str, Any]:
    """
    Generate enhanced histogram statistics with additional metrics
    
    Args:
        results_array: Array or list of simulation results
        bins: Number of histogram bins
        include_percentiles: Whether to include percentile information
        include_statistics: Whether to include basic statistics
        
    Returns:
        Dictionary containing enhanced histogram data and statistics
    """
    try:
        # Get basic histogram
        histogram_data = generate_histogram_statistics(results_array, bins)
        
        # If basic histogram failed, return it
        if not histogram_data["bins"]:
            return histogram_data
        
        # Convert to numpy array if needed
        if isinstance(results_array, list):
            results_array = np.array(results_array)
        
        # Remove any NaN or infinite values
        finite_mask = np.isfinite(results_array)
        finite_results = results_array[finite_mask]
        
        # Add enhanced statistics
        if include_statistics:
            histogram_data.update({
                "mean": float(np.mean(finite_results)),
                "median": float(np.median(finite_results)),
                "std_dev": float(np.std(finite_results)),
                "min_value": float(np.min(finite_results)),
                "max_value": float(np.max(finite_results)),
                "total_count": len(finite_results)
            })
        
        # Add percentiles
        if include_percentiles:
            percentiles = {}
            for p in [5, 10, 25, 50, 75, 90, 95]:
                percentiles[str(p)] = float(np.percentile(finite_results, p))
            histogram_data["percentiles"] = percentiles
        
        return histogram_data
        
    except Exception as e:
        logger.error(f"Error generating enhanced histogram statistics: {e}")
        return generate_histogram_statistics(results_array, bins)

def validate_histogram_data(histogram_data: Dict[str, Any]) -> bool:
    """
    Validate histogram data structure
    
    Args:
        histogram_data: Histogram data dictionary
        
    Returns:
        True if valid, False otherwise
    """
    try:
        required_keys = ["bins", "values", "bin_edges", "counts"]
        
        # Check required keys exist
        if not all(key in histogram_data for key in required_keys):
            return False
        
        # Check that bins and counts have compatible lengths
        bins = histogram_data["bins"]
        counts = histogram_data["counts"]
        
        if not isinstance(bins, list) or not isinstance(counts, list):
            return False
        
        # For proper histogram, bins should be counts + 1 (bin edges)
        if len(bins) != len(counts) + 1:
            return False
        
        # Check for empty data
        if len(counts) == 0:
            return False
        
        # Check for negative counts
        if any(count < 0 for count in counts):
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"Error validating histogram data: {e}")
        return False 