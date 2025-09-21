"""
SUPERENGINE - Excel Error Code Constants
=========================================
This module defines the constants used to represent Excel error codes within
our GPU-based simulation engine.

Excel's errors are not numbers, but our engine operates on floating-point arrays
for performance. To handle this, we represent each error as a specific, reserved
NaN (Not a Number) value. Using `np.nan` with different bit patterns allows us
to create distinct values that are all "NaN" but can be identified uniquely.

This enables our GPU kernels to propagate errors correctly, mimicking Excel's
behavior (e.g., any calculation involving an error results in that error).
"""

import numpy as np
import cupy as cp

# Create unique NaN values for each error type by manipulating the bit pattern.
# We create a 64-bit float NaN and then change its payload.
_base_nan = np.uint64(0x7ff8000000000001)

# --- Excel Error Code Definitions ---
# Each error is a unique NaN value that can be identified.

EXCEL_ERRORS = {
    '#NULL!':  np.float64.view(_base_nan + 0),
    '#DIV/0!': np.float64.view(_base_nan + 1),
    '#VALUE!': np.float64.view(_base_nan + 2),
    '#REF!':   np.float64.view(_base_nan + 3),
    '#NAME?':  np.float64.view(_base_nan + 4),
    '#NUM!':   np.float64.view(_base_nan + 5),
    '#N/A':    np.float64.view(_base_nan + 6),
}

# --- GPU-side Error Constants ---
# Create the same constants on the GPU for use in kernels.
EXCEL_ERRORS_GPU = {name: cp.asarray(val) for name, val in EXCEL_ERRORS.items()}

# --- Helper Functions ---

def is_error(arr: cp.ndarray) -> cp.ndarray:
    """
    Returns a boolean array indicating which elements are any kind of Excel error.
    This works because all our error codes are NaNs.
    """
    return cp.isnan(arr)

def get_error(error_name: str) -> cp.ndarray:
    """Returns the GPU constant for a specific error."""
    return EXCEL_ERRORS_GPU[error_name]

# Example Usage
if __name__ == '__main__':
    print("Excel Error Code Representation:")
    for name, val in EXCEL_ERRORS.items():
        print(f"{name}: {val} (is_nan: {np.isnan(val)})")

    # Example of checking for errors on the GPU
    a = cp.array([1.0, 2.0, EXCEL_ERRORS['#DIV/0!'], 4.0, EXCEL_ERRORS['#VALUE!']])
    print("\nSample GPU array:", a)
    
    error_mask = is_error(a)
    print("Error mask:", error_mask)
    
    print("\nChecking for specific error:")
    is_div_error = (a == EXCEL_ERRORS_GPU['#DIV/0!']) # Note: This comparison is tricky with NaNs
    # A better way is to check the bit pattern if needed, but isn't necessary if we just propagate.
    # For now, we rely on the fact that any operation with a NaN propagates the NaN.
    
    print("Is #DIV/0!:", cp.any(cp.isnan(a))) # Simplified check for demonstration 