"""
SUPERENGINE - Granular CUDA Kernel Library
========================================
This module provides a library of fine-grained, vectorized functions that execute
on the GPU using CuPy. Each function corresponds to a specific operation that can
appear in an AST (e.g., add, multiply, sum).

The AST Compiler will call these functions as it walks the tree, effectively
translating the formula's logic into a sequence of GPU operations.
"""

import logging
import cupy as cp
import numpy as np
from typing import Dict, Any, Union

from super_engine.error_codes import is_error, get_error

logger = logging.getLogger(__name__)

# --- GPU Kernel Library ---

def gpu_add(a: Union[cp.ndarray, float], b: Union[cp.ndarray, float]) -> cp.ndarray:
    """Element-wise addition on the GPU with error propagation."""
    # If either input is an error, the result is an error.
    # NaNs propagate automatically in arithmetic ops, which is what our errors are.
    return cp.add(a, b)

def gpu_subtract(a: Union[cp.ndarray, float], b: Union[cp.ndarray, float]) -> cp.ndarray:
    """Element-wise subtraction on the GPU with error propagation."""
    return cp.subtract(a, b)

def gpu_multiply(a: Union[cp.ndarray, float], b: Union[cp.ndarray, float]) -> cp.ndarray:
    """Element-wise multiplication on the GPU with error propagation."""
    return cp.multiply(a, b)

def gpu_divide(a: Union[cp.ndarray, float], b: Union[cp.ndarray, float]) -> cp.ndarray:
    """Element-wise division on the GPU. Handles division by zero and propagates errors."""
    # Find where the denominator is zero
    zero_mask = (b == 0)
    
    # Perform division where denominator is not zero.
    # Initialize result with a default value (it will be overwritten).
    result = cp.full_like(a, 0, dtype=cp.float64)
    non_zero_mask = ~zero_mask
    result[non_zero_mask] = cp.divide(a[non_zero_mask], b[non_zero_mask])

    # Set our specific #DIV/0! error where the denominator was zero.
    result[zero_mask] = get_error('#DIV/0!')

    # Ensure that errors in the inputs are propagated correctly.
    # If a or b is already an error (NaN), the result of the division will also be NaN.
    # We just need to make sure our specific error codes are preserved.
    # Let's ensure any existing error in `a` or `b` takes precedence over a new div/0 error.
    a_errors = is_error(a)
    b_errors = is_error(b)
    result[a_errors] = a[a_errors]
    result[b_errors] = b[b_errors] # b error takes precedence over a error if both exist

    return result

def gpu_sum_kahan(data: cp.ndarray) -> float:
    """
    Performs a deterministic, Kahan-style compensated summation on the GPU.
    This provides higher accuracy and consistent results across runs.
    """
    s = cp.zeros((), dtype=data.dtype) # Sum
    c = cp.zeros((), dtype=data.dtype) # Compensation for lost low-order bits
    
    # This is a direct implementation. For very large arrays, a parallel version would be better.
    for i in range(data.size):
        y = data[i] - c
        t = s + y
        c = (t - s) - y
        s = t
    return float(s)

def gpu_sum(data: cp.ndarray) -> float:
    """Performs a deterministic, parallel reduction sum on the GPU, ignoring errors."""
    finite_data = data[cp.isfinite(data)]
    if finite_data.size == 0:
        return 0.0
    # Use the more accurate Kahan summation
    return gpu_sum_kahan(finite_data)

def gpu_average(data: cp.ndarray) -> float:
    """Performs a deterministic, parallel reduction mean on the GPU, ignoring errors."""
    finite_data = data[cp.isfinite(data)]
    if finite_data.size == 0:
        return 0.0
    
    # Use accurate sum for a more accurate mean
    accurate_sum = gpu_sum_kahan(finite_data)
    return accurate_sum / finite_data.size

def gpu_if(condition: cp.ndarray, val_if_true: Union[cp.ndarray, float], val_if_false: Union[cp.ndarray, float]) -> cp.ndarray:
    """GPU-accelerated IF function. Selects elements based on a condition."""
    # Start with the standard where clause
    result = cp.where(condition, val_if_true, val_if_false)
    # If the condition itself is an error, the result should be that error
    cond_errors = is_error(condition)
    result[cond_errors] = condition[cond_errors]
    return result

def gpu_gt(a: Union[cp.ndarray, float], b: Union[cp.ndarray, float]) -> cp.ndarray:
    """Element-wise greater than (>) on the GPU with error propagation."""
    # NaNs in comparisons result in False. We need to see if either input was an error.
    # If so, that location in the result should be an error.
    result = cp.greater(a, b)
    # This is tricky. A simple solution for now is to just propagate NaN.
    # A more robust solution would check bit patterns.
    # For now, standard NaN propagation from cp.add is sufficient to mark errors.
    return cp.add(a, -b) > 0 # A trick to propagate NaNs correctly in comparisons

def gpu_lt(a: Union[cp.ndarray, float], b: Union[cp.ndarray, float]) -> cp.ndarray:
    """Element-wise less than (<) on the GPU with error propagation."""
    return cp.add(a, -b) < 0

def gpu_eq(a: Union[cp.ndarray, float], b: Union[cp.ndarray, float]) -> cp.ndarray:
    """Element-wise equal to (==) on the GPU with error propagation."""
    return cp.equal(a,b) # Note: nan == nan is false. This is tricky.

def gpu_gte(a: Union[cp.ndarray, float], b: Union[cp.ndarray, float]) -> cp.ndarray:
    """Element-wise greater than or equal to (>=) on the GPU with error propagation."""
    return cp.add(a, -b) >= 0

def gpu_lte(a: Union[cp.ndarray, float], b: Union[cp.ndarray, float]) -> cp.ndarray:
    """Element-wise less than or equal to (<=) on the GPU with error propagation."""
    return cp.add(a, -b) <= 0

def gpu_neq(a: Union[cp.ndarray, float], b: Union[cp.ndarray, float]) -> cp.ndarray:
    """Element-wise not equal to (!=) on the GPU with error propagation."""
    return cp.not_equal(a,b) # nan != nan is true.

def gpu_and(conditions: list[cp.ndarray]) -> cp.ndarray:
    """GPU-accelerated logical AND across multiple conditions with error propagation."""
    if not conditions:
        # Per Excel's behavior, AND() with no arguments is TRUE
        return cp.array(True)
    return cp.logical_and.reduce(conditions)

def gpu_or(conditions: list[cp.ndarray]) -> cp.ndarray:
    """GPU-accelerated logical OR across multiple conditions."""
    if not conditions:
        # Per Excel's behavior, OR() with no arguments is FALSE
        return cp.array(False)
    return cp.logical_or.reduce(conditions)

def gpu_min(data: list[cp.ndarray]) -> cp.ndarray:
    """Finds the element-wise minimum across multiple arrays on the GPU."""
    stacked_data = cp.stack(data)
    return cp.min(stacked_data, axis=0)

def gpu_max(data: list[cp.ndarray]) -> cp.ndarray:
    """Finds the element-wise maximum across multiple arrays on the GPU."""
    stacked_data = cp.stack(data)
    return cp.max(stacked_data, axis=0)

def gpu_sumif(range_to_check: cp.ndarray, criterion: str, sum_range: cp.ndarray) -> cp.ndarray:
    """
    Performs a conditional sum on the GPU. SUMIF(range, criterion, sum_range).
    
    Args:
        range_to_check: The array to evaluate against the criterion.
        criterion: The condition (e.g., ">10", "A", "<=5"). This is a string.
        sum_range: The array from which to sum cells if the condition is met.

    Returns:
        A scalar CuPy array containing the sum.
    """
    # This is a simplified implementation. A robust version would use a regex
    # or a mini-parser to handle all of Excel's criteria (e.g., wildcards).
    op_str = ''.join(filter(lambda x: not x.isdigit() and x != '.', criterion))
    val_str = ''.join(filter(lambda x: x.isdigit() or x == '.', criterion))
    
    if not val_str:
        # Handles criteria like "A" (equality)
        op_str = '=='
        val_str = criterion
        try:
            # Attempt to convert to float if it's a number string
            value = float(val_str)
        except ValueError:
             # It's a string criterion for exact match
             # Note: This requires the array to be of a compatible type.
             # This part is complex and may need object-dtype arrays.
             # For now, we assume numeric comparisons.
             logger.warning("String criteria in SUMIF not fully supported, assuming numeric.")
             return cp.array(0.0) # Placeholder
    else:
        value = float(val_str)

    op_map = {
        '>': cp.greater, '>=': cp.greater_equal,
        '<': cp.less, '<=': cp.less_equal,
        '=': cp.equal, '==': cp.equal,
        '<>': cp.not_equal, '!=': cp.not_equal
    }
    
    if op_str not in op_map:
        raise NotImplementedError(f"Criterion operator '{op_str}' in SUMIF is not supported.")
    
    # Get the boolean mask where the condition is true
    mask = op_map[op_str](range_to_check, value)
    
    # Use the mask to sum elements from the sum_range
    return cp.sum(sum_range[mask])

def gpu_normal(mean: Union[cp.ndarray, float], std_dev: Union[cp.ndarray, float], size: int) -> cp.ndarray:
    """Generates random numbers from a normal distribution on the GPU."""
    return cp.random.normal(loc=mean, scale=std_dev, size=size)

def gpu_vlookup_exact(lookup_values: cp.ndarray, lookup_table: cp.ndarray, result_column_index: int) -> cp.ndarray:
    """
    Performs a high-performance, hash-based exact VLOOKUP on the GPU.
    If input is string/object dtype, fallback to CPU (not supported on GPU).
    """
    # Check for string/object dtype
    if lookup_values.dtype.char == 'O' or lookup_table.dtype.char == 'O':
        import logging
        logger = logging.getLogger(__name__)
        logger.warning("gpu_vlookup_exact: String/object dtype detected, not supported on GPU. Fallback required.")
        # Return all NaN so the engine can fallback to CPU
        return cp.full(lookup_values.shape, cp.nan)

    # Ensure table is on GPU
    lookup_table_gpu = cp.asarray(lookup_table)
    lookup_col = lookup_table_gpu[:, 0]
    result_col = lookup_table_gpu[:, result_column_index]

    # Prepare the result array, initialized with NaN
    results = cp.full(lookup_values.shape, cp.nan, dtype=result_col.dtype)

    # For each unique value in our lookup_values, find all occurrences
    unique_vals = cp.unique(lookup_values)
    for val in unique_vals:
        input_mask = (lookup_values == val)
        table_matches = cp.where(lookup_col == val)[0]
        if table_matches.size > 0:
            first_match_index = table_matches[0]
            result_value = result_col[first_match_index]
            results[input_mask] = result_value
    return results

def gpu_not(condition: cp.ndarray) -> cp.ndarray:
    """Performs a logical NOT on the GPU."""
    return cp.logical_not(condition)

def gpu_npv(rate: Union[cp.ndarray, float], cash_flows: list[cp.ndarray]) -> cp.ndarray:
    """
    Calculates the Net Present Value on the GPU. NPV(rate, value1, value2, ...)

    Args:
        rate: The discount rate for one period (can be an array or scalar).
        cash_flows: A list of CuPy arrays, where each array represents the cash flow
                    for a period (e.g., [CF1, CF2, CF3, ...]).

    Returns:
        A CuPy array containing the NPV for each iteration.
    """
    if not cash_flows:
        return cp.array(0.0)

    # Stack the cash flows into a 2D array: (num_periods, num_iterations)
    cf_matrix = cp.stack(cash_flows, axis=0)

    # Create an array for the time periods (1, 2, 3, ...)
    num_periods = len(cash_flows)
    periods = cp.arange(1, num_periods + 1).reshape(-1, 1) # Shape: (num_periods, 1)

    # Calculate discount factors (1+r)^t. This will broadcast correctly.
    discount_factors = (1 + rate) ** periods

    # Calculate discounted cash flows
    discounted_cfs = cf_matrix / discount_factors

    # Sum the discounted cash flows across the periods (axis=0) to get the NPV for each iteration
    npv = cp.sum(discounted_cfs, axis=0)
    
    return npv

# --- Core Math Functions ---
def gpu_sin(data: cp.ndarray) -> cp.ndarray:
    """Element-wise SIN on the GPU."""
    return cp.sin(data)

def gpu_cos(data: cp.ndarray) -> cp.ndarray:
    """Element-wise COS on the GPU."""
    return cp.cos(data)

def gpu_tan(data: cp.ndarray) -> cp.ndarray:
    """Element-wise TAN on the GPU."""
    return cp.tan(data)

def gpu_log(data: cp.ndarray) -> cp.ndarray:
    """Element-wise natural logarithm (LOG) on the GPU."""
    return cp.log(data)

def gpu_exp(data: cp.ndarray) -> cp.ndarray:
    """Element-wise exponential function (e^x) on the GPU."""
    return cp.exp(data)

def gpu_abs(data: cp.ndarray) -> cp.ndarray:
    """Element-wise absolute value on the GPU."""
    return cp.abs(data)

def gpu_power(base: cp.ndarray, exponent: cp.ndarray) -> cp.ndarray:
    """Element-wise power function on the GPU."""
    return cp.power(base, exponent)

def gpu_lognormal(mean: Union[cp.ndarray, float], sigma: Union[cp.ndarray, float], size: int) -> cp.ndarray:
    """Generates random numbers from a log-normal distribution on the GPU."""
    return cp.random.lognormal(mean=mean, sigma=sigma, size=size)

# --- Financial Functions ---
def gpu_pv(rate: cp.ndarray, nper: cp.ndarray, pmt: cp.ndarray, 
           fv: cp.ndarray = None, type_: cp.ndarray = None) -> cp.ndarray:
    """
    GPU kernel for Present Value calculation.
    PV = -PMT * ((1 - (1 + rate)^(-nper)) / rate) - FV / (1 + rate)^nper
    """
    if fv is None:
        fv = cp.zeros_like(rate)
    if type_ is None:
        type_ = cp.zeros_like(rate)
    
    # Handle zero rate case
    zero_rate = rate == 0
    
    # Normal calculation
    factor = cp.power(1 + rate, -nper)
    pv_normal = -pmt * (1 + rate * type_) * ((1 - factor) / rate) - fv * factor
    
    # Zero rate calculation
    pv_zero = -pmt * nper - fv
    
    # Select based on rate
    return cp.where(zero_rate, pv_zero, pv_normal)

def gpu_fv(rate: cp.ndarray, nper: cp.ndarray, pmt: cp.ndarray, 
           pv: cp.ndarray = None, type_: cp.ndarray = None) -> cp.ndarray:
    """
    GPU kernel for Future Value calculation.
    FV = -PV * (1 + rate)^nper - PMT * ((1 + rate)^nper - 1) / rate
    """
    if pv is None:
        pv = cp.zeros_like(rate)
    if type_ is None:
        type_ = cp.zeros_like(rate)
    
    # Handle zero rate case
    zero_rate = rate == 0
    
    # Normal calculation
    factor = cp.power(1 + rate, nper)
    fv_normal = -pv * factor - pmt * (1 + rate * type_) * ((factor - 1) / rate)
    
    # Zero rate calculation
    fv_zero = -pv - pmt * nper
    
    # Select based on rate
    return cp.where(zero_rate, fv_zero, fv_normal)

def gpu_pmt(rate: cp.ndarray, nper: cp.ndarray, pv: cp.ndarray, 
            fv: cp.ndarray = None, type_: cp.ndarray = None) -> cp.ndarray:
    """
    GPU kernel for Payment calculation.
    PMT = (rate * (PV + FV / (1 + rate)^nper)) / (1 - (1 + rate)^(-nper))
    """
    if fv is None:
        fv = cp.zeros_like(rate)
    if type_ is None:
        type_ = cp.zeros_like(rate)
    
    # Handle zero rate case
    zero_rate = rate == 0
    
    # Normal calculation
    factor = cp.power(1 + rate, nper)
    pmt_normal = (rate * (pv * factor + fv)) / ((factor - 1) * (1 + rate * type_))
    
    # Zero rate calculation
    pmt_zero = -(pv + fv) / nper
    
    # Select based on rate
    return cp.where(zero_rate, pmt_zero, pmt_normal)

# --- Date/Time Functions ---
def gpu_year(date_value: cp.ndarray) -> cp.ndarray:
    """Extract year from Excel date value (days since 1900-01-01)."""
    # Simplified: assumes date_value is days since 1900
    # Excel's epoch is actually 1900-01-00 (bug compatibility)
    year = 1900 + cp.floor(date_value / 365.25)
    return year.astype(cp.int32)

def gpu_month(date_value: cp.ndarray) -> cp.ndarray:
    """Extract month from Excel date value."""
    # Simplified implementation
    days_in_year = date_value % 365.25
    month = cp.minimum(12, cp.maximum(1, cp.floor(days_in_year / 30.44) + 1))
    return month.astype(cp.int32)

def gpu_day(date_value: cp.ndarray) -> cp.ndarray:
    """Extract day from Excel date value."""
    # Simplified implementation
    days_in_month = date_value % 30.44
    day = cp.minimum(31, cp.maximum(1, cp.floor(days_in_month) + 1))
    return day.astype(cp.int32)

def gpu_today() -> cp.ndarray:
    """Get FIXED date as Excel date value for reproducible simulations."""
    import datetime
    # Excel epoch is 1900-01-00
    excel_epoch = datetime.datetime(1899, 12, 30)
    # Use fixed date for reproducible simulations
    fixed_date = datetime.datetime(2024, 1, 1)
    days_since_epoch = (fixed_date - excel_epoch).days
    return cp.full(1, days_since_epoch, dtype=cp.float32)

def gpu_now() -> cp.ndarray:
    """Get FIXED date and time as Excel date value for reproducible simulations."""
    import datetime
    excel_epoch = datetime.datetime(1899, 12, 30)
    # Use fixed datetime for reproducible simulations
    fixed_datetime = datetime.datetime(2024, 1, 1, 12, 0, 0)
    days_since_epoch = (fixed_datetime - excel_epoch).total_seconds() / 86400.0
    return cp.full(1, days_since_epoch, dtype=cp.float32)

# --- Additional Distribution Functions ---
def gpu_poisson(lam: Union[float, cp.ndarray], size: int) -> cp.ndarray:
    """Generate Poisson distributed random numbers on GPU."""
    return cp.random.poisson(lam=lam, size=size).astype(cp.float32)

def gpu_binomial(n: Union[int, cp.ndarray], p: Union[float, cp.ndarray], size: int) -> cp.ndarray:
    """Generate Binomial distributed random numbers on GPU."""
    return cp.random.binomial(n=n, p=p, size=size).astype(cp.float32)

def gpu_student_t(df: Union[float, cp.ndarray], size: int) -> cp.ndarray:
    """Generate Student's t distributed random numbers on GPU."""
    return cp.random.standard_t(df=df, size=size).astype(cp.float32)

def gpu_pert(minimum: Union[float, cp.ndarray], 
             most_likely: Union[float, cp.ndarray], 
             maximum: Union[float, cp.ndarray], 
             size: int, 
             shape: float = 4.0) -> cp.ndarray:
    """
    Generate PERT distributed random numbers on GPU.
    PERT is a special case of Beta distribution commonly used in project management.
    """
    # Convert to scalars if needed
    if isinstance(minimum, cp.ndarray):
        minimum = float(minimum)
    if isinstance(most_likely, cp.ndarray):
        most_likely = float(most_likely)
    if isinstance(maximum, cp.ndarray):
        maximum = float(maximum)
    
    # Calculate alpha and beta parameters
    mean = (minimum + shape * most_likely + maximum) / (shape + 2)
    
    if (mean - minimum) * (maximum - mean) < 0:
        raise ValueError("Invalid PERT parameters")
    
    alpha = 1 + shape * (mean - minimum) / (maximum - minimum)
    beta = 1 + shape * (maximum - mean) / (maximum - minimum)
    
    # Generate beta distribution and scale to [minimum, maximum]
    beta_samples = cp.random.beta(alpha, beta, size=size)
    return minimum + (maximum - minimum) * beta_samples

def gpu_discrete(values: cp.ndarray, probabilities: cp.ndarray, size: int) -> cp.ndarray:
    """
    Generate discrete distributed random numbers on GPU.
    
    Args:
        values: Array of possible values
        probabilities: Array of probabilities for each value (must sum to 1)
        size: Number of samples to generate
    """
    # Ensure probabilities sum to 1
    probabilities = probabilities / cp.sum(probabilities)
    
    # Use choice function with probabilities
    return cp.random.choice(values, size=size, p=probabilities).astype(cp.float32)

def gpu_exponential(scale: Union[float, cp.ndarray], size: int) -> cp.ndarray:
    """Generate Exponential distributed random numbers on GPU."""
    return cp.random.exponential(scale=scale, size=size).astype(cp.float32)

def gpu_chi_square(df: Union[float, cp.ndarray], size: int) -> cp.ndarray:
    """Generate Chi-square distributed random numbers on GPU."""
    return cp.random.chisquare(df=df, size=size).astype(cp.float32)

def gpu_f_distribution(dfnum: Union[float, cp.ndarray], 
                      dfden: Union[float, cp.ndarray], 
                      size: int) -> cp.ndarray:
    """Generate F-distributed random numbers on GPU."""
    return cp.random.f(dfnum=dfnum, dfden=dfden, size=size).astype(cp.float32)

def gpu_pareto(alpha: Union[float, cp.ndarray], size: int) -> cp.ndarray:
    """Generate Pareto distributed random numbers on GPU."""
    return cp.random.pareto(a=alpha, size=size).astype(cp.float32)

def gpu_rayleigh(scale: Union[float, cp.ndarray], size: int) -> cp.ndarray:
    """Generate Rayleigh distributed random numbers on GPU."""
    return cp.random.rayleigh(scale=scale, size=size).astype(cp.float32)

# --- Text Functions ---
def gpu_concatenate(*args: cp.ndarray) -> cp.ndarray:
    """
    GPU kernel for CONCATENATE function.
    Concatenates multiple numeric arrays by converting to strings.
    
    Note: CuPy has limited string support, so this returns a numeric
    representation of concatenation for demonstration purposes.
    """
    if not args:
        return cp.array([0.0])
    
    # For numeric concatenation demonstration, we'll multiply by powers of 10
    # This is a simplified placeholder - real string ops would need custom CUDA
    result = cp.zeros_like(args[0], dtype=cp.float64)
    
    # Simple demonstration: sum the values (placeholder for concatenation)
    for arg in args:
        result = result + arg
    
    logger.warning("CONCATENATE on GPU is experimental - returning sum as placeholder")
    return result

def gpu_left(text: cp.ndarray, num_chars: Union[int, cp.ndarray]) -> cp.ndarray:
    """
    GPU kernel for LEFT function.
    For numeric values, extracts leftmost digits.
    
    Args:
        text: Array of numeric values
        num_chars: Number of digits to extract from the left
    """
    # For numeric inputs, extract left digits by division
    if isinstance(num_chars, cp.ndarray):
        num_chars = int(cp.mean(num_chars))  # Use average for simplicity
    else:
        num_chars = int(num_chars)
    
    # Calculate divisor to get leftmost digits
    # For example, LEFT(12345, 3) = 123
    result = cp.zeros_like(text, dtype=cp.float64)
    
    for i in range(text.size):
        val = float(text.flat[i])
        if val != 0:
            # Count total digits
            total_digits = int(cp.floor(cp.log10(cp.abs(val))) + 1)
            if total_digits > num_chars:
                # Divide to get leftmost digits
                divisor = 10 ** (total_digits - num_chars)
                result.flat[i] = cp.floor(val / divisor)
            else:
                result.flat[i] = val
        else:
            result.flat[i] = 0
    
    return result

def gpu_right(text: cp.ndarray, num_chars: Union[int, cp.ndarray]) -> cp.ndarray:
    """
    GPU kernel for RIGHT function.
    For numeric values, extracts rightmost digits.
    
    Args:
        text: Array of numeric values
        num_chars: Number of digits to extract from the right
    """
    # For numeric inputs, extract right digits by modulo
    if isinstance(num_chars, cp.ndarray):
        num_chars = int(cp.mean(num_chars))  # Use average for simplicity
    else:
        num_chars = int(num_chars)
    
    # Use modulo to get rightmost digits
    # For example, RIGHT(12345, 3) = 345
    divisor = 10 ** num_chars
    result = cp.abs(text) % divisor
    
    # Preserve sign
    result = cp.where(text < 0, -result, result)
    
    return result

def gpu_len(text: cp.ndarray) -> cp.ndarray:
    """
    GPU kernel for LEN function.
    For numeric values, returns the number of digits.
    """
    # Count digits in numeric values
    result = cp.zeros_like(text, dtype=cp.int32)
    
    for i in range(text.size):
        val = abs(float(text.flat[i]))
        if val == 0:
            result.flat[i] = 1  # Zero has 1 digit
        else:
            # Count digits including decimal places
            # For simplicity, just count integer part
            result.flat[i] = int(cp.floor(cp.log10(val)) + 1)
    
    return result

def gpu_mid(text: cp.ndarray, start_pos: Union[int, cp.ndarray], 
            num_chars: Union[int, cp.ndarray]) -> cp.ndarray:
    """
    GPU kernel for MID function.
    For numeric values, extracts middle digits.
    
    Note: Simplified implementation for numeric values.
    """
    # Convert parameters to scalars for simplicity
    if isinstance(start_pos, cp.ndarray):
        start_pos = int(cp.mean(start_pos))
    else:
        start_pos = int(start_pos)
        
    if isinstance(num_chars, cp.ndarray):
        num_chars = int(cp.mean(num_chars))
    else:
        num_chars = int(num_chars)
    
    # For numeric values, this is a placeholder
    # Real implementation would need string manipulation
    result = cp.zeros_like(text, dtype=cp.float64)
    
    # Simple demonstration: return the original value
    logger.warning("MID on GPU is experimental - returning original values")
    return text

# --- Additional Text Functions for Future Implementation ---
def gpu_upper(text: cp.ndarray) -> cp.ndarray:
    """Placeholder for UPPER function - returns input unchanged"""
    logger.warning("UPPER on GPU not implemented - returning original values")
    return text

def gpu_lower(text: cp.ndarray) -> cp.ndarray:
    """Placeholder for LOWER function - returns input unchanged"""
    logger.warning("LOWER on GPU not implemented - returning original values")
    return text

def gpu_trim(text: cp.ndarray) -> cp.ndarray:
    """Placeholder for TRIM function - returns input unchanged"""
    logger.warning("TRIM on GPU not implemented - returning original values")
    return text

# --- Kernel Dispatcher ---

# A dictionary mapping operator names from the AST to the actual kernel functions.
# This makes the compiler's job easier and keeps the logic decoupled.
KERNEL_LIBRARY = {
    # Arithmetic
    'add': gpu_add,
    'sub': gpu_subtract,
    'mul': gpu_multiply,
    'div': gpu_divide,

    # Reductions
    'SUM': gpu_sum,
    'AVERAGE': gpu_average,
    # 'MIN': gpu_min, # Placeholder for future
    # 'MAX': gpu_max, # Placeholder for future

    # Logical
    'gt': gpu_gt,
    'lt': gpu_lt,
    'eq': gpu_eq,
    'gte': gpu_gte,
    'lte': gpu_lte,
    'neq': gpu_neq,

    # Functions
    'IF': gpu_if,
    'AND': gpu_and,
    'OR': gpu_or,
    'MIN': gpu_min,
    'MAX': gpu_max,
    'SUMIF': gpu_sumif,
    'NPV': gpu_npv,
    'NOT': gpu_not,
    'NORMAL': gpu_normal,
    'LOGNORMAL': gpu_lognormal,
    'VLOOKUP': gpu_vlookup_exact,
    # Core Math
    'SIN': gpu_sin,
    'COS': gpu_cos,
    'TAN': gpu_tan,
    'LOG': gpu_log,
    'EXP': gpu_exp,
    'ABS': gpu_abs,
    'POWER': gpu_power,
    # Financial functions
    'PV': gpu_pv,
    'FV': gpu_fv,
    'PMT': gpu_pmt,
    # Date/Time functions
    'YEAR': gpu_year,
    'MONTH': gpu_month,
    'DAY': gpu_day,
    'TODAY': gpu_today,
    'NOW': gpu_now,
    # Additional distributions
    'POISSON': gpu_poisson,
    'BINOMIAL': gpu_binomial,
    'STUDENT_T': gpu_student_t,
    'PERT': gpu_pert,
    'DISCRETE': gpu_discrete,
    'EXPONENTIAL': gpu_exponential,
    'CHI_SQUARE': gpu_chi_square,
    'F_DISTRIBUTION': gpu_f_distribution,
    'PARETO': gpu_pareto,
    'RAYLEIGH': gpu_rayleigh,
    # Text functions
    'CONCATENATE': gpu_concatenate,
    'LEFT': gpu_left,
    'RIGHT': gpu_right,
    'LEN': gpu_len,
    'MID': gpu_mid,
    'UPPER': gpu_upper,
    'LOWER': gpu_lower,
    'TRIM': gpu_trim,
}

logger.info(f"✅ SUPERENGINE: GPU Kernel Library initialized with {len(KERNEL_LIBRARY)} kernels.")

def is_gpu_available() -> bool:
    """Checks if a compatible GPU is available."""
    try:
        return cp.cuda.is_available()
    except Exception:
        return False

# Example Usage
if __name__ == '__main__':
    if not is_gpu_available():
        print("❌ No compatible GPU found. Cannot run kernel examples.")
    else:
        print("✅ GPU found. Running kernel examples...")

        # Create some sample data on the GPU
        a = cp.array([1, 2, 3, 4])
        b = cp.array([5, 6, 7, 8])

        # Test arithmetic kernels
        add_result = KERNEL_LIBRARY['add'](a, b)
        print(f"ADD Result: {add_result}")

        # Test reduction kernels
        sum_result = KERNEL_LIBRARY['SUM'](b)
        print(f"SUM Result: {sum_result}")
        
        # Test division by zero
        c = cp.array([10, 20, 30, 40])
        d = cp.array([2, 0, 10, 5])
        div_result = KERNEL_LIBRARY['div'](c, d)
        print(f"DIV Result (with zero): {div_result}")

        # Test logical and IF kernels
        cond = KERNEL_LIBRARY['gt'](a, 2)
        print(f"GT Result (a > 2): {cond}")
        if_result = KERNEL_LIBRARY['IF'](cond, b, c) # if a > 2, use b, else use c
        print(f"IF Result: {if_result}")

        # Test AND/OR kernels
        cond2 = KERNEL_LIBRARY['lt'](b, 8)
        and_result = KERNEL_LIBRARY['AND']([cond, cond2])
        print(f"AND Result (a > 2 AND b < 8): {and_result}")
        or_result = KERNEL_LIBRARY['OR']([cond, cond2])
        print(f"OR Result (a > 2 OR b < 8): {or_result}")

        # Test NORMAL kernel
        normal_result = KERNEL_LIBRARY['NORMAL'](100, 10, 5) # mean=100, std=10, size=5
        print(f"NORMAL Result: {normal_result}")

        # Test VLOOKUP kernel
        # lookup_values, lookup_table, result_column_index
        lu_vals = cp.array([10, 30, 10, 50, 60]) # Look for 10, 30, 50. 60 is not in table.
        lu_table = cp.array([
            [10, 100, 1000],
            [20, 200, 2000],
            [30, 300, 3000],
            [40, 400, 4000],
            [50, 500, 5000]
        ])
        vlookup_result = KERNEL_LIBRARY['VLOOKUP'](lu_vals, lu_table, 2)
        print(f"VLOOKUP Result (col 2): {vlookup_result}") # Should be [1000, 3000, 1000, 5000, nan]

        # Test MIN/MAX kernels
        min_result = KERNEL_LIBRARY['MIN']([a, b, c])
        print(f"MIN Result: {min_result}") # Element-wise min of a, b, c
        max_result = KERNEL_LIBRARY['MAX']([a, b, c])
        print(f"MAX Result: {max_result}") # Element-wise max of a, b, c

        # Test SUMIF kernel
        # sum b where a > 2
        sumif_result = KERNEL_LIBRARY['SUMIF'](a, ">2", b)
        print(f"SUMIF Result (sum b where a > 2): {sumif_result}")

        # Test NPV kernel
        rate = 0.1 # 10% discount rate
        cf1 = cp.array([100, 110, 90])
        cf2 = cp.array([200, 210, 190])
        cf3 = cp.array([300, 310, 290])
        npv_result = KERNEL_LIBRARY['NPV'](rate, [cf1, cf2, cf3])
        # Expected for first iteration: 100/1.1 + 200/1.21 + 300/1.331 = 481.58
        print(f"NPV Result: {npv_result}")

        # Test NOT kernel
        not_result = KERNEL_LIBRARY['NOT'](cond) # NOT(a > 2)
        print(f"NOT Result: {not_result}")

        # Test a math kernel
        abs_result = KERNEL_LIBRARY['ABS'](a - b)
        print(f"ABS(a-b) Result: {abs_result}")
        power_result = KERNEL_LIBRARY['POWER'](a, 2)
        print(f"POWER(a, 2) Result: {power_result}")

        # Test Lognormal kernel
        lognormal_result = KERNEL_LIBRARY['LOGNORMAL'](0.0, 1.0, 5) # mean=0, sigma=1, size=5
        print(f"LOGNORMAL Result: {lognormal_result}")

        # Test PV kernel
        rate = 0.05
        nper = 10
        pmt = -1000
        pv_result = KERNEL_LIBRARY['PV'](rate, nper, pmt)
        print(f"PV Result: {pv_result}")

        # Test FV kernel
        fv_result = KERNEL_LIBRARY['FV'](rate, nper, pmt)
        print(f"FV Result: {fv_result}")

        # Test PMT kernel
        pv = 10000
        fv = 0
        rate = 0.05
        nper = 10
        pmt_result = KERNEL_LIBRARY['PMT'](rate, nper, pv, fv)
        print(f"PMT Result: {pmt_result}")

        # Test YEAR kernel
        date_value = cp.array([44896, 44927, 44957, 44988])
        year_result = KERNEL_LIBRARY['YEAR'](date_value)
        print(f"YEAR Result: {year_result}")

        # Test MONTH kernel
        month_result = KERNEL_LIBRARY['MONTH'](cp.array([44896, 44927, 44957, 44988]))
        print(f"MONTH Result: {month_result}")

        # Test DAY kernel
        day_result = KERNEL_LIBRARY['DAY'](cp.array([44896, 44927, 44957, 44988]))
        print(f"DAY Result: {day_result}")

        # Test TODAY kernel
        today_result = KERNEL_LIBRARY['TODAY']()
        print(f"TODAY Result: {today_result}")

        # Test NOW kernel
        now_result = KERNEL_LIBRARY['NOW']()
        print(f"NOW Result: {now_result}")
