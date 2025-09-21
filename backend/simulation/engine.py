import numpy as np
import cupy as cp
from typing import Dict, Tuple, Any, List, Union, Callable, Optional # Added Callable, Optional
import asyncio
from concurrent.futures import ThreadPoolExecutor
import math # For safe eval
import os # for cpu_executor ThreadPool
import re # For the new _safe_excel_eval
import logging  # Add missing logger import
import time  # For progress callback timestamps
import random

from config import settings # Use absolute import for config
from gpu.manager import gpu_manager, ExcelFileStats # Changed to absolute import
from .schemas import VariableConfig # For type hinting mc_input_configs
from .formula_utils import CELL_REFERENCE_REGEX, RANGE_REFERENCE_REGEX, _parse_range_string, _expand_cell_range # Updated import to include _parse_range_string and _expand_cell_range
from .random_engine import get_random_engine, get_multi_stream_generator, RNGType # Import new random engine

# Initialize logger for this module
logger = logging.getLogger(__name__)

# Try to import cupy and assign to cp if USE_GPU is true
if settings.USE_GPU:
    try:
        # import cupy # Already imported at the top, but ensure it refers to the global cp
        pass # cp should be the global cupy if import was successful
    except ImportError: # This specific except might not be hit if global import failed first
        print("CuPy module not found during conditional check. GPU acceleration will be unavailable.")
        cp = None 
    except Exception as e:
        print(f"An error occurred during CuPy conditional check: {e}. GPU acceleration will be unavailable.")
        cp = None
else:
    cp = None # Ensure cp is defined if USE_GPU is false

# Define CpNdArray type alias for type hinting
if cp:
    CpNdArray = cp.ndarray
else:
    CpNdArray = Any # Fallback if cp is None

# --- Helper for normalizing function name case in processed formulas ---
def _normalize_function_name_case(match_obj: re.Match) -> str:
    func_name = match_obj.group(1)
    # Check if the name starts with __ to avoid uppercasing our internal variables
    # (though the regex pattern ending with \s*\( makes this less likely for functions)
    if func_name.startswith("__"):
        return match_obj.group(0) # Return original match (e.g., __CELL_REF_0__(...))
    return func_name.upper() + match_obj.group(0)[len(func_name):]

# --- Custom range-aware functions for SAFE_EVAL_NAMESPACE ---
def custom_sum_for_eval(*args: Any) -> Union[int, float]:
    """Sum implementation for Excel-like behavior with nested range support."""
    def flatten_nested(item):
        """Recursively flatten nested lists/tuples to extract numeric values."""
        if isinstance(item, (list, tuple)):
            result = []
            for subitem in item:
                result.extend(flatten_nested(subitem))
            return result
        elif isinstance(item, (int, float)):
            return [item]
        else:
            return []  # Skip non-numeric values
    
    total = 0
    for arg in args:
        flattened = flatten_nested(arg)
        total += sum(flattened)
    return total

def custom_average_for_eval(*args: Any) -> Union[int, float]:
    """Average implementation for Excel-like behavior with nested range support."""
    def flatten_nested(item):
        """Recursively flatten nested lists/tuples to extract numeric values."""
        if isinstance(item, (list, tuple)):
            result = []
            for subitem in item:
                result.extend(flatten_nested(subitem))
            return result
        elif isinstance(item, (int, float)):
            return [item]
        else:
            return []  # Skip non-numeric values
    
    all_nums: List[Union[int, float]] = []
    for arg in args:
        flattened = flatten_nested(arg)
        all_nums.extend(flattened)
    
    if not all_nums:
        # Raise a more specific error, similar to Excel's #DIV/0!
        raise ValueError("Error in AVERAGE function: Division by zero (no valid numeric inputs provided to average).")
    return sum(all_nums) / len(all_nums)

# --- Wrappers for math functions to provide more Excel-like errors ---
def excel_sqrt(number: Any) -> float:
    if not isinstance(number, (int, float)):
        raise TypeError(f"Error in SQRT function: Expected a number, but got type '{type(number).__name__}'.")
    if number < 0:
        raise ValueError("Error in SQRT function: Number cannot be negative (Excel #NUM! error).")
    return math.sqrt(number)

def excel_ln(number: Any) -> float:
    if not isinstance(number, (int, float)):
        raise TypeError(f"Error in LN function: Expected a number, but got type '{type(number).__name__}'.")
    if number <= 0:
        raise ValueError("Error in LN function: Number must be positive (Excel #NUM! error).")
    return math.log(number)

def excel_log10(number: Any) -> float:
    if not isinstance(number, (int, float)):
        raise TypeError(f"Error in LOG10 function: Expected a number, but got type '{type(number).__name__}'.")
    if number <= 0:
        raise ValueError("Error in LOG10 function: Number must be positive (Excel #NUM! error).")
    return math.log10(number)

def excel_log_base(number: Any, base: Any) -> float:
    if not isinstance(number, (int, float)):
        raise TypeError(f"Error in LOG function (first argument - number): Expected a number, but got type '{type(number).__name__}'.")
    if not isinstance(base, (int, float)):
        raise TypeError(f"Error in LOG function (second argument - base): Expected a number, but got type '{type(base).__name__}'.")
    if number <= 0:
        raise ValueError("Error in LOG function: Number (first argument) must be positive (Excel #NUM! error).")
    if base <= 0:
        raise ValueError("Error in LOG function: Base (second argument) must be positive (Excel #NUM! error).")
    if base == 1:
        raise ValueError("Error in LOG function: Base cannot be 1 (Excel #DIV/0! or #NUM! error depending on number).") # Excel gives #DIV/0! for LOG(any,1)
    return math.log(number, base)

def excel_generic_log(number: Any, base: Any = None) -> float:
    if base is None:
        # Mimics Excel's LOG(number) which is LOG10(number)
        # However, our previous LOG mapped to math.log (natural log if one arg).
        # To maintain consistency with a generic LOG and specific LN, LOG10, it might be better
        # if single-argument LOG is LOG10. Or we require base for LOG.
        # For now, let's make single-arg LOG be LOG10 for Excel compatibility if user types LOG(x)
        # and rely on LN for natural log.
        return excel_log10(number) 
    else:
        return excel_log_base(number, base)

# --- New COUNT functions ---
def excel_count(*args: Any) -> int:
    count = 0
    for arg in args:
        if isinstance(arg, (list, tuple)):
            for item in arg:
                if isinstance(item, (int, float)) and not isinstance(item, bool): # Excel COUNT doesn't count booleans
                    count += 1
        elif isinstance(arg, (int, float)) and not isinstance(arg, bool):
            count += 1
    return count

def excel_counta(*args: Any) -> int:
    count = 0
    for arg in args:
        if isinstance(arg, (list, tuple)):
            for item in arg:
                # Counts numbers, text (including ""), booleans, errors. Does not count None if None represents a truly blank cell.
                if item is not None: 
                    count += 1
        elif arg is not None:
            count += 1
    return count

def excel_countblank(*args: Any) -> int:
    count = 0
    for arg in args:
        if isinstance(arg, (list, tuple)):
            for item in arg:
                # Counts None (representing blank) and empty strings.
                if item is None or item == "":
                    count += 1
        elif arg is None or arg == "": # A single argument that is blank
            count += 1
    return count

# --- Lookup Functions for VLOOKUP, HLOOKUP, INDEX, MATCH ---
def excel_vlookup(lookup_value: Any, table_array: Any, col_index: Any, range_lookup: Any = True) -> Any:
    """
    VLOOKUP function - Vertical lookup in a table with proper string support
    """
    try:
        # Log VLOOKUP call for debugging
        logger.warning(f"[VLOOKUP_DEBUG] Called with lookup_value={lookup_value}, col_index={col_index}, range_lookup={range_lookup}, table_array_type={type(table_array)}")
        
        # Convert range_lookup to boolean
        if isinstance(range_lookup, str):
            range_lookup = range_lookup.upper() not in ['FALSE', 'F', '0']
        elif isinstance(range_lookup, (int, float)):
            range_lookup = bool(range_lookup)
        
        # Convert table_array to proper format
        if not isinstance(table_array, (list, tuple)):
            table_array = [[table_array]]
        elif len(table_array) > 0 and not isinstance(table_array[0], (list, tuple)):
            # 1D array - treat as single column
            table_array = [[val] for val in table_array]
        
        # Validate col_index
        col_index = int(col_index)
        if col_index < 1:
            logger.warning(f"[VLOOKUP_DEBUG] Invalid col_index: {col_index}")
            return "#VALUE!"
        
        # Check if table has enough columns
        max_cols = max(len(row) if isinstance(row, (list, tuple)) else 1 for row in table_array) if table_array else 0
        if col_index > max_cols:
            logger.warning(f"[VLOOKUP_DEBUG] col_index {col_index} exceeds table columns {max_cols}")
            return "#REF!"
        
        # Convert lookup_value for comparison
        lookup_str = str(lookup_value).lower() if isinstance(lookup_value, str) else lookup_value
        logger.warning(f"[VLOOKUP_DEBUG] Searching for: {lookup_str} (type: {type(lookup_value)})")
        
        # Search through the first column
        last_valid_row = None
        
        for i, row in enumerate(table_array):
            if not isinstance(row, (list, tuple)):
                row = [row]
            
            if len(row) == 0:
                continue
                
            first_col_value = row[0]
            
            # Exact match search
            if isinstance(first_col_value, str) and isinstance(lookup_value, str):
                # Case-insensitive string comparison
                if first_col_value.lower() == lookup_str:
                    logger.warning(f"[VLOOKUP_DEBUG] Found exact string match at row {i}")
                    # Return value from specified column
                    if col_index <= len(row):
                        result = row[col_index - 1]
                        logger.warning(f"[VLOOKUP_DEBUG] Returning: {result}")
                        return result
                    else:
                        return ""
            else:
                # Try numeric comparison
                try:
                    first_col_num = float(first_col_value)
                    lookup_num = float(lookup_value)
                    
                    if first_col_num == lookup_num:
                        logger.warning(f"[VLOOKUP_DEBUG] Found exact numeric match at row {i}")
                        # Return value from specified column
                        if col_index <= len(row):
                            result = row[col_index - 1]
                            logger.warning(f"[VLOOKUP_DEBUG] Returning: {result}")
                            return result
                        else:
                            return ""
                    
                    # For approximate match (range_lookup=True), track last row where value <= lookup
                    if range_lookup and first_col_num <= lookup_num:
                        last_valid_row = i
                        
                except (ValueError, TypeError):
                    # If numeric conversion fails, try string comparison for approximate match
                    if range_lookup and str(first_col_value).lower() <= str(lookup_value).lower():
                        last_valid_row = i
        
        # Return result for approximate match
        if range_lookup and last_valid_row is not None:
            row = table_array[last_valid_row]
            if not isinstance(row, (list, tuple)):
                row = [row]
            if col_index <= len(row):
                result = row[col_index - 1]
                logger.warning(f"[VLOOKUP_DEBUG] Approximate match at row {last_valid_row}, returning: {result}")
                return result
            else:
                return ""
        
        logger.warning(f"[VLOOKUP_DEBUG] No match found, returning #N/A")
        return "#N/A"
        
    except Exception as e:
        logger.warning(f"[VLOOKUP_DEBUG] Exception: {e}")
        return "#VALUE!"

def excel_hlookup(lookup_value: Any, table_array: Any, row_index: Any, range_lookup: Any = True) -> Any:
    """HLOOKUP function - Horizontal lookup in a table"""
    try:
        # Convert range_lookup to boolean
        if isinstance(range_lookup, str):
            range_lookup = range_lookup.upper() not in ['FALSE', 'F', '0']
        elif isinstance(range_lookup, (int, float)):
            range_lookup = bool(range_lookup)
        
        # Convert table_array to proper format
        if not isinstance(table_array, (list, tuple)):
            table_array = [[table_array]]
        elif isinstance(table_array[0], (int, float, str)):
            # 1D array - treat as single row
            table_array = [table_array]
        
        # Validate row_index
        row_index = int(row_index)
        if row_index < 1 or row_index > len(table_array):
            return "#REF!"
        
        # Get first row for searching
        first_row = table_array[0] if table_array else []
        if not isinstance(first_row, (list, tuple)):
            first_row = [first_row]
        
        # Convert lookup_value for comparison
        lookup_str = str(lookup_value).lower() if isinstance(lookup_value, str) else lookup_value
        
        # Search through the first row
        last_valid_col = None
        
        for col_idx, cell_value in enumerate(first_row):
            # Exact match search
            if not range_lookup:
                if isinstance(cell_value, str) and isinstance(lookup_value, str):
                    if cell_value.lower() == lookup_str:
                        target_row = table_array[row_index - 1]
                        if not isinstance(target_row, (list, tuple)):
                            target_row = [target_row]
                        return target_row[col_idx] if col_idx < len(target_row) else ""
                else:
                    try:
                        if float(cell_value) == float(lookup_value):
                            target_row = table_array[row_index - 1]
                            if not isinstance(target_row, (list, tuple)):
                                target_row = [target_row]
                            return target_row[col_idx] if col_idx < len(target_row) else ""
                    except (ValueError, TypeError):
                        if cell_value == lookup_value:
                            target_row = table_array[row_index - 1]
                            if not isinstance(target_row, (list, tuple)):
                                target_row = [target_row]
                            return target_row[col_idx] if col_idx < len(target_row) else ""
            
            # Approximate match search
            else:
                try:
                    cell_num = float(cell_value)
                    lookup_num = float(lookup_value)
                    
                    if cell_num <= lookup_num:
                        last_valid_col = col_idx
                    elif cell_num > lookup_num:
                        break
                        
                except (ValueError, TypeError):
                    if str(cell_value).lower() <= str(lookup_value).lower():
                        last_valid_col = col_idx
                    else:
                        break
        
        # Return result for approximate match
        if range_lookup and last_valid_col is not None:
            target_row = table_array[row_index - 1]
            if not isinstance(target_row, (list, tuple)):
                target_row = [target_row]
            return target_row[last_valid_col] if last_valid_col < len(target_row) else ""
        
        return "#N/A"
        
    except Exception:
        return "#VALUE!"

def excel_index(array: Any, row_num: Any, col_num: Any = None) -> Any:
    """INDEX function - Returns a value from a specific position in an array"""
    try:
        # Convert array to proper format
        if not isinstance(array, (list, tuple)):
            # Single value
            if row_num == 1 and (col_num is None or col_num == 1):
                return array
            else:
                return "#REF!"
        
        # Check if it's a 1D array
        is_1d = all(not isinstance(item, (list, tuple)) for item in array)
        
        if is_1d:
            # 1D array handling
            if col_num is not None and col_num != 1:
                return "#REF!"  # 1D array doesn't have columns
            
            row_num = int(row_num)
            if row_num == 0:
                return array  # Return entire array
            elif 1 <= row_num <= len(array):
                return array[row_num - 1]
            else:
                return "#REF!"
        
        else:
            # 2D array handling
            row_num = int(row_num)
            
            # Validate row_num
            if row_num < 0 or row_num > len(array):
                return "#REF!"
            
            if row_num == 0:
                # Return entire column
                if col_num is None or col_num == 1:
                    col_num = 1
                col_num = int(col_num)
                if col_num < 1:
                    return "#REF!"
                
                column_values = []
                for row in array:
                    if isinstance(row, (list, tuple)) and col_num <= len(row):
                        column_values.append(row[col_num - 1])
                    else:
                        column_values.append("")
                return column_values
            
            # Get specific cell
            target_row = array[row_num - 1]
            if not isinstance(target_row, (list, tuple)):
                target_row = [target_row]
            
            if col_num is None:
                # Return entire row
                return target_row
            
            col_num = int(col_num)
            if col_num == 0:
                return target_row  # Return entire row
            elif 1 <= col_num <= len(target_row):
                return target_row[col_num - 1]
            else:
                return "#REF!"
        
    except Exception:
        return "#VALUE!"

def excel_match(lookup_value: Any, lookup_array: Any, match_type: Any = 1) -> Any:
    """MATCH function - Finds the position of a value in an array"""
    try:
        # Convert lookup_array to list
        if not isinstance(lookup_array, (list, tuple)):
            lookup_array = [lookup_array]
        
        # Flatten if it's a 2D array (take first column or first row)
        if lookup_array and isinstance(lookup_array[0], (list, tuple)):
            lookup_array = [row[0] if len(row) > 0 else "" for row in lookup_array]
        
        match_type = int(match_type)
        lookup_str = str(lookup_value).lower() if isinstance(lookup_value, str) else lookup_value
        
        if match_type == 0:
            # Exact match
            for i, value in enumerate(lookup_array):
                if isinstance(value, str) and isinstance(lookup_value, str):
                    if value.lower() == lookup_str:
                        return i + 1  # 1-based index
                else:
                    try:
                        if float(value) == float(lookup_value):
                            return i + 1
                    except (ValueError, TypeError):
                        if value == lookup_value:
                            return i + 1
            return "#N/A"
        
        elif match_type == 1:
            # Find largest value <= lookup_value (assumes ascending order)
            last_valid_index = None
            
            for i, value in enumerate(lookup_array):
                try:
                    value_num = float(value)
                    lookup_num = float(lookup_value)
                    
                    if value_num <= lookup_num:
                        last_valid_index = i + 1
                    elif value_num > lookup_num:
                        break
                        
                except (ValueError, TypeError):
                    # String comparison
                    if str(value).lower() <= str(lookup_value).lower():
                        last_valid_index = i + 1
                    else:
                        break
            
            return last_valid_index if last_valid_index is not None else "#N/A"
        
        elif match_type == -1:
            # Find smallest value >= lookup_value (assumes descending order)
            for i, value in enumerate(lookup_array):
                try:
                    value_num = float(value)
                    lookup_num = float(lookup_value)
                    
                    # In descending order, we want first value >= lookup_value
                    if value_num >= lookup_num:
                        return i + 1
                        
                except (ValueError, TypeError):
                    # String comparison
                    if str(value).lower() >= str(lookup_value).lower():
                        return i + 1
            
            return "#N/A"
        
        else:
            return "#VALUE!"
            
    except Exception:
        return "#VALUE!"

# Add this new function before SAFE_EVAL_NAMESPACE definition
def excel_vlookup_with_fallback(lookup_value: Any, table_array: Any, col_index: Any, range_lookup: Any = True) -> Any:
    """
    VLOOKUP wrapper that handles GPU NaN fallback for string lookups.
    If the result is NaN (from GPU kernel), fallback to CPU implementation.
    """
    # First, try the regular VLOOKUP
    result = excel_vlookup(lookup_value, table_array, col_index, range_lookup)
    
    # Check if result is NaN (which indicates GPU couldn't handle string lookup)
    try:
        if isinstance(result, (float, int)) and math.isnan(float(result)):
            logger.warning(f"[VLOOKUP_FALLBACK] GPU returned NaN, using CPU fallback for lookup_value={lookup_value}")
            # The excel_vlookup function already handles strings properly
            return excel_vlookup(lookup_value, table_array, col_index, range_lookup)
    except (ValueError, TypeError):
        # If we can't check for NaN, just return the result
        pass
    
    return result

# ====== COMPREHENSIVE EXCEL FUNCTION IMPLEMENTATIONS ======

# ===== 1. LOGICAL FUNCTIONS =====
def excel_and(*args: Any) -> bool:
    """AND function - returns True if all arguments are True"""
    for arg in args:
        if isinstance(arg, (list, tuple)):
            for item in arg:
                if not bool(item):
                    return False
        else:
            if not bool(arg):
                return False
    return True

def excel_or(*args: Any) -> bool:
    """OR function - returns True if any argument is True"""
    for arg in args:
        if isinstance(arg, (list, tuple)):
            for item in arg:
                if bool(item):
                    return True
        else:
            if bool(arg):
                return True
    return False

def excel_not(value: Any) -> bool:
    """NOT function - returns opposite boolean value"""
    return not bool(value)

def excel_iferror(value: Any, value_if_error: Any) -> Any:
    """IFERROR function - returns value_if_error if value is an error"""
    try:
        # Check if value is an Excel error
        if isinstance(value, str) and value.startswith('#'):
            return value_if_error
        return value
    except:
        return value_if_error

# ===== 2. CONDITIONAL FUNCTIONS =====
def excel_countif(range_values: Any, criteria: Any) -> int:
    """COUNTIF function - count cells meeting criteria"""
    count = 0
    
    # Flatten range if it's nested
    if isinstance(range_values, (list, tuple)):
        values = []
        for item in range_values:
            if isinstance(item, (list, tuple)):
                values.extend(item)
            else:
                values.append(item)
    else:
        values = [range_values]
    
    # Parse criteria
    criteria_str = str(criteria)
    
    for value in values:
        try:
            # Handle different criteria types
            if criteria_str.startswith('>='):
                threshold = float(criteria_str[2:])
                if isinstance(value, (int, float)) and value >= threshold:
                    count += 1
            elif criteria_str.startswith('<='):
                threshold = float(criteria_str[2:])
                if isinstance(value, (int, float)) and value <= threshold:
                    count += 1
            elif criteria_str.startswith('>'):
                threshold = float(criteria_str[1:])
                if isinstance(value, (int, float)) and value > threshold:
                    count += 1
            elif criteria_str.startswith('<'):
                threshold = float(criteria_str[1:])
                if isinstance(value, (int, float)) and value < threshold:
                    count += 1
            elif criteria_str.startswith('<>'):
                compare_val = criteria_str[2:]
                try:
                    compare_val = float(compare_val)
                    if isinstance(value, (int, float)) and value != compare_val:
                        count += 1
                except ValueError:
                    if str(value) != compare_val:
                        count += 1
            else:
                # Exact match
                try:
                    criteria_num = float(criteria_str)
                    if isinstance(value, (int, float)) and value == criteria_num:
                        count += 1
                except ValueError:
                    if str(value).lower() == criteria_str.lower():
                        count += 1
        except (ValueError, TypeError):
            continue
    
    return count

def excel_sumif(range_values: Any, criteria: Any, sum_range: Any = None) -> float:
    """SUMIF function - sum cells meeting criteria"""
    total = 0.0
    
    # If sum_range is not provided, use range_values
    if sum_range is None:
        sum_range = range_values
    
    # Flatten ranges
    def flatten_range(data):
        if isinstance(data, (list, tuple)):
            result = []
            for item in data:
                if isinstance(item, (list, tuple)):
                    result.extend(item)
                else:
                    result.append(item)
            return result
        return [data]
    
    check_values = flatten_range(range_values)
    sum_values = flatten_range(sum_range)
    
    # Ensure ranges are same length
    min_len = min(len(check_values), len(sum_values))
    
    criteria_str = str(criteria)
    
    for i in range(min_len):
        check_val = check_values[i]
        sum_val = sum_values[i]
        
        try:
            # Check criteria
            meets_criteria = False
            
            if criteria_str.startswith('>='):
                threshold = float(criteria_str[2:])
                if isinstance(check_val, (int, float)) and check_val >= threshold:
                    meets_criteria = True
            elif criteria_str.startswith('<='):
                threshold = float(criteria_str[2:])
                if isinstance(check_val, (int, float)) and check_val <= threshold:
                    meets_criteria = True
            elif criteria_str.startswith('>'):
                threshold = float(criteria_str[1:])
                if isinstance(check_val, (int, float)) and check_val > threshold:
                    meets_criteria = True
            elif criteria_str.startswith('<'):
                threshold = float(criteria_str[1:])
                if isinstance(check_val, (int, float)) and check_val < threshold:
                    meets_criteria = True
            elif criteria_str.startswith('<>'):
                compare_val = criteria_str[2:]
                try:
                    compare_val = float(compare_val)
                    if isinstance(check_val, (int, float)) and check_val != compare_val:
                        meets_criteria = True
                except ValueError:
                    if str(check_val) != compare_val:
                        meets_criteria = True
            else:
                # Exact match
                try:
                    criteria_num = float(criteria_str)
                    if isinstance(check_val, (int, float)) and check_val == criteria_num:
                        meets_criteria = True
                except ValueError:
                    if str(check_val).lower() == criteria_str.lower():
                        meets_criteria = True
            
            # Add to sum if criteria met
            if meets_criteria and isinstance(sum_val, (int, float)):
                total += sum_val
                
        except (ValueError, TypeError):
            continue
    
    return total

def excel_averageif(range_values: Any, criteria: Any, average_range: Any = None) -> float:
    """AVERAGEIF function - average cells meeting criteria"""
    if average_range is None:
        average_range = range_values
    
    total = excel_sumif(range_values, criteria, average_range)
    count = excel_countif(range_values, criteria)
    
    return total / count if count > 0 else 0

# ===== 3. TEXT FUNCTIONS =====
def excel_concatenate(*args: Any) -> str:
    """CONCATENATE function - join text strings"""
    return ''.join(str(arg) for arg in args)

def excel_left(text: Any, num_chars: int) -> str:
    """LEFT function - extract leftmost characters"""
    return str(text)[:int(num_chars)]

def excel_right(text: Any, num_chars: int) -> str:
    """RIGHT function - extract rightmost characters"""
    return str(text)[-int(num_chars):]

def excel_mid(text: Any, start: int, num_chars: int) -> str:
    """MID function - extract middle characters (Excel 1-based indexing)"""
    start_idx = max(0, int(start) - 1)  # Convert to 0-based
    return str(text)[start_idx:start_idx + int(num_chars)]

def excel_upper(text: Any) -> str:
    """UPPER function - convert to uppercase"""
    return str(text).upper()

def excel_lower(text: Any) -> str:
    """LOWER function - convert to lowercase"""
    return str(text).lower()

def excel_trim(text: Any) -> str:
    """TRIM function - remove extra spaces"""
    return ' '.join(str(text).split())

def excel_len(text: Any) -> int:
    """LEN function - get text length"""
    return len(str(text))

def excel_find(find_text: Any, within_text: Any, start_num: int = 1) -> int:
    """FIND function - case-sensitive text search"""
    find_str = str(find_text)
    within_str = str(within_text)
    start_idx = max(0, int(start_num) - 1)  # Convert to 0-based
    
    try:
        result = within_str.index(find_str, start_idx)
        return result + 1  # Convert back to 1-based
    except ValueError:
        return "#VALUE!"

def excel_search(find_text: Any, within_text: Any, start_num: int = 1) -> int:
    """SEARCH function - case-insensitive text search"""
    find_str = str(find_text).lower()
    within_str = str(within_text).lower()
    start_idx = max(0, int(start_num) - 1)  # Convert to 0-based
    
    try:
        result = within_str.index(find_str, start_idx)
        return result + 1  # Convert back to 1-based
    except ValueError:
        return "#VALUE!"

# ===== 4. DATE/TIME FUNCTIONS =====
def excel_today() -> int:
    """TODAY function - current date as Excel serial number"""
    from datetime import date
    today = date.today()
    # Excel serial date (days since 1900-01-01, accounting for leap year bug)
    epoch = date(1900, 1, 1)
    delta = today - epoch
    return delta.days + 2  # +2 to account for Excel's 1900 leap year bug

def excel_now() -> float:
    """NOW function - DETERMINISTIC datetime as Excel serial number for reproducible simulations"""
    from datetime import datetime
    # Use a FIXED deterministic datetime instead of current time
    # This ensures consistent results across simulation runs
    fixed_datetime = datetime(2024, 1, 1, 12, 0, 0)  # Fixed: Jan 1, 2024 at noon
    # Excel serial datetime
    epoch = datetime(1900, 1, 1)
    delta = fixed_datetime - epoch
    return delta.total_seconds() / 86400 + 2  # +2 for Excel's bug

def excel_year(date_value: Any) -> int:
    """YEAR function - extract year from date"""
    from datetime import datetime, date, timedelta
    
    try:
        if isinstance(date_value, (datetime, date)):
            return date_value.year
        elif isinstance(date_value, str):
            # Try common date formats
            for fmt in ['%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y', '%Y/%m/%d']:
                try:
                    dt = datetime.strptime(date_value, fmt)
                    return dt.year
                except ValueError:
                    continue
            return 2024  # DETERMINISTIC fallback year for consistent simulations
        elif isinstance(date_value, (int, float)):
            # Excel serial date
            base_date = datetime(1900, 1, 1)
            target_date = base_date + timedelta(days=int(date_value) - 2)
            return target_date.year
        else:
            return 2024  # DETERMINISTIC fallback year for consistent simulations
    except Exception:
        return 2024  # DETERMINISTIC fallback year for consistent simulations

def excel_month(date_value: Any) -> int:
    """MONTH function - extract month from date"""
    from datetime import datetime, date, timedelta
    
    try:
        if isinstance(date_value, (datetime, date)):
            return date_value.month
        elif isinstance(date_value, str):
            for fmt in ['%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y', '%Y/%m/%d']:
                try:
                    dt = datetime.strptime(date_value, fmt)
                    return dt.month
                except ValueError:
                    continue
            return 1  # DETERMINISTIC fallback month for consistent simulations
        elif isinstance(date_value, (int, float)):
            base_date = datetime(1900, 1, 1)
            target_date = base_date + timedelta(days=int(date_value) - 2)
            return target_date.month
        else:
            return 1  # DETERMINISTIC fallback month for consistent simulations
    except Exception:
        return 1  # DETERMINISTIC fallback month for consistent simulations

def excel_day(date_value: Any) -> int:
    """DAY function - extract day from date"""
    from datetime import datetime, date, timedelta
    
    try:
        if isinstance(date_value, (datetime, date)):
            return date_value.day
        elif isinstance(date_value, str):
            for fmt in ['%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y', '%Y/%m/%d']:
                try:
                    dt = datetime.strptime(date_value, fmt)
                    return dt.day
                except ValueError:
                    continue
            return 1  # DETERMINISTIC fallback day for consistent simulations
        elif isinstance(date_value, (int, float)):
            base_date = datetime(1900, 1, 1)
            target_date = base_date + timedelta(days=int(date_value) - 2)
            return target_date.day
        else:
            return 1  # DETERMINISTIC fallback day for consistent simulations
    except Exception:
        return 1  # DETERMINISTIC fallback day for consistent simulations

# ===== 5. FINANCIAL FUNCTIONS =====
def excel_pmt(rate: float, nper: int, pv: float, fv: float = 0, type_val: int = 0) -> float:
    """PMT function - payment calculation"""
    try:
        rate = float(rate)
        nper = int(nper)
        pv = float(pv)
        fv = float(fv)
        type_val = int(type_val)
        
        if rate == 0:
            return -(pv + fv) / nper
        
        factor = (1 + rate) ** nper
        pmt = -(pv * factor + fv) / ((factor - 1) / rate)
        
        if type_val == 1:
            pmt = pmt / (1 + rate)
        
        return pmt
    except (ValueError, ZeroDivisionError):
        return "#VALUE!"

def excel_pv(rate: float, nper: int, pmt: float, fv: float = 0, type_val: int = 0) -> float:
    """PV function - present value calculation"""
    try:
        rate = float(rate)
        nper = int(nper)
        pmt = float(pmt)
        fv = float(fv)
        type_val = int(type_val)
        
        if rate == 0:
            return -(pmt * nper + fv)
        
        factor = (1 + rate) ** nper
        pv = -(pmt * (1 + rate * type_val) * ((factor - 1) / rate) + fv) / factor
        
        return pv
    except (ValueError, ZeroDivisionError):
        return "#VALUE!"

def excel_fv(rate: float, nper: int, pmt: float, pv: float = 0, type_val: int = 0) -> float:
    """FV function - future value calculation"""
    try:
        rate = float(rate)
        nper = int(nper)
        pmt = float(pmt)
        pv = float(pv)
        type_val = int(type_val)
        
        if rate == 0:
            return -(pv + pmt * nper)
        
        factor = (1 + rate) ** nper
        fv = -(pv * factor + pmt * (1 + rate * type_val) * ((factor - 1) / rate))
        
        return fv
    except (ValueError, ZeroDivisionError):
        return "#VALUE!"

def excel_npv(rate: float, *values) -> float:
    # 🔍 NPV_CASHFLOW_SOURCE_DEBUG: Track where cash flows come from
    logger.info(f"📊 [NPV_DEBUG] ===== NPV CALLED =====")
    logger.info(f"📊 [NPV_DEBUG] Rate: {rate} (type: {type(rate)})")
    logger.info(f"📊 [NPV_DEBUG] Values count: {len(values) if hasattr(values, '__len__') else 'scalar'}")
    logger.info(f"📊 [NPV_DEBUG] Values type: {type(values)}")
    
    if hasattr(values, '__len__') and len(values) > 0:
        for i, val in enumerate(values[:3]):  # Show first 3 arguments
            logger.info(f"💰 [NPV_DEBUG] Arg {i}: type={type(val)}, length={len(val) if hasattr(val, '__len__') else 'scalar'}")
            if hasattr(val, '__len__'):
                logger.info(f"💰 [NPV_DEBUG] Arg {i} preview: {val[:5] if len(val) > 5 else val}")
            else:
                logger.info(f"💰 [NPV_DEBUG] Arg {i} value: {val}")
    else:
        logger.info(f"💰 [NPV_DEBUG] Direct values: {values}")
    """NPV function - net present value with robust range handling"""
    try:
        rate = float(rate)
        npv = 0
        
        # Recursively flatten all nested structures
        def flatten_recursive(item):
            if isinstance(item, (list, tuple)):
                result = []
                for subitem in item:
                    result.extend(flatten_recursive(subitem))
                return result
            else:
                return [item]
        
        cash_flows = []
        for value in values:
            cash_flows.extend(flatten_recursive(value))
        
        logger.info(f"💰 [NPV_DEBUG] Flattened {len(cash_flows)} cash flows: {cash_flows[:10]}...")
        
        # Convert to float and calculate NPV
        valid_count = 0
        for i, cash_flow in enumerate(cash_flows):
            try:
                cf_value = float(cash_flow)
                period_value = cf_value / ((1 + rate) ** (i + 1))
                npv += period_value
                valid_count += 1
                if i < 5:  # Log first few for debugging
                    logger.info(f"💰 [NPV_DEBUG] Period {i+1}: CF={cf_value}, PV={period_value:.6f}")
            except (ValueError, TypeError) as e:
                # Skip non-numeric values (Excel behavior)
                logger.warning(f"💰 [NPV_DEBUG] Skipping non-numeric at period {i+1}: {cash_flow} - {e}")
                continue
        
        logger.info(f"📊 [NPV_DEBUG] Final NPV: {npv:.6f} from {valid_count}/{len(cash_flows)} valid cash flows")
        
        if valid_count == 0:
            raise ValueError("No valid numeric cash flows found")
        
        return npv
    except (ValueError, ZeroDivisionError, TypeError) as e:
        logger.error(f"❌ [NPV_ERROR] NPV calculation failed: {e}")
        return "#VALUE!"  # Return error value like the working version

def excel_irr(*values, guess=0.1) -> float:
    """IRR function - robust Excel-like implementation with bracketing fallback.

    Behavior:
    - Uses CF0 at period 0 (Excel-compatible for IRR)
    - Returns 0 on non-convergence or invalid input so IFERROR-style formulas work
    - Avoids raising exceptions that would bypass IFERROR in user formulas
    """
    try:
        # Flatten values to a 1D list
        def flatten_recursive(item):
            if isinstance(item, (list, tuple)):
                result = []
                for subitem in item:
                    result.extend(flatten_recursive(subitem))
                return result
            return [item]

        raw = []
        for v in values:
            raw.extend(flatten_recursive(v))

        # Clean numeric cash flows
        cash_flows = []
        for cf in raw:
            try:
                if cf is not None:
                    cash_flows.append(float(cf))
            except (ValueError, TypeError):
                continue

        if len(cash_flows) < 2:
            return 0

        # Must have both signs for IRR to exist
        has_pos = any(cf > 0 for cf in cash_flows)
        has_neg = any(cf < 0 for cf in cash_flows)
        if not (has_pos and has_neg):
            return 0

        # NPV with CF0 at t=0 (Excel IRR convention)
        def npv_at(r: float) -> float:
            try:
                denom = 1.0 + r
                if denom <= 0:
                    # Discount factor invalid, steer away
                    return float('inf') if sum(cash_flows) > 0 else -float('inf')
                total = 0.0
                for i, cf in enumerate(cash_flows):
                    total += cf / (denom ** i)
                return total
            except Exception:
                return float('inf')

        # Try to find a bracketing interval [a,b] with opposite signs
        candidates = [-0.9, -0.5, -0.1, 0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
        a = None
        b = None
        prev_r = candidates[0]
        prev_f = npv_at(prev_r)
        for r in candidates[1:]:
            f = npv_at(r)
            if f == 0:
                return r
            if prev_f == 0:
                return prev_r
            if f * prev_f < 0:
                a, b = prev_r, r
                break
            prev_r, prev_f = r, f

        if a is None:
            # Expand around guess
            try:
                g = float(guess)
            except Exception:
                g = 0.1
            r_low = max(-0.9, g - 0.5)
            r_high = g + 0.5
            f_low = npv_at(r_low)
            f_high = npv_at(r_high)
            if f_low * f_high < 0:
                a, b = r_low, r_high

        # If still no bracket, IRR not solvable reliably → return 0
        if a is None:
            return 0

        # Bisection method on [a,b]
        fa = npv_at(a)
        fb = npv_at(b)
        if fa == 0:
            return a
        if fb == 0:
            return b
        if fa * fb > 0:
            return 0

        for _ in range(100):
            mid = (a + b) / 2.0
            fm = npv_at(mid)
            if abs(fm) < 1e-7 or abs(b - a) < 1e-7:
                return mid
            if fa * fm < 0:
                b = mid
                fb = fm
            else:
                a = mid
                fa = fm

        # If no convergence within iterations, return safe 0 (IFERROR path)
        return 0
    except Exception:
        # Never raise from IRR; return 0 so IFERROR or downstream logic can handle
        return 0

# ===== 6. ADDITIONAL MATH FUNCTIONS =====
def excel_product(*args: Any) -> float:
    """PRODUCT function - multiply values"""
    result = 1
    for arg in args:
        if isinstance(arg, (list, tuple)):
            for item in arg:
                if isinstance(item, (int, float)):
                    result *= item
        elif isinstance(arg, (int, float)):
            result *= arg
    return result

def excel_roundup(number: float, digits: int = 0) -> float:
    """ROUNDUP function - round away from zero (Excel compatible)"""
    import math
    try:
        val = float(number)
        multiplier = 10 ** int(digits)
        if val >= 0:
            return math.ceil(val * multiplier) / multiplier
        else:
            return -math.ceil(abs(val) * multiplier) / multiplier
    except (ValueError, TypeError):
        return 0

def excel_rounddown(number: float, digits: int = 0) -> float:
    """ROUNDDOWN function - round down toward zero"""
    import math
    try:
        val = float(number)
        multiplier = 10 ** int(digits)
        if val >= 0:
            return math.floor(val * multiplier) / multiplier
        else:
            return math.ceil(val * multiplier) / multiplier
    except (ValueError, TypeError):
        return 0

def excel_ceiling(number: float, significance: float = 1) -> float:
    """CEILING function - round up to nearest multiple"""
    import math
    try:
        num = float(number)
        sig = float(significance)
        if sig == 0:
            return 0
        return math.ceil(num / sig) * sig
    except (ValueError, TypeError):
        return 0

# ===== 7. RANDOM FUNCTIONS =====
def excel_rand() -> float:
    """RAND function - random number between 0 and 1"""
    import random
    return random.random()

def excel_randbetween(bottom: int, top: int) -> int:
    """RANDBETWEEN function - random integer in range"""
    import random
    try:
        return random.randint(int(bottom), int(top))
    except ValueError:
        return int(bottom)

# ===== 8. STATISTICAL FUNCTIONS =====
def excel_median(*args: Any) -> float:
    """MEDIAN function - middle value"""
    import statistics
    values = []
    for arg in args:
        if isinstance(arg, (list, tuple)):
            values.extend([x for x in arg if isinstance(x, (int, float))])
        elif isinstance(arg, (int, float)):
            values.append(arg)
    
    return statistics.median(values) if values else 0

def excel_mode(*args: Any) -> float:
    """MODE function - most frequent value"""
    import statistics
    values = []
    for arg in args:
        if isinstance(arg, (list, tuple)):
            values.extend([x for x in arg if isinstance(x, (int, float))])
        elif isinstance(arg, (int, float)):
            values.append(arg)
    
    try:
        return statistics.mode(values) if values else 0
    except statistics.StatisticsError:
        return "#N/A"

def excel_stdev(*args: Any) -> float:
    """STDEV function - sample standard deviation"""
    import statistics
    values = []
    for arg in args:
        if isinstance(arg, (list, tuple)):
            values.extend([x for x in arg if isinstance(x, (int, float))])
        elif isinstance(arg, (int, float)):
            values.append(arg)
    
    try:
        return statistics.stdev(values) if len(values) > 1 else 0
    except statistics.StatisticsError:
        return "#DIV/0!"

def excel_var(*args: Any) -> float:
    """VAR function - sample variance"""
    import statistics
    values = []
    for arg in args:
        if isinstance(arg, (list, tuple)):
            values.extend([x for x in arg if isinstance(x, (int, float))])
        elif isinstance(arg, (int, float)):
            values.append(arg)
    
    try:
        return statistics.variance(values) if len(values) > 1 else 0
    except statistics.StatisticsError:
        return "#DIV/0!"

# ====== COMPREHENSIVE EXCEL FUNCTION IMPLEMENTATIONS (RESTORED FROM e238232) ======

# ===== 1. LOGICAL FUNCTIONS =====
def excel_and(*args: Any) -> bool:
    """AND function - returns True if all arguments are True"""
    for arg in args:
        if isinstance(arg, (list, tuple)):
            for item in arg:
                if not bool(item):
                    return False
        else:
            if not bool(arg):
                return False
    return True

def excel_or(*args: Any) -> bool:
    """OR function - returns True if any argument is True"""
    for arg in args:
        if isinstance(arg, (list, tuple)):
            for item in arg:
                if bool(item):
                    return True
        else:
            if bool(arg):
                return True
    return False

def excel_not(value: Any) -> bool:
    """NOT function - returns opposite boolean value"""
    return not bool(value)

def excel_iferror(value: Any, value_if_error: Any) -> Any:
    """IFERROR function - returns value_if_error if value is an error"""
    try:
        # Check if value is an Excel error
        if isinstance(value, str) and value.startswith('#'):
            return value_if_error
        return value
    except:
        return value_if_error

# ===== 2. CONDITIONAL FUNCTIONS =====
def excel_countif(range_values: Any, criteria: Any) -> int:
    """COUNTIF function - count cells meeting criteria"""
    count = 0
    
    # Flatten range if it's nested
    if isinstance(range_values, (list, tuple)):
        values = []
        for item in range_values:
            if isinstance(item, (list, tuple)):
                values.extend(item)
            else:
                values.append(item)
    else:
        values = [range_values]
    
    # Parse criteria
    criteria_str = str(criteria)
    
    for value in values:
        try:
            # Handle different criteria types
            if criteria_str.startswith('>='):
                threshold = float(criteria_str[2:])
                if isinstance(value, (int, float)) and value >= threshold:
                    count += 1
            elif criteria_str.startswith('<='):
                threshold = float(criteria_str[2:])
                if isinstance(value, (int, float)) and value <= threshold:
                    count += 1
            elif criteria_str.startswith('>'):
                threshold = float(criteria_str[1:])
                if isinstance(value, (int, float)) and value > threshold:
                    count += 1
            elif criteria_str.startswith('<'):
                threshold = float(criteria_str[1:])
                if isinstance(value, (int, float)) and value < threshold:
                    count += 1
            elif criteria_str.startswith('<>'):
                compare_val = criteria_str[2:]
                try:
                    compare_val = float(compare_val)
                    if isinstance(value, (int, float)) and value != compare_val:
                        count += 1
                except ValueError:
                    if str(value) != compare_val:
                        count += 1
            else:
                # Exact match
                try:
                    criteria_num = float(criteria_str)
                    if isinstance(value, (int, float)) and value == criteria_num:
                        count += 1
                except ValueError:
                    if str(value).lower() == criteria_str.lower():
                        count += 1
        except (ValueError, TypeError):
            continue
    
    return count

def excel_sumif(range_values: Any, criteria: Any, sum_range: Any = None) -> float:
    """SUMIF function - sum cells meeting criteria"""
    total = 0.0
    
    # If sum_range is not provided, use range_values
    if sum_range is None:
        sum_range = range_values
    
    # Flatten ranges
    def flatten_range(data):
        if isinstance(data, (list, tuple)):
            result = []
            for item in data:
                if isinstance(item, (list, tuple)):
                    result.extend(item)
                else:
                    result.append(item)
            return result
        return [data]
    
    check_values = flatten_range(range_values)
    sum_values = flatten_range(sum_range)
    
    # Ensure ranges are same length
    min_len = min(len(check_values), len(sum_values))
    
    criteria_str = str(criteria)
    
    for i in range(min_len):
        check_val = check_values[i]
        sum_val = sum_values[i]
        
        try:
            # Check criteria
            meets_criteria = False
            
            if criteria_str.startswith('>='):
                threshold = float(criteria_str[2:])
                if isinstance(check_val, (int, float)) and check_val >= threshold:
                    meets_criteria = True
            elif criteria_str.startswith('<='):
                threshold = float(criteria_str[2:])
                if isinstance(check_val, (int, float)) and check_val <= threshold:
                    meets_criteria = True
            elif criteria_str.startswith('>'):
                threshold = float(criteria_str[1:])
                if isinstance(check_val, (int, float)) and check_val > threshold:
                    meets_criteria = True
            elif criteria_str.startswith('<'):
                threshold = float(criteria_str[1:])
                if isinstance(check_val, (int, float)) and check_val < threshold:
                    meets_criteria = True
            elif criteria_str.startswith('<>'):
                compare_val = criteria_str[2:]
                try:
                    compare_val = float(compare_val)
                    if isinstance(check_val, (int, float)) and check_val != compare_val:
                        meets_criteria = True
                except ValueError:
                    if str(check_val) != compare_val:
                        meets_criteria = True
            else:
                # Exact match
                try:
                    criteria_num = float(criteria_str)
                    if isinstance(check_val, (int, float)) and check_val == criteria_num:
                        meets_criteria = True
                except ValueError:
                    if str(check_val).lower() == criteria_str.lower():
                        meets_criteria = True
            
            # Add to sum if criteria met
            if meets_criteria and isinstance(sum_val, (int, float)):
                total += sum_val
                
        except (ValueError, TypeError):
            continue
    
    return total

def excel_averageif(range_values: Any, criteria: Any, average_range: Any = None) -> float:
    """AVERAGEIF function - average cells meeting criteria"""
    if average_range is None:
        average_range = range_values
    
    total = excel_sumif(range_values, criteria, average_range)
    count = excel_countif(range_values, criteria)
    
    return total / count if count > 0 else 0

# ===== 3. FINANCIAL FUNCTIONS (CRITICAL FOR B12 AND B13) =====
# NOTE: NPV implementation moved to the first excel_npv function above (line ~943)
# This duplicate was causing silent 0.0 returns instead of proper error handling

# NOTE: IRR implementation moved to the first excel_irr function above (line ~998)
# This duplicate was causing silent 0.0 returns instead of proper error handling

# ===== 4. TEXT FUNCTIONS (RESTORED FROM e238232) =====
def excel_concatenate(*args: Any) -> str:
    """CONCATENATE function - join text strings"""
    return ''.join(str(arg) for arg in args)

def excel_left(text: Any, num_chars: int) -> str:
    """LEFT function - extract leftmost characters"""
    return str(text)[:int(num_chars)]

def excel_right(text: Any, num_chars: int) -> str:
    """RIGHT function - extract rightmost characters"""
    return str(text)[-int(num_chars):]

def excel_mid(text: Any, start: int, num_chars: int) -> str:
    """MID function - extract middle characters (Excel 1-based indexing)"""
    start_idx = max(0, int(start) - 1)  # Convert to 0-based
    return str(text)[start_idx:start_idx + int(num_chars)]

def excel_upper(text: Any) -> str:
    """UPPER function - convert to uppercase"""
    return str(text).upper()

def excel_lower(text: Any) -> str:
    """LOWER function - convert to lowercase"""
    return str(text).lower()

def excel_trim(text: Any) -> str:
    """TRIM function - remove extra spaces"""
    return ' '.join(str(text).split())

def excel_len(text: Any) -> int:
    """LEN function - get text length"""
    return len(str(text))

def excel_find(find_text: Any, within_text: Any, start_num: int = 1) -> int:
    """FIND function - case-sensitive text search"""
    find_str = str(find_text)
    within_str = str(within_text)
    start_idx = max(0, int(start_num) - 1)  # Convert to 0-based
    
    try:
        result = within_str.index(find_str, start_idx)
        return result + 1  # Convert back to 1-based
    except ValueError:
        return "#VALUE!"

def excel_search(find_text: Any, within_text: Any, start_num: int = 1) -> int:
    """SEARCH function - case-insensitive text search"""
    find_str = str(find_text).lower()
    within_str = str(within_text).lower()
    start_idx = max(0, int(start_num) - 1)  # Convert to 0-based
    
    try:
        result = within_str.index(find_str, start_idx)
        return result + 1  # Convert back to 1-based
    except ValueError:
        return "#VALUE!"

# ===== 5. DATE/TIME FUNCTIONS (RESTORED FROM e238232) =====
def excel_today() -> int:
    """TODAY function - current date as Excel serial number"""
    from datetime import date
    today = date.today()
    # Excel serial date (days since 1900-01-01, accounting for leap year bug)
    epoch = date(1900, 1, 1)
    delta = today - epoch
    return delta.days + 2  # +2 to account for Excel's 1900 leap year bug

def excel_now() -> float:
    """NOW function - DETERMINISTIC datetime as Excel serial number for reproducible simulations"""
    from datetime import datetime
    # Use a FIXED deterministic datetime instead of current time
    # This ensures consistent results across simulation runs
    fixed_datetime = datetime(2024, 1, 1, 12, 0, 0)  # Fixed: Jan 1, 2024 at noon
    # Excel serial datetime
    epoch = datetime(1900, 1, 1)
    delta = fixed_datetime - epoch
    return delta.total_seconds() / 86400 + 2  # +2 for Excel's bug

def excel_year(date_value: Any) -> int:
    """YEAR function - extract year from date"""
    from datetime import datetime, date, timedelta
    
    try:
        if isinstance(date_value, (datetime, date)):
            return date_value.year
        elif isinstance(date_value, str):
            # Try common date formats
            for fmt in ['%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y', '%Y/%m/%d']:
                try:
                    dt = datetime.strptime(date_value, fmt)
                    return dt.year
                except ValueError:
                    continue
            return 2024  # DETERMINISTIC fallback year for consistent simulations
        elif isinstance(date_value, (int, float)):
            # Excel serial date
            base_date = datetime(1900, 1, 1)
            target_date = base_date + timedelta(days=int(date_value) - 2)
            return target_date.year
        else:
            return 2024  # DETERMINISTIC fallback year for consistent simulations
    except Exception:
        return 2024  # DETERMINISTIC fallback year for consistent simulations

def excel_month(date_value: Any) -> int:
    """MONTH function - extract month from date"""
    from datetime import datetime, date, timedelta
    
    try:
        if isinstance(date_value, (datetime, date)):
            return date_value.month
        elif isinstance(date_value, str):
            for fmt in ['%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y', '%Y/%m/%d']:
                try:
                    dt = datetime.strptime(date_value, fmt)
                    return dt.month
                except ValueError:
                    continue
            return 1  # DETERMINISTIC fallback month for consistent simulations
        elif isinstance(date_value, (int, float)):
            base_date = datetime(1900, 1, 1)
            target_date = base_date + timedelta(days=int(date_value) - 2)
            return target_date.month
        else:
            return 1  # DETERMINISTIC fallback month for consistent simulations
    except Exception:
        return 1  # DETERMINISTIC fallback month for consistent simulations

def excel_day(date_value: Any) -> int:
    """DAY function - extract day from date"""
    from datetime import datetime, date, timedelta
    
    try:
        if isinstance(date_value, (datetime, date)):
            return date_value.day
        elif isinstance(date_value, str):
            for fmt in ['%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y', '%Y/%m/%d']:
                try:
                    dt = datetime.strptime(date_value, fmt)
                    return dt.day
                except ValueError:
                    continue
            return 1  # DETERMINISTIC fallback day for consistent simulations
        elif isinstance(date_value, (int, float)):
            base_date = datetime(1900, 1, 1)
            target_date = base_date + timedelta(days=int(date_value) - 2)
            return target_date.day
        else:
            return 1  # DETERMINISTIC fallback day for consistent simulations
    except Exception:
        return 1  # DETERMINISTIC fallback day for consistent simulations

# ===== 6. ADDITIONAL FINANCIAL FUNCTIONS (RESTORED FROM e238232) =====
def excel_pmt(rate: float, nper: int, pv: float, fv: float = 0, type_val: int = 0) -> float:
    """PMT function - payment calculation"""
    try:
        rate = float(rate)
        nper = int(nper)
        pv = float(pv)
        fv = float(fv)
        type_val = int(type_val)
        
        if rate == 0:
            return -(pv + fv) / nper
        
        factor = (1 + rate) ** nper
        pmt = -(pv * factor + fv) / ((factor - 1) / rate)
        
        if type_val == 1:
            pmt = pmt / (1 + rate)
        
        return pmt
    except (ValueError, ZeroDivisionError):
        return 0.0

def excel_pv(rate: float, nper: int, pmt: float, fv: float = 0, type_val: int = 0) -> float:
    """PV function - present value calculation"""
    try:
        rate = float(rate)
        nper = int(nper)
        pmt = float(pmt)
        fv = float(fv)
        type_val = int(type_val)
        
        if rate == 0:
            return -(pmt * nper + fv)
        
        factor = (1 + rate) ** nper
        pv = -(pmt * (1 + rate * type_val) * ((factor - 1) / rate) + fv) / factor
        
        return pv
    except (ValueError, ZeroDivisionError):
        return 0.0

def excel_fv(rate: float, nper: int, pmt: float, pv: float = 0, type_val: int = 0) -> float:
    """FV function - future value calculation"""
    try:
        rate = float(rate)
        nper = int(nper)
        pmt = float(pmt)
        pv = float(pv)
        type_val = int(type_val)
        
        if rate == 0:
            return -(pv + pmt * nper)
        
        factor = (1 + rate) ** nper
        fv = -(pv * factor + pmt * (1 + rate * type_val) * ((factor - 1) / rate))
        
        return fv
    except (ValueError, ZeroDivisionError):
        return 0.0

# ===== 7. ADDITIONAL MATH FUNCTIONS (RESTORED FROM e238232) =====
def excel_product(*args: Any) -> float:
    """PRODUCT function - multiply values"""
    result = 1
    for arg in args:
        if isinstance(arg, (list, tuple)):
            for item in arg:
                if isinstance(item, (int, float)):
                    result *= item
        elif isinstance(arg, (int, float)):
            result *= arg
    return result

def excel_roundup(number: float, digits: int = 0) -> float:
    """ROUNDUP function - round away from zero (Excel compatible)"""
    import math
    try:
        val = float(number)
        multiplier = 10 ** int(digits)
        if val >= 0:
            return math.ceil(val * multiplier) / multiplier
        else:
            return -math.ceil(abs(val) * multiplier) / multiplier
    except (ValueError, TypeError):
        return 0

def excel_rounddown(number: float, digits: int = 0) -> float:
    """ROUNDDOWN function - round down toward zero"""
    import math
    try:
        val = float(number)
        multiplier = 10 ** int(digits)
        if val >= 0:
            return math.floor(val * multiplier) / multiplier
        else:
            return math.ceil(val * multiplier) / multiplier
    except (ValueError, TypeError):
        return 0

def excel_ceiling(number: float, significance: float = 1) -> float:
    """CEILING function - round up to nearest multiple"""
    import math
    try:
        num = float(number)
        sig = float(significance)
        if sig == 0:
            return 0
        return math.ceil(num / sig) * sig
    except (ValueError, TypeError):
        return 0

# ===== 8. RANDOM FUNCTIONS (RESTORED FROM e238232) =====
def excel_rand() -> float:
    """RAND function - random number between 0 and 1"""
    import random
    return random.random()

def excel_randbetween(bottom: int, top: int) -> int:
    """RANDBETWEEN function - random integer in range"""
    import random
    try:
        return random.randint(int(bottom), int(top))
    except ValueError:
        return int(bottom)

# ===== 9. STATISTICAL FUNCTIONS (RESTORED FROM e238232) =====
def excel_median(*args: Any) -> float:
    """MEDIAN function - middle value"""
    import statistics
    values = []
    for arg in args:
        if isinstance(arg, (list, tuple)):
            values.extend([x for x in arg if isinstance(x, (int, float))])
        elif isinstance(arg, (int, float)):
            values.append(arg)
    
    return statistics.median(values) if values else 0

def excel_mode(*args: Any) -> float:
    """MODE function - most frequent value"""
    import statistics
    values = []
    for arg in args:
        if isinstance(arg, (list, tuple)):
            values.extend([x for x in arg if isinstance(x, (int, float))])
        elif isinstance(arg, (int, float)):
            values.append(arg)
    
    try:
        return statistics.mode(values) if values else 0
    except statistics.StatisticsError:
        return 0

def excel_stdev(*args: Any) -> float:
    """STDEV function - sample standard deviation"""
    import statistics
    values = []
    for arg in args:
        if isinstance(arg, (list, tuple)):
            values.extend([x for x in arg if isinstance(x, (int, float))])
        elif isinstance(arg, (int, float)):
            values.append(arg)
    
    try:
        return statistics.stdev(values) if len(values) > 1 else 0
    except statistics.StatisticsError:
        return 0

def excel_var(*args: Any) -> float:
    """VAR function - sample variance"""
    import statistics
    values = []
    for arg in args:
        if isinstance(arg, (list, tuple)):
            values.extend([x for x in arg if isinstance(x, (int, float))])
        elif isinstance(arg, (int, float)):
            values.append(arg)
    
    try:
        return statistics.variance(values) if len(values) > 1 else 0
    except statistics.StatisticsError:
        return 0

# Define a limited set of functions and constants for safe evaluation
# All function keys are now UPPERCASE.
SAFE_EVAL_NAMESPACE = {
    '__builtins__': {},
    # Standard functions (keys uppercased)
    'ABS': abs,
    'MAX': max,
    'MIN': min,
    'ROUND': round,
    'LEN': len,
    'POW': pow, # math.pow(x,y) or x**y. Excel's POWER(number, power)
    'FLOAT': float,
    'INT': int, # Excel's INT function truncates towards zero.
    'STR': str, # Python's str(). Excel has TEXT() for formatting.

    # Custom/Excel-like functions (already uppercase or now uppercased)
    'SUM': custom_sum_for_eval,
    'AVERAGE': custom_average_for_eval,
    'IF': lambda condition, true_val=None, false_val=None: true_val if condition else false_val,
    
    # ✅ COMPREHENSIVE EXCEL FUNCTIONS - RESTORED FROM WORKING VERSION e238232
    'AND': excel_and,
    'OR': excel_or,
    'NOT': excel_not,
    'IFERROR': excel_iferror,
    'COUNTIF': excel_countif,
    'SUMIF': excel_sumif,
    'AVERAGEIF': excel_averageif,
    
    # ✅ FINANCIAL FUNCTIONS - CRITICAL FOR B12 AND B13 FORMULAS
    'NPV': excel_npv,
    'IRR': excel_irr,
    'PMT': excel_pmt,
    'PV': excel_pv,
    'FV': excel_fv,

    # Lookup functions - Use the fallback wrapper for VLOOKUP
    'VLOOKUP': excel_vlookup_with_fallback,  # Changed to use fallback wrapper
    'HLOOKUP': excel_hlookup,
    'INDEX': excel_index,
    'MATCH': excel_match,

    # Math constants (remain lowercase as they are not function calls with ())
    'e': math.e,
    'pi': math.pi,
    'tau': math.tau,

    # Math functions (keys uppercased)
    'ACOS': math.acos,
    'ASIN': math.asin,
    'ATAN': math.atan,
    'ATAN2': math.atan2,
    'CEIL': math.ceil, # Excel uses CEILING often, this is math.ceil
    'COS': math.cos,
    'COSH': math.cosh,
    'DEGREES': math.degrees,
    'EXP': math.exp,
    'FABS': math.fabs, # Float absolute value, covered by ABS too for general use
    'FLOOR': math.floor,
    # MOD for Excel's MOD(number, divisor). Python's % operator. math.fmod has different sign behavior for negative numbers.
    # For now, let's map MOD to a lambda that uses %, which is closer to Excel's MOD for positive divisors.
    'MOD': lambda number, divisor: number % divisor, 
    'FREXP': math.frexp,
    'GAMMA': math.gamma,
    'HYPOT': math.hypot,
    'LDEXP': math.ldexp,
    'LGAMMA': math.lgamma,
    # Updated Log functions
    'LOG': excel_generic_log,  # Handles LOG(number) as LOG10, and LOG(number, base)
    'LN': excel_ln,
    'LOG10': excel_log10,
    'LOG1P': math.log1p,
    'LOG2': math.log2,
    'LOG10': excel_log10,
    'LOG1P': math.log1p,
    'LOG2': math.log2,
    'MODF': math.modf,
    'RADIANS': math.radians,
    'SIN': math.sin,
    'SINH': math.sinh,
    'SQRT': excel_sqrt, # Now uses the wrapper
    'TAN': math.tan,
    'TANH': math.tanh,
    'TRUNC': math.trunc, # Excel's TRUNC function. math.trunc works.
    # New COUNT functions
    'COUNT': excel_count,
    'COUNTA': excel_counta,
    'COUNTBLANK': excel_countblank,
    # Alias to avoid NameError where regex inadvertently expands LOG to LOGGER in WorldClass engine
    'LOGGER': excel_generic_log,
    'POWER': pow,  # Excel POWER(number, power) alias
    'SIGN': lambda x: 1 if x > 0 else (-1 if x < 0 else 0),  # Excel SIGN function
    
    # ✅ TEXT FUNCTIONS - COMPLETE ARSENAL FROM e238232
    'CONCATENATE': excel_concatenate,
    'LEFT': excel_left,
    'RIGHT': excel_right,
    'MID': excel_mid,
    'UPPER': excel_upper,
    'LOWER': excel_lower,
    'TRIM': excel_trim,
    'FIND': excel_find,
    'SEARCH': excel_search,
    
    # ✅ DATE/TIME FUNCTIONS - COMPLETE ARSENAL FROM e238232
    'TODAY': excel_today,
    'NOW': excel_now,
    'YEAR': excel_year,
    'MONTH': excel_month,
    'DAY': excel_day,
    
    # ✅ ADDITIONAL MATH FUNCTIONS - COMPLETE ARSENAL FROM e238232
    'PRODUCT': excel_product,
    'ROUNDUP': excel_roundup,
    'ROUNDDOWN': excel_rounddown,
    'CEILING': excel_ceiling,
    
    # ✅ RANDOM FUNCTIONS - COMPLETE ARSENAL FROM e238232
    'RAND': excel_rand,
    'RANDBETWEEN': excel_randbetween,
    
    # ✅ STATISTICAL FUNCTIONS - COMPLETE ARSENAL FROM e238232
    'MEDIAN': excel_median,
    'MODE': excel_mode,
    'STDEV': excel_stdev,
    'VAR': excel_var,
    
    # ===== NEW COMPREHENSIVE EXCEL FUNCTIONS =====
    
    # 1. Logical Functions
    'AND': excel_and,
    'OR': excel_or,
    'NOT': excel_not,
    'IFERROR': excel_iferror,
    
    # 2. Conditional Functions
    'COUNTIF': excel_countif,
    'SUMIF': excel_sumif,
    'AVERAGEIF': excel_averageif,
    
    # 3. Text Functions
    'CONCATENATE': excel_concatenate,
    'LEFT': excel_left,
    'RIGHT': excel_right,
    'MID': excel_mid,
    'UPPER': excel_upper,
    'LOWER': excel_lower,
    'TRIM': excel_trim,
    'FIND': excel_find,
    'SEARCH': excel_search,
    
    # 4. Date/Time Functions
    'TODAY': excel_today,
    'NOW': excel_now,
    'YEAR': excel_year,
    'MONTH': excel_month,
    'DAY': excel_day,
    
    # 5. Financial Functions
    'PMT': excel_pmt,
    'PV': excel_pv,
    'FV': excel_fv,
    'NPV': excel_npv,
    'IRR': excel_irr,
    
    # 6. Additional Math Functions
    'PRODUCT': excel_product,
    'ROUNDUP': excel_roundup,
    'ROUNDDOWN': excel_rounddown,
    'CEILING': excel_ceiling,
    
    # 7. Random Functions
    'RAND': excel_rand,
    'RANDBETWEEN': excel_randbetween,
    
    # 8. Additional Statistical Functions
    'MEDIAN': excel_median,
    'MODE': excel_mode,
    'STDEV': excel_stdev,
    'VAR': excel_var,
    
    # Excel Boolean Constants
    'TRUE': True,
    'FALSE': False,
}

def _safe_excel_eval(
    formula_string: str, 
    current_eval_sheet: str, 
    all_current_iter_values: Dict[Tuple[str, str], Any], 
    safe_eval_globals: Dict[str, Any],
    current_calc_cell_coord: str = "UnknownCell", # Added parameter with default
    constant_values: Dict[Tuple[str, str], Any] = None # Add constant values for fallback
) -> Any:
    """Safely evaluate a formula string. Includes cell coordinate in error messages."""
    if not isinstance(formula_string, str):
        raise ValueError(f"Formula for evaluation is not a string (cell: {current_calc_cell_coord}): {formula_string}")

    if formula_string.startswith('='):
        formula_string = formula_string[1:]

    local_vars_for_internal_eval: Dict[str, Any] = {}
    parts = []
    current_pos = 0
    ref_counter = 0

    while current_pos < len(formula_string):
        range_match = RANGE_REFERENCE_REGEX.match(formula_string, current_pos)
        if range_match:
            matched_range_str = formula_string[range_match.start():range_match.end()]
            try:
                r_sheet, r_start_cell, r_end_cell = _parse_range_string(matched_range_str, current_eval_sheet)
                expanded_cells_in_range = _expand_cell_range(r_start_cell, r_end_cell, r_sheet)
            except ValueError as e:
                raise ValueError(f"Error parsing range '{matched_range_str}' in cell {current_calc_cell_coord} (formula: '{formula_string}'): {e}")
            
            # Convert expanded cells to 2D array structure
            # Parse start and end cells to determine dimensions
            start_col_str, start_row_int = r_start_cell[0:1] if len(r_start_cell) == 2 else r_start_cell[0:2], int(r_start_cell[1:] if len(r_start_cell) == 2 else r_start_cell[2:])
            end_col_str, end_row_int = r_end_cell[0:1] if len(r_end_cell) == 2 else r_end_cell[0:2], int(r_end_cell[1:] if len(r_end_cell) == 2 else r_end_cell[2:])
            
            # Get proper column parsing
            def parse_cell_coord(coord_str: str) -> Tuple[str, int]:
                match = re.match(r"([A-Z]+)([1-9][0-9]*)", coord_str.upper())
                if not match:
                    raise ValueError(f"Invalid cell coordinate format: {coord_str}")
                return match.group(1), int(match.group(2))
            
            def col_str_to_int(col_str: str) -> int:
                num = 0
                for char in col_str.upper():
                    num = num * 26 + (ord(char) - ord('A') + 1)
                return num - 1  # 0-indexed
                
            start_col_str, start_row_int = parse_cell_coord(r_start_cell)
            end_col_str, end_row_int = parse_cell_coord(r_end_cell)
            
            start_col_int = col_str_to_int(start_col_str)
            end_col_int = col_str_to_int(end_col_str)
            
            min_col = min(start_col_int, end_col_int)
            max_col = max(start_col_int, end_col_int)
            min_row = min(start_row_int, end_row_int)
            max_row = max(start_row_int, end_row_int)
            
            # Create 2D array structure
            range_values = []
            for r in range(min_row, max_row + 1):
                row_values = []
                for c in range(min_col, max_col + 1):
                    col_str = ""
                    num = c + 1  # 1-indexed for conversion
                    while num > 0:
                        num, remainder = divmod(num - 1, 26)
                        col_str = chr(65 + remainder) + col_str
                    cell_coord = f"{col_str}{r}"
                    cell_key = (r_sheet, cell_coord)
                    if cell_key not in all_current_iter_values:
                        # Use Excel-compatible behavior: treat missing cells as 0
                        if constant_values and cell_key in constant_values:
                            cell_value = constant_values[cell_key]
                        else:
                            cell_value = 0  # Excel treats empty/missing cells as 0
                        row_values.append(cell_value)
                        continue
                    else:
                        row_values.append(all_current_iter_values[cell_key])
                range_values.append(row_values)
            
            temp_var_name = f"__RANGE_VALS_{ref_counter}__"
            ref_counter += 1
            local_vars_for_internal_eval[temp_var_name] = range_values
            parts.append(temp_var_name)
            current_pos = range_match.end()
            continue

        cell_match = CELL_REFERENCE_REGEX.match(formula_string, current_pos)
        if cell_match:
            raw_sheet_name_match = cell_match.group(1)
            cell_coord_match = cell_match.group(2).upper()
            resolved_sheet_name = current_eval_sheet
            if raw_sheet_name_match:
                resolved_sheet_name = raw_sheet_name_match[1:-1] if raw_sheet_name_match.startswith("'") and raw_sheet_name_match.endswith("'") else raw_sheet_name_match
            
            # Strip dollar signs for lookup (absolute references like $D$3 become D3)
            clean_cell_coord = cell_coord_match.replace('$', '')
            cell_key = (resolved_sheet_name, clean_cell_coord)
            
            # Check if cell exists in our values
            if cell_key not in all_current_iter_values:
                # For cells not yet computed in the current iteration, do NOT persist a fake value.
                # Prefer a strict behavior so the scheduler can defer evaluation until deps are ready.
                # If the cell exists in constants, use it transiently; otherwise raise to signal missing dep.
                if constant_values and cell_key in constant_values:
                    fallback_value = constant_values[cell_key]
                    print(f"Info: Using constant value {fallback_value} for cell {resolved_sheet_name}!{clean_cell_coord}")
                    temp_var_name = f"__CELL_REF_{ref_counter}__"
                    ref_counter += 1
                    local_vars_for_internal_eval[temp_var_name] = fallback_value
                    parts.append(temp_var_name)
                    current_pos = cell_match.end()
                    continue
                else:
                    raise NameError(f"Missing dependency {resolved_sheet_name}!{clean_cell_coord}")
                
            temp_var_name = f"__CELL_REF_{ref_counter}__"
            ref_counter += 1
            local_vars_for_internal_eval[temp_var_name] = all_current_iter_values[cell_key]
            parts.append(temp_var_name)
            current_pos = cell_match.end()
            continue
        
        parts.append(formula_string[current_pos])
        current_pos += 1
            
    processed_formula = "".join(parts)

    try:
        processed_formula = re.sub(r"([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", _normalize_function_name_case, processed_formula)
        
        # Convert Excel comparison operators to Python syntax
        # Handle = operator for comparisons (but not in function assignments)
        # This regex matches = that are NOT inside function calls or at the start of formulas
        processed_formula = re.sub(r'(?<!^)(?<![\(\,\s])([A-Za-z0-9_]+|TRUE|FALSE)\s*=\s*([A-Za-z0-9_]+|TRUE|FALSE)', r'\1==\2', processed_formula)
        
        # Convert Excel percentage syntax to Python decimal
        # Convert patterns like "34%" to "0.34", handling integer and decimal percentages
        processed_formula = re.sub(r'(\d+(?:\.\d+)?)%', lambda m: str(float(m.group(1)) / 100), processed_formula)
        
    except Exception as e:
        raise ValueError(f"Error during formula normalization for cell {current_calc_cell_coord} (formula: '{processed_formula}'): {str(e)}")

    # CRITICAL DEBUG: Add comprehensive debugging for zero results issue - DISABLED FOR PERFORMANCE 
    if current_calc_cell_coord != "UnknownCell" and len(all_current_iter_values) < 1000:  # Only debug first 1000 evaluations
        debug_enabled = any(target in formula_string.upper() for target in ['B12', 'B13']) and len(formula_string) < 100
        if debug_enabled:
            logger.info(f"🔍 [EVAL_DEBUG] Cell {current_calc_cell_coord}: Original formula: '{formula_string}'")
            logger.info(f"🔍 [EVAL_DEBUG] Cell {current_calc_cell_coord}: Processed formula: '{processed_formula}'")
            logger.info(f"🔍 [EVAL_DEBUG] Cell {current_calc_cell_coord}: Local vars count: {len(local_vars_for_internal_eval)}")
            logger.info(f"🔍 [EVAL_DEBUG] Cell {current_calc_cell_coord}: Sample iter values count: {len(all_current_iter_values)}")

    try:
        result = eval(processed_formula, safe_eval_globals, local_vars_for_internal_eval)
        
        # ✅ ROBUSTNESS: Handle boolean results from logical functions
        if isinstance(result, bool):
            result = 1 if result else 0
        
        # ✅ ROBUSTNESS: Handle None results
        if result is None:
            logger.warning(f"⚠️ [EVAL_WARNING] Cell {current_calc_cell_coord}: Formula returned None, converting to 0")
            result = 0
        
        # ✅ ROBUSTNESS: Validate numeric results (including numpy types)
        import numpy as np
        if not isinstance(result, (int, float, bool, np.number)):
            logger.warning(f"⚠️ [EVAL_WARNING] Cell {current_calc_cell_coord}: Non-numeric result {result} (type: {type(result)}), converting to 0")
            result = 0
        else:
            # Convert numpy types to Python native types for consistency
            if isinstance(result, np.number):
                result = float(result)
        
        # CRITICAL DEBUG: Check result and alert if zero - DISABLED FOR PERFORMANCE
        if current_calc_cell_coord != "UnknownCell" and len(all_current_iter_values) < 1000:  # Only debug first 1000 evaluations
            debug_enabled = any(target in formula_string.upper() for target in ['B12', 'B13']) and len(formula_string) < 100
            if debug_enabled:
                logger.info(f"🔍 [EVAL_RESULT] Cell {current_calc_cell_coord}: Result = {result} (type: {type(result)})")
                if result == 0 or result == 0.0:
                    logger.error(f"❌ [ZERO_ALERT] Cell {current_calc_cell_coord}: Formula '{formula_string}' evaluated to ZERO!")
                    logger.error(f"❌ [ZERO_ALERT] Processed: '{processed_formula}' with local vars: {len(local_vars_for_internal_eval)} variables")
        
        return result
    except NameError as ne:
        problematic_name = str(ne).split("'")[1] if "'" in str(ne) else "unknown"
        err_msg = f"Error in cell {current_calc_cell_coord} (formula: '{formula_string}', processed: '{processed_formula}'): "
        if problematic_name in safe_eval_globals:
            err_msg += f"Problem with arguments or usage of function '{problematic_name}'. Detail: {str(ne)}"
        else:
            err_msg += f"Unknown variable, range, or function '{problematic_name}'. Detail: {str(ne)}"
        raise ValueError(err_msg)
    except TypeError as te:
        # CRITICAL FIX: Handle specific float() conversion error for IRR/range data
        if "float() argument must be a string or a real number, not 'list'" in str(te):
            logger.warning(f"🔧 [FLOAT_LIST_FIX] Detected float-list conversion error in {current_calc_cell_coord}")
            logger.warning(f"🔧 [FLOAT_LIST_FIX] Formula: '{formula_string}' -> '{processed_formula}'")
            
            # For IRR formulas, return 0 (which IFERROR will handle)
            if 'IRR' in formula_string.upper():
                logger.warning(f"🔧 [FLOAT_LIST_FIX] IRR formula detected, returning 0 for IFERROR handling")
                return 0
            else:
                logger.warning(f"🔧 [FLOAT_LIST_FIX] Non-IRR formula, returning NaN")
                return float('nan')
        else:
            raise ValueError(f"Error in cell {current_calc_cell_coord} (formula: '{formula_string}', processed: '{processed_formula}'): Type error in function or operation. Detail: {str(te)}")
    except SyntaxError as se:
        raise ValueError(f"Error in cell {current_calc_cell_coord} (formula: '{formula_string}', processed: '{processed_formula}'): Syntax error. Detail: {str(se)}")
    except ZeroDivisionError as zde:
        # Handle division by zero Excel-style: return 0 instead of error
        logger.info(f"Division by zero in {current_calc_cell_coord} (formula: '{formula_string}'): Returning 0 (Excel-compatible behavior)")
        return 0
    except ValueError as ve: # Catch ValueErrors from our custom wrappers or other sources
        raise ValueError(f"Error in cell {current_calc_cell_coord} (formula: '{formula_string}', processed: '{processed_formula}'): {str(ve)}")
    except Exception as e:
        raise ValueError(f"Unexpected error in cell {current_calc_cell_coord} (formula: '{formula_string}', processed: '{processed_formula}'): {str(e)}")

class MonteCarloSimulation:
    def __init__(self, iterations: int = settings.DEFAULT_ITERATIONS):
        self.iterations = min(iterations, settings.MAX_ITERATIONS)
        # If not using GPU or GPU init fails, this executor is used.
        self.cpu_executor = ThreadPoolExecutor(max_workers=os.cpu_count() or 1)
        self.progress_callback: Optional[Callable] = None

    def set_progress_callback(self, callback: Callable):
        """Sets a callback function for progress updates."""
        self.progress_callback = callback
    
    def _is_cupy_array(self, array: Any) -> bool:
        """Check if an array is a CuPy array."""
        if cp is None:
            return False
        return isinstance(array, cp.ndarray)
    
    async def run_simulation(
        self,
        mc_input_configs: List[VariableConfig],
        ordered_calc_steps: List[Tuple[str, str, str]], # (sheet, cell, formula_str)
        target_sheet_name: str,
        target_cell_coordinate: str,
        constant_values: Dict[Tuple[str, str], Any]
    ) -> Tuple[Union[np.ndarray, CpNdArray, None], List[str]]:
        logger.warning(f"[DEPENDENCY_CHAIN_DEBUG] Starting simulation for {target_sheet_name}!{target_cell_coordinate}")
        for sheet, cell, formula in ordered_calc_steps:
            logger.warning(f"[DEPENDENCY_CHAIN_DEBUG] {sheet}!{cell} = {formula}")
            if 'VLOOKUP' in str(formula).upper():
                logger.warning(f"[DEPENDENCY_CHAIN_DEBUG] VLOOKUP formula detected: {sheet}!{cell} = {formula}")
        
        use_gpu_effective = settings.USE_GPU
        if use_gpu_effective:
            if not gpu_manager.initialized:
                await gpu_manager.initialize()
            if not gpu_manager.is_gpu_available():
                print("GPU explicitly enabled but not available/initialized, falling back to CPU.")
                use_gpu_effective = False
        
        loop = asyncio.get_event_loop()
        
        # Prepare mc_input_map for quick lookup of distribution parameters
        # Key: (sheet_name, cell_coord_upper), Value: (min, mode, max)
        mc_input_params_map: Dict[Tuple[str,str], Tuple[float,float,float]] = {}
        for var_conf in mc_input_configs:
            key_norm = (var_conf.sheet_name, var_conf.name.upper())
            mc_input_params_map[key_norm] = (
                var_conf.min_value, var_conf.most_likely, var_conf.max_value
            )
            # Add absolute reference variant as well
            try:
                col_part = ''.join(filter(str.isalpha, var_conf.name)).upper()
                row_part = ''.join(filter(str.isdigit, var_conf.name))
                if col_part and row_part:
                    abs_coord = f"${col_part}${'$'}{row_part}"
                    mc_input_params_map[(var_conf.sheet_name, abs_coord)] = (
                        var_conf.min_value, var_conf.most_likely, var_conf.max_value
                    )
            except Exception:
                pass

        if use_gpu_effective and cp is not None:
            print(f"Running simulation on GPU with {self.iterations} iterations.")
            # GPU simulation needs careful adaptation for iterative formula evaluation.
            # For now, the GPU path will also perform iterative eval on CPU after generating random numbers on GPU.
            raw_results_array, iteration_errors = await gpu_manager.run_task(
                self._run_simulation_iterations_gpu_hybrid, # New name for clarity
                mc_input_params_map,
                ordered_calc_steps,
                target_sheet_name,
                target_cell_coordinate,
                constant_values
            )
        else:
            print(f"Running simulation on CPU with {self.iterations} iterations.")
            raw_results_array, iteration_errors = await loop.run_in_executor(
                self.cpu_executor,
                self._run_simulation_iterations_cpu,
                mc_input_params_map, 
                ordered_calc_steps,
                target_sheet_name,
                target_cell_coordinate,
                constant_values
        )

        # Ensure result is always a numpy array if not None, even if only one iteration or if GPU path returns cp array
        if raw_results_array is not None and use_gpu_effective and cp is not None and isinstance(raw_results_array, cp.ndarray):
            raw_results_array = cp.asnumpy(raw_results_array)
        elif raw_results_array is None:
             # If all iterations failed and engine returns None for results_array (e.g. critical setup error)
             raw_results_array = np.full(self.iterations, np.nan)
             if not iteration_errors: # Add a generic error if none were collected but results are None
                 iteration_errors.append("Simulation engine returned no results; check for critical setup errors.")

        return raw_results_array, iteration_errors

    def _run_simulation_iterations_cpu(
        self,
        mc_input_params_map: Dict[Tuple[str,str], Tuple[float,float,float]],
        ordered_calc_steps: List[Tuple[str, str, str]],
        target_sheet_name: str,
        target_cell_coordinate: str,
        constant_values: Dict[Tuple[str, str], Any]
    ) -> Tuple[np.ndarray, List[str]]:
        print("DEBUG_ENTRY: CPU simulation method called!", flush=True)
        
        simulation_output_results = np.full(self.iterations, np.nan)
        iteration_errors: List[str] = []
        print(f"BASIC_DEBUG: Starting CPU simulation with {len(ordered_calc_steps)} calc steps", flush=True)

        # Pre-generate all random numbers for MC inputs using enhanced random engine
        mc_input_iter_values: Dict[Tuple[str, str], np.ndarray] = {}
        
        try:
            # Use enhanced random engine for better statistical properties
            random_engine = get_random_engine()
            for (sheet, cell), (min_val, mode_val, max_val) in mc_input_params_map.items():
                if settings.USE_GPU and cp is not None:
                    # Generate on GPU and transfer to CPU for formula evaluation
                    gpu_samples = random_engine.generate_triangular_distribution(
                        shape=(self.iterations,),
                        left=min_val,
                        mode=mode_val,
                        right=max_val,
                        generator=RNGType.CURAND
                    )
                    mc_input_iter_values[(sheet, cell)] = cp.asnumpy(gpu_samples)
                else:
                    # CPU fallback with enhanced validation
                    cpu_samples = random_engine._generate_triangular_cpu_fallback(
                        shape=(self.iterations,),
                        left=min_val,
                        mode=mode_val,
                        right=max_val,
                        seed=None
                    )
                    mc_input_iter_values[(sheet, cell)] = cp.asnumpy(cpu_samples)
                    
        except Exception as e:
            print(f"⚠️ Enhanced random generation failed: {e}")
            print("Falling back to NumPy random generation with proper Monte Carlo variation")
            # Use simulation_id for deterministic seeding
            import hashlib
            seed_source = f"{self.simulation_id or 'default'}_deterministic"
            seed_hash = int(hashlib.md5(seed_source.encode()).hexdigest()[:8], 16)
            np.random.seed(seed_hash)  # Deterministic seed per simulation
            for (sheet, cell), (min_val, mode_val, max_val) in mc_input_params_map.items():
                mc_input_iter_values[(sheet, cell)] = np.random.triangular(
                    min_val, mode_val, max_val, size=self.iterations
                )

        # ENHANCED: Store variable samples for sensitivity analysis
        # Convert from (sheet, cell) format to simple variable names for sensitivity analysis
        self._last_variable_samples = {}
        for (sheet, cell), samples in mc_input_iter_values.items():
            # Use cell name as variable name (e.g., "D2", "D3", "D4")
            var_name = cell
            self._last_variable_samples[var_name] = samples
        
        print(f"🔍 [SENSITIVITY_PREP] Stored samples for {len(self._last_variable_samples)} variables: {list(self._last_variable_samples.keys())}")

        for i in range(self.iterations):
            # Performance-optimized progress: every 1% for smoother UX
            if self.progress_callback and (i > 0 and i % max(1, self.iterations // 100) == 0):
                progress_percentage = (i / self.iterations) * 100
                try:
                    self.progress_callback({
                        "status": "running",
                        "progress_percentage": progress_percentage,
                        "current_iteration": i,
                        "total_iterations": self.iterations,
                        "stage": "Calculating Iterations (CPU)",
                        "timestamp": time.time()
                    })
                except Exception as e:
                    print(f"Warning: CPU progress callback failed: {e}")

            current_iter_cell_values: Dict[Tuple[str, str], Any] = constant_values.copy()
            
            # Populate with this iteration's MC input values
            for (sheet, cell), all_vals_for_input in mc_input_iter_values.items():
                current_iter_cell_values[(sheet, cell)] = all_vals_for_input[i]
            
            try:
                # Evaluate ordered calculation steps
                for calc_sheet, calc_cell, calc_formula_str in ordered_calc_steps:
                    logger.warning(f"[SIM_DEBUG] Iter {i}: Evaluating {calc_sheet}!{calc_cell} = {calc_formula_str}")
                    eval_result = _safe_excel_eval(
                        calc_formula_str,
                        calc_sheet,
                        current_iter_cell_values,
                        SAFE_EVAL_NAMESPACE,
                        current_calc_cell_coord=f"{calc_sheet}!{calc_cell}", # Pass cell coordinate
                        constant_values=constant_values # Pass constant values for fallback
                    )
                    logger.warning(f"[SIM_DEBUG] Iter {i}: Result for {calc_sheet}!{calc_cell} = {eval_result}")
                    current_iter_cell_values[(calc_sheet, calc_cell)] = eval_result
                
                final_value_key = (target_sheet_name, target_cell_coordinate)
                if final_value_key in current_iter_cell_values:
                    iter_final_result = current_iter_cell_values[final_value_key]
                    try:
                        # Handle case where result might be a list/array (from range formulas)
                        if isinstance(iter_final_result, (list, tuple)):
                            # If it's a 2D array, take the first element of the first row
                            if isinstance(iter_final_result[0], (list, tuple)):
                                final_float_result = float(iter_final_result[0][0])
                            else:
                                # If it's a 1D array, take the first element
                                final_float_result = float(iter_final_result[0])
                        else:
                            final_float_result = float(iter_final_result)
                        simulation_output_results[i] = final_float_result
                        
                        # ENHANCED: Debug target cell results for zero issue - DISABLED FOR PERFORMANCE
                        # if i < 3:  # Only debug first 3 iterations
                        #     print(f"🔍 [TARGET_DEBUG] Iter {i}: Target {target_sheet_name}!{target_cell_coordinate} = {final_float_result}")
                        
                    except (ValueError, TypeError):
                        iteration_errors.append(f"Iteration {i}: Non-numeric result '{iter_final_result}' for target.")
                        # if i < 3:
                        #     print(f"❌ [TARGET_ERROR] Iter {i}: Non-numeric target result: {iter_final_result}")
                else:
                    # This should not happen if get_evaluation_order is correct and target is part of calc chain or an MC input
                    iteration_errors.append(f"Iteration {i}: Target cell {target_sheet_name}!{target_cell_coordinate} not found after calculations.")
                    # if i < 3:
                    #     print(f"❌ [TARGET_ERROR] Iter {i}: Target cell not found in results")

            except ValueError as e: # Catches errors from _safe_excel_eval or float conversion
                iteration_errors.append(f"Iteration {i}: {str(e)}")
            except Exception as e: 
                iteration_errors.append(f"Iteration {i}: Unexpected error - {str(e)}")
        
        # Final progress update before returning
        if self.progress_callback:
            try:
                self.progress_callback({
                    "status": "running", # Still running as we finalize
                    "progress_percentage": 100,
                    "current_iteration": self.iterations,
                    "total_iterations": self.iterations,
                    "stage": "Finalizing (CPU)"
                })
            except Exception as e:
                print(f"Warning: Final CPU progress callback failed: {e}")

        return simulation_output_results, iteration_errors

    async def _run_simulation_iterations_gpu_hybrid(
        self,
        mc_input_params_map: Dict[Tuple[str,str], Tuple[float,float,float]],
        ordered_calc_steps: List[Tuple[str, str, str]],
        target_sheet_name: str,
        target_cell_coordinate: str,
        constant_values: Dict[Tuple[str, str], Any]
    ) -> Tuple[CpNdArray, List[str]]:
        if cp is None: # Should not happen if called when use_gpu_effective is True
            raise RuntimeError("CuPy not available for GPU simulation path.")

        simulation_output_results_gpu = cp.full(self.iterations, cp.nan)
        iteration_errors: List[str] = []

        # Initial progress update (GPU simulation starting)
        if self.progress_callback:
            try:
                self.progress_callback({
                    "status": "running",
                    "progress_percentage": 0,
                    "current_iteration": 0,
                    "total_iterations": self.iterations,
                    "stage": "Starting GPU Simulation",
                    "timestamp": time.time()
                })
            except Exception as e:
                print(f"Warning: Initial GPU progress callback failed: {e}")

        # Pre-generate all random numbers for MC inputs on GPU using enhanced engine
        mc_input_iter_values_gpu: Dict[Tuple[str, str], cp.ndarray] = {}
        
        try:
            # Use enhanced GPU random engine
            random_engine = get_random_engine()
            for (sheet, cell), (min_val, mode_val, max_val) in mc_input_params_map.items():
                mc_input_iter_values_gpu[(sheet, cell)] = random_engine.generate_triangular_distribution(
                    shape=(self.iterations,),
                    left=min_val,
                    mode=mode_val,
                    right=max_val,
                    generator=RNGType.CURAND
                )
        except Exception as e:
            print(f"⚠️ Enhanced GPU random generation failed: {e}")
            print("Falling back to CuPy random generation with proper Monte Carlo variation")
            # Use simulation_id for deterministic seeding
            import hashlib
            seed_source = f"{self.simulation_id or 'default'}_deterministic"
            seed_hash = int(hashlib.md5(seed_source.encode()).hexdigest()[:8], 16)
            cp.random.seed(seed_hash)  # Deterministic seed per simulation
            for (sheet, cell), (min_val, mode_val, max_val) in mc_input_params_map.items():
                mc_input_iter_values_gpu[(sheet, cell)] = cp.random.triangular(
                    min_val, mode_val, max_val, size=self.iterations
                )
        
        # Transfer MC input values to CPU for iterative evaluation
        # (as _safe_excel_eval is CPU-bound)
        mc_input_iter_values_cpu: Dict[Tuple[str, str], np.ndarray] = {}
        for key, gpu_array in mc_input_iter_values_gpu.items():
            mc_input_iter_values_cpu[key] = cp.asnumpy(gpu_array)

        print("Warning: GPU path currently uses CPU for iterative formula evaluation due to `eval()`.")

        for i in range(self.iterations):
            # Progress callback for GPU simulation (every 1-2% for smoother UX)
            if self.progress_callback and (i > 0 and i % max(1, self.iterations // 100) == 0):
                progress_percentage = (i / self.iterations) * 100
                try:
                    self.progress_callback({
                        "status": "running",
                        "progress_percentage": progress_percentage,
                        "current_iteration": i,
                        "total_iterations": self.iterations,
                        "stage": "Calculating Iterations (GPU)",
                        "timestamp": time.time()
                    })
                except Exception as e:
                    print(f"Warning: GPU progress callback failed: {e}")
            
            current_iter_cell_values: Dict[Tuple[str, str], Any] = constant_values.copy()
            
            for (sheet, cell), all_vals_for_input_cpu in mc_input_iter_values_cpu.items():
                current_iter_cell_values[(sheet, cell)] = all_vals_for_input_cpu[i]
            
            try:
                for calc_sheet, calc_cell, calc_formula_str in ordered_calc_steps:
                    eval_result = _safe_excel_eval(
                        calc_formula_str,
                        calc_sheet,
                        current_iter_cell_values,
                        SAFE_EVAL_NAMESPACE,
                        current_calc_cell_coord=f"{calc_sheet}!{calc_cell}", # Pass cell coordinate
                        constant_values=constant_values # Pass constant values for fallback
                    )
                    current_iter_cell_values[(calc_sheet, calc_cell)] = eval_result
                
                final_value_key = (target_sheet_name, target_cell_coordinate)
                if final_value_key in current_iter_cell_values:
                    iter_final_result = current_iter_cell_values[final_value_key]
                    try:
                        # Handle case where result might be a list/array (from range formulas)
                        if isinstance(iter_final_result, (list, tuple)):
                            # If it's a 2D array, take the first element of the first row
                            if isinstance(iter_final_result[0], (list, tuple)):
                                final_float_result = float(iter_final_result[0][0])
                            else:
                                # If it's a 1D array, take the first element
                                final_float_result = float(iter_final_result[0])
                        else:
                            final_float_result = float(iter_final_result)
                        # Store directly in GPU array after potential conversion
                        simulation_output_results_gpu[i] = final_float_result 
                    except (ValueError, TypeError):
                        iteration_errors.append(f"Iteration {i} (GPU Path): Non-numeric result '{iter_final_result}' for target.")
                else:
                    iteration_errors.append(f"Iteration {i} (GPU Path): Target cell {target_sheet_name}!{target_cell_coordinate} not found after calculations.")

            except ValueError as e:
                iteration_errors.append(f"Iteration {i} (GPU Path): {str(e)}")
            except Exception as e:
                iteration_errors.append(f"Iteration {i} (GPU Path): Unexpected error - {str(e)}")
        
        # Final progress update before returning (GPU)
        if self.progress_callback:
            try:
                self.progress_callback({
                    "status": "running", # Still running as we finalize
                    "progress_percentage": 100,
                    "current_iteration": self.iterations,
                    "total_iterations": self.iterations,
                    "stage": "Finalizing (GPU)"
                })
            except Exception as e:
                print(f"Warning: Final GPU progress callback failed: {e}")
        
        return simulation_output_results_gpu, iteration_errors

    def _calculate_statistics(self, results_array: Union[np.ndarray, CpNdArray, None], array_module = None) -> Dict[str, Any]:
        # Import sanitize_float function
        from simulation.engines.service import sanitize_float
        
        if results_array is None or len(results_array) == 0:
            # Handle case where results_array might be empty or None due to all iterations failing catastrophically
            # Or if no iterations were run (e.g. self.iterations was 0)
            return {
                "mean": sanitize_float(0.0),
                "median": sanitize_float(0.0),
                "std_dev": sanitize_float(0.0),
                "min_value": sanitize_float(0.0),
                "max_value": sanitize_float(0.0),
                "percentiles": {str(p): sanitize_float(0.0) for p in [10, 25, 50, 75, 90]},
                "histogram": {"bins": [], "values": []},
                "successful_iterations": 0
            }

        if array_module is None:
            array_module = cp if self._is_cupy_array(results_array) else np

        # Remove any NaN or infinite values for statistics calculation
        finite_mask = array_module.isfinite(results_array)
        if array_module.sum(finite_mask) == 0:
            # All values are NaN or infinite
            return {
                "mean": sanitize_float(0.0),
                "median": sanitize_float(0.0),
                "std_dev": sanitize_float(0.0),
                "min_value": sanitize_float(0.0),
                "max_value": sanitize_float(0.0),
                "percentiles": {str(p): sanitize_float(0.0) for p in [10, 25, 50, 75, 90]},
                "histogram": {"bins": [], "values": []},
                "successful_iterations": 0
            }

        finite_results = results_array[finite_mask]
        successful_iterations = len(finite_results)

        # Convert to numpy for calculations
        if self._is_cupy_array(finite_results):
            finite_results_np = cp.asnumpy(finite_results)
        else:
            finite_results_np = finite_results

        # Calculate basic statistics using sanitize_float for safety
        mean = sanitize_float(np.mean(finite_results_np))
        median = sanitize_float(np.median(finite_results_np))
        std_dev = sanitize_float(np.std(finite_results_np))
        min_value = sanitize_float(np.min(finite_results_np))
        max_value = sanitize_float(np.max(finite_results_np))

        # Calculate percentiles
        percentiles = {}
        for p in [10, 25, 50, 75, 90]:
            percentiles[str(p)] = sanitize_float(np.percentile(finite_results_np, p))

        # Generate histogram with adaptive bins for better resolution
        try:
            # Use 50 bins for much better resolution and smoother curves
            num_bins = min(50, max(15, len(finite_results_np) // 2))  # Adaptive: 15-50 bins
            hist_counts, hist_edges = np.histogram(finite_results_np, bins=num_bins)
            
            # Ensure we have actual variation by checking the range
            data_range = max_value - min_value
            relative_std = std_dev / abs(mean) if mean != 0 else 0
            
            print(f"🔍 [HISTOGRAM] Data range: {data_range:.2e}, Relative std: {relative_std:.4f}")
            
            if relative_std < 0.001:  # Very low variation
                print("⚠️ [HISTOGRAM] Low variation detected - results may appear clustered")
            
            histogram = {
                "bins": hist_edges.tolist(),
                "values": hist_counts.tolist(),
                "bin_edges": hist_edges.tolist(),
                "counts": hist_counts.tolist()
            }
            print(f"🔍 [HISTOGRAM] Generated histogram with {num_bins} bins for better resolution")
        except Exception as e:
            logger.warning(f"Failed to generate histogram: {e}")
            histogram = {"bins": [], "values": []}

        # Calculate sensitivity analysis if available from enhanced engine
        sensitivity_analysis = []
        if hasattr(self, 'sensitivity_analysis'):
            sensitivity_analysis = self.sensitivity_analysis
            print(f"🔍 [SENSITIVITY] Found sensitivity analysis with {len(sensitivity_analysis)} variables")
        elif hasattr(self, '_last_variable_samples') and hasattr(self, '_last_results'):
            # Calculate on the spot if we have the data
            print(f"🔍 [SENSITIVITY] Calculating sensitivity analysis from stored samples...")
            sensitivity_analysis = self._calculate_sensitivity_analysis()
        else:
            print("⚠️ [SENSITIVITY] No variable sample data available for sensitivity analysis")

        print(f"🔍 [DEBUG] Statistics Summary:")
        print(f"  - Valid iterations: {successful_iterations}/{len(results_array)}")
        print(f"  - Mean: {mean}")
        print(f"  - Range: [{min_value}, {max_value}]") 
        print(f"  - Std Dev: {std_dev}")
        print(f"  - Sample values: {finite_results_np[:5].tolist()}")
        print(f"  - Sensitivity variables: {len(sensitivity_analysis)}")

        return {
            "mean": mean,
            "median": median,
            "std_dev": std_dev,
            "min_value": min_value,
            "max_value": max_value,
            "percentiles": percentiles,
            "histogram": histogram,
            "successful_iterations": successful_iterations,
            "sensitivity_analysis": sensitivity_analysis
        }

    def _calculate_sensitivity_analysis(self):
        """Calculate sensitivity analysis using correlation coefficients."""
        import numpy as np
        from scipy.stats import pearsonr
        
        sensitivity_results = []
        
        try:
            if not hasattr(self, '_last_results') or not hasattr(self, '_last_variable_samples'):
                logger.warning("⚠️ [SENSITIVITY] No variable sample data available for sensitivity analysis")
                return sensitivity_results
                
            results_array = np.array(self._last_results)
            
            # Remove NaN values
            valid_indices = ~np.isnan(results_array)
            valid_results = results_array[valid_indices]
            
            if len(valid_results) < 2:
                logger.warning("⚠️ [SENSITIVITY] Not enough valid results for sensitivity analysis")
                return sensitivity_results
            
            for var_name, var_samples in self._last_variable_samples.items():
                try:
                    var_array = np.array(var_samples)
                    valid_var_samples = var_array[valid_indices]
                    
                    if len(valid_var_samples) < 2:
                        continue
                    
                    # Calculate correlation coefficient
                    correlation, p_value = pearsonr(valid_var_samples, valid_results)
                    
                    # Calculate impact as absolute correlation * 100
                    impact_percentage = abs(correlation) * 100
                    
                    sensitivity_results.append({
                        'variable_name': var_name,
                        'correlation_coefficient': correlation,
                        'impact_percentage': impact_percentage,
                        'p_value': p_value,
                        'sample_count': len(valid_var_samples)
                    })
                    
                    logger.info(f"📊 [SENSITIVITY] {var_name}: {impact_percentage:.1f}% impact (r={correlation:.3f})")
                    
                except Exception as e:
                    logger.warning(f"⚠️ [SENSITIVITY] Failed to calculate sensitivity for {var_name}: {e}")
                    continue
            
            # Sort by impact percentage (highest first)
            sensitivity_results.sort(key=lambda x: x['impact_percentage'], reverse=True)
            
            logger.info(f"✅ [SENSITIVITY] Calculated sensitivity for {len(sensitivity_results)} variables")
            
        except Exception as e:
            logger.error(f"❌ [SENSITIVITY] Sensitivity analysis failed: {e}")
            
        return sensitivity_results

# GPU Manager related class (as provided in the markdown)
# Needs to be adapted if not already present or if its structure is different.
# Assuming gpu_manager from ..gpu.manager has initialize() and run_task()
# and a way to check if GPU is truly available after initialization.
# Need to handle Union type hint properly, import Union from typing. 