"""
Enhanced Excel Compatibility Module

This module makes our Monte Carlo simulation engine more intelligent about common Excel scenarios:
1. SUM formulas with ranges beyond actual data (should return sum of available cells)
2. Division by zero handling (should return Excel-compatible errors or default behavior)
3. Empty cell handling (should treat as 0 for calculations)
4. Range references that extend beyond sheet boundaries
"""

import logging
import re
from typing import Any, Dict, Tuple, List, Union, Optional
import numpy as np

logger = logging.getLogger(__name__)

class ExcelCompatibilityEngine:
    """
    Enhanced Excel compatibility engine that handles common Excel scenarios intelligently
    """
    
    @staticmethod
    def safe_range_lookup(
        all_current_iter_values: Dict[Tuple[str, str], Any],
        constant_values: Dict[Tuple[str, str], Any],
        sheet_name: str,
        start_row: int,
        end_row: int,
        start_col: int,
        end_col: int,
        formula_string: str = "",
        cell_coord: str = "UnknownCell"
    ) -> List[List[Any]]:
        """
        Safely look up a range of cells, treating missing cells as 0 (like Excel does)
        
        Args:
            all_current_iter_values: Current iteration values
            constant_values: Constant values from Excel file
            sheet_name: Name of the sheet
            start_row, end_row: Row range (inclusive)
            start_col, end_col: Column range (inclusive, 0-indexed)
            formula_string: Original formula for error reporting
            cell_coord: Cell coordinate for error reporting
            
        Returns:
            2D array with values, missing cells filled with 0
        """
        range_values = []
        missing_cells = []
        total_cells = 0
        found_cells = 0
        
        for r in range(start_row, end_row + 1):
            row_values = []
            for c in range(start_col, end_col + 1):
                total_cells += 1
                
                # Convert column number to letter
                col_str = ""
                num = c + 1  # 1-indexed for conversion
                while num > 0:
                    num, remainder = divmod(num - 1, 26)
                    col_str = chr(65 + remainder) + col_str
                
                cell_coord_lookup = f"{col_str}{r}"
                cell_key = (sheet_name, cell_coord_lookup)
                
                # Try to find the cell value
                cell_value = None
                
                # 1. Check current iteration values first
                if cell_key in all_current_iter_values:
                    cell_value = all_current_iter_values[cell_key]
                    found_cells += 1
                # 2. Check constant values (from original Excel)
                elif constant_values and cell_key in constant_values:
                    cell_value = constant_values[cell_key]
                    found_cells += 1
                # 3. Default to 0 (Excel behavior for empty cells)
                else:
                    cell_value = 0
                    missing_cells.append(cell_coord_lookup)
                
                # Convert to numeric if possible
                if isinstance(cell_value, str):
                    try:
                        cell_value = float(cell_value)
                    except (ValueError, TypeError):
                        cell_value = 0
                elif cell_value is None:
                    cell_value = 0
                
                row_values.append(cell_value)
            range_values.append(row_values)
        
        # Log intelligence - only warn if we're missing a significant portion
        missing_percentage = (len(missing_cells) / total_cells) * 100
        if missing_percentage > 50:
            logger.warning(
                f"Range lookup in {cell_coord} (formula: '{formula_string}'): "
                f"{len(missing_cells)}/{total_cells} cells ({missing_percentage:.1f}%) "
                f"not found, using 0 values. Consider adjusting range size."
            )
        elif missing_cells:
            logger.info(
                f"Range lookup in {cell_coord}: {len(missing_cells)}/{total_cells} "
                f"cells treated as 0 (Excel-compatible behavior)"
            )
        
        return range_values
    
    @staticmethod
    def safe_division(numerator: Any, denominator: Any, 
                     formula_string: str = "", cell_coord: str = "UnknownCell") -> Any:
        """
        Safely perform division with Excel-compatible behavior
        
        Args:
            numerator: The dividend
            denominator: The divisor
            formula_string: Original formula for error reporting
            cell_coord: Cell coordinate for error reporting
            
        Returns:
            Division result or Excel-compatible error value
        """
        try:
            # Convert to numeric
            num_val = float(numerator) if numerator is not None else 0
            denom_val = float(denominator) if denominator is not None else 0
            
            # Handle division by zero Excel-style
            if denom_val == 0:
                logger.info(
                    f"Division by zero in {cell_coord} (formula: '{formula_string}'): "
                    f"Returning 0 (Excel-compatible behavior)"
                )
                return 0  # Excel-compatible: return 0 instead of error
            
            return num_val / denom_val
            
        except (ValueError, TypeError) as e:
            logger.warning(
                f"Division error in {cell_coord} (formula: '{formula_string}'): "
                f"Non-numeric values, returning 0"
            )
            return 0
    
    @staticmethod
    def enhance_sum_function(*args: Any) -> Union[int, float]:
        """
        Enhanced SUM function that handles various input types intelligently
        
        Args:
            *args: Values to sum (can be numbers, strings, lists, 2D arrays)
            
        Returns:
            Sum of all valid numeric values
        """
        total = 0
        processed_values = 0
        
        def process_value(val: Any) -> float:
            """Convert a single value to numeric for summing"""
            if val is None:
                return 0
            elif isinstance(val, (int, float)):
                return float(val)
            elif isinstance(val, str):
                try:
                    return float(val)
                except (ValueError, TypeError):
                    return 0  # Excel treats non-numeric strings as 0 in SUM
            else:
                return 0
        
        for arg in args:
            if isinstance(arg, (list, tuple)):
                # Handle 2D arrays (ranges)
                for row in arg:
                    if isinstance(row, (list, tuple)):
                        for cell_val in row:
                            total += process_value(cell_val)
                            processed_values += 1
                    else:
                        total += process_value(row)
                        processed_values += 1
            else:
                # Single value
                total += process_value(arg)
                processed_values += 1
        
        return total
    
    @staticmethod
    def auto_detect_range_issues(formula_string: str, 
                                sheet_data_bounds: Dict[str, Tuple[int, int]]) -> List[str]:
        """
        Analyze a formula for potential range issues and suggest fixes
        
        Args:
            formula_string: The Excel formula to analyze
            sheet_data_bounds: Dict mapping sheet names to (max_row, max_col)
            
        Returns:
            List of suggestions for improving the formula
        """
        suggestions = []
        
        # Find range references in formula
        range_pattern = r'([A-Z]+)(\d+):([A-Z]+)(\d+)'
        ranges = re.findall(range_pattern, formula_string.upper())
        
        for start_col, start_row, end_col, end_row in ranges:
            start_row_num = int(start_row)
            end_row_num = int(end_row)
            
            # Check if range is suspiciously large
            range_size = end_row_num - start_row_num + 1
            if range_size > 1000:
                suggestions.append(
                    f"Large range detected: {start_col}{start_row}:{end_col}{end_row} "
                    f"({range_size} rows). Consider using dynamic ranges or adjusting to actual data size."
                )
            
            # Check for common problematic ranges
            if end_row_num == 10000:
                suggestions.append(
                    f"Range ends at row 10000: {start_col}{start_row}:{end_col}{end_row}. "
                    f"This might be larger than your actual data. "
                    f"Consider using =SUM({start_col}{start_row}:{end_col}100) if your data has ~100 rows."
                )
        
        # Check for division operations
        if '/' in formula_string and 'IF' not in formula_string.upper():
            suggestions.append(
                "Division detected without error checking. "
                "Consider using IF function to prevent division by zero: =IF(denominator=0, 0, numerator/denominator)"
            )
        
        return suggestions

def create_enhanced_safe_eval_globals() -> Dict[str, Any]:
    """
    Create enhanced evaluation globals with intelligent Excel functions
    
    Returns:
        Dictionary of enhanced Excel functions
    """
    # Import the original functions
    from simulation.engine import (
        custom_sum_for_eval, custom_average_for_eval, excel_sqrt, excel_ln,
        excel_log10, excel_log_base, excel_generic_log, excel_count, excel_counta,
        excel_countblank, excel_vlookup, excel_hlookup, excel_index, excel_match
    )
    
    # Enhanced compatibility engine
    compat = ExcelCompatibilityEngine()
    
    return {
        # Basic math
        'abs': abs,
        'min': min,
        'max': max,
        'round': round,
        'pow': pow,
        'sqrt': excel_sqrt,
        'ln': excel_ln,
        'log10': excel_log10,
        'log': excel_generic_log,
        
        # Enhanced Excel functions
        'SUM': compat.enhance_sum_function,  # Use enhanced SUM
        'AVERAGE': custom_average_for_eval,
        'COUNT': excel_count,
        'COUNTA': excel_counta,
        'COUNTBLANK': excel_countblank,
        
        # Lookup functions
        'VLOOKUP': excel_vlookup,
        'HLOOKUP': excel_hlookup,
        'INDEX': excel_index,
        'MATCH': excel_match,
        
        # Utility functions
        'IF': lambda condition, if_true, if_false: if_true if condition else if_false,
        'IFERROR': lambda value, if_error: if_error if str(value).startswith('#') else value,
        
        # Safe division function
        'SAFEDIV': compat.safe_division,
        
        # Constants
        'PI': 3.14159265359,
        'E': 2.71828182846,
        'TRUE': True,
        'FALSE': False,
    }

def enhance_formula_with_intelligence(formula_string: str, 
                                    sheet_name: str,
                                    cell_coord: str) -> str:
    """
    Enhance a formula with intelligent error handling
    
    Args:
        formula_string: Original Excel formula
        sheet_name: Sheet name for context
        cell_coord: Cell coordinate for context
        
    Returns:
        Enhanced formula with better error handling
    """
    enhanced = formula_string
    
    # Auto-wrap divisions with safety
    division_pattern = r'([A-Z]+\d+)/([A-Z]+\d+)'
    divisions = re.findall(division_pattern, enhanced)
    
    for numerator, denominator in divisions:
        original = f"{numerator}/{denominator}"
        safe_div = f"SAFEDIV({numerator},{denominator},'{formula_string}','{cell_coord}')"
        enhanced = enhanced.replace(original, safe_div)
    
    return enhanced 