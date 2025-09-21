"""
Advanced Formula Engine for Excel-like calculations
Handles formulas, cell dependencies, and Monte Carlo integration
"""

import re
import ast
import math
import statistics
import networkx as nx
from typing import Dict, Any, List, Tuple, Set, Optional
from formulas import Parser
import numpy as np
import pandas as pd
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class CellDependency:
    """Represents a cell dependency relationship"""
    source: str  # e.g., "A1"
    target: str  # e.g., "B2"
    sheet: str
    
@dataclass
class FormulaResult:
    """Result of formula evaluation"""
    value: Any
    dependencies: List[str]
    error: Optional[str] = None

class ExcelFormulaEngine:
    """
    Advanced Excel formula engine with dependency tracking
    """
    
    def __init__(self):
        self.parser = Parser()
        self.dependency_graph = nx.DiGraph()
        self.cell_values = {}  # {sheet_name: {coordinate: value}}
        self.cell_formulas = {}  # {sheet_name: {coordinate: formula}}
        self.formula_cache = {}  # Cache for parsed formulas
        
        # Current context for formula evaluation
        self._current_sheet_name = None
        self._current_context = None
        
        # Supported Excel functions
        self.excel_functions = {
            'SUM': self._sum,
            'AVERAGE': self._average,
            'COUNT': self._count,
            'MAX': self._max,
            'MIN': self._min,
            'SQRT': self._sqrt,
            'ABS': self._abs,
            'ROUND': self._round,
            'IF': self._if,
            'VLOOKUP': self._vlookup,
            'HLOOKUP': self._hlookup,
            'INDEX': self._index,
            'MATCH': self._match,
            'CONCATENATE': self._concatenate,
            'LEN': self._len,
            'LEFT': self._left,
            'RIGHT': self._right,
            'MID': self._mid,
            'UPPER': self._upper,
            'LOWER': self._lower,
            'TODAY': self._today,
            'NOW': self._now,
            'YEAR': self._year,
            'MONTH': self._month,
            'DAY': self._day,
            'RAND': self._rand,
            'RANDBETWEEN': self._randbetween,
            # Phase 1 - Math Functions
            'PRODUCT': self._product,
            'POWER': self._power,
            'INT': self._int,
            'MOD': self._mod,
            'TRUNC': self._trunc,
            'ROUNDUP': self._roundup,
            'ROUNDDOWN': self._rounddown,
            'SIGN': self._sign,
            # Phase 1 - Statistical Functions
            'COUNTA': self._counta,
            'COUNTBLANK': self._countblank,
            'STDEV': self._stdev_s,
            'STDEV.S': self._stdev_s,
            'STDEV.P': self._stdev_p,
            'VAR': self._var_s,
            'VAR.S': self._var_s,
            'VAR.P': self._var_p,
            'MEDIAN': self._median,
            'MODE': self._mode,
            'PERCENTILE': self._percentile,
            'QUARTILE': self._quartile,
        }

    def load_workbook_data(self, workbook_data: Dict[str, Dict[str, Any]]):
        """
        Load workbook data from parsed Excel file
        
        Args:
            workbook_data: {sheet_name: {coordinate: {value, formula, ...}}}
        """
        self.cell_values = {}
        self.cell_formulas = {}
        self.dependency_graph.clear()
        
        for sheet_name, sheet_data in workbook_data.items():
            self.cell_values[sheet_name] = {}
            self.cell_formulas[sheet_name] = {}
            
            for coordinate, cell_data in sheet_data.items():
                if isinstance(cell_data, dict):
                    value = cell_data.get('display_value') or cell_data.get('value')
                    formula = cell_data.get('formula')
                    
                    self.cell_values[sheet_name][coordinate] = value
                    
                    if formula:
                        self.cell_formulas[sheet_name][coordinate] = formula
                        # Build dependency graph
                        dependencies = self._extract_cell_references(formula)
                        for dep in dependencies:
                            self.dependency_graph.add_edge(
                                f"{sheet_name}!{dep}", 
                                f"{sheet_name}!{coordinate}"
                            )

    def _extract_cell_references(self, formula: str) -> List[str]:
        """Extract cell references from formula (e.g., A1, B2:D4)"""
        # Pattern to match cell references like A1, $A$1, A1:B5, etc.
        pattern = r'\$?[A-Z]+\$?\d+(?::\$?[A-Z]+\$?\d+)?'
        matches = re.findall(pattern, formula.upper())
        
        cell_refs = []
        for match in matches:
            if ':' in match:
                # Handle ranges like A1:B5
                start, end = match.split(':')
                start_col, start_row = self._parse_cell_reference(start)
                end_col, end_row = self._parse_cell_reference(end)
                
                for row in range(start_row, end_row + 1):
                    for col in range(start_col, end_col + 1):
                        cell_refs.append(f"{chr(65 + col - 1)}{row}")
            else:
                cell_refs.append(match.replace('$', ''))
        
        return cell_refs

    def _parse_cell_reference(self, cell_ref: str) -> Tuple[int, int]:
        """Parse cell reference like A1 into column and row numbers"""
        cell_ref = cell_ref.replace('$', '')
        col_str = ''.join(filter(str.isalpha, cell_ref))
        row_str = ''.join(filter(str.isdigit, cell_ref))
        
        # Convert column letters to number (A=1, B=2, ..., Z=26, AA=27)
        col_num = 0
        for char in col_str:
            col_num = col_num * 26 + (ord(char) - ord('A') + 1)
        
        return col_num, int(row_str)

    def evaluate_formula(self, formula: str, sheet_name: str, context: Dict[str, Any] = None) -> FormulaResult:
        """
        Evaluate an Excel formula
        
        Args:
            formula: Excel formula string (e.g., "=A1+B2")
            sheet_name: Name of the sheet
            context: Additional context variables for Monte Carlo
            
        Returns:
            FormulaResult with value, dependencies, and any errors
        """
        if not formula.startswith('='):
            # Not a formula, just return the value
            try:
                return FormulaResult(
                    value=float(formula),
                    dependencies=[]
                )
            except (ValueError, TypeError):
                return FormulaResult(
                    value=formula,
                    dependencies=[]
                )
        
        try:
            # Remove the = sign
            formula_body = formula[1:]
            
            # Extract dependencies
            dependencies = self._extract_cell_references(formula_body)
            
            # Try using the formulas library first
            try:
                parsed_formula = self.parser.ast(formula)[1].compile()
                
                # Create input data for the formula
                inputs = {}
                for dep in dependencies:
                    if sheet_name in self.cell_values and dep in self.cell_values[sheet_name]:
                        inputs[dep] = self.cell_values[sheet_name][dep]
                    elif context and dep in context:
                        inputs[dep] = context[dep]
                    else:
                        inputs[dep] = 0  # Default value
                
                result = parsed_formula(**inputs)
                
                return FormulaResult(
                    value=result,
                    dependencies=dependencies
                )
                
            except Exception as formula_lib_error:
                logger.warning(f"Formulas library failed: {formula_lib_error}")
                # Fallback to custom evaluation
                return self._evaluate_custom_formula(formula_body, sheet_name, dependencies, context)
                
        except Exception as e:
            logger.error(f"Formula evaluation failed: {e}")
            return FormulaResult(
                value=0,
                dependencies=[],
                error=str(e)
            )

    def _evaluate_custom_formula(self, formula: str, sheet_name: str, dependencies: List[str], context: Dict[str, Any] = None) -> FormulaResult:
        """Custom formula evaluation for cases where formulas library fails"""
        
        # Set current context for lookup function processing
        self._current_sheet_name = sheet_name
        self._current_context = context
        
        try:
            # Check if this is a lookup function that needs special handling
            lookup_functions = ['VLOOKUP', 'HLOOKUP', 'INDEX', 'MATCH']
            is_lookup_formula = any(func.upper() in formula.upper() for func in lookup_functions)
            
            if is_lookup_formula:
                # Handle lookup functions directly without cell replacement
                evaluated_formula = formula
                
                # Handle Excel functions
                for func_name, func_impl in self.excel_functions.items():
                    pattern = rf'{func_name}\s*\('
                    if re.search(pattern, evaluated_formula, re.IGNORECASE):
                        evaluated_formula = self._replace_excel_function(evaluated_formula, func_name, func_impl)
                
                # For lookup functions, the result should be directly available after function replacement
                try:
                    # Try to evaluate if it's a simple numeric result
                    if isinstance(evaluated_formula, (int, float)):
                        result = evaluated_formula
                    elif isinstance(evaluated_formula, str) and evaluated_formula.replace('.', '').replace('-', '').isdigit():
                        result = float(evaluated_formula)
                    else:
                        result = evaluated_formula
                    
                    return FormulaResult(
                        value=result,
                        dependencies=dependencies
                    )
                except Exception as e:
                    return FormulaResult(
                        value=0,
                        dependencies=dependencies,
                        error=str(e)
                    )
            
            else:
                # Original logic for non-lookup functions
                # Replace cell references with values
                evaluated_formula = formula
                
                for dep in dependencies:
                    cell_value = None
                    
                    # Check if it's in the current sheet data
                    if sheet_name in self.cell_values and dep in self.cell_values[sheet_name]:
                        cell_value = self.cell_values[sheet_name][dep]
                    elif context and dep in context:
                        cell_value = context[dep]
                    else:
                        cell_value = 0
                    
                    # Convert to numeric if possible
                    if isinstance(cell_value, str):
                        try:
                            cell_value = float(cell_value)
                        except (ValueError, TypeError):
                            cell_value = 0
                    
                    # Replace in formula
                    evaluated_formula = re.sub(
                        rf'\b{re.escape(dep)}\b',
                        str(cell_value),
                        evaluated_formula
                    )
                
                # Handle Excel functions
                for func_name, func_impl in self.excel_functions.items():
                    pattern = rf'{func_name}\s*\('
                    if re.search(pattern, evaluated_formula, re.IGNORECASE):
                        evaluated_formula = self._replace_excel_function(evaluated_formula, func_name, func_impl)
                
                try:
                    # Safe evaluation
                    result = self._safe_eval(evaluated_formula)
                    return FormulaResult(
                        value=result,
                        dependencies=dependencies
                    )
                except Exception as e:
                    return FormulaResult(
                        value=0,
                        dependencies=dependencies,
                        error=str(e)
                    )
        
        finally:
            # Clear context
            self._current_sheet_name = None
            self._current_context = None

    def _safe_eval(self, expression: str) -> Any:
        """Safely evaluate mathematical expressions"""
        # Allow only safe operations
        allowed_names = {
            '__builtins__': {},
            'abs': abs,
            'min': min,
            'max': max,
            'round': round,
            'sum': sum,
            'len': len,
            'pow': pow,
            'sqrt': lambda x: x ** 0.5,
            'pi': 3.14159265359,
            'e': 2.71828182846,
        }
        
        # Add all Excel functions to the allowed namespace
        for func_name, func_impl in self.excel_functions.items():
            allowed_names[func_name] = func_impl
            allowed_names[func_name.lower()] = func_impl  # Also add lowercase versions
        
        # Parse and validate the expression
        try:
            node = ast.parse(expression, mode='eval')
            # Additional safety checks could be added here
            result = eval(compile(node, '<string>', 'eval'), allowed_names)
            return result
        except Exception as e:
            raise ValueError(f"Invalid expression: {expression}") from e

    def _replace_excel_function(self, formula: str, func_name: str, func_impl) -> str:
        """Replace Excel function calls with Python equivalents"""
        pattern = rf'{func_name}\s*\([^)]+\)'
        matches = re.finditer(pattern, formula, re.IGNORECASE)
        
        for match in matches:
            try:
                func_call = match.group()
                # Extract arguments
                args_str = func_call[func_call.find('(')+1:func_call.rfind(')')]
                
                # Check if this is a lookup function that needs special handling
                lookup_functions = ['VLOOKUP', 'HLOOKUP', 'INDEX', 'MATCH']
                if func_name.upper() in lookup_functions:
                    # Use intelligent argument parsing for lookup functions
                    args = self._parse_function_arguments(args_str)
                    converted_args = []
                    
                    for arg in args:
                        converted_arg = self._convert_argument_for_lookup(arg, self._current_sheet_name, self._current_context)
                        converted_args.append(converted_arg)
                    
                    # Call the lookup function
                    result = func_impl(*converted_args)
                    formula = formula.replace(func_call, str(result))
                    
                else:
                    # Use original simple parsing for other functions
                    args = [arg.strip() for arg in args_str.split(',')]
                    
                    # Convert arguments to appropriate types
                    converted_args = []
                    for arg in args:
                        try:
                            converted_args.append(float(arg))
                        except ValueError:
                            converted_args.append(arg.strip('"\''))
                    
                    # Call the function
                    result = func_impl(*converted_args)
                    formula = formula.replace(func_call, str(result))
                
            except Exception as e:
                logger.warning(f"Failed to evaluate {func_name}: {e}")
                formula = formula.replace(match.group(), "0")
        
        return formula

    # Excel function implementations
    def _sum(self, *args):
        """SUM function"""
        total = 0
        for arg in args:
            if isinstance(arg, (int, float)):
                total += arg
        return total

    def _average(self, *args):
        """AVERAGE function"""
        values = [arg for arg in args if isinstance(arg, (int, float))]
        return sum(values) / len(values) if values else 0

    def _count(self, *args):
        """COUNT function"""
        return len([arg for arg in args if isinstance(arg, (int, float))])

    def _max(self, *args):
        """MAX function"""
        values = [arg for arg in args if isinstance(arg, (int, float))]
        return max(values) if values else 0

    def _min(self, *args):
        """MIN function"""
        values = [arg for arg in args if isinstance(arg, (int, float))]
        return min(values) if values else 0

    def _sqrt(self, value):
        """SQRT function"""
        return float(value) ** 0.5

    def _abs(self, value):
        """ABS function"""
        return abs(float(value))

    def _round(self, value, digits=0):
        """ROUND function"""
        return round(float(value), int(digits))

    def _if(self, condition, true_value, false_value):
        """IF function"""
        return true_value if condition else false_value

    def _concatenate(self, *args):
        """CONCATENATE function"""
        return ''.join(str(arg) for arg in args)

    def _len(self, text):
        """LEN function"""
        return len(str(text))

    def _left(self, text, num_chars):
        """LEFT function"""
        return str(text)[:int(num_chars)]

    def _right(self, text, num_chars):
        """RIGHT function"""
        return str(text)[-int(num_chars):]

    def _mid(self, text, start, num_chars):
        """MID function"""
        start_idx = max(0, int(start) - 1)  # Excel uses 1-based indexing
        return str(text)[start_idx:start_idx + int(num_chars)]

    def _upper(self, text):
        """UPPER function"""
        return str(text).upper()

    def _lower(self, text):
        """LOWER function"""
        return str(text).lower()

    def _today(self):
        """TODAY function"""
        from datetime import date
        return date.today()

    def _now(self):
        """NOW function"""
        from datetime import datetime
        return datetime.now()

    def _year(self, date_value):
        """YEAR function"""
        # Simplified implementation
        return 2024

    def _month(self, date_value):
        """MONTH function"""
        # Simplified implementation
        return 1

    def _day(self, date_value):
        """DAY function"""
        # Simplified implementation
        return 1

    def _rand(self):
        """RAND function"""
        return np.random.random()

    def _randbetween(self, bottom, top):
        """RANDBETWEEN function"""
        return np.random.randint(int(bottom), int(top) + 1)

    def _vlookup(self, lookup_value, table_array, col_index, range_lookup=True):
        """
        VLOOKUP function - Vertical lookup in a table
        
        Args:
            lookup_value: The value to search for in the first column
            table_array: The table to search in (2D array/list)
            col_index: Column number to return value from (1-based)
            range_lookup: True for approximate match, False for exact match
        """
        try:
            # Convert range_lookup to boolean if it's a string or number
            if isinstance(range_lookup, str):
                range_lookup = range_lookup.upper() not in ['FALSE', 'F', '0']
            elif isinstance(range_lookup, (int, float)):
                range_lookup = bool(range_lookup)
            
            # Convert table_array to proper format
            if not isinstance(table_array, (list, tuple)):
                # Single value - treat as 1x1 table
                table_array = [[table_array]]
            elif isinstance(table_array[0], (int, float, str)):
                # 1D array - treat as single column
                table_array = [[val] for val in table_array]
            
            # Validate col_index
            col_index = int(col_index)
            if col_index < 1:
                logger.warning("VLOOKUP: col_index must be >= 1")
                return "#VALUE!"
            
            max_cols = max(len(row) if isinstance(row, (list, tuple)) else 1 for row in table_array)
            if col_index > max_cols:
                logger.warning(f"VLOOKUP: col_index {col_index} exceeds table width {max_cols}")
                return "#REF!"
            
            # Convert lookup_value for comparison
            lookup_str = str(lookup_value).lower() if isinstance(lookup_value, str) else lookup_value
            
            # Search through the table
            last_valid_row = None
            
            for row in table_array:
                if not isinstance(row, (list, tuple)):
                    row = [row]
                
                if len(row) == 0:
                    continue
                    
                first_col_value = row[0]
                
                # Exact match search
                if not range_lookup:
                    # Case-insensitive string comparison or exact numeric comparison
                    if isinstance(first_col_value, str) and isinstance(lookup_value, str):
                        if first_col_value.lower() == lookup_str:
                            return row[col_index - 1] if col_index <= len(row) else ""
                    else:
                        try:
                            if float(first_col_value) == float(lookup_value):
                                return row[col_index - 1] if col_index <= len(row) else ""
                        except (ValueError, TypeError):
                            if first_col_value == lookup_value:
                                return row[col_index - 1] if col_index <= len(row) else ""
                
                # Approximate match search (range_lookup = True)
                else:
                    try:
                        first_col_num = float(first_col_value)
                        lookup_num = float(lookup_value)
                        
                        if first_col_num <= lookup_num:
                            last_valid_row = row
                        elif first_col_num > lookup_num:
                            break  # Table should be sorted for approximate match
                            
                    except (ValueError, TypeError):
                        # Handle string comparison for approximate match
                        if str(first_col_value).lower() <= str(lookup_value).lower():
                            last_valid_row = row
                        else:
                            break
            
            # Return result for approximate match
            if range_lookup and last_valid_row:
                return last_valid_row[col_index - 1] if col_index <= len(last_valid_row) else ""
            
            # No match found
            return "#N/A"
            
        except Exception as e:
            logger.warning(f"VLOOKUP function error: {e}")
            return "#VALUE!"

    def _hlookup(self, lookup_value, table_array, row_index, range_lookup=True):
        """
        HLOOKUP function - Horizontal lookup in a table
        
        Args:
            lookup_value: The value to search for in the first row
            table_array: The table to search in (2D array/list)
            row_index: Row number to return value from (1-based)
            range_lookup: True for approximate match, False for exact match
        """
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
            if row_index < 1:
                logger.warning("HLOOKUP: row_index must be >= 1")
                return "#VALUE!"
            
            if row_index > len(table_array):
                logger.warning(f"HLOOKUP: row_index {row_index} exceeds table height {len(table_array)}")
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
            
        except Exception as e:
            logger.warning(f"HLOOKUP function error: {e}")
            return "#VALUE!"

    def _index(self, array, row_num, col_num=None):
        """
        INDEX function - Returns a value from a specific position in an array
        
        Args:
            array: The array or range to index into
            row_num: Row number (1-based), 0 to return entire column
            col_num: Column number (1-based), 0 to return entire row, None for 1D arrays
        """
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
            
        except Exception as e:
            logger.warning(f"INDEX function error: {e}")
            return "#VALUE!"

    def _match(self, lookup_value, lookup_array, match_type=1):
        """
        MATCH function - Finds the position of a value in an array
        
        Args:
            lookup_value: The value to search for
            lookup_array: The array to search in
            match_type: 1 (default) = largest value <= lookup_value (sorted ascending)
                       0 = exact match
                       -1 = smallest value >= lookup_value (sorted descending)
        """
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
                logger.warning(f"MATCH: Invalid match_type {match_type}")
                return "#VALUE!"
                
        except Exception as e:
            logger.warning(f"MATCH function error: {e}")
            return "#VALUE!"

    # ========== Phase 1 Functions - Math & Statistical ==========
    
    # Math Functions
    def _product(self, *args):
        """PRODUCT function - multiply range of values"""
        try:
            values = [float(arg) for arg in args if isinstance(arg, (int, float)) and arg != 0]
            if not values:
                return 0
            result = 1
            for value in values:
                result *= value
            return result
        except (ValueError, TypeError) as e:
            logger.warning(f"PRODUCT function error: {e}")
            return 0

    def _power(self, number, power):
        """POWER function - exponentiation"""
        try:
            return float(number) ** float(power)
        except (ValueError, TypeError, OverflowError) as e:
            logger.warning(f"POWER function error: {e}")
            return 0

    def _int(self, value):
        """INT function - integer truncation (floor for positive, ceiling for negative)"""
        try:
            val = float(value)
            return math.floor(val)
        except (ValueError, TypeError) as e:
            logger.warning(f"INT function error: {e}")
            return 0

    def _mod(self, number, divisor):
        """MOD function - modulo operation"""
        try:
            return float(number) % float(divisor)
        except (ValueError, TypeError, ZeroDivisionError) as e:
            logger.warning(f"MOD function error: {e}")
            return "#DIV/0!"  # Return proper Excel error instead of zero

    def _trunc(self, number, digits=0):
        """TRUNC function - truncate to specified decimal places"""
        try:
            val = float(number)
            multiplier = 10 ** int(digits)
            return math.trunc(val * multiplier) / multiplier
        except (ValueError, TypeError) as e:
            logger.warning(f"TRUNC function error: {e}")
            return 0

    def _roundup(self, number, digits=0):
        """ROUNDUP function - round up away from zero"""
        try:
            val = float(number)
            multiplier = 10 ** int(digits)
            if val >= 0:
                return math.ceil(val * multiplier) / multiplier
            else:
                return math.floor(val * multiplier) / multiplier
        except (ValueError, TypeError) as e:
            logger.warning(f"ROUNDUP function error: {e}")
            return 0

    def _rounddown(self, number, digits=0):
        """ROUNDDOWN function - round down toward zero"""
        try:
            val = float(number)
            multiplier = 10 ** int(digits)
            if val >= 0:
                return math.floor(val * multiplier) / multiplier
            else:
                return math.ceil(val * multiplier) / multiplier
        except (ValueError, TypeError) as e:
            logger.warning(f"ROUNDDOWN function error: {e}")
            return 0

    def _sign(self, value):
        """SIGN function - return sign of number"""
        try:
            val = float(value)
            if val > 0:
                return 1
            elif val < 0:
                return -1
            else:
                return 0
        except (ValueError, TypeError) as e:
            logger.warning(f"SIGN function error: {e}")
            return 0

    # Statistical Functions
    def _counta(self, *args):
        """COUNTA function - count non-empty cells"""
        count = 0
        for arg in args:
            if arg is not None and str(arg).strip() != '':
                count += 1
        return count

    def _countblank(self, *args):
        """COUNTBLANK function - count empty cells"""
        count = 0
        for arg in args:
            if arg is None or str(arg).strip() == '':
                count += 1
        return count

    def _stdev_s(self, *args):
        """STDEV.S function - sample standard deviation"""
        try:
            values = [float(arg) for arg in args if isinstance(arg, (int, float))]
            if len(values) < 2:
                return 0  # Excel returns #DIV/0! but we return 0
            return statistics.stdev(values)
        except (ValueError, TypeError, statistics.StatisticsError) as e:
            logger.warning(f"STDEV.S function error: {e}")
            return 0

    def _stdev_p(self, *args):
        """STDEV.P function - population standard deviation"""
        try:
            values = [float(arg) for arg in args if isinstance(arg, (int, float))]
            if len(values) < 1:
                return 0
            return statistics.pstdev(values)
        except (ValueError, TypeError, statistics.StatisticsError) as e:
            logger.warning(f"STDEV.P function error: {e}")
            return 0

    def _var_s(self, *args):
        """VAR.S function - sample variance"""
        try:
            values = [float(arg) for arg in args if isinstance(arg, (int, float))]
            if len(values) < 2:
                return 0
            return statistics.variance(values)
        except (ValueError, TypeError, statistics.StatisticsError) as e:
            logger.warning(f"VAR.S function error: {e}")
            return 0

    def _var_p(self, *args):
        """VAR.P function - population variance"""
        try:
            values = [float(arg) for arg in args if isinstance(arg, (int, float))]
            if len(values) < 1:
                return 0
            return statistics.pvariance(values)
        except (ValueError, TypeError, statistics.StatisticsError) as e:
            logger.warning(f"VAR.P function error: {e}")
            return 0

    def _median(self, *args):
        """MEDIAN function - middle value"""
        try:
            values = [float(arg) for arg in args if isinstance(arg, (int, float))]
            if not values:
                return 0
            return statistics.median(values)
        except (ValueError, TypeError, statistics.StatisticsError) as e:
            logger.warning(f"MEDIAN function error: {e}")
            return 0

    def _mode(self, *args):
        """MODE function - most frequent value"""
        try:
            values = [float(arg) for arg in args if isinstance(arg, (int, float))]
            if not values:
                return 0
            return statistics.mode(values)
        except (ValueError, TypeError, statistics.StatisticsError) as e:
            logger.warning(f"MODE function error: {e}")
            return 0

    def _percentile(self, array, k):
        """PERCENTILE function - value at percentile k"""
        try:
            # array should be a range of values, k should be between 0 and 1
            if isinstance(array, (list, tuple)):
                values = [float(val) for val in array if isinstance(val, (int, float))]
            else:
                values = [float(array)]
            
            if not values:
                return 0
                
            k_val = float(k)
            if k_val < 0 or k_val > 1:
                return 0  # Excel returns #NUM! error
                
            # Use numpy for percentile calculation (Excel compatible)
            return np.percentile(values, k_val * 100)
        except (ValueError, TypeError) as e:
            logger.warning(f"PERCENTILE function error: {e}")
            return 0

    def _quartile(self, array, quart):
        """QUARTILE function - quartile calculations"""
        try:
            if isinstance(array, (list, tuple)):
                values = [float(val) for val in array if isinstance(val, (int, float))]
            else:
                values = [float(array)]
                
            if not values:
                return 0
                
            quart_val = int(quart)
            if quart_val < 0 or quart_val > 4:
                return 0  # Excel returns #NUM! error
                
            # Excel quartile mapping
            percentiles = {0: 0, 1: 25, 2: 50, 3: 75, 4: 100}
            return np.percentile(values, percentiles[quart_val])
        except (ValueError, TypeError) as e:
            logger.warning(f"QUARTILE function error: {e}")
            return 0

    # ========== End Phase 1 Functions ==========

    def get_calculation_order(self, sheet_name: str) -> List[str]:
        """
        Get the order in which cells should be calculated based on dependencies
        """
        try:
            # Get all cells in the sheet that have formulas
            formula_cells = []
            if sheet_name in self.cell_formulas:
                formula_cells = [f"{sheet_name}!{coord}" for coord in self.cell_formulas[sheet_name].keys()]
            
            if not formula_cells:
                return []
            
            # Create subgraph for this sheet
            subgraph = self.dependency_graph.subgraph(formula_cells)
            
            # Topological sort to get calculation order
            if nx.is_directed_acyclic_graph(subgraph):
                calculation_order = list(nx.topological_sort(subgraph))
                # Remove sheet prefix and return just coordinates
                return [coord.split('!')[1] for coord in calculation_order if coord.startswith(f"{sheet_name}!")]
            else:
                logger.warning(f"Circular dependency detected in sheet {sheet_name}")
                return list(self.cell_formulas[sheet_name].keys())
                
        except Exception as e:
            logger.error(f"Error calculating order for sheet {sheet_name}: {e}")
            return list(self.cell_formulas.get(sheet_name, {}).keys())

    def recalculate_sheet(self, sheet_name: str, variable_overrides: Dict[str, Any] = None):
        """
        Recalculate all formulas in a sheet with optional variable overrides
        Used for Monte Carlo simulations
        """
        if sheet_name not in self.cell_formulas:
            return
        
        # Get calculation order
        calc_order = self.get_calculation_order(sheet_name)
        
        # Recalculate in order
        for coordinate in calc_order:
            if coordinate in self.cell_formulas[sheet_name]:
                formula = self.cell_formulas[sheet_name][coordinate]
                result = self.evaluate_formula(formula, sheet_name, variable_overrides)
                
                if result.error:
                    logger.warning(f"Error in {coordinate}: {result.error}")
                    self.cell_values[sheet_name][coordinate] = 0
                else:
                    self.cell_values[sheet_name][coordinate] = result.value

    def get_cell_value(self, sheet_name: str, coordinate: str, variable_overrides: Dict[str, Any] = None) -> Any:
        """Get the value of a cell, with optional variable overrides"""
        if variable_overrides and coordinate in variable_overrides:
            return variable_overrides[coordinate]
        
        if sheet_name in self.cell_values and coordinate in self.cell_values[sheet_name]:
            return self.cell_values[sheet_name][coordinate]
        
        # DETERMINISTIC FIX: Return zero for missing cells
        return 0.0 

    def _get_range_data(self, range_ref: str, sheet_name: str, context: Dict[str, Any] = None) -> List[List[Any]]:
        """Convert a range reference like 'A1:C5' to actual data array"""
        try:
            if ':' not in range_ref:
                # Single cell reference
                if sheet_name in self.cell_values and range_ref in self.cell_values[sheet_name]:
                    value = self.cell_values[sheet_name][range_ref]
                    return [[value]]
                elif context and range_ref in context:
                    value = context[range_ref]
                    return [[value]]
                else:
                    return [[0]]
            
            # Range reference like A1:C5
            start_cell, end_cell = range_ref.split(':')
            start_col, start_row = self._parse_cell_reference(start_cell)
            end_col, end_row = self._parse_cell_reference(end_cell)
            
            # Build 2D array from the range
            data_array = []
            for row in range(start_row, end_row + 1):
                row_data = []
                for col in range(start_col, end_col + 1):
                    cell_ref = f"{chr(65 + col - 1)}{row}"
                    
                    # Get cell value
                    cell_value = 0  # Default
                    if sheet_name in self.cell_values and cell_ref in self.cell_values[sheet_name]:
                        cell_value = self.cell_values[sheet_name][cell_ref]
                    elif context and cell_ref in context:
                        cell_value = context[cell_ref]
                    
                    row_data.append(cell_value)
                data_array.append(row_data)
            
            return data_array
            
        except Exception as e:
            logger.warning(f"Error processing range {range_ref}: {e}")
            return [[0]]

    def _parse_function_arguments(self, args_str: str) -> List[str]:
        """Parse function arguments more intelligently, handling quoted strings and ranges"""
        args = []
        current_arg = ""
        paren_depth = 0
        in_quotes = False
        quote_char = None
        
        for char in args_str:
            if char in ['"', "'"] and not in_quotes:
                in_quotes = True
                quote_char = char
                current_arg += char
            elif char == quote_char and in_quotes:
                in_quotes = False
                quote_char = None
                current_arg += char
            elif char == '(' and not in_quotes:
                paren_depth += 1
                current_arg += char
            elif char == ')' and not in_quotes:
                paren_depth -= 1
                current_arg += char
            elif char == ',' and paren_depth == 0 and not in_quotes:
                args.append(current_arg.strip())
                current_arg = ""
            else:
                current_arg += char
        
        if current_arg.strip():
            args.append(current_arg.strip())
        
        return args

    def _convert_argument_for_lookup(self, arg: str, sheet_name: str, context: Dict[str, Any] = None) -> Any:
        """Convert an argument for lookup functions, handling ranges, strings, and numbers"""
        arg = arg.strip()
        
        # Handle quoted strings
        if (arg.startswith('"') and arg.endswith('"')) or (arg.startswith("'") and arg.endswith("'")):
            return arg[1:-1]  # Remove quotes
        
        # Handle numbers
        try:
            if '.' in arg:
                return float(arg)
            else:
                return int(arg)
        except ValueError:
            pass
        
        # Handle range references (A1:B5)
        if ':' in arg or re.match(r'[A-Z]+\d+', arg.upper()):
            return self._get_range_data(arg.upper(), sheet_name, context)
        
        # Handle boolean values
        if arg.upper() == 'TRUE':
            return True
        elif arg.upper() == 'FALSE':
            return False
        
        # Default to 0 for unknown arguments
        return 0 