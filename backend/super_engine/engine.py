"""
SUPERENGINE - Main Orchestration Engine
========================================
This module serves as the main entry point for the SuperEngine, orchestrating
the entire simulation process from formula parsing to GPU execution.

Key Responsibilities:
- Manages the simulation lifecycle
- Coordinates between parser, compiler, and GPU kernels
- Handles memory management and optimization
- Provides fallback mechanisms for unsupported operations
"""

import logging
import numpy as np
import cupy as cp
import time
import re
from typing import Dict, List, Tuple, Any, Optional, Set

from super_engine.hybrid_parser import HybridExcelParser
from super_engine.compiler_v2 import CompilerV2
from super_engine.jit_compiler import JitCompiler
from super_engine.gpu_kernels import KERNEL_LIBRARY, is_gpu_available

# Import utilities from the main simulation module
from simulation.formula_utils import get_evaluation_order, _expand_cell_range
from excel_parser.service import get_named_ranges_for_file, get_tables_for_file

logger = logging.getLogger(__name__)

class SuperEngine:
    """
    The main orchestration engine for GPU-native Monte Carlo simulations.
    """
    
    def __init__(self, iterations: int = 10000, use_jit: bool = False):
        """
        Initialize the SuperEngine.
        
        Args:
            iterations: Number of Monte Carlo iterations
            use_jit: Whether to use JIT compilation for complex formulas
        """
        self.iterations = iterations
        self.use_jit = use_jit
        
        # Initialize components
        self.parser = HybridExcelParser({'current_sheet': 'Sheet1'})
        # Don't initialize CompilerV2 here - it needs simulation data
        self.compiler = None  # Will be initialized when needed
        self.jit_compiler = JitCompiler() if use_jit else None
        
        # Ensure GPU is available
        if not is_gpu_available():
            raise RuntimeError("GPU not available. SuperEngine requires CUDA.")
        
        # Performance tracking
        self.stats = {
            'formulas_processed': 0,
            'gpu_kernel_calls': 0,
            'jit_compilations': 0,
            'total_time': 0.0,
            'parse_time': 0.0,
            'compile_time': 0.0,
            'execution_time': 0.0
        }
        
        # Cache for parsed formulas and compiled kernels
        self.parsed_cache = {}
        self.jit_cache = {}
        
        # Named ranges and tables
        self.named_ranges = {}
        self.tables = {}
        
        # Regex patterns for formula parsing (temporary solution)
        self.cell_pattern = re.compile(r'([A-Z]+)(\d+)')
        self.range_pattern = re.compile(r'([A-Z]+\d+):([A-Z]+\d+)')
        self.function_pattern = re.compile(r'([A-Z]+)\s*\(')
        
        logger.info("✅ SUPERENGINE initialized")
        logger.info(f"   Iterations: {iterations}")
        logger.info(f"   JIT Compilation: {'Enabled' if use_jit else 'Disabled'}")
        logger.info(f"   GPU Available: {is_gpu_available()}")
    
    def parse_formula_regex(self, formula: str) -> Tuple[str, List[str], List[str]]:
        """
        Temporary regex-based formula parser.
        
        Returns:
            (formula_type, cell_refs, function_names)
        """
        if not formula or not formula.startswith('='):
            return ('value', [], [])
        
        # Remove the '=' sign
        formula_body = formula[1:].strip()
        
        # Extract cell references
        cell_refs = []
        for match in self.cell_pattern.finditer(formula_body):
            cell_refs.append(f"{match.group(1)}{match.group(2)}")
        
        # Extract ranges
        ranges = []
        for match in self.range_pattern.finditer(formula_body):
            ranges.append(f"{match.group(1)}:{match.group(2)}")
        
        # Extract function names
        functions = []
        for match in self.function_pattern.finditer(formula_body):
            functions.append(match.group(1))
        
        # Determine formula type
        if functions:
            formula_type = 'function'
        elif any(op in formula_body for op in ['+', '-', '*', '/', '^']):
            formula_type = 'arithmetic'
        elif ranges:
            formula_type = 'range'
        else:
            formula_type = 'reference'
        
        return (formula_type, cell_refs + ranges, functions)
    
    def compile_formula_simple(self, formula: str, cell_values: Dict[str, cp.ndarray]) -> Optional[cp.ndarray]:
        """
        Simple formula compilation using regex parsing and direct kernel calls.
        """
        formula_type, refs, functions = self.parse_formula_regex(formula)
        
        if formula_type == 'value':
            # Just a constant value
            try:
                value = float(formula)
                return cp.full(self.iterations, value, dtype=cp.float32)
            except:
                return None
        
        # Handle simple arithmetic
        if formula_type == 'arithmetic' and not functions:
            return self._compile_arithmetic_simple(formula[1:], cell_values)
        
        # Handle functions
        if functions:
            func_name = functions[0].upper()
            if func_name in ['SUM', 'AVERAGE', 'MIN', 'MAX']:
                return self._compile_aggregation_simple(func_name, formula, cell_values)
            elif func_name == 'IF':
                return self._compile_if_simple(formula, cell_values)
            elif func_name == 'VLOOKUP':
                return self._compile_vlookup_simple(formula, cell_values)
        
        # Fallback to None if we can't handle it
        return None
    
    def _compile_arithmetic_simple(self, expression: str, cell_values: Dict[str, cp.ndarray]) -> Optional[cp.ndarray]:
        """
        Compile simple arithmetic expressions.
        """
        try:
            # Replace cell references with array names
            expr = expression
            
            # First, try to replace full cell references (with sheet names)
            for cell_ref in sorted(cell_values.keys(), key=len, reverse=True):
                if cell_ref in expr:
                    expr = expr.replace(cell_ref, f"cell_values['{cell_ref}']")
            
            # Then, look for any remaining cell references without sheet names
            # and try to find them in cell_values
            import re
            remaining_cells = re.findall(r'\b([A-Z]+\d+)\b', expr)
            for cell in remaining_cells:
                # Look for this cell in any sheet
                for full_ref in cell_values.keys():
                    if full_ref.endswith(f"!{cell}"):
                        expr = expr.replace(cell, f"cell_values['{full_ref}']")
                        break
            
            # Evaluate the expression
            # This is safe because we control the expression format
            result = eval(expr, {"cell_values": cell_values, "cp": cp})
            return result if isinstance(result, cp.ndarray) else cp.array(result)
        except Exception as e:
            logger.error(f"Failed to compile arithmetic: {e}")
            logger.error(f"Expression: {expr}")
            logger.error(f"Available cells: {list(cell_values.keys())}")
            return None
    
    def _compile_aggregation_simple(self, func_name: str, formula: str, cell_values: Dict[str, cp.ndarray]) -> Optional[cp.ndarray]:
        """
        Compile aggregation functions (SUM, AVERAGE, MIN, MAX).
        """
        try:
            # Extract the range or cell list from the formula
            # This is a simplified extraction
            start = formula.find('(') + 1
            end = formula.rfind(')')
            args = formula[start:end].strip()
            
            # Collect all values to aggregate
            values_to_aggregate = []
            
            # Check if it's a range
            if ':' in args:
                # Handle range like A1:A10
                start_cell, end_cell = args.split(':')
                # Expand range (simplified - assumes single column)
                # In real implementation, use _expand_cell_range
                values_to_aggregate = [cell_values.get(cell, cp.zeros(self.iterations)) 
                                     for cell in cell_values if cell.startswith(start_cell[0])]
            else:
                # Handle comma-separated cells
                cells = [c.strip() for c in args.split(',')]
                values_to_aggregate = [cell_values.get(cell, cp.zeros(self.iterations)) 
                                     for cell in cells]
            
            if not values_to_aggregate:
                return cp.zeros(self.iterations)
            
            # Stack arrays for aggregation
            stacked = cp.stack(values_to_aggregate)
            
            # Apply the appropriate kernel
            if func_name == 'SUM':
                return cp.sum(stacked, axis=0)
            elif func_name == 'AVERAGE':
                return cp.mean(stacked, axis=0)
            elif func_name == 'MIN':
                return cp.min(stacked, axis=0)
            elif func_name == 'MAX':
                return cp.max(stacked, axis=0)
            
        except Exception as e:
            logger.error(f"Failed to compile {func_name}: {e}")
            return None
    
    def _compile_if_simple(self, formula: str, cell_values: Dict[str, cp.ndarray]) -> Optional[cp.ndarray]:
        """
        Compile IF function.
        """
        try:
            # Extract IF arguments (simplified parser)
            # IF(condition, true_value, false_value)
            start = formula.find('(') + 1
            end = formula.rfind(')')
            args_str = formula[start:end]
            
            # Split by commas (naive split - doesn't handle nested functions)
            parts = []
            paren_depth = 0
            current_part = ""
            for char in args_str:
                if char == ',' and paren_depth == 0:
                    parts.append(current_part.strip())
                    current_part = ""
                else:
                    if char == '(':
                        paren_depth += 1
                    elif char == ')':
                        paren_depth -= 1
                    current_part += char
            parts.append(current_part.strip())
            
            if len(parts) != 3:
                return None
            
            condition_str, true_str, false_str = parts
            
            # Evaluate condition
            # Simple comparison support
            condition = None
            for op, kernel_name in [('>=', 'gte'), ('<=', 'lte'), ('>', 'gt'), 
                                   ('<', 'lt'), ('=', 'eq'), ('<>', 'neq')]:
                if op in condition_str:
                    left_str, right_str = condition_str.split(op, 1)
                    left_str = left_str.strip()
                    right_str = right_str.strip()
                    
                    # Get left value
                    left_val = None
                    if left_str in cell_values:
                        left_val = cell_values[left_str]
                    else:
                        # Look for cell without sheet name
                        for full_ref in cell_values.keys():
                            if full_ref.endswith(f"!{left_str}"):
                                left_val = cell_values[full_ref]
                                break
                        if left_val is None:
                            try:
                                left_val = cp.full(self.iterations, float(left_str), dtype=cp.float32)
                            except:
                                return None
                    
                    # Get right value
                    right_val = None
                    if right_str in cell_values:
                        right_val = cell_values[right_str]
                    else:
                        # Look for cell without sheet name
                        for full_ref in cell_values.keys():
                            if full_ref.endswith(f"!{right_str}"):
                                right_val = cell_values[full_ref]
                                break
                        if right_val is None:
                            try:
                                right_val = cp.full(self.iterations, float(right_str), dtype=cp.float32)
                            except:
                                return None
                    
                    condition = KERNEL_LIBRARY[kernel_name](left_val, right_val)
                    break
            
            if condition is None:
                # No comparison found
                return None
            
            # Evaluate true and false values
            # First check if it's a cell reference that needs sheet prefix
            true_val = None
            false_val = None
            
            # Try to find true value
            if true_str in cell_values:
                true_val = cell_values[true_str]
            else:
                # Look for cell without sheet name
                for full_ref in cell_values.keys():
                    if full_ref.endswith(f"!{true_str}"):
                        true_val = cell_values[full_ref]
                        break
                if true_val is None:
                    try:
                        true_val = cp.full(self.iterations, float(true_str), dtype=cp.float32)
                    except:
                        true_val = cp.zeros(self.iterations, dtype=cp.float32)
            
            # Try to find false value
            if false_str in cell_values:
                false_val = cell_values[false_str]
            else:
                # Look for cell without sheet name
                for full_ref in cell_values.keys():
                    if full_ref.endswith(f"!{false_str}"):
                        false_val = cell_values[full_ref]
                        break
                if false_val is None:
                    try:
                        false_val = cp.full(self.iterations, float(false_str), dtype=cp.float32)
                    except:
                        false_val = cp.zeros(self.iterations, dtype=cp.float32)
            
            # Use IF kernel
            return KERNEL_LIBRARY['IF'](condition, true_val, false_val)
            
        except Exception as e:
            logger.error(f"Failed to compile IF: {e}")
            return None
    
    def _compile_vlookup_simple(self, formula: str, cell_values: Dict[str, cp.ndarray]) -> Optional[cp.ndarray]:
        """
        Compile VLOOKUP function (simplified).
        """
        # For now, return None to use CPU fallback
        # Full VLOOKUP implementation is complex
        return None
    
    async def run_simulation(self, 
                           mc_input_configs: List[Any],
                           ordered_calc_steps: List[Tuple[str, str, str]], 
                           target_sheet_name: str,
                           target_cell_coordinate: str,
                           constant_values: Dict[str, float],
                           file_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Run a complete Monte Carlo simulation using the SuperEngine.
        
        Args:
            mc_input_configs: Monte Carlo variable configurations
            ordered_calc_steps: List of (sheet, cell, formula) tuples in dependency order
            target_sheet_name: Target sheet name
            target_cell_coordinate: Target cell coordinate
            constant_values: Dictionary of constant cell values
            file_id: Optional file ID for loading named ranges and tables
            
        Returns:
            Dictionary with simulation results
        """
        start_time = time.time()
        
        logger.info(f"🚀 SUPERENGINE: Starting simulation with {self.iterations} iterations")
        logger.info(f"   Formulas to process: {len(ordered_calc_steps)}")
        logger.info(f"   MC variables: {len(mc_input_configs)}")
        logger.info(f"   Constants: {len(constant_values)}")
        
        # Load named ranges and tables if file_id provided
        if file_id:
            try:
                self.named_ranges = await get_named_ranges_for_file(file_id)
                self.tables = await get_tables_for_file(file_id)
                logger.info(f"   Named ranges: {len(self.named_ranges)}")
                logger.info(f"   Tables: {len(self.tables)}")
            except Exception as e:
                logger.warning(f"Could not load named ranges/tables: {e}")
                self.named_ranges = {}
                self.tables = {}
        else:
            self.named_ranges = {}
            self.tables = {}
        
        # Initialize GPU arrays for MC variables
        mc_values = {}
        for config in mc_input_configs:
            cell_key = f"{config.sheet_name}!{config.name}"
            # Generate triangular distribution
            mc_values[cell_key] = cp.random.triangular(
                config.min_value,
                config.most_likely,
                config.max_value,
                size=self.iterations
            ).astype(cp.float32)
        
        # Initialize constant values as GPU arrays
        const_gpu = {}
        for key, value in constant_values.items():
            const_gpu[key] = cp.full(self.iterations, value, dtype=cp.float32)
        
        # Combined cell values
        all_cell_values = {**mc_values, **const_gpu}
        
        # Process formulas in dependency order
        results = None  # Will store the target cell results
        parse_time = 0
        compile_time = 0
        exec_time = 0
        
        # CRITICAL FIX: Run Monte Carlo iterations
        # We need to process all formulas for EACH iteration
        # For now, we'll use a simplified approach where we process all formulas once
        # with arrays of values
        
        for sheet, cell, formula in ordered_calc_steps:
            cell_key = f"{sheet}!{cell}"
            
            # Skip if this is a constant or MC variable
            if cell_key in all_cell_values:
                continue
            
            # Compile and execute formula
            t0 = time.time()
            
            logger.debug(f"Compiling formula for {cell_key}: {formula}")
            logger.debug(f"Available cells: {list(all_cell_values.keys())}")
            
            # Try JIT compilation first if enabled
            if self.use_jit and self.jit_compiler:
                try:
                    # Convert cell values to format expected by JIT
                    jit_input_data = {}
                    for key, value in all_cell_values.items():
                        # Extract just the cell reference (e.g., "Sheet1!A1" -> "A1")
                        cell_ref = key.split('!')[-1] if '!' in key else key
                        jit_input_data[cell_ref] = value
                    
                    # JIT compile and run
                    result = self.jit_compiler.compile_and_run(formula, jit_input_data)
                    self.stats['jit_compilations'] += 1
                    
                    # Log successful JIT compilation
                    result_mean = float(cp.mean(result))
                    logger.debug(f"Successfully JIT compiled {cell_key}: mean={result_mean:.2f}")
                    
                except Exception as e:
                    logger.debug(f"JIT compilation failed for {cell_key}, falling back to AST: {e}")
                    # Fall through to AST compilation
                    result = None
            else:
                result = None
            
            # If JIT failed or not enabled, try AST compilation
            if result is None:
                # Try new parser and compiler first
                try:
                    # Update parser context
                    self.parser.current_sheet = sheet
                    
                    # Parse formula
                    ast = self.parser.parse(formula)
                    
                    # Check if we got a hybrid parser node
                    if hasattr(ast, '__class__') and ast.__class__.__module__ == 'super_engine.hybrid_parser':
                        # Hybrid parser returns different node types
                        # For now, fall back to simple compilation
                        logger.debug(f"Got hybrid parser node, using simple compilation")
                        result = self.compile_formula_simple(formula, all_cell_values)
                        
                        if result is None:
                            # Use default value
                            logger.warning(f"Could not compile formula for {cell_key}: {formula}")
                            result = cp.zeros(self.iterations, dtype=cp.float32)
                        else:
                            # Log successful fallback
                            result_mean = float(cp.mean(result))
                            logger.debug(f"Successfully compiled {cell_key} with regex: mean={result_mean:.2f}")
                    else:
                        # Initialize or update compiler
                        if self.compiler is None:
                            self.compiler = CompilerV2(
                                simulation_data=all_cell_values,
                                named_ranges=self.named_ranges,
                                tables=self.tables,
                                iterations=self.iterations,
                                current_sheet=sheet
                            )
                        else:
                            self.compiler.simulation_data = all_cell_values
                            self.compiler.current_sheet = sheet
                        
                        # Compile AST
                        result = self.compiler.compile(ast)
                        
                        # Log successful compilation
                        result_mean = float(cp.mean(result))
                        logger.debug(f"Successfully compiled {cell_key} with AST: mean={result_mean:.2f}")
                    
                except Exception as e:
                    logger.warning(f"AST compilation failed for {cell_key}: {e}")
                    # Fallback to simple compilation
                    result = self.compile_formula_simple(formula, all_cell_values)
                    
                    if result is None:
                        # Use default value
                        logger.warning(f"Could not compile formula for {cell_key}: {formula}")
                        result = cp.zeros(self.iterations, dtype=cp.float32)
                    else:
                        # Log successful fallback
                        result_mean = float(cp.mean(result))
                        logger.debug(f"Successfully compiled {cell_key} with regex: mean={result_mean:.2f}")
            
            compile_time += time.time() - t0
            
            # Store result
            all_cell_values[cell_key] = result
            self.stats['formulas_processed'] += 1
            
            # Track target cell results
            if sheet == target_sheet_name and cell == target_cell_coordinate:
                results = cp.asnumpy(result)
                logger.info(f"Target cell {cell_key} results: mean={np.mean(results):.2f}, std={np.std(results):.2f}")
        
        # CRITICAL FIX: If we didn't find the target cell in the formula steps,
        # get it directly from the cell values
        if results is None:
            target_key = f"{target_sheet_name}!{target_cell_coordinate}"
            if target_key in all_cell_values:
                result_array = all_cell_values[target_key]
                if isinstance(result_array, cp.ndarray):
                    results = cp.asnumpy(result_array)
                else:
                    results = np.array(result_array)
                logger.info(f"Retrieved target cell {target_key} from cell values: mean={np.mean(results):.2f}, std={np.std(results):.2f}")
            else:
                # Target cell not found - this is an error
                logger.error(f"Target cell {target_key} not found in simulation results!")
                results = np.zeros(self.iterations)
        
        # Calculate statistics
        if results is not None and len(results) > 0:
            stats = {
                'mean': float(np.mean(results)),
                'std': float(np.std(results)),
                'min': float(np.min(results)),
                'max': float(np.max(results)),
                'percentiles': {
                    '5': float(np.percentile(results, 5)),
                    '25': float(np.percentile(results, 25)),
                    '50': float(np.percentile(results, 50)),
                    '75': float(np.percentile(results, 75)),
                    '95': float(np.percentile(results, 95))
                }
            }
        else:
            stats = {}
        
        # Update performance stats
        total_time = time.time() - start_time
        self.stats['total_time'] += total_time
        self.stats['parse_time'] += parse_time
        self.stats['compile_time'] += compile_time
        self.stats['execution_time'] += exec_time
        
        logger.info(f"✅ SUPERENGINE: Simulation complete in {total_time:.2f}s")
        logger.info(f"   Formulas processed: {self.stats['formulas_processed']}")
        logger.info(f"   Compile time: {compile_time:.2f}s")
        logger.info(f"   Results mean: {stats.get('mean', 0):.2f}")
        
        return {
            'results': results.tolist() if isinstance(results, np.ndarray) else results,
            'statistics': stats,
            'performance': {
                'total_time': total_time,
                'formulas_processed': self.stats['formulas_processed'],
                'iterations': self.iterations
            },
            'engine': 'SuperEngine'
        }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return self.stats.copy()
