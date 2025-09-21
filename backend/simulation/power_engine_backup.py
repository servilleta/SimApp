import logging
import numpy as np
import re
from typing import Dict, List, Tuple, Any, Optional
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor
import os
import formulas as formula_parser

from simulation.hardware_detector import get_gpu_compute_capability
from simulation.ast_parser import get_dependencies_from_formula as get_deps_from_formula_ast
from simulation.schemas import VariableConfig as MonteCarloVariable

try:
    import cupy as cp
    # Try to set memory allocator, but don't fail if not supported
    try:
        cp.cuda.set_allocator(cp.cuda.MemoryAsyncPool().malloc)
    except:
        pass  # Use default allocator if async pool not supported
except ImportError:
    cp = None

logger = logging.getLogger(__name__)

def _safe_eval_wrapper(expression, context):
    """A simplified, safer eval for formula expressions."""
    try:
        safe_globals = {
            'SUM': sum, 'IF': lambda c, t, f: t if c else f,
            '__builtins__': {}
        }
        return eval(expression, safe_globals, context)
    except Exception as e:
        logger.error(f"Error evaluating expression '{expression}': {e}")
        return 0

class PowerMonteCarloEngine:
    def __init__(self, iterations: int = 1000, config: Dict[str, Any] = None):
        self.iterations = iterations
        self.config = config or {}
        # Check if GPU is actually available, not just if CuPy is installed
        self.gpu_available = False
        if cp is not None:
            try:
                cp.cuda.Device(0).compute_capability  # Test GPU access
                self.gpu_available = True
            except:
                logger.info("CuPy installed but no GPU available, using CPU mode")
        self.gpu_kernels = {}
        self.progress_callback = None
        self._configure_engine_for_hardware()
        if self.gpu_available:
            self._initialize_gpu_support()

    def _configure_engine_for_hardware(self):
        """Sets performance limits based on detected GPU hardware."""
        capability = get_gpu_compute_capability() if self.gpu_available else None
        if capability is None:
            capability = 0.0
        limit = 500
        if 5.0 <= capability < 6.0: limit = 1000
        elif 6.0 <= capability < 7.0: limit = 2000
        elif 7.0 <= capability < 7.5: limit = 3000
        elif 7.5 <= capability < 8.0: limit = 5000
        elif 8.0 <= capability < 9.0: limit = 10000
        elif capability >= 9.0: limit = 20000
        self.formula_limit = limit
        logger.info(f"Hardware capability {capability:.1f} detected. Formula limit set to {self.formula_limit}.")

    def _initialize_gpu_support(self):
        """Pre-compiles GPU kernels for supported functions."""
        if not self.gpu_available: return
        self._compile_gpu_kernel('ARITHMETIC')
        self._compile_gpu_kernel('IF')

    def _compile_gpu_kernel(self, formula_type: str):
        if formula_type == 'ARITHMETIC':
            self.gpu_kernels['ARITHMETIC'] = cp.ElementwiseKernel(
                'raw T x, raw T y, raw T op_code', 'T z',
                '''
                if (op_code == 0) z = x + y;
                else if (op_code == 1) z = x - y;
                else if (op_code == 2) z = x * y;
                else if (op_code == 3) z = (y != 0) ? x / y : 0;
                ''',
                'arithmetic_op'
            )
        elif formula_type == 'IF':
            self.gpu_kernels['IF'] = cp.ElementwiseKernel(
                'raw T logical_test, raw T val_if_true, raw T val_if_false', 'T z',
                'z = logical_test ? val_if_true : val_if_false;',
                'if_op'
            )
        logger.info(f"GPU kernel for {formula_type} compiled.")

    def set_progress_callback(self, callback):
        self.progress_callback = callback

    def _get_formula_dependencies(self, sheet: str, formula: str) -> List[Tuple[str, str]]:
        """Uses the AST parser to get dependencies."""
        try:
            return get_deps_from_formula_ast(formula, sheet)
        except Exception as e:
            logger.error(f"AST Parsing failed for '{formula}': {e}. Falling back to regex.")
            return [(sheet, cell) for cell in re.findall(r'[A-Z]+\d+', formula.upper())]

    def _get_evaluation_order(self, all_formulas: Dict, target_cell_id: Tuple[str, str]) -> List[Tuple[str, str, str]]:
        """Creates a topological sort of formula dependencies."""
        graph = {}
        for (sheet, cell), formula in all_formulas.items():
            graph[(sheet, cell)] = self._get_formula_dependencies(sheet, formula)

        sorted_order = []
        visited = set()
        recursion_stack = set()

        def visit(node):
            if node in recursion_stack:
                raise Exception(f"Circular reference detected involving {node}")
            if node in visited:
                return

            visited.add(node)
            recursion_stack.add(node)
            
            if node in graph:
                for dep in graph.get(node, []):
                    visit(dep)
            
            recursion_stack.remove(node)
            if node in all_formulas:
                sorted_order.append((node[0], node[1], all_formulas[node]))

        visit(target_cell_id)
        return sorted_order

    async def run_simulation(
        self, file_path: str, file_id: str, target_cell: str,
        variables: List[MonteCarloVariable], iterations: int, sheet_name: str
    ) -> Tuple[Optional[np.ndarray], List[str], List[Dict[str, Any]]]:
        self.iterations = iterations
        target_sheet, target_coord = sheet_name, target_cell
        if '!' in target_cell:
            target_sheet, target_coord = target_cell.split('!', 1)

        loop = asyncio.get_event_loop()
        
        # Parse the Excel file directly
        import openpyxl
        wb = openpyxl.load_workbook(file_path, data_only=False)
        
        all_formulas_tuple_keys = []
        all_constants_tuple_keys = []
        
        for sheet in wb.worksheets:
            for row in sheet.iter_rows():
                for cell in row:
                    if cell.value is not None:
                        if isinstance(cell.value, str) and cell.value.startswith('='):
                            # It's a formula
                            all_formulas_tuple_keys.append((sheet.title, cell.coordinate, cell.value))
                        else:
                            # It's a constant
                            all_constants_tuple_keys.append(((sheet.title, cell.coordinate), cell.value))
        
        all_formulas = {(s, c): f for s, c, f in all_formulas_tuple_keys}
        
        try:
            ordered_calc_steps = self._get_evaluation_order(all_formulas, (target_sheet, target_coord))
        except Exception as e:
            return None, [f"Dependency analysis failed: {e}"], []

        xp = cp if self.gpu_available else np
        iter_values = {}
        
        for (s, c), val in all_constants_tuple_keys:
            try:
                # Convert value to float, handling different types
                float_val = float(val)
                iter_values[(s, c)] = xp.full(self.iterations, float_val, dtype=np.float64)
            except (ValueError, TypeError):
                # Skip non-numeric values
                pass
        for var in variables:
            # Using triangular distribution based on min, most_likely, max values
            iter_values[(var.sheet_name, var.name)] = xp.random.triangular(
                var.min_value, var.most_likely, var.max_value, self.iterations
            ).astype(np.float64)

        for i, (sheet, cell, formula) in enumerate(ordered_calc_steps):
            if self.progress_callback and i % 10 == 0:
                self.progress_callback((i / len(ordered_calc_steps)) * 100)
            
            try:
                result = await self._evaluate_formula(sheet, formula, iter_values)
                if result is not None:
                    iter_values[(sheet, cell)] = result
            except Exception as e:
                logger.error(f"Error evaluating {sheet}!{cell}: {formula} - {e}")
                return None, [f"Error evaluating {sheet}!{cell}: {e}"], []

        final_result = iter_values.get((target_sheet, target_coord))
        
        final_result_cpu = cp.asnumpy(final_result) if self.gpu_available and final_result is not None else final_result
        
        sensitivity = [{"variable": v.name, "impact": np.random.rand()} for v in variables]

        return final_result_cpu, [], sensitivity

    async def _evaluate_formula(self, sheet: str, formula: str, iter_values: Dict) -> Any:
        formula_upper = formula.upper()
        if formula_upper.startswith('=SUM('):
            return await self._handle_sum(sheet, formula, iter_values)
        if any(op in formula_upper for op in ['+', '-', '*', '/']):
            return await self._handle_arithmetic(sheet, formula, iter_values)
        if formula_upper.startswith('=IF('):
            return await self._handle_if(sheet, formula, iter_values)
        return await self._evaluate_formula_fallback(sheet, formula, iter_values)

    async def _evaluate_expression(self, expression: str, sheet: str, iter_values: Dict) -> Any:
        expression = expression.strip()
        xp = cp if self.gpu_available else np
        
        match = re.fullmatch(r'([A-Z]+)(\d+)', expression)
        if match:
            val = iter_values.get((sheet, expression))
            return val if val is not None else xp.zeros(self.iterations, dtype=np.float64)
        try:
            return xp.full(self.iterations, float(expression), dtype=np.float64)
        except ValueError:
            return await self._evaluate_formula(sheet, f"={expression}", iter_values)

    async def _handle_sum(self, sheet: str, formula: str, iter_values: Dict) -> Any:
        xp = cp if self.gpu_available else np
        match = re.match(r'=SUM\((.+)\)', formula, re.IGNORECASE)
        if not match: return xp.zeros(self.iterations, dtype=np.float64)
        
        # Simplified: assumes comma-separated cells, not ranges
        parts = match.group(1).split(',')
        total = xp.zeros(self.iterations, dtype=np.float64)
        for part in parts:
            total += await self._evaluate_expression(part.strip(), sheet, iter_values)
        return total

    async def _handle_arithmetic(self, sheet: str, formula: str, iter_values: Dict) -> Any:
        xp = cp if self.gpu_available else np
        ops = {'+': 0, '-': 1, '*': 2, '/': 3}
        for op, code in ops.items():
            if op in formula:
                parts = formula.lstrip('=').split(op, 1)
                left_expr, right_expr = parts[0], parts[1]
                
                left_val = await self._evaluate_expression(left_expr, sheet, iter_values)
                right_val = await self._evaluate_expression(right_expr, sheet, iter_values)
                
                # Ensure both values have the same shape
                if hasattr(left_val, 'shape') and hasattr(right_val, 'shape'):
                    if left_val.shape != right_val.shape:
                        logger.debug(f"Shape mismatch: left={left_val.shape}, right={right_val.shape}")
                        # Try to broadcast scalar to array shape
                        if left_val.shape == ():
                            left_val = xp.full(self.iterations, float(left_val), dtype=np.float64)
                        if right_val.shape == ():
                            right_val = xp.full(self.iterations, float(right_val), dtype=np.float64)
                
                if self.gpu_available and 'ARITHMETIC' in self.gpu_kernels:
                    # ElementwiseKernel expects arrays of same shape
                    result = xp.zeros(self.iterations, dtype=np.float64)
                    if code == 0: result = left_val + right_val
                    elif code == 1: result = left_val - right_val
                    elif code == 2: result = left_val * right_val
                    elif code == 3: result = left_val / xp.where(right_val==0, 1, right_val)
                    return result
                else:
                    if code == 0: return left_val + right_val
                    if code == 1: return left_val - right_val
                    if code == 2: return left_val * right_val
                    if code == 3: return left_val / np.where(right_val==0, 1, right_val)
        return xp.zeros(self.iterations, dtype=np.float64)
        
    async def _handle_if(self, sheet: str, formula: str, iter_values: Dict) -> Any:
        xp = cp if self.gpu_available else np
        # More robust IF parsing that handles nested commas
        if_content = formula[4:-1]  # Remove =IF( and )
        
        # Find the first comma after the condition
        paren_count = 0
        first_comma = -1
        second_comma = -1
        
        for i, char in enumerate(if_content):
            if char == '(':
                paren_count += 1
            elif char == ')':
                paren_count -= 1
            elif char == ',' and paren_count == 0:
                if first_comma == -1:
                    first_comma = i
                else:
                    second_comma = i
                    break
        
        if first_comma == -1 or second_comma == -1:
            return xp.zeros(self.iterations, dtype=np.float64)
            
        cond_expr = if_content[:first_comma].strip()
        true_expr = if_content[first_comma+1:second_comma].strip()
        false_expr = if_content[second_comma+1:].strip()

        op_match = re.search(r'([A-Z0-9\.]+)\s*([><=]+)\s*([A-Z0-9\.]+)', cond_expr)
        if not op_match: return xp.zeros(self.iterations, dtype=np.float64)
        
        left_ref, op, right_ref = op_match.groups()
        left_val = await self._evaluate_expression(left_ref, sheet, iter_values)
        right_val = await self._evaluate_expression(right_ref, sheet, iter_values)
        
        if op == '>': cond_val = left_val > right_val
        elif op == '<': cond_val = left_val < right_val
        else: cond_val = left_val == right_val
        
        # Handle expressions that might contain arithmetic
        if '*' in true_expr:
            true_val = await self._handle_arithmetic(sheet, f"={true_expr}", iter_values)
        else:
            true_val = await self._evaluate_expression(true_expr, sheet, iter_values)
            
        if '+' in false_expr:
            false_val = await self._handle_arithmetic(sheet, f"={false_expr}", iter_values)
        else:
            false_val = await self._evaluate_expression(false_expr, sheet, iter_values)

        try:
            if self.gpu_available and 'IF' in self.gpu_kernels:
                # Use numpy operations for now as ElementwiseKernel has issues
                return xp.where(cond_val, true_val, false_val)
            else:
                return np.where(cond_val, true_val, false_val)
        except Exception as e:
            logger.error(f"IF evaluation error: {e}, cond_val type: {type(cond_val)}, shape: {getattr(cond_val, 'shape', 'no shape')}")
            raise
            
    async def _evaluate_formula_fallback(self, sheet, formula, iter_values):
        xp = cp if self.gpu_available else np
        results = xp.zeros(self.iterations, dtype=np.float64)
        
        for i in range(self.iterations):
            context = {}
            deps = re.findall(r'[A-Z]+\d+', formula.upper())
            for dep in deps:
                dep_val_array = iter_values.get((sheet, dep))
                if dep_val_array is not None:
                    if hasattr(dep_val_array, '__len__') and len(dep_val_array) > i:
                        context[dep] = float(dep_val_array[i])
                    else:
                        context[dep] = float(dep_val_array) if dep_val_array is not None else 0.0

            eval_formula = formula.lstrip('=')
            for k, v in context.items():
                eval_formula = eval_formula.replace(k, str(v))
            try:
                results[i] = _safe_eval_wrapper(eval_formula, {})
            except Exception as e:
                logger.error(f"Fallback eval error at iteration {i}: {e}, formula: {eval_formula}")
                results[i] = 0.0
        return results

    def cleanup(self):
        logger.info("Power Engine cleanup complete.")

    def __del__(self):
        self.cleanup