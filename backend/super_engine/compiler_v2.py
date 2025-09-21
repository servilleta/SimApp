"""
SUPERENGINE - AST Compiler V2
=============================
Enhanced compiler that works with the new parser and provides better error handling
and performance optimization.
"""

import logging
import cupy as cp
from typing import Dict, Any, Union, Optional, List
import numpy as np

from super_engine.parser_v2 import (
    ASTNode, NumberNode, StringNode, BooleanNode, ErrorNode,
    CellNode, RangeNode, NamedRangeNode, TableRefNode,
    FunctionNode, BinaryOpNode, UnaryOpNode, ArrayNode,
    TokenType
)
from super_engine.gpu_kernels import KERNEL_LIBRARY, is_gpu_available

logger = logging.getLogger(__name__)

class CompilerV2:
    """
    Enhanced AST compiler that executes parsed formulas on GPU.
    """
    
    def __init__(self, 
                 simulation_data: Dict[str, cp.ndarray],
                 named_ranges: Dict[str, Any],
                 tables: Dict[str, Any],
                 iterations: int,
                 current_sheet: str = 'Sheet1'):
        """
        Initialize the compiler with simulation context.
        
        Args:
            simulation_data: Cell reference -> GPU array mapping
            named_ranges: Named range definitions
            tables: Table definitions
            iterations: Number of Monte Carlo iterations
            current_sheet: Current worksheet name
        """
        if not is_gpu_available():
            raise RuntimeError("GPU not available for CompilerV2")
        
        self.simulation_data = simulation_data
        self.named_ranges = named_ranges
        self.tables = tables
        self.iterations = iterations
        self.current_sheet = current_sheet
        
        # Statistics
        self.stats = {
            'nodes_compiled': 0,
            'gpu_kernels_called': 0,
            'cache_hits': 0,
            'errors': 0
        }
        
        # Cache for compiled expressions
        self.cache = {}
        
        logger.info("✅ CompilerV2 initialized")
    
    def compile(self, node: ASTNode) -> cp.ndarray:
        """
        Compile an AST node to GPU array result.
        
        Args:
            node: AST node to compile
            
        Returns:
            GPU array with results
        """
        try:
            result = self._compile_node(node)
            self.stats['nodes_compiled'] += 1
            return result
        except Exception as e:
            self.stats['errors'] += 1
            logger.error(f"Compilation error: {e}")
            # Return error array
            return cp.full(self.iterations, cp.nan, dtype=cp.float32)
    
    def _compile_node(self, node: ASTNode) -> cp.ndarray:
        """Compile a single AST node."""
        
        # Numbers
        if isinstance(node, NumberNode):
            return cp.full(self.iterations, node.value, dtype=cp.float32)
        
        # Strings (convert to NaN for now)
        elif isinstance(node, StringNode):
            # In a full implementation, we'd handle string operations
            return cp.full(self.iterations, cp.nan, dtype=cp.float32)
        
        # Booleans
        elif isinstance(node, BooleanNode):
            return cp.full(self.iterations, float(node.value), dtype=cp.float32)
        
        # Errors
        elif isinstance(node, ErrorNode):
            # Return NaN for errors
            return cp.full(self.iterations, cp.nan, dtype=cp.float32)
        
        # Cell references
        elif isinstance(node, CellNode):
            return self._compile_cell(node)
        
        # Ranges
        elif isinstance(node, RangeNode):
            return self._compile_range(node)
        
        # Named ranges
        elif isinstance(node, NamedRangeNode):
            return self._compile_named_range(node)
        
        # Table references
        elif isinstance(node, TableRefNode):
            return self._compile_table_ref(node)
        
        # Functions
        elif isinstance(node, FunctionNode):
            return self._compile_function(node)
        
        # Binary operations
        elif isinstance(node, BinaryOpNode):
            return self._compile_binary_op(node)
        
        # Unary operations
        elif isinstance(node, UnaryOpNode):
            return self._compile_unary_op(node)
        
        # Arrays
        elif isinstance(node, ArrayNode):
            return self._compile_array(node)
        
        else:
            raise ValueError(f"Unknown node type: {type(node)}")
    
    def _compile_cell(self, node: CellNode) -> cp.ndarray:
        """Compile cell reference."""
        # Build cell key
        sheet = node.sheet or self.current_sheet
        cell_key = f"{sheet}!{node.column}{node.row}"
        
        # Look up in simulation data
        if cell_key in self.simulation_data:
            return self.simulation_data[cell_key]
        
        # Try without sheet prefix
        simple_key = f"{node.column}{node.row}"
        if simple_key in self.simulation_data:
            return self.simulation_data[simple_key]
        
        # Cell not found - return zeros
        logger.warning(f"Cell {cell_key} not found in simulation data")
        return cp.zeros(self.iterations, dtype=cp.float32)
    
    def _compile_range(self, node: RangeNode) -> cp.ndarray:
        """Compile range reference."""
        # For now, collect all cells in range and stack them
        # In a full implementation, we'd expand the range properly
        start_result = self._compile_node(node.start)
        end_result = self._compile_node(node.end)
        
        # Return stacked array
        return cp.stack([start_result, end_result])
    
    def _compile_named_range(self, node: NamedRangeNode) -> cp.ndarray:
        """Compile named range reference."""
        if node.name in self.named_ranges:
            # Resolve named range to cells/range
            range_def = self.named_ranges[node.name]
            # In a full implementation, we'd parse the range definition
            # For now, return placeholder
            logger.info(f"Named range {node.name} -> {range_def}")
            return cp.zeros(self.iterations, dtype=cp.float32)
        
        logger.warning(f"Named range {node.name} not found")
        return cp.zeros(self.iterations, dtype=cp.float32)
    
    def _compile_table_ref(self, node: TableRefNode) -> cp.ndarray:
        """Compile table reference."""
        # In a full implementation, we'd resolve table columns
        logger.warning(f"Table references not yet implemented: {node.table_name}")
        return cp.zeros(self.iterations, dtype=cp.float32)
    
    def _compile_function(self, node: FunctionNode) -> cp.ndarray:
        """Compile function call."""
        func_name = node.name.upper()
        
        # Check if we have a GPU kernel for this function
        if func_name not in KERNEL_LIBRARY:
            logger.error(f"Unknown function: {func_name}")
            return cp.full(self.iterations, cp.nan, dtype=cp.float32)
        
        # Compile arguments
        args = [self._compile_node(arg) for arg in node.args]
        
        # Call appropriate kernel
        self.stats['gpu_kernels_called'] += 1
        
        # Special handling for different function types
        if func_name == 'IF':
            if len(args) != 3:
                raise ValueError(f"IF requires 3 arguments, got {len(args)}")
            return KERNEL_LIBRARY['IF'](args[0], args[1], args[2])
        
        elif func_name in ['SUM', 'AVERAGE', 'MIN', 'MAX']:
            # Aggregate functions
            if len(args) == 1 and args[0].ndim > 1:
                # Operating on a range
                return KERNEL_LIBRARY[func_name](args[0])
            else:
                # Multiple arguments
                stacked = cp.stack(args) if len(args) > 1 else args[0]
                return KERNEL_LIBRARY[func_name](stacked)
        
        elif func_name in ['AND', 'OR']:
            # Logical functions with multiple arguments
            if len(args) < 2:
                raise ValueError(f"{func_name} requires at least 2 arguments")
            result = args[0]
            for arg in args[1:]:
                result = KERNEL_LIBRARY[func_name]([result, arg])
            return result
        
        elif func_name == 'VLOOKUP':
            import logging
            logger = logging.getLogger(__name__)
            logger.warning("[VLOOKUP DEBUG] Entered VLOOKUP handler in compiler_v2.py for function: VLOOKUP")
            if len(args) < 3:
                raise ValueError("VLOOKUP requires at least 3 arguments")
            # Try GPU VLOOKUP first
            gpu_result = KERNEL_LIBRARY['VLOOKUP'](args[0], args[1], args[2])
            logger.warning(f"[VLOOKUP DEBUG] GPU result: {gpu_result}")
            # Check if all values are NaN or result is empty (string/object fallback needed)
            if (hasattr(gpu_result, 'size') and gpu_result.size == 0) or (hasattr(gpu_result, 'all') and cp.isnan(gpu_result).all()):
                logger.warning("[VLOOKUP FALLBACK] All-NaN or empty detected from GPU VLOOKUP. Triggering CPU fallback for string/object lookup.")
                # Import the excel_vlookup function from simulation engine
                import sys
                import os
                sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                from simulation.engine import excel_vlookup
                
                # Convert GPU arrays to CPU for processing
                lookup_values_cpu = cp.asnumpy(args[0]) if hasattr(args[0], 'get') else args[0]
                table_array_cpu = cp.asnumpy(args[1]) if hasattr(args[1], 'get') else args[1]
                col_index_cpu = int(args[2]) if hasattr(args[2], 'item') else args[2]
                
                # If lookup_values is an array, do a vectorized CPU fallback
                if hasattr(lookup_values_cpu, '__len__') and not isinstance(lookup_values_cpu, str):
                    cpu_results = []
                    for val in lookup_values_cpu:
                        result = excel_vlookup(val, table_array_cpu, col_index_cpu, True)
                        # Convert #N/A or other Excel errors to NaN
                        if isinstance(result, str) and result.startswith('#'):
                            cpu_results.append(float('nan'))
                        else:
                            try:
                                cpu_results.append(float(result))
                            except (ValueError, TypeError):
                                cpu_results.append(float('nan'))
                    # Convert back to GPU array
                    cpu_results_array = cp.array(cpu_results, dtype=cp.float32)
                    logger.warning(f"[VLOOKUP FALLBACK] CPU results (vectorized): {cpu_results_array}")
                    return cpu_results_array
                else:
                    cpu_result = excel_vlookup(lookup_values_cpu, table_array_cpu, col_index_cpu, True)
                    # Convert result to float
                    if isinstance(cpu_result, str) and cpu_result.startswith('#'):
                        cpu_result = float('nan')
                    else:
                        try:
                            cpu_result = float(cpu_result)
                        except (ValueError, TypeError):
                            cpu_result = float('nan')
                    logger.warning(f"[VLOOKUP FALLBACK] CPU result (single): {cpu_result}")
                    # Return as GPU array
                    return cp.full(self.iterations, cpu_result, dtype=cp.float32)
            logger.warning("[VLOOKUP DEBUG] Returning GPU result.")
            return gpu_result
        
        elif func_name in ['NORMAL', 'LOGNORMAL']:
            # Distribution functions
            if len(args) != 2:
                raise ValueError(f"{func_name} requires 2 arguments")
            return KERNEL_LIBRARY[func_name](args[0], args[1], size=self.iterations)
        
        elif func_name in ['TRIANGULAR']:
            if len(args) != 3:
                raise ValueError("TRIANGULAR requires 3 arguments")
            return KERNEL_LIBRARY[func_name](args[0], args[1], args[2], size=self.iterations)
        
        elif func_name == 'RAYLEIGH':
            # RAYLEIGH(scale)
            if len(args) >= 1:
                scale = self._compile_node(args[0])
                return KERNEL_LIBRARY['RAYLEIGH'](scale, self.iterations)
            else:
                raise ValueError("RAYLEIGH requires 1 argument: scale")
        
        # --- Text Functions ---
        elif func_name == 'CONCATENATE':
            # CONCATENATE(text1, text2, ...)
            if len(args) >= 1:
                compiled_args = [self._compile_node(arg) for arg in args]
                return KERNEL_LIBRARY['CONCATENATE'](*compiled_args)
            else:
                raise ValueError("CONCATENATE requires at least 1 argument")
        
        elif func_name == 'LEFT':
            # LEFT(text, num_chars)
            if len(args) >= 2:
                text = self._compile_node(args[0])
                num_chars = self._compile_node(args[1])
                return KERNEL_LIBRARY['LEFT'](text, num_chars)
            elif len(args) == 1:
                # Default to 1 character if num_chars not specified
                text = self._compile_node(args[0])
                return KERNEL_LIBRARY['LEFT'](text, 1)
            else:
                raise ValueError("LEFT requires 1-2 arguments: text, [num_chars]")
        
        elif func_name == 'RIGHT':
            # RIGHT(text, num_chars)
            if len(args) >= 2:
                text = self._compile_node(args[0])
                num_chars = self._compile_node(args[1])
                return KERNEL_LIBRARY['RIGHT'](text, num_chars)
            elif len(args) == 1:
                # Default to 1 character if num_chars not specified
                text = self._compile_node(args[0])
                return KERNEL_LIBRARY['RIGHT'](text, 1)
            else:
                raise ValueError("RIGHT requires 1-2 arguments: text, [num_chars]")
        
        elif func_name == 'LEN':
            # LEN(text)
            if len(args) >= 1:
                text = self._compile_node(args[0])
                return KERNEL_LIBRARY['LEN'](text)
            else:
                raise ValueError("LEN requires 1 argument: text")
        
        elif func_name == 'MID':
            # MID(text, start_num, num_chars)
            if len(args) >= 3:
                text = self._compile_node(args[0])
                start_num = self._compile_node(args[1])
                num_chars = self._compile_node(args[2])
                return KERNEL_LIBRARY['MID'](text, start_num, num_chars)
            else:
                raise ValueError("MID requires 3 arguments: text, start_num, num_chars")
        
        else:
            # Default: pass all arguments to kernel
            kernel = KERNEL_LIBRARY[func_name]
            return kernel(*args)
    
    def _compile_binary_op(self, node: BinaryOpNode) -> cp.ndarray:
        """Compile binary operation."""
        left = self._compile_node(node.left)
        right = self._compile_node(node.right)
        
        # Map token type to kernel name
        op_map = {
            TokenType.PLUS: 'add',
            TokenType.MINUS: 'subtract',
            TokenType.MULTIPLY: 'multiply',
            TokenType.DIVIDE: 'divide',
            TokenType.POWER: 'power',
            TokenType.EQ: 'eq',
            TokenType.NEQ: 'neq',
            TokenType.LT: 'lt',
            TokenType.GT: 'gt',
            TokenType.LTE: 'lte',
            TokenType.GTE: 'gte',
            TokenType.AND: 'AND',
            TokenType.OR: 'OR',
        }
        
        if node.op in op_map:
            kernel_name = op_map[node.op]
            if kernel_name in KERNEL_LIBRARY:
                self.stats['gpu_kernels_called'] += 1
                if kernel_name in ['AND', 'OR']:
                    # Logical operations take list of args
                    return KERNEL_LIBRARY[kernel_name]([left, right])
                else:
                    return KERNEL_LIBRARY[kernel_name](left, right)
        
        # Concatenation (return NaN for now)
        if node.op == TokenType.CONCAT:
            return cp.full(self.iterations, cp.nan, dtype=cp.float32)
        
        raise ValueError(f"Unknown binary operation: {node.op}")
    
    def _compile_unary_op(self, node: UnaryOpNode) -> cp.ndarray:
        """Compile unary operation."""
        operand = self._compile_node(node.operand)
        
        if node.op == TokenType.MINUS:
            return -operand
        elif node.op == TokenType.NOT:
            return KERNEL_LIBRARY['NOT'](operand)
        else:
            raise ValueError(f"Unknown unary operation: {node.op}")
    
    def _compile_array(self, node: ArrayNode) -> cp.ndarray:
        """Compile array constant."""
        # Compile all elements
        compiled_rows = []
        for row in node.elements:
            compiled_row = [self._compile_node(elem) for elem in row]
            compiled_rows.append(compiled_row)
        
        # Stack into 2D array
        # For Monte Carlo, we'd need to broadcast this properly
        # For now, return first element
        if compiled_rows and compiled_rows[0]:
            return compiled_rows[0][0]
        
        return cp.zeros(self.iterations, dtype=cp.float32)
    
    def get_stats(self) -> Dict[str, int]:
        """Get compilation statistics."""
        return self.stats.copy()

# Testing
if __name__ == '__main__':
    from super_engine.parser_v2 import WorldClassExcelParser
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Create test data
    iterations = 1000
    sim_data = {
        'Sheet1!A1': cp.random.rand(iterations) * 100,
        'Sheet1!B1': cp.random.rand(iterations) * 200,
        'Sheet1!C1': cp.random.rand(iterations) * 300,
    }
    
    # Create parser and compiler
    parser = WorldClassExcelParser({'current_sheet': 'Sheet1'})
    compiler = CompilerV2(
        simulation_data=sim_data,
        named_ranges={},
        tables={},
        iterations=iterations
    )
    
    # Test formulas
    test_formulas = [
        '=A1+B1',
        '=A1*2+B1/C1',
        '=IF(A1>50,B1,C1)',
        '=SUM(A1,B1,C1)',
        '=MAX(A1,B1)-MIN(A1,C1)',
    ]
    
    for formula in test_formulas:
        try:
            print(f"\nCompiling: {formula}")
            
            # Parse
            ast = parser.parse(formula)
            print(f"Parsed successfully")
            
            # Compile
            result = compiler.compile(ast)
            print(f"Compiled successfully")
            print(f"Result shape: {result.shape}")
            print(f"Result mean: {cp.mean(result):.2f}")
            print(f"Result std: {cp.std(result):.2f}")
            
        except Exception as e:
            print(f"Error: {e}")
    
    # Print stats
    print(f"\nCompiler stats: {compiler.get_stats()}") 