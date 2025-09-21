"""
SUPERENGINE - JIT Formula Compiler
==================================
This module implements the Just-In-Time (JIT) formula compilation, the core
of the Tier 4 performance enhancements.

Key Innovation: Instead of walking the AST and calling multiple small kernels,
this compiler translates the entire AST into a single, monolithic CUDA C++
kernel. This kernel is then compiled on the fly using CuPy's RawKernel.

Benefits:
- Eliminates kernel launch overhead.
- Maximizes GPU occupancy.
- Enables the GPU's instruction-level parallelism and register optimizations.
- Results in a significant performance boost for complex formulas.
"""

import logging
import cupy as cp
from typing import Dict, Any, Tuple, List, Union, Set

from super_engine.parser import FormulaParser
from super_engine.monolithic_jit import MonolithicJitCompiler, CacheManager

logger = logging.getLogger(__name__)

class JitCompiler:
    """
    Translates an AST into a single CUDA kernel and executes it.
    Enhanced with monolithic kernel fusion support.
    """
    def __init__(self, enable_monolithic: bool = True):
        self.parser = FormulaParser()
        self._code_cache = {} # Cache for previously compiled kernels
        self.enable_monolithic = enable_monolithic
        
        # Initialize monolithic compiler if enabled
        if enable_monolithic:
            self.monolithic_compiler = MonolithicJitCompiler(
                max_cache_size=2000,
                enable_profiling=True
            )
            self.cache_manager = CacheManager(
                max_memory_gb=2.0,
                cache_policy='adaptive'
            )
            logger.info("✅ SUPERENGINE: JIT Compiler initialized with Monolithic Fusion")
        else:
            self.monolithic_compiler = None
            self.cache_manager = None
            logger.info("✅ SUPERENGINE: JIT Compiler initialized (Standard Mode)")

    def compile_batch(self, formulas: List[Tuple[str, str]], 
                     input_data: Dict[str, cp.ndarray],
                     shared_inputs: Set[str] = None) -> Dict[str, cp.ndarray]:
        """
        Compile and execute multiple formulas as a monolithic kernel.
        
        Args:
            formulas: List of (output_name, formula_string) tuples
            input_data: Dictionary of input arrays
            shared_inputs: Set of inputs that are frequently accessed
            
        Returns:
            Dictionary mapping output names to result arrays
        """
        if not self.enable_monolithic or not self.monolithic_compiler:
            # Fall back to individual compilation
            results = {}
            for output_name, formula in formulas:
                results[output_name] = self.compile_and_run(formula, input_data)
            return results
        
        # Use monolithic compilation
        iterations = list(input_data.values())[0].size
        if shared_inputs is None:
            # Auto-detect shared inputs (those used in multiple formulas)
            shared_inputs = self._detect_shared_inputs(formulas)
        
        # Compile batch
        metadata = self.monolithic_compiler.compile_formula_batch(
            formulas, shared_inputs, iterations
        )
        
        # Execute monolithic kernel
        results = self.monolithic_compiler.execute_monolithic_kernel(
            metadata, input_data, iterations
        )
        
        return results
    
    def _detect_shared_inputs(self, formulas: List[Tuple[str, str]]) -> Set[str]:
        """Auto-detect inputs that are shared across multiple formulas."""
        input_counts = {}
        
        for _, formula in formulas:
            # Simple regex to find cell references
            import re
            refs = re.findall(r'\b[A-Z]+\d+\b', formula)
            for ref in refs:
                input_counts[ref] = input_counts.get(ref, 0) + 1
        
        # Consider inputs used in 2+ formulas as shared
        return {inp for inp, count in input_counts.items() if count >= 2}

    def compile_and_run(self, formula_string: str, input_data: Dict[str, cp.ndarray]) -> cp.ndarray:
        """
        JIT-compiles and runs a formula.
        Enhanced with result caching support.

        Args:
            formula_string: The Excel formula string.
            input_data: A dictionary mapping all input variable names (e.g., "A1",
                        "B1", "MyNamedRange", "Table1[Column1]") to their
                        corresponding CuPy data arrays.

        Returns:
            A CuPy array with the final result.
        """
        # Check result cache if available
        if self.cache_manager:
            cache_key = formula_string
            dependencies = {k: v[0].item() if v.size > 0 else 0 
                          for k, v in input_data.items()}
            
            cached_result = self.cache_manager.get(cache_key, dependencies)
            if cached_result is not None:
                logger.debug(f"🎯 Cache hit for formula: {formula_string[:50]}...")
                return cached_result
        
        # Standard compilation path
        if formula_string in self._code_cache:
            kernel, output_name, input_names = self._code_cache[formula_string]
        else:
            # We need the size of the arrays for the CUDA kernel code's boundary check.
            # We can get it from any of the input arrays.
            if not input_data:
                raise ValueError("Input data cannot be empty for JIT compilation.")
            
            array_size = list(input_data.values())[0].size
            ast = self.parser.parse(formula_string)
            cuda_code, output_name, input_names = self._ast_to_cuda(ast, array_size)
            
            # Define the kernel using CuPy's RawKernel
            # The kernel name ('jit_formula') must match the name in the code string.
            kernel = cp.RawKernel(cuda_code, 'jit_formula')
            
            self._code_cache[formula_string] = (kernel, output_name, input_names)
            logger.info(f"JIT compiled formula: {formula_string}")
        
        # Prepare kernel arguments
        output_array = cp.zeros(input_data[input_names[0]].shape, dtype=cp.float64)
        args = [output_array] + [input_data[name] for name in input_names]
        
        # Configure grid and block dimensions for GPU execution
        # These dimensions can be tuned for optimal performance.
        block_size = 256
        grid_size = (output_array.size + block_size - 1) // block_size
        
        # Execute the JIT-compiled kernel
        kernel((grid_size,), (block_size,), args)
        
        # Cache result if available
        if self.cache_manager:
            self.cache_manager.put(cache_key, output_array, dependencies)
        
        return output_array

    def _ast_to_cuda(self, ast: Union[Tuple, Any], array_size: int) -> Tuple[str, str, List[str]]:
        """
        Walks the AST and generates a CUDA C++ source code string.
        """
        # This is the core translation logic. It converts the AST into C++ expressions.
        # It keeps track of inputs and assigns temporary variables for intermediate results.
        
        cuda_body = []
        input_names = []
        temp_var_count = 0

        alias_map = {}

        def get_input_name(cell_ref: str) -> str:
            """Return a safe local variable name for a cell reference and track pointer inputs."""
            if cell_ref not in input_names:
                input_names.append(cell_ref)  # pointer param name
            # Use cached alias to ensure consistent naming inside expression string
            if cell_ref not in alias_map:
                alias_map[cell_ref] = f"{cell_ref}_val"
            return alias_map[cell_ref]

        def walk(node: Union[Tuple, Any]) -> str:
            nonlocal temp_var_count
            
            # Handle non-tuple nodes
            if not isinstance(node, tuple):
                # If it's a simple value, convert to string
                return str(node)
            
            # Handle empty tuples
            if len(node) == 0:
                return "0.0"
            
            node_type = node[0]

            # Handle literal values
            if node_type == 'number':
                return str(node[1])
            elif node_type == 'string':
                # For now, strings are not supported in calculations
                return "0.0"
            elif node_type == 'bool':
                return "1.0" if node[1] else "0.0"
            
            # Handle cell references
            elif node_type in ('cell', 'named_range'):
                return get_input_name(node[1])
            elif node_type == 'table_ref':
                # Create a single reference name like "Table1[Column1]"
                ref_name = f"{node[1]}[{node[2]}]"
                return get_input_name(ref_name)
            
            # Handle unary operations
            elif node_type == 'neg':
                operand = walk(node[1])
                temp_var = f"temp{temp_var_count}"
                temp_var_count += 1
                cuda_body.append(f"    double {temp_var} = -{operand};")
                return temp_var
            
            # Handle binary operations
            elif node_type in ('add', 'sub', 'mul', 'div', 'power'):
                if len(node) < 3:
                    return "0.0"  # Invalid node structure
                    
                left = walk(node[1])
                right = walk(node[2])
                
                if node_type == 'power':
                    temp_var = f"temp{temp_var_count}"
                    temp_var_count += 1
                    cuda_body.append(f"    double {temp_var} = pow({left}, {right});")
                    return temp_var
                else:
                    op_map = {'add': '+', 'sub': '-', 'mul': '*', 'div': '/'}
                    op = op_map[node_type]
                    
                    # Create a temporary variable for the result of this operation.
                    temp_var = f"temp{temp_var_count}"
                    temp_var_count += 1
                    cuda_body.append(f"    double {temp_var} = {left} {op} {right};")
                    return temp_var
            
            # Handle comparison operations
            elif node_type in ('gt', 'lt', 'gte', 'lte', 'eq', 'neq'):
                if len(node) < 3:
                    return "0.0"
                    
                left = walk(node[1])
                right = walk(node[2])
                
                op_map = {
                    'gt': '>', 'lt': '<', 'gte': '>=', 
                    'lte': '<=', 'eq': '==', 'neq': '!='
                }
                op = op_map[node_type]
                
                temp_var = f"temp{temp_var_count}"
                temp_var_count += 1
                cuda_body.append(f"    double {temp_var} = ({left} {op} {right}) ? 1.0 : 0.0;")
                return temp_var
            
            # Handle function calls
            elif node_type == 'function_call':
                func_name = node[1]
                args = node[2] if len(node) > 2 else []
                
                # Support more Excel functions
                if func_name == 'SUM':
                    if args and isinstance(args[0], tuple) and args[0][0] == 'range':
                        # For ranges, we'd need to expand them - for now, return 0
                        return "0.0"
                    else:
                        # Sum individual arguments
                        arg_vars = [walk(arg) for arg in args]
                        temp_var = f"temp{temp_var_count}"
                        temp_var_count += 1
                        sum_expr = ' + '.join(arg_vars) if arg_vars else "0.0"
                        cuda_body.append(f"    double {temp_var} = {sum_expr};")
                        return temp_var
                
                elif func_name == 'AVERAGE':
                    arg_vars = [walk(arg) for arg in args]
                    if arg_vars:
                        temp_var = f"temp{temp_var_count}"
                        temp_var_count += 1
                        sum_expr = ' + '.join(arg_vars)
                        cuda_body.append(f"    double {temp_var} = ({sum_expr}) / {len(arg_vars)}.0;")
                        return temp_var
                    return "0.0"
                
                elif func_name in ('MIN', 'MAX'):
                    arg_vars = [walk(arg) for arg in args]
                    if arg_vars:
                        temp_var = f"temp{temp_var_count}"
                        temp_var_count += 1
                        if func_name == 'MIN':
                            cuda_body.append(f"    double {temp_var} = {arg_vars[0]};")
                            for av in arg_vars[1:]:
                                cuda_body.append(f"    {temp_var} = fmin({temp_var}, {av});")
                        else:  # MAX
                            cuda_body.append(f"    double {temp_var} = {arg_vars[0]};")
                            for av in arg_vars[1:]:
                                cuda_body.append(f"    {temp_var} = fmax({temp_var}, {av});")
                        return temp_var
                    return "0.0"
                
                elif func_name == 'IF':
                    if len(args) >= 3:
                        condition = walk(args[0])
                        true_val = walk(args[1])
                        false_val = walk(args[2])
                        temp_var = f"temp{temp_var_count}"
                        temp_var_count += 1
                        cuda_body.append(f"    double {temp_var} = ({condition} > 0.0) ? {true_val} : {false_val};")
                        return temp_var
                    return "0.0"
                
                elif func_name == 'ABS':
                    if args:
                        arg_var = walk(args[0])
                        temp_var = f"temp{temp_var_count}"
                        temp_var_count += 1
                        cuda_body.append(f"    double {temp_var} = fabs({arg_var});")
                        return temp_var
                    return "0.0"
                
                elif func_name == 'SQRT':
                    if args:
                        arg_var = walk(args[0])
                        temp_var = f"temp{temp_var_count}"
                        temp_var_count += 1
                        cuda_body.append(f"    double {temp_var} = sqrt({arg_var});")
                        return temp_var
                    return "0.0"
                
                elif func_name in ('SIN', 'COS', 'TAN'):
                    if args:
                        arg_var = walk(args[0])
                        temp_var = f"temp{temp_var_count}"
                        temp_var_count += 1
                        cuda_body.append(f"    double {temp_var} = {func_name.lower()}({arg_var});")
                        return temp_var
                    return "0.0"
                
                elif func_name in ('LOG', 'LN'):
                    if args:
                        arg_var = walk(args[0])
                        temp_var = f"temp{temp_var_count}"
                        temp_var_count += 1
                        cuda_body.append(f"    double {temp_var} = log({arg_var});")
                        return temp_var
                    return "0.0"
                
                elif func_name == 'EXP':
                    if args:
                        arg_var = walk(args[0])
                        temp_var = f"temp{temp_var_count}"
                        temp_var_count += 1
                        cuda_body.append(f"    double {temp_var} = exp({arg_var});")
                        return temp_var
                    return "0.0"
                
                elif func_name == 'POWER':
                    if len(args) >= 2:
                        base = walk(args[0])
                        exponent = walk(args[1])
                        temp_var = f"temp{temp_var_count}"
                        temp_var_count += 1
                        cuda_body.append(f"    double {temp_var} = pow({base}, {exponent});")
                        return temp_var
                    return "0.0"
                
                # Text functions (simplified for numeric values)
                elif func_name == 'CONCATENATE':
                    # For JIT, we'll handle numeric concatenation only
                    logger.warning(f"JIT: CONCATENATE is experimental - numeric concatenation only")
                    if args:
                        # Convert all args to string representation and concatenate
                        # This is a placeholder - actual implementation would be more complex
                        temp_var = f"temp{temp_var_count}"
                        temp_var_count += 1
                        # For now, just sum the values as a placeholder
                        arg_vars = [walk(arg) for arg in args]
                        sum_expr = ' + '.join(arg_vars) if arg_vars else "0.0"
                        cuda_body.append(f"    double {temp_var} = {sum_expr}; // CONCATENATE placeholder")
                        return temp_var
                    return "0.0"
                
                elif func_name == 'LEFT':
                    # For numeric values, we'll simulate by dividing
                    logger.warning(f"JIT: LEFT is experimental - numeric simulation only")
                    if args:
                        value = walk(args[0])
                        num_chars = walk(args[1]) if len(args) > 1 else "1.0"
                        temp_var = f"temp{temp_var_count}"
                        temp_var_count += 1
                        # Placeholder: divide by 10^(total_digits - num_chars)
                        cuda_body.append(f"    double {temp_var} = floor({value} / pow(10.0, floor(log10(fabs({value}) + 1.0)) - {num_chars} + 1.0)); // LEFT placeholder")
                        return temp_var
                    return "0.0"
                
                elif func_name == 'RIGHT':
                    # For numeric values, we'll simulate with modulo
                    logger.warning(f"JIT: RIGHT is experimental - numeric simulation only")
                    if args:
                        value = walk(args[0])
                        num_chars = walk(args[1]) if len(args) > 1 else "1.0"
                        temp_var = f"temp{temp_var_count}"
                        temp_var_count += 1
                        # Placeholder: use modulo to get rightmost digits
                        cuda_body.append(f"    double {temp_var} = fmod({value}, pow(10.0, {num_chars})); // RIGHT placeholder")
                        return temp_var
                    return "0.0"
                
                elif func_name == 'LEN':
                    # For numeric values, count digits
                    logger.warning(f"JIT: LEN is experimental - numeric digit count only")
                    if args:
                        value = walk(args[0])
                        temp_var = f"temp{temp_var_count}"
                        temp_var_count += 1
                        # Count digits in the number
                        cuda_body.append(f"    double {temp_var} = floor(log10(fabs({value}) + 1.0)) + 1.0; // LEN placeholder")
                        return temp_var
                    return "0.0"
                
                elif func_name == 'MID':
                    # For numeric values, extract middle digits
                    logger.warning(f"JIT: MID is experimental - numeric simulation only")
                    if len(args) >= 3:
                        value = walk(args[0])
                        start_pos = walk(args[1])
                        num_chars = walk(args[2])
                        temp_var = f"temp{temp_var_count}"
                        temp_var_count += 1
                        # Placeholder implementation
                        cuda_body.append(f"    double {temp_var} = {value}; // MID placeholder - not fully implemented")
                        return temp_var
                    return "0.0"
                
                else:
                    # Unsupported function - return 0
                    logger.warning(f"Unsupported function in JIT: {func_name}")
                    return "0.0"
            
            # Handle ranges (for now, just return 0)
            elif node_type == 'range':
                return "0.0"
            
            # Unknown node type
            else:
                logger.warning(f"Unknown node type in JIT compiler: {node_type}")
                return "0.0"

        # Start walking the tree from the root
        final_var = walk(ast)
        output_name = "output"

        # Construct the full CUDA kernel string.
        # It takes pointers to the output array and all input arrays.
        param_declarations = f"double* {output_name}"
        if input_names:
            param_declarations += ", " + ', '.join([f"const double* {name}" for name in input_names])
        
        # This creates the lines like 'double A1 = A1_in[i];'
        input_loading_lines = []
        for name in input_names:
            alias = alias_map.get(name, f"{name}_val")
            input_loading_lines.append(f"    const double {alias} = {name}[i];")
        input_loading = '\n'.join(input_loading_lines)
        # This creates the lines like 'double temp0 = A1 + B1;'
        formula_body = '\n'.join(cuda_body)

        cuda_code = f"""
        extern "C" __global__ void jit_formula({param_declarations}) {{
            int i = blockDim.x * blockIdx.x + threadIdx.x;
            if (i >= {array_size}) return;

            // Load inputs
{input_loading}

            // Formula body
{formula_body}

            // Store result
            {output_name}[i] = {final_var};
        }}
        """
        return cuda_code, output_name, input_names

    def get_statistics(self) -> Dict[str, Any]:
        """Get JIT compiler statistics."""
        stats = {
            'kernel_cache_size': len(self._code_cache),
            'monolithic_enabled': self.enable_monolithic
        }
        
        if self.monolithic_compiler:
            stats['monolithic_stats'] = self.monolithic_compiler.get_cache_statistics()
        
        if self.cache_manager:
            stats['cache_manager_stats'] = self.cache_manager.get_statistics()
        
        return stats


# Example usage for testing and demonstration
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    try:
        jit = JitCompiler()
        
        sim_data = {
            "A1": cp.arange(5, dtype=cp.float64),
            "B1": cp.arange(5, dtype=cp.float64) * 2,
            "C1": cp.arange(5, dtype=cp.float64) * 3,
        }
        
        formula = "=A1 + B1 * C1"
        result = jit.compile_and_run(formula, sim_data)
        
        print(f"\nFormula: {formula}")
        print(f"Inputs:\nA1: {sim_data['A1']}\nB1: {sim_data['B1']}\nC1: {sim_data['C1']}")
        print(f"JIT Result: {result}")
        # Expected: 0+0*0=0, 1+2*3=7, 2+4*6=26, 3+6*9=57, 4+8*12=100
        print(f"Expected: {[0, 7, 26, 57, 100]}")

        # Test simple addition
        formula2 = "=A1 + B1"
        result2 = jit.compile_and_run(formula2, sim_data)
        print(f"\nFormula: {formula2}")
        print(f"Result: {result2}")
        print(f"Expected: {sim_data['A1'] + sim_data['B1']}")

        # Test with table references
        table_input_data = {**sim_data, "Sales[Revenue]": cp.random.rand(5) * 1000}
        formula_table = "=Sales[Revenue] * A1"
        try:
            result_table = jit.compile_and_run(formula_table, table_input_data)
            print(f"\nFormula: {formula_table}")
            print(f"JIT Result: {result_table}")
        except Exception as e:
            print(f"An error occurred during Table Reference test: {e}")

    except Exception as e:
        print(f"An error occurred during JIT Compiler test: {e}") 