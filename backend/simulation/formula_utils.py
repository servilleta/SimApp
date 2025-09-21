import re
from typing import Set, Tuple, Dict, List, Any
from collections import deque
import time
import logging

# ------------------------------------------------------------------
# CONFIGURABLE LIMITS FOR LARGE/COMPLEX FILES
# ------------------------------------------------------------------
# MAX_DEPENDENCY_NODES : hard cap on breadth-first exploration.
# ANALYSIS_TIMEOUT_SECONDS : wall-clock timeout for full analysis.
# These can be tuned without touching logic below.

MAX_DEPENDENCY_NODES = 100_000  # was 10_000 - Standard limit
MAX_DEPENDENCY_NODES_ARROW = 500_000  # Higher limit for Arrow engine (5x more)
ANALYSIS_TIMEOUT_SECONDS = 600  # was 300

# --- Column/Cell Coordinate Parsing Utilities ---

def _col_str_to_int(col_str: str) -> int:
    """Converts Excel column letters (A, B, ..., Z, AA, AB, ...) to a 0-indexed integer."""
    num = 0
    for char in col_str.upper():
        num = num * 26 + (ord(char) - ord('A') + 1)
    return num - 1 # 0-indexed

def _col_int_to_str(col_int: int) -> str:
    """Converts a 0-indexed integer back to Excel column letters."""
    col_str = ""
    num = col_int + 1 # 1-indexed for conversion
    while num > 0:
        num, remainder = divmod(num - 1, 26)
        col_str = chr(65 + remainder) + col_str
    return col_str

def _parse_cell_coord(coord_str: str) -> Tuple[str, int]:
    """Parses a cell coordinate string (e.g., "A1", "BC23") into its column string and 1-indexed row number."""
    match = re.match(r"([A-Z]+)([1-9][0-9]*)", coord_str.upper())
    if not match:
        raise ValueError(f"Invalid cell coordinate format: {coord_str}")
    return match.group(1), int(match.group(2))

def _expand_cell_range(start_cell_coord: str, end_cell_coord: str, sheet_name: str) -> Set[Tuple[str, str]]:
    """Expands a cell range (e.g., "A1:B2" on "Sheet1") into a set of (sheet_name, cell_coord_str) tuples."""
    start_col_str, start_row_int = _parse_cell_coord(start_cell_coord)
    end_col_str, end_row_int = _parse_cell_coord(end_cell_coord)

    start_col_int = _col_str_to_int(start_col_str)
    end_col_int = _col_str_to_int(end_col_str)

    # Ensure start is top-left and end is bottom-right
    min_col = min(start_col_int, end_col_int)
    max_col = max(start_col_int, end_col_int)
    min_row = min(start_row_int, end_row_int)
    max_row = max(start_row_int, end_row_int)

    expanded_cells: Set[Tuple[str, str]] = set()
    for r in range(min_row, max_row + 1):
        for c in range(min_col, max_col + 1):
            cell_c_str = _col_int_to_str(c)
            expanded_cells.add((sheet_name, f"{cell_c_str}{r}"))
    return expanded_cells

# --- Regular Expressions for Cell and Range References ---

# Group 1 (optional): The raw sheet name (e.g., "'My Sheet'" or "Sheet1")
# Group 2: Cell coordinate (e.g., "A1", "B2", "$D$3")
# Uses re.IGNORECASE for formula function names like SUM, AVERAGE, etc.
# Cell references themselves are typically case-insensitive in Excel but often written uppercase.
# We will uppercase cell_coord internally for consistency.
CELL_REFERENCE_REGEX = re.compile(
    r"(?:('[\w\s]+'|[A-Za-z0-9_]+)!)?(\$?[A-Z]+\$?[1-9][0-9]*)", re.IGNORECASE
)

# Group 1 (optional): Raw sheet name
# Group 2: Start cell of range (e.g., "A1", "$A$1")  
# Group 3: End cell of range (e.g., "B2", "$B$2")
# Updated to handle absolute references with $ signs
RANGE_REFERENCE_REGEX = re.compile(
    r"(?:('[\w\s]+'|[A-Za-z0-9_]+)!)?(\$?[A-Z]+\$?[1-9][0-9]*):(\$?[A-Z]+\$?[1-9][0-9]*)", re.IGNORECASE
)

def _parse_range_string(range_str: str, current_sheet_name: str) -> Tuple[str, str, str]:
    """
    Parses a range string (e.g., "Sheet1!A1:B2" or "A1:B2") into (resolved_sheet_name, start_cell, end_cell).
    Uses the global RANGE_REFERENCE_REGEX.
    """
    match = RANGE_REFERENCE_REGEX.fullmatch(range_str) # Use fullmatch to ensure the whole string is a range
    if not match:
        raise ValueError(f"Invalid range string format: {range_str}")

    raw_sheet_name = match.group(1)
    start_cell = match.group(2).upper()
    end_cell = match.group(3).upper()

    resolved_sheet_name = current_sheet_name
    if raw_sheet_name:
        if raw_sheet_name.startswith("'") and raw_sheet_name.endswith("'"):
            resolved_sheet_name = raw_sheet_name[1:-1]
        else:
            resolved_sheet_name = raw_sheet_name
    
    return resolved_sheet_name, start_cell, end_cell

def extract_cell_dependencies(formula_string: str, current_sheet_name: str) -> Set[Tuple[str, str]]:
    """
    Extracts unique cell dependencies (sheet_name, cell_coordinate) from a formula string.
    Supports single cells (A1, Sheet1!B2) and ranges (A1:C5, Sheet2!B2:D10).
    """
    dependencies: Set[Tuple[str, str]] = set()
    
    if not isinstance(formula_string, str):
        return dependencies
        
    if formula_string.startswith('='):
        formula_string = formula_string[1:]

    # Phase 1: Extract and expand ranges
    # We need to process ranges first and potentially remove them from the string 
    # to avoid single cell regex matching parts of a range later, though set handles duplicates.
    # A more robust way is to iterate and replace matched parts, but for now, two passes with a set is simpler.

    for match in RANGE_REFERENCE_REGEX.finditer(formula_string):
        raw_sheet_name = match.group(1)
        start_cell = match.group(2).upper()
        end_cell = match.group(3).upper()
        
        resolved_sheet_name = current_sheet_name
        if raw_sheet_name:
            if raw_sheet_name.startswith("'") and raw_sheet_name.endswith("'"):
                resolved_sheet_name = raw_sheet_name[1:-1]
            else:
                resolved_sheet_name = raw_sheet_name
        
        dependencies.update(_expand_cell_range(start_cell, end_cell, resolved_sheet_name))

    # Phase 2: Extract single cell references
    for match in CELL_REFERENCE_REGEX.finditer(formula_string):
        # Check if the matched single cell is part of an already processed range. 
        # This is tricky without modifying the string or more complex logic.
        # For now, we rely on the fact that `_expand_cell_range` and `add` to set handles overlaps.
        # A potential issue: if CELL_REFERENCE_REGEX matches "A1" in "A1:B5". 
        # Let's ensure ranges are processed to avoid this. 
        # A better approach would be one pass tokenizing, but this is an incremental step.
        # To avoid CELL_REFERENCE_REGEX matching parts of a range like A1 in A1:B5, we should ensure
        # the single cell regex does not match if it's followed by a colon (part of a range).
        # However, this is complex. A simpler way: the set `dependencies` will naturally handle overlaps.
        # If A1 from A1:B5 is added, and then A1 is matched singly, the set won't change.

        raw_sheet_name = match.group(1) 
        cell_coord = match.group(2).upper()

        resolved_sheet_name = current_sheet_name
        if raw_sheet_name:
            if raw_sheet_name.startswith("'") and raw_sheet_name.endswith("'"):
                resolved_sheet_name = raw_sheet_name[1:-1]
            else:
                resolved_sheet_name = raw_sheet_name
        
        dependencies.add((resolved_sheet_name, cell_coord))
        
    return dependencies

def get_evaluation_order(
    target_sheet_name: str, 
    target_cell_coord: str, 
    all_formulas: Dict[str, Dict[str, str]], 
    mc_input_cells: Set[Tuple[str, str]],
    engine_type: str = "enhanced"  # CRITICAL FIX: Add engine_type parameter
) -> List[Tuple[str, str, str]]: # Returns ordered list of (sheet, cell, formula)
    """
    Determines the correct evaluation order for a target cell and its precedent formulas
    using topological sort. Detects circular dependencies.
    `mc_input_cells` are cells whose values are provided by Monte Carlo simulation, not by their own formulas.
    `engine_type` determines the dependency analysis limits (Arrow engine gets higher limits).
    """
    logger = logging.getLogger(__name__)
    start_time = time.time()
    TIMEOUT_SECONDS = ANALYSIS_TIMEOUT_SECONDS
    
    logger.info(f"🔍 Starting formula dependency analysis for {target_sheet_name}!{target_cell_coord}")
    
    graph: Dict[Tuple[str, str], Set[Tuple[str, str]]] = {}
    reverse_graph: Dict[Tuple[str, str], Set[Tuple[str, str]]] = {}
    all_relevant_cells: Set[Tuple[str, str]] = set()
    
    queue = deque([(target_sheet_name, target_cell_coord.upper())])
    visited_for_exploration: Set[Tuple[str, str]] = set()
    
    # CRITICAL FIX: Use higher limits for large-file engines (Arrow and Power)
    if engine_type in ["arrow", "power"]:
        max_iterations = MAX_DEPENDENCY_NODES_ARROW
        logger.info(f"🚀 [{engine_type.upper()}] Using enhanced dependency analysis limit: {max_iterations:,} nodes")
    else:
        max_iterations = MAX_DEPENDENCY_NODES
        logger.info(f"🔧 [STANDARD] Using standard dependency analysis limit: {max_iterations:,} nodes")
    
    iteration_count = 0

    while queue and iteration_count < max_iterations:
        # Check timeout every 100 iterations
        if iteration_count % 100 == 0:
            elapsed = time.time() - start_time
            if elapsed > TIMEOUT_SECONDS:
                logger.error(f"❌ Formula dependency analysis timed out after {elapsed:.2f}s")
                raise ValueError(f"Formula dependency analysis timed out after {elapsed:.2f}s. This may indicate a very complex dependency tree or circular references.")
        
        iteration_count += 1
        current_s_name, current_c_coord = queue.popleft()

        if (current_s_name, current_c_coord) in visited_for_exploration:
            continue
        visited_for_exploration.add((current_s_name, current_c_coord))
        all_relevant_cells.add((current_s_name, current_c_coord))

        if (current_s_name, current_c_coord) in mc_input_cells:
            continue

        formula = all_formulas.get(current_s_name, {}).get(current_c_coord)
        if not formula:
            continue 

        try:
            dependencies_for_node = extract_cell_dependencies(formula, current_s_name)
        except Exception as e:
            logger.warning(f"⚠️ Failed to extract dependencies from formula in {current_s_name}!{current_c_coord}: {e}")
            continue
        
        current_node = (current_s_name, current_c_coord)
        if current_node not in graph:
            graph[current_node] = set()
            
        for dep_s_name, dep_c_coord in dependencies_for_node:
            dep_node = (dep_s_name, dep_c_coord)
            graph[current_node].add(dep_node)
            
            if dep_node not in reverse_graph:
                reverse_graph[dep_node] = set()
            reverse_graph[dep_node].add(current_node)
            
            if dep_node not in visited_for_exploration:
                queue.append(dep_node)
                all_relevant_cells.add(dep_node)
    
    # Check if we hit the iteration limit
    if iteration_count >= max_iterations:
        logger.error(f"❌ Formula dependency analysis exceeded maximum iterations ({max_iterations})")
        raise ValueError(f"Formula dependency analysis exceeded maximum iterations ({max_iterations}). This indicates a very complex or circular dependency structure.")

    logger.info(f"🔍 Found {len(all_relevant_cells)} relevant cells in {iteration_count} iterations")

    nodes_to_calc: Dict[Tuple[str, str], Set[Tuple[str, str]]] = {}
    for node in all_relevant_cells:
        node_formula = all_formulas.get(node[0], {}).get(node[1])
        # ✅ CRITICAL FIX: Include Monte Carlo input cells in calculation if they have formulas
        # This ensures Monte Carlo variable values propagate to dependent cells
        if node_formula:  # Calculate ANY cell that has a formula, including MC inputs
            nodes_to_calc[node] = set()
            try:
                actual_deps = extract_cell_dependencies(node_formula, node[0])
                for dep_node in actual_deps:
                    dep_node_formula = all_formulas.get(dep_node[0], {}).get(dep_node[1])
                    # Only include dependencies that have actual formulas
                    # Monte Carlo input cells without formulas stay as injected constants
                    if dep_node_formula:
                        nodes_to_calc[node].add(dep_node)
            except Exception as e:
                logger.warning(f"⚠️ Failed to process dependencies for {node[0]}!{node[1]}: {e}")

    # ✅ MONTE CARLO FIX: Force inclusion of cells that reference MC inputs
    # Even if they're not in the target dependency chain, they need to be calculated
    mc_reference_cells = set()
    for sheet_name, sheet_formulas in all_formulas.items():
        for cell_coord, formula in sheet_formulas.items():
            if formula and isinstance(formula, str):
                # Check if this formula references any Monte Carlo input cells
                try:
                    for mc_sheet, mc_cell in mc_input_cells:
                        if f"${mc_cell}" in formula or f"{mc_cell}" in formula:
                            cell_key = (sheet_name, cell_coord)
                            mc_reference_cells.add(cell_key)
                            if cell_key not in nodes_to_calc:
                                nodes_to_calc[cell_key] = set()
                                logger.info(f"🔧 [MC_FIX] Force-included MC reference: {sheet_name}!{cell_coord} = {formula}")
                except Exception as e:
                    continue
    
    logger.info(f"🔧 [MC_FIX] Added {len(mc_reference_cells)} Monte Carlo reference cells")
    logger.info(f"🔍 Building calculation order for {len(nodes_to_calc)} formula cells")

    in_degree_kahn: Dict[Tuple[str, str], int] = {node: 0 for node in nodes_to_calc}
    graph_kahn: Dict[Tuple[str, str], List[Tuple[str, str]]] = {node: [] for node in nodes_to_calc}

    for node, node_dependencies in nodes_to_calc.items():
        for dep in node_dependencies:
            if dep in nodes_to_calc: 
                in_degree_kahn[node] += 1
                graph_kahn[dep].append(node)
    
    kahn_queue = deque([node for node in nodes_to_calc if in_degree_kahn[node] == 0])
    ordered_eval_list: List[Tuple[str, str, str]] = []
    
    # Add timeout protection for topological sort as well
    topo_iteration_count = 0
    max_topo_iterations = max(len(nodes_to_calc) * 2, MAX_DEPENDENCY_NODES)
    
    while kahn_queue and topo_iteration_count < max_topo_iterations:
        topo_iteration_count += 1
        
        # Check timeout
        elapsed = time.time() - start_time
        if elapsed > TIMEOUT_SECONDS:
            logger.error(f"❌ Topological sort timed out after {elapsed:.2f}s")
            raise ValueError(f"Topological sort timed out after {elapsed:.2f}s")
        
        u_node = kahn_queue.popleft()
        u_formula = all_formulas.get(u_node[0], {}).get(u_node[1])
        if u_formula:
            ordered_eval_list.append((u_node[0], u_node[1], u_formula))
        
        for v_node in graph_kahn.get(u_node, []):
            in_degree_kahn[v_node] -= 1
            if in_degree_kahn[v_node] == 0:
                kahn_queue.append(v_node)
                
    if len(ordered_eval_list) < len(nodes_to_calc):
        problem_nodes = [n for n in nodes_to_calc if n not in {(s,c) for s,c,_ in ordered_eval_list}]
        logger.error(f"❌ Circular dependency detected. Problem nodes: {problem_nodes[:5]}")  # Limit output
        raise ValueError(f"Circular dependency or unresolvable formula detected. Problematic formula cells: {problem_nodes[:10]}")  # Limit to first 10

    # ✅ CRITICAL FIX: Don't add placeholder formulas for Monte Carlo inputs
    # Monte Carlo values are injected directly into current_values by the engine
    # Adding placeholder formulas like "=F4" would overwrite the injected values with 0
    # Only include MC cells that actually have real formulas in the Excel file
    mc_calc_order = []
    existing_cells = {(s, c) for s, c, _ in ordered_eval_list}
    
    for mc_cell in mc_input_cells:
        sheet, cell = mc_cell
        # Only include if the MC input cell actually has a real formula in Excel
        if mc_cell in existing_cells:
            # This MC input has a real formula and is already in ordered_eval_list
            continue
        # Don't add MC inputs without formulas - they get their values from direct injection
    
    # No need to prepend MC cells - they get their values from injection, not formula evaluation
    final_order = ordered_eval_list
    
    elapsed = time.time() - start_time
    logger.info(f"✅ Formula dependency analysis completed in {elapsed:.3f}s - {len(ordered_eval_list)} formulas total (MC inputs handled by direct injection)")
    
    # CRITICAL FIX: Only show first 10 formulas to prevent performance bottleneck
    # (Previously this was logging all 69,954+ formulas causing the hang)
    logger.warning(f"[EVAL_ORDER_DEBUG] Ordered evaluation list for {target_sheet_name}!{target_cell_coord} (showing first 10 of {len(final_order)}):")
    for i, (s, c, f) in enumerate(final_order[:10]):  # Only show first 10
        logger.warning(f"[EVAL_ORDER_DEBUG] {s}!{c} = {f}")
        if 'VLOOKUP' in str(f).upper():
            logger.warning(f"[EVAL_ORDER_DEBUG] VLOOKUP formula detected: {s}!{c} = {f}")
    
    if len(final_order) > 10:
        logger.warning(f"[EVAL_ORDER_DEBUG] ... and {len(final_order) - 10} more formulas")
    
    return final_order

# Placeholder for graph cycle detection
# def _has_cycle(graph, start_node, visited, recursion_stack):
#     pass 