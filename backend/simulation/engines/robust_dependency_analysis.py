"""
Robust Dependency Analysis Module for Power Engine
Implements Phase 1 of the Power Engine Robustness Plan
"""

import logging
import time
from typing import Dict, List, Tuple, Set, Any, Optional, Callable
from collections import deque, defaultdict
import gc
import signal
from contextlib import contextmanager

from ..formula_utils import extract_cell_dependencies, MAX_DEPENDENCY_NODES_ARROW, MAX_DEPENDENCY_NODES, ANALYSIS_TIMEOUT_SECONDS

logger = logging.getLogger(__name__)

class DependencyAnalysisTimeout(Exception):
    """Raised when dependency analysis exceeds timeout"""
    pass

class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is in OPEN state"""
    pass

class DependencyAnalysisCircuitBreaker:
    """Circuit breaker for dependency analysis to detect repeated failures"""
    
    def __init__(self, failure_threshold: int = 3, recovery_timeout: int = 300):
        self.failure_count = 0
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.last_failure_time = 0.0
        self.state = 'CLOSED'  # CLOSED, OPEN, HALF_OPEN
        
    def call_with_circuit_breaker(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        if self.state == 'OPEN':
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = 'HALF_OPEN'
                logger.info("[POWER_CIRCUIT_BREAKER] Moving to HALF_OPEN state")
            else:
                raise CircuitBreakerOpenError("Dependency analysis circuit breaker is OPEN")
        
        try:
            result = func(*args, **kwargs)
            if self.state == 'HALF_OPEN':
                self.state = 'CLOSED'
                self.failure_count = 0
                logger.info("[POWER_CIRCUIT_BREAKER] Moving to CLOSED state - recovery successful")
            return result
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = 'OPEN'
                logger.error(f"[POWER_CIRCUIT_BREAKER] Circuit breaker OPEN after {self.failure_count} failures")
            
            raise e

class CompactFormulaGraph:
    """Memory-efficient representation of formula dependencies using integer IDs"""
    
    def __init__(self):
        self.cell_to_id: Dict[Tuple[str, str], int] = {}
        self.id_to_cell: Dict[int, Tuple[str, str]] = {}
        self.dependencies: List[Tuple[int, int]] = []
        self.next_id = 0
        
    def add_cell(self, sheet: str, cell: str) -> int:
        """Add cell and return its integer ID"""
        key = (sheet, cell.upper())
        if key not in self.cell_to_id:
            self.cell_to_id[key] = self.next_id
            self.id_to_cell[self.next_id] = key
            self.next_id += 1
        return self.cell_to_id[key]
    
    def add_dependency(self, from_cell: Tuple[str, str], to_cell: Tuple[str, str]):
        """Add dependency relationship"""
        from_id = self.add_cell(*from_cell)
        to_id = self.add_cell(*to_cell)
        self.dependencies.append((from_id, to_id))
    
    def get_cell_count(self) -> int:
        """Get total number of cells in graph"""
        return len(self.cell_to_id)
    
    def get_dependency_count(self) -> int:
        """Get total number of dependencies"""
        return len(self.dependencies)

class RobustDependencyAnalyzer:
    """
    Robust dependency analyzer that can handle large Excel files with fallback strategies
    """
    
    def __init__(self, progress_callback: Optional[Callable] = None):
        self.progress_callback = progress_callback
        self.circuit_breaker = DependencyAnalysisCircuitBreaker()
        self.stats = {
            'total_formulas': 0,
            'processed_formulas': 0,
            'chunks_processed': 0,
            'fallback_used': False,
            'analysis_time': 0.0
        }
        
    def analyze_dependencies_robust(
        self,
        target_sheet_name: str,
        target_cell_coord: str,
        all_formulas: Dict[str, Dict[str, str]],
        mc_input_cells: Set[Tuple[str, str]],
        engine_type: str = "power"
    ) -> List[Tuple[str, str, str]]:
        """
        Main entry point for robust dependency analysis with multiple fallback levels
        """
        start_time = time.time()
        
        try:
            # Level 1: Full dependency analysis with optimizations
            result = self.circuit_breaker.call_with_circuit_breaker(
                self._analyze_dependencies_full,
                target_sheet_name, target_cell_coord, all_formulas, mc_input_cells, engine_type
            )
            logger.info("[POWER_ROBUST] Full dependency analysis successful")
            return result
            
        except (DependencyAnalysisTimeout, CircuitBreakerOpenError) as e:
            logger.warning(f"[POWER_ROBUST] Fallback Level 1: {str(e)}")
            self.stats['fallback_used'] = True
            
            try:
                # Level 2: Chunked dependency analysis
                result = self._analyze_dependencies_chunked(
                    target_sheet_name, target_cell_coord, all_formulas, mc_input_cells, engine_type
                )
                logger.info("[POWER_ROBUST] Chunked dependency analysis successful")
                return result
                
            except Exception as e2:
                logger.warning(f"[POWER_ROBUST] Fallback Level 2: {str(e2)}")
                
                try:
                    # Level 3: Simplified dependency analysis
                    result = self._analyze_dependencies_simplified(
                        target_sheet_name, target_cell_coord, all_formulas, mc_input_cells
                    )
                    logger.info("[POWER_ROBUST] Simplified dependency analysis successful")
                    return result
                    
                except Exception as e3:
                    logger.warning(f"[POWER_ROBUST] Fallback Level 3: {str(e3)}")
                    
                    # Level 4: Basic sheet-based ordering (last resort)
                    result = self._analyze_dependencies_basic(
                        target_sheet_name, target_cell_coord, all_formulas
                    )
                    logger.info("[POWER_ROBUST] Basic dependency analysis successful")
                    return result
        
        finally:
            self.stats['analysis_time'] = time.time() - start_time
    
    def _analyze_dependencies_full(
        self,
        target_sheet_name: str,
        target_cell_coord: str,
        all_formulas: Dict[str, Dict[str, str]],
        mc_input_cells: Set[Tuple[str, str]],
        engine_type: str
    ) -> List[Tuple[str, str, str]]:
        """Full dependency analysis with timeout and progress tracking"""
        start_time = time.time()
        timeout_seconds = 180  # POWER ENGINE FIX: Reduced from 5 minutes to 3 minutes to prevent hangs
        
        logger.info(f"[POWER_ROBUST] Starting full dependency analysis for {target_sheet_name}!{target_cell_coord}")
        
        # POWER ENGINE FIX: Add heartbeat mechanism during dependency analysis
        last_heartbeat = time.time()
        heartbeat_interval = 30  # Emit heartbeat every 30 seconds
        
        def emit_analysis_heartbeat(phase: str):
            nonlocal last_heartbeat
            current_time = time.time()
            if current_time - last_heartbeat > heartbeat_interval:
                logger.info(f"💓 [POWER_ROBUST] Heartbeat - {phase} - {current_time - start_time:.1f}s elapsed")
                last_heartbeat = current_time
        
        with self._timeout_handler(timeout_seconds):
            # Phase 1: Build dependency graph with progress tracking
            emit_analysis_heartbeat("building_dependency_graph")
            graph, all_relevant_cells = self._build_dependency_graph_with_progress(
                target_sheet_name, target_cell_coord, all_formulas, mc_input_cells, engine_type
            )
            
            # Phase 2: Calculate evaluation order with progress tracking
            emit_analysis_heartbeat("calculating_evaluation_order")
            ordered_eval_list = self._calculate_evaluation_order_with_progress(
                graph, all_relevant_cells, all_formulas, mc_input_cells
            )
            
            elapsed = time.time() - start_time
            logger.info(f"[POWER_ROBUST] Full analysis completed in {elapsed:.3f}s - {len(ordered_eval_list)} formulas")
            
            return ordered_eval_list
    
    def _analyze_dependencies_chunked(
        self,
        target_sheet_name: str,
        target_cell_coord: str,
        all_formulas: Dict[str, Dict[str, str]],
        mc_input_cells: Set[Tuple[str, str]],
        engine_type: str,
        chunk_size: int = 5000
    ) -> List[Tuple[str, str, str]]:
        """Chunked dependency analysis to prevent memory issues"""
        start_time = time.time()
        logger.info(f"[POWER_ROBUST] Starting chunked dependency analysis (chunk_size={chunk_size})")
        
        # Build compact graph representation
        compact_graph = CompactFormulaGraph()
        
        # Group formulas by sheet for processing
        sheets_formulas = defaultdict(list)
        for sheet_name, sheet_formulas in all_formulas.items():
            for cell, formula in sheet_formulas.items():
                sheets_formulas[sheet_name].append((cell, formula))
        
        total_sheets = len(sheets_formulas)
        processed_sheets = 0
        
        # Process each sheet as a chunk
        for sheet_name, formulas in sheets_formulas.items():
            self._update_progress(f"Processing sheet {sheet_name}: {len(formulas)} formulas")
            
            # Process formulas in this sheet
            for cell, formula in formulas:
                try:
                    deps = extract_cell_dependencies(formula, sheet_name)
                    for dep_sheet, dep_cell in deps:
                        compact_graph.add_dependency((sheet_name, cell), (dep_sheet, dep_cell))
                except Exception as e:
                    logger.warning(f"[POWER_ROBUST] Error processing {sheet_name}!{cell}: {e}")
            
            processed_sheets += 1
            
            # Force garbage collection between sheets
            if processed_sheets % 5 == 0:
                gc.collect()
                self._update_progress(
                    f"Processed {processed_sheets}/{total_sheets} sheets, "
                    f"{compact_graph.get_cell_count()} cells, "
                    f"{compact_graph.get_dependency_count()} dependencies"
                )
        
        # Convert compact graph to evaluation order using simplified algorithm
        ordered_eval_list = self._compact_graph_to_evaluation_order(
            compact_graph, all_formulas, target_sheet_name, target_cell_coord
        )
        
        elapsed = time.time() - start_time
        logger.info(f"[POWER_ROBUST] Chunked analysis completed in {elapsed:.3f}s - {len(ordered_eval_list)} formulas")
        
        return ordered_eval_list
    
    def _analyze_dependencies_simplified(
        self,
        target_sheet_name: str,
        target_cell_coord: str,
        all_formulas: Dict[str, Dict[str, str]],
        mc_input_cells: Set[Tuple[str, str]]
    ) -> List[Tuple[str, str, str]]:
        """Simplified dependency analysis using sheet-based natural ordering"""
        start_time = time.time()
        logger.info("[POWER_ROBUST] Starting simplified dependency analysis")
        
        # Group formulas by sheet and sort by natural Excel order
        sheets_formulas = {}
        for sheet_name, sheet_formulas in all_formulas.items():
            if sheet_name not in sheets_formulas:
                sheets_formulas[sheet_name] = []
            
            for cell, formula in sheet_formulas.items():
                if (sheet_name, cell.upper()) not in mc_input_cells:
                    sheets_formulas[sheet_name].append((cell, formula))
        
        # Sort formulas within each sheet by row then column (natural Excel order)
        evaluation_order = []
        for sheet_name, sheet_formulas in sheets_formulas.items():
            sorted_formulas = sorted(sheet_formulas, key=lambda x: self._excel_sort_key(x[0]))
            
            for cell, formula in sorted_formulas:
                evaluation_order.append((sheet_name, cell, formula))
        
        elapsed = time.time() - start_time
        logger.info(f"[POWER_ROBUST] Simplified analysis completed in {elapsed:.3f}s - {len(evaluation_order)} formulas")
        
        return evaluation_order
    
    def _analyze_dependencies_basic(
        self,
        target_sheet_name: str,
        target_cell_coord: str,
        all_formulas: Dict[str, Dict[str, str]]
    ) -> List[Tuple[str, str, str]]:
        """Basic dependency analysis - just return all formulas in sheet order"""
        start_time = time.time()
        logger.info("[POWER_ROBUST] Starting basic dependency analysis")
        
        evaluation_order = []
        for sheet_name, sheet_formulas in all_formulas.items():
            for cell, formula in sheet_formulas.items():
                evaluation_order.append((sheet_name, cell, formula))
        
        elapsed = time.time() - start_time
        logger.info(f"[POWER_ROBUST] Basic analysis completed in {elapsed:.3f}s - {len(evaluation_order)} formulas")
        
        return evaluation_order
    
    def _build_dependency_graph_with_progress(
        self,
        target_sheet_name: str,
        target_cell_coord: str,
        all_formulas: Dict[str, Dict[str, str]],
        mc_input_cells: Set[Tuple[str, str]],
        engine_type: str
    ) -> Tuple[Dict[Tuple[str, str], Set[Tuple[str, str]]], Set[Tuple[str, str]]]:
        """Build dependency graph with real-time progress updates"""
        graph: Dict[Tuple[str, str], Set[Tuple[str, str]]] = {}
        all_relevant_cells: Set[Tuple[str, str]] = set()
        
        queue = deque([(target_sheet_name, target_cell_coord.upper())])
        visited: Set[Tuple[str, str]] = set()
        
        # Use enhanced limits for Power engine
        max_iterations = MAX_DEPENDENCY_NODES_ARROW if engine_type == "power" else MAX_DEPENDENCY_NODES
        
        iteration_count = 0
        last_progress_update = 0
        
        while queue and iteration_count < max_iterations:
            iteration_count += 1
            
            # Update progress every 1000 iterations
            if iteration_count - last_progress_update >= 1000:
                self._update_progress(f"Building dependency graph: {iteration_count} nodes processed")
                last_progress_update = iteration_count
            
            current_sheet, current_cell = queue.popleft()
            
            if (current_sheet, current_cell) in visited:
                continue
                
            visited.add((current_sheet, current_cell))
            all_relevant_cells.add((current_sheet, current_cell))
            
            if (current_sheet, current_cell) in mc_input_cells:
                continue
            
            formula = all_formulas.get(current_sheet, {}).get(current_cell)
            if not formula:
                continue
            
            try:
                dependencies = extract_cell_dependencies(formula, current_sheet)
                current_node = (current_sheet, current_cell)
                
                if current_node not in graph:
                    graph[current_node] = set()
                
                for dep_sheet, dep_cell in dependencies:
                    dep_node = (dep_sheet, dep_cell)
                    graph[current_node].add(dep_node)
                    
                    if dep_node not in visited:
                        queue.append(dep_node)
                        all_relevant_cells.add(dep_node)
                        
            except Exception as e:
                logger.warning(f"[POWER_ROBUST] Error processing {current_sheet}!{current_cell}: {e}")
        
        if iteration_count >= max_iterations:
            raise DependencyAnalysisTimeout(f"Dependency graph building exceeded {max_iterations} iterations")
        
        logger.info(f"[POWER_ROBUST] Dependency graph built: {len(all_relevant_cells)} cells, {iteration_count} iterations")
        return graph, all_relevant_cells
    
    def _calculate_evaluation_order_with_progress(
        self,
        graph: Dict[Tuple[str, str], Set[Tuple[str, str]]],
        all_relevant_cells: Set[Tuple[str, str]],
        all_formulas: Dict[str, Dict[str, str]],
        mc_input_cells: Set[Tuple[str, str]]
    ) -> List[Tuple[str, str, str]]:
        """Calculate evaluation order with progress tracking"""
        # Filter nodes that need calculation
        nodes_to_calc = {}
        for node in all_relevant_cells:
            sheet, cell = node
            formula = all_formulas.get(sheet, {}).get(cell)
            # ✅ CRITICAL FIX: Include Monte Carlo input cells in calculation if they have formulas
            # This ensures Monte Carlo variable values propagate to dependent cells
            if formula:  # Calculate ANY cell that has a formula, including MC inputs
                nodes_to_calc[node] = set()
                
                try:
                    deps = extract_cell_dependencies(formula, sheet)
                    for dep_node in deps:
                        if dep_node in all_relevant_cells:
                            # Only include dependencies that have actual formulas
                            # Monte Carlo input cells without formulas stay as injected constants
                            dep_formula = all_formulas.get(dep_node[0], {}).get(dep_node[1])
                            if dep_formula:
                                nodes_to_calc[node].add(dep_node)
                except Exception as e:
                    logger.warning(f"[POWER_ROBUST] Error calculating dependencies for {sheet}!{cell}: {e}")
        
        self._update_progress(f"Calculating evaluation order for {len(nodes_to_calc)} formulas")
        
        # Use Kahn's algorithm for topological sort with progress tracking
        in_degree = {node: 0 for node in nodes_to_calc}
        adjacency_list = {node: [] for node in nodes_to_calc}
        
        # Build adjacency list and calculate in-degrees
        for node, dependencies in nodes_to_calc.items():
            for dep in dependencies:
                if dep in nodes_to_calc:
                    in_degree[node] += 1
                    adjacency_list[dep].append(node)
        
        # POWER ENGINE FIX: Topological sort with iteration limits to prevent infinite loops
        queue = deque([node for node in nodes_to_calc if in_degree[node] == 0])
        ordered_eval_list = []
        processed = 0
        max_iterations = len(nodes_to_calc) * 2  # Safety limit: 2x the number of nodes
        iteration_count = 0
        
        while queue and iteration_count < max_iterations:
            iteration_count += 1
            node = queue.popleft()
            sheet, cell = node
            formula = all_formulas.get(sheet, {}).get(cell)
            
            if formula:
                ordered_eval_list.append((sheet, cell, formula))
            
            processed += 1
            
            # POWER ENGINE FIX: Update progress every 10% of formulas with heartbeat
            if processed % max(1, len(nodes_to_calc) // 10) == 0:
                progress_pct = (processed / len(nodes_to_calc)) * 100
                self._update_progress(f"Topological sort: {progress_pct:.1f}% complete")
                logger.info(f"💓 [POWER_ROBUST] Topological sort heartbeat - {processed}/{len(nodes_to_calc)} processed")
            
            # Process neighbors
            for neighbor in adjacency_list[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        # POWER ENGINE FIX: Check for infinite loop condition
        if iteration_count >= max_iterations:
            logger.error(f"[POWER_ROBUST] Topological sort exceeded maximum iterations ({max_iterations})")
            raise DependencyAnalysisTimeout(f"Topological sort exceeded maximum iterations ({max_iterations}) - possible infinite loop")
        
        if len(ordered_eval_list) < len(nodes_to_calc):
            remaining = len(nodes_to_calc) - len(ordered_eval_list)
            logger.warning(f"[POWER_ROBUST] Possible circular dependencies: {remaining} formulas not ordered")
            
            # Add remaining formulas in arbitrary order
            remaining_nodes = [n for n in nodes_to_calc if n not in {(s,c) for s,c,_ in ordered_eval_list}]
            for node in remaining_nodes:
                sheet, cell = node
                formula = all_formulas.get(sheet, {}).get(cell)
                if formula:
                    ordered_eval_list.append((sheet, cell, formula))
        
        # ✅ CRITICAL FIX: Don't add placeholder formulas for Monte Carlo inputs
        # Monte Carlo values are injected directly into current_values by the engine
        # Adding placeholder formulas like "=F4" would overwrite the injected values with 0
        # Only include MC cells that actually have real formulas in the Excel file
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
        
        logger.info(f"[POWER_ROBUST] Final evaluation order: {len(ordered_eval_list)} formulas total (MC inputs handled by direct injection)")
        
        return final_order
    
    def _compact_graph_to_evaluation_order(
        self,
        compact_graph: CompactFormulaGraph,
        all_formulas: Dict[str, Dict[str, str]],
        target_sheet_name: str,
        target_cell_coord: str
    ) -> List[Tuple[str, str, str]]:
        """Convert compact graph to evaluation order using simplified algorithm"""
        # Simple strategy: process by dependency levels
        evaluation_order = []
        processed_ids = set()
        
        # Start with cells that have no dependencies
        no_deps = []
        dep_count = defaultdict(int)
        
        # Count dependencies for each cell
        for from_id, to_id in compact_graph.dependencies:
            dep_count[from_id] += 1
        
        # Find cells with no dependencies
        for cell_id in compact_graph.id_to_cell.keys():
            if dep_count[cell_id] == 0:
                no_deps.append(cell_id)
        
        # Process in levels
        queue = deque(no_deps)
        
        while queue:
            cell_id = queue.popleft()
            if cell_id in processed_ids:
                continue
                
            processed_ids.add(cell_id)
            sheet, cell = compact_graph.id_to_cell[cell_id]
            formula = all_formulas.get(sheet, {}).get(cell)
            
            if formula:
                evaluation_order.append((sheet, cell, formula))
            
            # Add cells that depend on this one
            for from_id, to_id in compact_graph.dependencies:
                if to_id == cell_id and from_id not in processed_ids:
                    dep_count[from_id] -= 1
                    if dep_count[from_id] == 0:
                        queue.append(from_id)
        
        return evaluation_order
    
    @contextmanager
    def _timeout_handler(self, timeout_seconds: int):
        """Context manager for timeout handling"""
        def timeout_handler(signum, frame):
            raise DependencyAnalysisTimeout(f"Analysis exceeded {timeout_seconds} seconds")
        
        # Set up timeout
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout_seconds)
        
        try:
            yield
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
    
    def _excel_sort_key(self, cell_coord: str) -> Tuple[int, str]:
        """Generate sort key for Excel cell coordinates (A1, B1, A2, etc.)"""
        # Extract column letters and row number
        col_match = ""
        row_match = ""
        
        for char in cell_coord:
            if char.isalpha():
                col_match += char
            elif char.isdigit():
                row_match += char
        
        try:
            row_num = int(row_match) if row_match else 0
            return (row_num, col_match)
        except ValueError:
            return (0, col_match)
    
    def _update_progress(self, message: str):
        """Update progress via callback if available"""
        if self.progress_callback:
            self.progress_callback({
                'stage': 'dependency_analysis',
                'message': message,
                'stats': self.stats.copy()
            })
        logger.info(f"[POWER_ROBUST] {message}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get analysis statistics"""
        return self.stats.copy()

# Main function to be used by Power Engine
def get_evaluation_order_robust(
    target_sheet_name: str,
    target_cell_coord: str,
    all_formulas: Dict[str, Dict[str, str]],
    mc_input_cells: Set[Tuple[str, str]],
    engine_type: str = "power",
    progress_callback: Optional[Callable] = None
) -> List[Tuple[str, str, str]]:
    """
    Robust wrapper for get_evaluation_order with multiple fallback strategies
    """
    analyzer = RobustDependencyAnalyzer(progress_callback)
    
    return analyzer.analyze_dependencies_robust(
        target_sheet_name, target_cell_coord, all_formulas, mc_input_cells, engine_type
    )
