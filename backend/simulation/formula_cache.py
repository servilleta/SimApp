"""
Formula Dependency Caching System for Enterprise Excel Processing
Implements intelligent formula analysis, dependency mapping, and selective recalculation.
"""

import re
import numpy as np
from typing import Dict, List, Set, Any, Tuple, Optional
import logging
import time
from collections import defaultdict, deque
import hashlib
import pickle
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class FormulaNode:
    """Represents a formula in the dependency graph."""
    cell_address: str
    formula: str
    dependencies: Set[str]  # Cells this formula depends on
    dependents: Set[str]   # Cells that depend on this formula
    complexity_score: float
    last_modified: float
    cached_result: Any = None
    cache_valid: bool = False

class FormulaDependencyAnalyzer:
    """
    Analyzes Excel formulas to build dependency graphs and enable smart caching.
    """
    
    def __init__(self):
        self.formula_nodes: Dict[str, FormulaNode] = {}
        self.input_variables: Set[str] = set()
        self.calculation_order: List[str] = []
        self.dependency_levels: Dict[str, int] = {}
        self.logger = logging.getLogger(__name__)
        
        # Performance tracking
        self.analysis_stats = {
            'formulas_analyzed': 0,
            'dependencies_found': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'recalculations_avoided': 0
        }
        
    def analyze_excel_model(self, sheet_data: Dict[str, Any], input_variables: List[Dict[str, Any]]) -> None:
        """
        Analyze an Excel model to build the complete dependency graph.
        
        Args:
            sheet_data: Excel sheet data with formulas
            input_variables: List of input variable configurations
        """
        start_time = time.time()
        
        self.logger.info("🔍 [FormulaAnalysis] Starting comprehensive Excel model analysis")
        
        # Extract input variable cell addresses
        self.input_variables = {var['cell'] for var in input_variables if 'cell' in var}
        self.logger.info(f"📊 [FormulaAnalysis] Input variables: {len(self.input_variables)} cells")
        
        # Analyze all formulas in the sheet
        formulas_found = 0
        for row_data in sheet_data.get('data', []):
            for cell_data in row_data:
                if isinstance(cell_data, dict) and 'formula' in cell_data:
                    cell_address = self._get_cell_address(cell_data)
                    formula = cell_data['formula']
                    
                    if formula and formula.startswith('='):
                        self._analyze_single_formula(cell_address, formula)
                        formulas_found += 1
        
        self.analysis_stats['formulas_analyzed'] = formulas_found
        
        # Build dependency relationships
        self._build_dependency_relationships()
        
        # Calculate optimal calculation order
        self._calculate_optimal_order()
        
        # Assign complexity scores
        self._assign_complexity_scores()
        
        analysis_time = time.time() - start_time
        self.logger.info(f"⚡ [FormulaAnalysis] Analyzed {formulas_found} formulas in {analysis_time:.3f}s")
        self.logger.info(f"🔗 [FormulaAnalysis] Found {self.analysis_stats['dependencies_found']} dependencies")
        self.logger.info(f"📈 [FormulaAnalysis] Calculation levels: {len(set(self.dependency_levels.values()))}")
        
    def _analyze_single_formula(self, cell_address: str, formula: str) -> None:
        """Analyze a single formula to extract its dependencies."""
        dependencies = self._extract_cell_references(formula)
        complexity = self._calculate_formula_complexity(formula)
        
        node = FormulaNode(
            cell_address=cell_address,
            formula=formula,
            dependencies=dependencies,
            dependents=set(),
            complexity_score=complexity,
            last_modified=time.time()
        )
        
        self.formula_nodes[cell_address] = node
        self.analysis_stats['dependencies_found'] += len(dependencies)
        
    def _extract_cell_references(self, formula: str) -> Set[str]:
        """Extract all cell references from a formula."""
        # Regex pattern for Excel cell references (e.g., A1, $B$2, Sheet1!C3)
        cell_pattern = r'(?:[A-Za-z]+\w*!)?(?:\$?[A-Z]+\$?\d+)'
        matches = re.findall(cell_pattern, formula, re.IGNORECASE)
        
        # Clean up references (remove $ signs, normalize case)
        clean_refs = set()
        for match in matches:
            clean_ref = match.replace('$', '').upper()
            # Remove sheet name if present (for now, assume single sheet)
            if '!' in clean_ref:
                clean_ref = clean_ref.split('!')[-1]
            clean_refs.add(clean_ref)
            
        return clean_refs
    
    def _calculate_formula_complexity(self, formula: str) -> float:
        """Calculate a complexity score for a formula."""
        complexity = 0.0
        
        # Base complexity from length
        complexity += len(formula) * 0.1
        
        # Function complexity
        functions = re.findall(r'[A-Z]+\(', formula, re.IGNORECASE)
        complexity += len(functions) * 2.0
        
        # Nested function complexity
        nesting_level = max(formula.count('('), formula.count('['))
        complexity += nesting_level * 1.5
        
        # Array formula complexity
        if '{' in formula and '}' in formula:
            complexity += 5.0
            
        # Conditional complexity (IF, CHOOSE, etc.)
        conditionals = ['IF', 'CHOOSE', 'SWITCH', 'IFS']
        for conditional in conditionals:
            if conditional in formula.upper():
                complexity += 3.0
                
        # Lookup complexity (VLOOKUP, INDEX/MATCH, etc.)
        lookups = ['VLOOKUP', 'HLOOKUP', 'INDEX', 'MATCH', 'XLOOKUP']
        for lookup in lookups:
            if lookup in formula.upper():
                complexity += 4.0
                
        return complexity
    
    def _build_dependency_relationships(self) -> None:
        """Build bidirectional dependency relationships."""
        for cell_addr, node in self.formula_nodes.items():
            for dependency in node.dependencies:
                if dependency in self.formula_nodes:
                    self.formula_nodes[dependency].dependents.add(cell_addr)
    
    def _calculate_optimal_order(self) -> None:
        """Calculate optimal calculation order using topological sort."""
        # Create a graph for topological sorting
        in_degree = defaultdict(int)
        graph = defaultdict(list)
        
        # Initialize in-degrees
        for cell_addr in self.formula_nodes:
            in_degree[cell_addr] = 0
            
        # Build graph and calculate in-degrees
        for cell_addr, node in self.formula_nodes.items():
            for dependency in node.dependencies:
                if dependency in self.formula_nodes:
                    graph[dependency].append(cell_addr)
                    in_degree[cell_addr] += 1
                    
        # Topological sort using Kahn's algorithm
        queue = deque([cell for cell in self.formula_nodes if in_degree[cell] == 0])
        calculation_order = []
        level = 0
        
        while queue:
            level_size = len(queue)
            current_level = []
            
            for _ in range(level_size):
                cell = queue.popleft()
                calculation_order.append(cell)
                current_level.append(cell)
                self.dependency_levels[cell] = level
                
                # Process neighbors
                for neighbor in graph[cell]:
                    in_degree[neighbor] -= 1
                    if in_degree[neighbor] == 0:
                        queue.append(neighbor)
                        
            level += 1
            
        self.calculation_order = calculation_order
        
    def _assign_complexity_scores(self) -> None:
        """Assign complexity scores considering dependencies."""
        for cell_addr, node in self.formula_nodes.items():
            # Add dependency complexity
            dependency_complexity = sum(
                self.formula_nodes[dep].complexity_score 
                for dep in node.dependencies 
                if dep in self.formula_nodes
            )
            node.complexity_score += dependency_complexity * 0.1
    
    def get_affected_cells(self, changed_cells: Set[str]) -> List[str]:
        """
        Get list of cells that need recalculation when specific cells change.
        Returns cells in optimal calculation order.
        """
        affected = set()
        queue = deque(changed_cells)
        
        # Find all cells affected by the changes
        while queue:
            cell = queue.popleft()
            if cell in affected:
                continue
                
            affected.add(cell)
            
            # Add all cells that depend on this cell
            if cell in self.formula_nodes:
                for dependent in self.formula_nodes[cell].dependents:
                    if dependent not in affected:
                        queue.append(dependent)
                        
        # Return in calculation order
        ordered_affected = [
            cell for cell in self.calculation_order 
            if cell in affected
        ]
        
        self.logger.info(f"🔄 [SelectiveRecalc] {len(changed_cells)} cells changed → {len(ordered_affected)} need recalc")
        
        return ordered_affected
    
    def invalidate_cache(self, changed_cells: Set[str]) -> None:
        """Invalidate cache for affected cells."""
        affected_cells = self.get_affected_cells(changed_cells)
        
        for cell in affected_cells:
            if cell in self.formula_nodes:
                self.formula_nodes[cell].cache_valid = False
                self.formula_nodes[cell].cached_result = None
                
        self.logger.info(f"🗑️ [Cache] Invalidated cache for {len(affected_cells)} cells")
    
    def get_calculation_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the formula analysis."""
        total_complexity = sum(node.complexity_score for node in self.formula_nodes.values())
        avg_complexity = total_complexity / len(self.formula_nodes) if self.formula_nodes else 0
        
        return {
            **self.analysis_stats,
            'total_formulas': len(self.formula_nodes),
            'total_complexity': total_complexity,
            'average_complexity': avg_complexity,
            'calculation_levels': len(set(self.dependency_levels.values())),
            'input_variables': len(self.input_variables),
            'max_dependency_level': max(self.dependency_levels.values()) if self.dependency_levels else 0
        }
    
    def _get_cell_address(self, cell_data: Dict[str, Any]) -> str:
        """Extract cell address from cell data."""
        if 'address' in cell_data:
            return cell_data['address']
        elif 'row' in cell_data and 'col' in cell_data:
            # Convert row/col to Excel address (A1 notation)
            col_letter = self._number_to_column_letter(cell_data['col'])
            return f"{col_letter}{cell_data['row']}"
        else:
            return f"UNKNOWN_{hash(str(cell_data)) % 10000}"
    
    def _number_to_column_letter(self, col_num: int) -> str:
        """Convert column number to Excel column letter."""
        result = ""
        while col_num > 0:
            col_num -= 1
            result = chr(col_num % 26 + ord('A')) + result
            col_num //= 26
        return result

class EnterpriseFormulaCache:
    """
    High-performance caching system for formula results with intelligent invalidation.
    """
    
    def __init__(self, max_cache_size: int = 10000):
        self.cache: Dict[str, Any] = {}
        self.cache_metadata: Dict[str, Dict[str, Any]] = {}
        self.max_cache_size = max_cache_size
        self.hit_count = 0
        self.miss_count = 0
        self.logger = logging.getLogger(__name__)
        
    def get(self, cache_key: str) -> Tuple[bool, Any]:
        """
        Get cached result if available and valid.
        
        Returns:
            (cache_hit: bool, result: Any)
        """
        if cache_key in self.cache:
            metadata = self.cache_metadata.get(cache_key, {})
            if metadata.get('valid', False):
                self.hit_count += 1
                # Update access time for LRU
                metadata['last_accessed'] = time.time()
                return True, self.cache[cache_key]
                
        self.miss_count += 1
        return False, None
    
    def set(self, cache_key: str, result: Any, dependencies: Set[str] = None) -> None:
        """Cache a formula result with metadata."""
        # Evict if cache is full
        if len(self.cache) >= self.max_cache_size:
            self._evict_lru()
            
        self.cache[cache_key] = result
        self.cache_metadata[cache_key] = {
            'valid': True,
            'timestamp': time.time(),
            'last_accessed': time.time(),
            'dependencies': dependencies or set(),
            'size_estimate': self._estimate_size(result)
        }
        
    def invalidate(self, cache_keys: Set[str]) -> None:
        """Invalidate specific cache entries."""
        for key in cache_keys:
            if key in self.cache_metadata:
                self.cache_metadata[key]['valid'] = False
                
    def _evict_lru(self) -> None:
        """Evict least recently used cache entry."""
        if not self.cache_metadata:
            return
            
        lru_key = min(
            self.cache_metadata.keys(),
            key=lambda k: self.cache_metadata[k].get('last_accessed', 0)
        )
        
        del self.cache[lru_key]
        del self.cache_metadata[lru_key]
        
    def _estimate_size(self, obj: Any) -> int:
        """Estimate memory size of cached object."""
        try:
            return len(pickle.dumps(obj))
        except:
            return 1000  # Default estimate
            
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        total_requests = self.hit_count + self.miss_count
        hit_rate = self.hit_count / total_requests if total_requests > 0 else 0
        
        return {
            'hit_count': self.hit_count,
            'miss_count': self.miss_count,
            'hit_rate': hit_rate,
            'cache_size': len(self.cache),
            'max_cache_size': self.max_cache_size
        }

def create_formula_analyzer() -> FormulaDependencyAnalyzer:
    """Factory function to create a formula dependency analyzer."""
    analyzer = FormulaDependencyAnalyzer()
    logger.info("🧠 Created enterprise formula dependency analyzer")
    return analyzer

def create_formula_cache(max_size: int = 10000) -> EnterpriseFormulaCache:
    """Factory function to create a formula cache."""
    cache = EnterpriseFormulaCache(max_cache_size=max_size)
    logger.info(f"💾 Created enterprise formula cache (max_size={max_size})")
    return cache 