"""
SUPERENGINE - Model Optimization Analyzer
========================================
This module analyzes Excel models and suggests optimization paths to improve
GPU execution performance. It's part of the Model Intelligence feature in Tier 0.
"""

import logging
import re
from typing import Dict, List, Set, Tuple, Any, Optional
from collections import defaultdict, Counter
import numpy as np

logger = logging.getLogger(__name__)

class OptimizationSuggestion:
    """Represents a single optimization suggestion."""
    def __init__(self, 
                 category: str,
                 severity: str,  # 'high', 'medium', 'low'
                 title: str,
                 description: str,
                 affected_cells: List[str],
                 estimated_speedup: float,
                 implementation_effort: str):  # 'easy', 'medium', 'hard'
        self.category = category
        self.severity = severity
        self.title = title
        self.description = description
        self.affected_cells = affected_cells
        self.estimated_speedup = estimated_speedup
        self.implementation_effort = implementation_effort
        
    def to_dict(self) -> dict:
        return {
            'category': self.category,
            'severity': self.severity,
            'title': self.title,
            'description': self.description,
            'affected_cells': self.affected_cells[:10],  # Limit to first 10
            'affected_count': len(self.affected_cells),
            'estimated_speedup': f"{self.estimated_speedup:.1f}x",
            'implementation_effort': self.implementation_effort
        }

class ModelOptimizationAnalyzer:
    """
    Analyzes Excel models to suggest optimization paths for GPU execution.
    This implements the "Suggested optimization paths" feature from Tier 0.
    """
    
    def __init__(self):
        self.suggestions = []
        self.model_stats = {}
        self.gpu_incompatible_functions = {
            'INDIRECT', 'OFFSET', 'GETPIVOTDATA', 'CELL', 'INFO',
            'HYPERLINK', 'TRANSPOSE', 'FILTER', 'SORT', 'UNIQUE'
        }
        self.slow_functions = {
            'VLOOKUP': 'Consider using INDEX/MATCH for better GPU performance',
            'HLOOKUP': 'Consider using INDEX/MATCH for better GPU performance',
            'SUMPRODUCT': 'Can be slow with large ranges, consider breaking down',
            'ARRAY': 'Array formulas can be GPU-intensive, consider alternatives'
        }
        
    def analyze_model(self, 
                     formulas: Dict[str, Dict[str, str]], 
                     dependencies: List[Tuple[str, str, str]],
                     mc_inputs: Set[Tuple[str, str]]) -> Dict[str, Any]:
        """
        Analyze an Excel model and provide optimization suggestions.
        
        Args:
            formulas: Dictionary of sheet -> cell -> formula
            dependencies: List of (sheet, cell, formula) in dependency order
            mc_inputs: Set of (sheet, cell) that are Monte Carlo inputs
            
        Returns:
            Dictionary with analysis results and suggestions
        """
        logger.info("🔍 Starting model optimization analysis...")
        
        # Reset state
        self.suggestions = []
        self.model_stats = {
            'total_formulas': sum(len(sheet_formulas) for sheet_formulas in formulas.values()),
            'total_cells': 0,
            'gpu_compatible': 0,
            'gpu_incompatible': 0,
            'optimization_potential': 0.0
        }
        
        # Analyze different aspects
        self._analyze_formula_patterns(formulas)
        self._analyze_dependencies(dependencies)
        self._analyze_volatility(formulas)
        self._analyze_lookup_patterns(formulas)
        self._analyze_array_formulas(formulas)
        self._analyze_calculation_chains(dependencies)
        self._analyze_monte_carlo_setup(mc_inputs, dependencies)
        
        # Calculate overall optimization potential
        self._calculate_optimization_score()
        
        # Sort suggestions by potential impact
        self.suggestions.sort(key=lambda s: (
            {'high': 3, 'medium': 2, 'low': 1}[s.severity],
            s.estimated_speedup
        ), reverse=True)
        
        return {
            'stats': self.model_stats,
            'suggestions': [s.to_dict() for s in self.suggestions],
            'optimization_score': self.model_stats['optimization_potential'],
            'summary': self._generate_summary()
        }
    
    def _analyze_formula_patterns(self, formulas: Dict[str, Dict[str, str]]):
        """Analyze formula patterns for GPU compatibility."""
        pattern_counter = Counter()
        gpu_incompatible_cells = []
        
        for sheet, sheet_formulas in formulas.items():
            for cell, formula in sheet_formulas.items():
                formula_upper = formula.upper()
                
                # Check for GPU-incompatible functions
                for func in self.gpu_incompatible_functions:
                    if func + '(' in formula_upper:
                        gpu_incompatible_cells.append(f"{sheet}!{cell}")
                        self.model_stats['gpu_incompatible'] += 1
                        break
                else:
                    self.model_stats['gpu_compatible'] += 1
                
                # Count function usage
                functions = re.findall(r'([A-Z]+)\s*\(', formula_upper)
                pattern_counter.update(functions)
        
        # Add suggestion for GPU-incompatible functions
        if gpu_incompatible_cells:
            self.suggestions.append(OptimizationSuggestion(
                category='GPU Compatibility',
                severity='high',
                title='GPU-Incompatible Functions Detected',
                description=f'Found {len(gpu_incompatible_cells)} cells using functions that cannot run on GPU. '
                           f'Consider replacing with GPU-compatible alternatives.',
                affected_cells=gpu_incompatible_cells,
                estimated_speedup=2.0,
                implementation_effort='medium'
            ))
        
        # Check for slow functions
        slow_function_cells = defaultdict(list)
        for sheet, sheet_formulas in formulas.items():
            for cell, formula in sheet_formulas.items():
                formula_upper = formula.upper()
                for func, suggestion in self.slow_functions.items():
                    if func + '(' in formula_upper:
                        slow_function_cells[func].append(f"{sheet}!{cell}")
        
        for func, cells in slow_function_cells.items():
            if cells:
                self.suggestions.append(OptimizationSuggestion(
                    category='Performance',
                    severity='medium',
                    title=f'Optimize {func} Usage',
                    description=self.slow_functions[func],
                    affected_cells=cells,
                    estimated_speedup=1.5,
                    implementation_effort='easy'
                ))
    
    def _analyze_dependencies(self, dependencies: List[Tuple[str, str, str]]):
        """Analyze dependency patterns for optimization opportunities."""
        dependency_depth = defaultdict(int)
        dependency_breadth = defaultdict(int)
        
        # Build dependency graph
        graph = defaultdict(set)
        reverse_graph = defaultdict(set)
        
        for sheet, cell, formula in dependencies:
            cell_ref = f"{sheet}!{cell}"
            # Extract dependencies from formula
            deps = self._extract_cell_references(formula, sheet)
            for dep in deps:
                graph[cell_ref].add(dep)
                reverse_graph[dep].add(cell_ref)
                dependency_breadth[cell_ref] = len(graph[cell_ref])
        
        # Calculate dependency depths
        def calculate_depth(cell, visited=None):
            if visited is None:
                visited = set()
            if cell in visited:
                return 0
            visited.add(cell)
            
            if cell not in graph or not graph[cell]:
                return 0
            
            max_depth = 0
            for dep in graph[cell]:
                depth = calculate_depth(dep, visited.copy())
                max_depth = max(max_depth, depth + 1)
            
            dependency_depth[cell] = max_depth
            return max_depth
        
        for cell in graph:
            calculate_depth(cell)
        
        # Find cells with deep dependencies
        deep_cells = [cell for cell, depth in dependency_depth.items() if depth > 10]
        if deep_cells:
            self.suggestions.append(OptimizationSuggestion(
                category='Structure',
                severity='medium',
                title='Deep Calculation Chains Detected',
                description='Some cells have very deep dependency chains (>10 levels). '
                           'Consider flattening calculations or using intermediate results.',
                affected_cells=deep_cells,
                estimated_speedup=1.3,
                implementation_effort='medium'
            ))
        
        # Find cells with broad dependencies
        broad_cells = [cell for cell, breadth in dependency_breadth.items() if breadth > 20]
        if broad_cells:
            self.suggestions.append(OptimizationSuggestion(
                category='Structure',
                severity='low',
                title='Broad Dependencies Detected',
                description='Some cells depend on many other cells (>20). '
                           'Consider consolidating ranges or using array formulas.',
                affected_cells=broad_cells,
                estimated_speedup=1.2,
                implementation_effort='easy'
            ))
    
    def _analyze_volatility(self, formulas: Dict[str, Dict[str, str]]):
        """Analyze volatile functions that recalculate frequently."""
        volatile_functions = {'NOW', 'TODAY', 'RAND', 'RANDBETWEEN', 'OFFSET', 'INDIRECT'}
        volatile_cells = []
        
        for sheet, sheet_formulas in formulas.items():
            for cell, formula in sheet_formulas.items():
                formula_upper = formula.upper()
                for func in volatile_functions:
                    if func + '(' in formula_upper:
                        volatile_cells.append(f"{sheet}!{cell}")
                        break
        
        if volatile_cells:
            self.suggestions.append(OptimizationSuggestion(
                category='Performance',
                severity='medium',
                title='Volatile Functions Detected',
                description='Volatile functions recalculate on every change, impacting performance. '
                           'Consider using non-volatile alternatives where possible.',
                affected_cells=volatile_cells,
                estimated_speedup=1.4,
                implementation_effort='easy'
            ))
    
    def _analyze_lookup_patterns(self, formulas: Dict[str, Dict[str, str]]):
        """Analyze VLOOKUP/HLOOKUP patterns for optimization."""
        lookup_patterns = defaultdict(list)
        
        for sheet, sheet_formulas in formulas.items():
            for cell, formula in sheet_formulas.items():
                formula_upper = formula.upper()
                
                # Check for repeated lookups on same table
                vlookup_matches = re.findall(r'VLOOKUP\s*\([^,]+,\s*([^,]+)', formula_upper)
                for table_ref in vlookup_matches:
                    lookup_patterns[table_ref].append(f"{sheet}!{cell}")
        
        # Find tables that are looked up many times
        for table_ref, cells in lookup_patterns.items():
            if len(cells) > 10:
                self.suggestions.append(OptimizationSuggestion(
                    category='Lookup Optimization',
                    severity='high',
                    title=f'Repeated Lookups on Same Table',
                    description=f'Table {table_ref} is used in {len(cells)} VLOOKUP operations. '
                               f'Consider using INDEX/MATCH or creating a sorted lookup table.',
                    affected_cells=cells,
                    estimated_speedup=2.5,
                    implementation_effort='medium'
                ))
    
    def _analyze_array_formulas(self, formulas: Dict[str, Dict[str, str]]):
        """Analyze array formulas for GPU optimization."""
        array_formula_cells = []
        large_sum_ranges = []
        
        for sheet, sheet_formulas in formulas.items():
            for cell, formula in sheet_formulas.items():
                formula_upper = formula.upper()
                
                # Check for array formula indicators
                if '{' in formula and '}' in formula:
                    array_formula_cells.append(f"{sheet}!{cell}")
                
                # Check for large SUM ranges
                sum_matches = re.findall(r'SUM\s*\(([^)]+)\)', formula_upper)
                for range_ref in sum_matches:
                    if ':' in range_ref:
                        # Try to parse range size
                        try:
                            parts = range_ref.split(':')
                            if len(parts) == 2:
                                # Simple heuristic: if range spans more than 1000 rows
                                if any(c.isdigit() and int(''.join(filter(str.isdigit, c))) > 1000 
                                      for c in parts):
                                    large_sum_ranges.append(f"{sheet}!{cell}")
                        except:
                            pass
        
        if array_formula_cells:
            self.suggestions.append(OptimizationSuggestion(
                category='GPU Optimization',
                severity='low',
                title='Array Formulas Can Be GPU-Accelerated',
                description='Array formulas are well-suited for GPU parallelization. '
                           'Ensure they use GPU-compatible functions.',
                affected_cells=array_formula_cells,
                estimated_speedup=3.0,
                implementation_effort='easy'
            ))
        
        if large_sum_ranges:
            self.suggestions.append(OptimizationSuggestion(
                category='GPU Optimization',
                severity='medium',
                title='Large Range Operations Detected',
                description='Large SUM/AVERAGE ranges benefit significantly from GPU acceleration.',
                affected_cells=large_sum_ranges,
                estimated_speedup=5.0,
                implementation_effort='easy'
            ))
    
    def _analyze_calculation_chains(self, dependencies: List[Tuple[str, str, str]]):
        """Analyze calculation chains for parallelization opportunities."""
        # Group formulas by dependency level
        levels = defaultdict(list)
        processed = set()
        
        # Simple level assignment (not perfect but good enough)
        level = 0
        remaining = list(dependencies)
        
        while remaining:
            current_level = []
            next_remaining = []
            
            for sheet, cell, formula in remaining:
                cell_ref = f"{sheet}!{cell}"
                # Check if all dependencies are processed
                deps = self._extract_cell_references(formula, sheet)
                if all(dep in processed or dep == cell_ref for dep in deps):
                    current_level.append(cell_ref)
                    processed.add(cell_ref)
                else:
                    next_remaining.append((sheet, cell, formula))
            
            if current_level:
                levels[level] = current_level
                level += 1
            
            remaining = next_remaining
            
            # Prevent infinite loop
            if level > 100 or (not current_level and remaining):
                break
        
        # Find levels with many parallel calculations
        parallel_opportunities = []
        for level, cells in levels.items():
            if len(cells) > 50:
                parallel_opportunities.append((level, len(cells)))
        
        if parallel_opportunities:
            self.suggestions.append(OptimizationSuggestion(
                category='Parallelization',
                severity='high',
                title='High Parallelization Potential',
                description=f'Found {len(parallel_opportunities)} calculation levels with 50+ independent formulas. '
                           f'These will benefit greatly from GPU parallel execution.',
                affected_cells=[f"Level {level}: {count} cells" for level, count in parallel_opportunities[:5]],
                estimated_speedup=4.0,
                implementation_effort='easy'
            ))
    
    def _analyze_monte_carlo_setup(self, mc_inputs: Set[Tuple[str, str]], 
                                  dependencies: List[Tuple[str, str, str]]):
        """Analyze Monte Carlo simulation setup for optimization."""
        # Check if MC inputs are at the beginning of calculation chain
        mc_input_refs = {f"{sheet}!{cell}" for sheet, cell in mc_inputs}
        
        early_deps = []
        for i, (sheet, cell, formula) in enumerate(dependencies[:20]):
            deps = self._extract_cell_references(formula, sheet)
            if any(dep in mc_input_refs for dep in deps):
                early_deps.append(i)
        
        if early_deps and min(early_deps) > 5:
            self.suggestions.append(OptimizationSuggestion(
                category='Monte Carlo',
                severity='medium',
                title='Monte Carlo Inputs Not Optimally Placed',
                description='Monte Carlo input variables are not at the start of the calculation chain. '
                           'Consider restructuring to minimize recalculation.',
                affected_cells=list(mc_input_refs)[:10],
                estimated_speedup=1.5,
                implementation_effort='hard'
            ))
        
        # Check for efficient random number usage
        if len(mc_inputs) > 100:
            self.suggestions.append(OptimizationSuggestion(
                category='Monte Carlo',
                severity='low',
                title='Large Number of Random Variables',
                description=f'Model uses {len(mc_inputs)} random variables. '
                           f'Consider using correlated random number generation for better performance.',
                affected_cells=list(mc_input_refs)[:10],
                estimated_speedup=1.3,
                implementation_effort='medium'
            ))
    
    def _extract_cell_references(self, formula: str, current_sheet: str) -> Set[str]:
        """Extract cell references from a formula."""
        refs = set()
        
        # Pattern for cell references (simplified)
        cell_pattern = r'(?:([A-Za-z]+\w*)!)?(\$?[A-Z]+\$?\d+)'
        matches = re.findall(cell_pattern, formula)
        
        for sheet, cell in matches:
            if sheet:
                refs.add(f"{sheet}!{cell}")
            else:
                refs.add(f"{current_sheet}!{cell}")
        
        return refs
    
    def _calculate_optimization_score(self):
        """Calculate overall optimization potential score (0-100)."""
        score = 50.0  # Base score
        
        # Adjust based on GPU compatibility
        if self.model_stats['total_formulas'] > 0:
            gpu_ratio = self.model_stats['gpu_compatible'] / self.model_stats['total_formulas']
            score += (gpu_ratio - 0.5) * 20  # +/-10 points
        
        # Adjust based on suggestions
        high_severity = sum(1 for s in self.suggestions if s.severity == 'high')
        medium_severity = sum(1 for s in self.suggestions if s.severity == 'medium')
        
        score -= high_severity * 5
        score -= medium_severity * 2
        
        # Bonus for parallelization opportunities
        parallel_suggestions = [s for s in self.suggestions if s.category == 'Parallelization']
        if parallel_suggestions:
            score += 10
        
        # Clamp to 0-100
        score = max(0, min(100, score))
        self.model_stats['optimization_potential'] = score
        
    def _generate_summary(self) -> str:
        """Generate a summary of the optimization analysis."""
        if self.model_stats['optimization_potential'] >= 80:
            summary = "Excellent! This model is well-optimized for GPU execution."
        elif self.model_stats['optimization_potential'] >= 60:
            summary = "Good. This model will perform well on GPU with minor optimizations."
        elif self.model_stats['optimization_potential'] >= 40:
            summary = "Fair. Several optimization opportunities available for better GPU performance."
        else:
            summary = "Poor. Significant optimization needed for efficient GPU execution."
        
        summary += f" Found {len(self.suggestions)} optimization suggestions."
        
        return summary


# Example usage
if __name__ == '__main__':
    analyzer = ModelOptimizationAnalyzer()
    
    # Test with sample data
    test_formulas = {
        'Sheet1': {
            'A1': '=B1+C1',
            'A2': '=VLOOKUP(A1,D:E,2,FALSE)',
            'A3': '=SUM(A1:A1000)',
            'A4': '=NOW()',
            'A5': '=INDIRECT("B"&ROW())'
        }
    }
    
    test_deps = [
        ('Sheet1', 'A1', '=B1+C1'),
        ('Sheet1', 'A2', '=VLOOKUP(A1,D:E,2,FALSE)'),
        ('Sheet1', 'A3', '=SUM(A1:A1000)'),
        ('Sheet1', 'A4', '=NOW()'),
        ('Sheet1', 'A5', '=INDIRECT("B"&ROW())')
    ]
    
    test_mc_inputs = {('Sheet1', 'B1'), ('Sheet1', 'C1')}
    
    results = analyzer.analyze_model(test_formulas, test_deps, test_mc_inputs)
    
    print("\n=== Model Optimization Analysis ===")
    print(f"Optimization Score: {results['optimization_score']:.1f}/100")
    print(f"Summary: {results['summary']}")
    print(f"\nStats: {results['stats']}")
    print(f"\nTop Suggestions:")
    for suggestion in results['suggestions'][:3]:
        print(f"- [{suggestion['severity'].upper()}] {suggestion['title']}")
        print(f"  {suggestion['description']}")
        print(f"  Estimated speedup: {suggestion['estimated_speedup']}")
        print() 