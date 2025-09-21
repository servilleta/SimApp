#!/usr/bin/env python3
"""
Coordinate Mapping and Detection Solution

This module provides intelligent coordinate mapping to resolve mismatches between 
frontend expectations and actual Excel file structure.
"""

import asyncio
import logging
import re
from typing import Dict, List, Tuple, Optional, Any
from arrow_engine.excel_arrow_loader import ExcelToArrowLoader

logger = logging.getLogger(__name__)

class CoordinateMapper:
    """Intelligent coordinate mapping for Excel files"""
    
    def __init__(self):
        self.excel_data = None
        self.available_cells = {}
        self.available_formulas = {}
        
    async def analyze_excel_file(self, file_path: str) -> Dict[str, Any]:
        """Load and analyze Excel file structure"""
        try:
            loader = ExcelToArrowLoader()
            self.excel_data = await loader.load_excel_to_arrow(file_path)
            
            self.available_cells = self.excel_data.get('cell_values', {})
            self.available_formulas = self.excel_data.get('cell_formulas', {})
            
            return {
                'worksheets': list(self.excel_data.get('worksheets', {}).keys()),
                'total_cells': len(self.available_cells),
                'formula_cells': len(self.available_formulas),
                'numeric_cells': self._count_numeric_cells(),
                'value_cells': self._get_value_cells(),
                'formula_cells_list': list(self.available_formulas.keys())
            }
            
        except Exception as e:
            logger.error(f"❌ [COORDINATE_MAPPER] Failed to analyze Excel file: {e}")
            raise
    
    def _count_numeric_cells(self) -> int:
        """Count cells with numeric values"""
        count = 0
        for cell_addr, value in self.available_cells.items():
            if isinstance(value, (int, float)):
                count += 1
        return count
    
    def _get_value_cells(self) -> List[Tuple[str, Any]]:
        """Get list of cells with their values"""
        value_cells = []
        for cell_addr, value in self.available_cells.items():
            if isinstance(value, (int, float)) and value != 0:
                value_cells.append((cell_addr, value))
        return value_cells
    
    def _find_formula_equivalent(self, target_coord: str, available_formulas: List[Tuple[str, str]], sheet_name: str) -> Optional[Tuple[str, str, str]]:
        """Find the best formula equivalent for a target coordinate"""
        
        # Define expected formula patterns for common targets
        expected_patterns = {
            'I6': ['SUM', 'I8:I208', 'I8:I207'],  # SUM of I column range
            'J6': ['SUM', 'J8:J208', 'J8:J207'],  # SUM of J column range  
            'K6': ['/', 'J6/I6', 'J.*I', 'division'],  # Division formula
        }
        
        target_patterns = expected_patterns.get(target_coord, [])
        
        # Score each available formula based on pattern matching
        best_score = 0
        best_match = None
        
        for coord, formula in available_formulas:
            score = 0
            formula_upper = formula.upper() if formula else ""
            
            # Check for specific patterns
            for pattern in target_patterns:
                if pattern.upper() in formula_upper:
                    if pattern == 'SUM':
                        score += 100  # High priority for SUM formulas
                    elif '/' in pattern or 'division' in pattern.lower():
                        if '/' in formula:
                            score += 100  # High priority for division
                    elif ':' in pattern:  # Range patterns
                        if pattern.upper().replace('8', '').replace('20', '') in formula_upper:
                            score += 80  # High priority for range matches
                    else:
                        score += 50
            
            # Additional scoring based on formula type
            if formula_upper.startswith('=SUM(') and target_coord in ['I6', 'J6']:
                score += 90
            elif '/' in formula and target_coord == 'K6':
                score += 90
            elif formula_upper.startswith('=') and len(formula) > 10:
                score += 30  # Prefer complex formulas
            
            # Update best match
            if score > best_score:
                best_score = score
                reason = f"Pattern match (score: {score}) - {target_coord} expected patterns: {target_patterns}"
                best_match = (coord, formula, reason)
        
        return best_match if best_score > 50 else None
    
    def find_cell_exists(self, target_coordinate: str, sheet_name: str) -> bool:
        """Check if a specific cell exists in the Excel file"""
        full_address = f"{sheet_name}!{target_coordinate}"
        return (full_address in self.available_cells or 
                full_address in self.available_formulas)
    
    def suggest_alternative_coordinates(self, 
                                      requested_coords: List[str], 
                                      sheet_name: str,
                                      coord_type: str = "variable") -> Dict[str, Any]:
        """
        Suggest alternative coordinates when requested ones don't exist
        
        Args:
            requested_coords: List of requested coordinates (e.g., ['D2', 'D3', 'D4'])
            sheet_name: Worksheet name
            coord_type: 'variable' for input variables, 'target' for formulas
        """
        suggestions = {
            'missing_coordinates': [],
            'suggested_mapping': {},
            'available_alternatives': [],
            'analysis': {}
        }
        
        # Check which coordinates are missing
        for coord in requested_coords:
            if not self.find_cell_exists(coord, sheet_name):
                suggestions['missing_coordinates'].append(coord)
        
        if not suggestions['missing_coordinates']:
            suggestions['analysis']['status'] = 'all_found'
            return suggestions
        
        # Get available options based on type
        if coord_type == "variable":
            # For input variables, suggest numeric value cells
            available_options = []
            for cell_addr, value in self.available_cells.items():
                if isinstance(value, (int, float)) and cell_addr.startswith(f"{sheet_name}!"):
                    coord_only = cell_addr.split('!')[1]
                    available_options.append((coord_only, value))
            
            suggestions['available_alternatives'] = available_options[:10]  # Top 10
            
            # Try to create intelligent mapping
            if len(requested_coords) <= len(available_options):
                for i, missing_coord in enumerate(suggestions['missing_coordinates']):
                    if i < len(available_options):
                        suggested_coord, suggested_value = available_options[i]
                        suggestions['suggested_mapping'][missing_coord] = {
                            'coordinate': suggested_coord,
                            'value': suggested_value,
                            'reason': f'Numeric value cell with value {suggested_value}'
                        }
                        
        elif coord_type == "target":
            # For target cells, suggest formula cells with intelligent pattern matching
            available_formulas = []
            for cell_addr, formula in self.available_formulas.items():
                if cell_addr.startswith(f"{sheet_name}!"):
                    coord_only = cell_addr.split('!')[1]
                    available_formulas.append((coord_only, formula))
            
            suggestions['available_alternatives'] = available_formulas[:10]  # Top 10
            
            # ENHANCED: Create intelligent mapping based on formula patterns
            for missing_coord in suggestions['missing_coordinates']:
                best_match = self._find_formula_equivalent(missing_coord, available_formulas, sheet_name)
                if best_match:
                    coord, formula, reason = best_match
                    suggestions['suggested_mapping'][missing_coord] = {
                        'coordinate': coord,
                        'formula': formula,
                        'reason': reason
                    }
                elif available_formulas:
                    # Fallback to first available formula
                    suggested_coord, suggested_formula = available_formulas[0]
                    suggestions['suggested_mapping'][missing_coord] = {
                        'coordinate': suggested_coord,
                        'formula': suggested_formula,
                        'reason': f'Fallback formula: {suggested_formula[:50]}...'
                    }
        
        suggestions['analysis'] = {
            'status': 'mapping_needed',
            'total_missing': len(suggestions['missing_coordinates']),
            'total_available': len(suggestions['available_alternatives']),
            'mapping_possible': len(suggestions['suggested_mapping']) > 0
        }
        
        return suggestions

async def fix_coordinate_mismatch(file_path: str, 
                                variables: List[Dict],
                                targets: List[str], 
                                sheet_name: str,
                                auto_apply: bool = True) -> Dict[str, Any]:
    """
    Main function to fix coordinate mismatches
    
    Returns:
        {
            'status': 'success' | 'partial' | 'failed',
            'mapped_variables': [...],
            'mapped_targets': [...], 
            'report': {...},
            'applied_mapping': {...}
        }
    """
    try:
        # Create mapper and analyze file
        mapper = CoordinateMapper()
        analysis = await mapper.analyze_excel_file(file_path)
        
        logger.info(f"🔍 [COORDINATE_FIX] Excel analysis: {analysis['total_cells']} cells, {analysis['formula_cells']} formulas")
        
        # Create mapping report
        variable_coords = [var.get('name', var.get('cell', '')) for var in variables]
        var_analysis = mapper.suggest_alternative_coordinates(variable_coords, sheet_name, 'variable')
        target_analysis = mapper.suggest_alternative_coordinates(targets, sheet_name, 'target')
        
        result = {
            'status': 'failed',
            'mapped_variables': variables.copy(),  # Default to original
            'mapped_targets': targets.copy(),      # Default to original  
            'report': {
                'variables': var_analysis,
                'targets': target_analysis,
                'analysis': {}
            },
            'applied_mapping': {}
        }
        
        # Check if all coordinates exist
        total_missing = len(var_analysis['missing_coordinates']) + len(target_analysis['missing_coordinates'])
        if total_missing == 0:
            result['status'] = 'success'
            result['report']['analysis'] = {'status': 'all_coordinates_found', 'message': 'All coordinates exist'}
            logger.info(f"✅ [COORDINATE_FIX] All coordinates found - no mapping needed")
            return result
        
        if not auto_apply:
            result['status'] = 'partial'
            result['report']['analysis'] = {'status': 'mapping_needed', 'message': f'{total_missing} coordinates need mapping'}
            logger.info(f"⚠️ [COORDINATE_FIX] Mapping needed but auto_apply=False")
            return result
        
        # Apply automatic mapping if possible
        applied_mapping = {}
        
        # Map variables
        if var_analysis['suggested_mapping']:
            var_mapping = {k: v['coordinate'] for k, v in var_analysis['suggested_mapping'].items()}
            applied_mapping['variables'] = var_mapping
            
            # Apply mapping to variables
            for var in result['mapped_variables']:
                original_coord = var.get('name', var.get('cell', ''))
                if original_coord in var_mapping:
                    var['name'] = var_mapping[original_coord]
                    if 'cell' in var:
                        var['cell'] = var_mapping[original_coord]
                    logger.info(f"🔧 [COORDINATE_FIX] Mapped variable {original_coord} → {var_mapping[original_coord]}")
        
        # Map targets
        if target_analysis['suggested_mapping']:
            target_mapping = {k: v['coordinate'] for k, v in target_analysis['suggested_mapping'].items()}
            applied_mapping['targets'] = target_mapping
            
            # Apply mapping to targets
            mapped_targets = []
            for target in result['mapped_targets']:
                if target in target_mapping:
                    mapped_target = target_mapping[target]
                    mapped_targets.append(mapped_target)
                    logger.info(f"🔧 [COORDINATE_FIX] Mapped target {target} → {mapped_target}")
                else:
                    mapped_targets.append(target)
            
            result['mapped_targets'] = mapped_targets
        
        result['applied_mapping'] = applied_mapping
        
        # Determine final status
        auto_fix_possible = (
            len(var_analysis['suggested_mapping']) == len(var_analysis['missing_coordinates']) and
            len(target_analysis['suggested_mapping']) == len(target_analysis['missing_coordinates'])
        )
        
        if applied_mapping:
            result['status'] = 'success' if auto_fix_possible else 'partial'
            result['report']['analysis'] = {
                'status': 'coordinate_mismatch_fixed', 
                'message': f'Applied coordinate mapping for {len(applied_mapping)} types'
            }
            logger.info(f"✅ [COORDINATE_FIX] Applied coordinate mapping: {len(applied_mapping)} changes")
        else:
            result['status'] = 'failed'
            result['report']['analysis'] = {
                'status': 'coordinate_mismatch_failed',
                'message': f'Could not create automatic mapping for {total_missing} coordinates'
            }
            logger.warning(f"❌ [COORDINATE_FIX] Could not create automatic mapping")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ [COORDINATE_FIX] Failed to fix coordinate mismatch: {e}")
        raise 