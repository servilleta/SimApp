"""
Excel Intelligence Agent - AI-powered Excel file analysis
Understands Excel structure, formulas, and business logic patterns
"""

import logging
import json
import re
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
import os
from .deepseek_client import get_deepseek_client, initialize_deepseek_client
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)

class CellType(Enum):
    INPUT_VARIABLE = "input_variable"
    OUTPUT_TARGET = "output_target"
    INTERMEDIATE_CALC = "intermediate_calculation"
    CONSTANT = "constant"
    LOOKUP_TABLE = "lookup_table"
    ASSUMPTION = "assumption"

class DistributionType(Enum):
    TRIANGULAR = "triangular"
    NORMAL = "normal"
    UNIFORM = "uniform"
    LOGNORMAL = "lognormal"
    BETA = "beta"
    CUSTOM = "custom"

@dataclass
class CellAnalysis:
    """Analysis result for a single cell"""
    sheet_name: str
    cell_address: str
    cell_type: CellType
    confidence_score: float  # 0.0 to 1.0
    description: str
    formula: Optional[str] = None
    current_value: Optional[Any] = None
    dependencies: List[str] = None
    dependents: List[str] = None
    business_context: Optional[str] = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []
        if self.dependents is None:
            self.dependents = []

@dataclass
class VariableSuggestion:
    """AI suggestion for a Monte Carlo variable"""
    cell_analysis: CellAnalysis
    suggested_distribution: DistributionType
    distribution_params: Dict[str, float]
    reasoning: str
    risk_impact: str  # "high", "medium", "low"
    business_rationale: str

@dataclass
class ModelInsights:
    """High-level insights about the Excel model"""
    model_type: str  # e.g., "Financial Projection", "Valuation Model", "Budget"
    complexity_score: float = 0.0  # 0.0 to 1.0
    key_drivers: List[str] = None
    output_variables: List[str] = None
    potential_risks: List[str] = None
    model_quality_issues: List[str] = None
    recommended_iterations: int = 5000
    business_purpose: str = "Financial model analysis"  # Business purpose from DeepSeek
    key_assumptions: List[str] = None  # Key assumptions from DeepSeek
    
    def __post_init__(self):
        """Initialize default values for lists"""
        if self.key_drivers is None:
            self.key_drivers = []
        if self.output_variables is None:
            self.output_variables = []
        if self.potential_risks is None:
            self.potential_risks = []
        if self.model_quality_issues is None:
            self.model_quality_issues = []
        if self.key_assumptions is None:
            self.key_assumptions = []

class ExcelIntelligenceAgent:
    """
    AI-powered agent that analyzes Excel files to understand:
    - Business logic and model structure
    - Appropriate input variables for Monte Carlo simulation
    - Target output variables to track
    - Suggested probability distributions
    """
    
    def __init__(self, use_deepseek: bool = True):
        self.use_deepseek = use_deepseek
        self.deepseek_client = None
        
        if use_deepseek:
            api_key = os.getenv('DEEPSEEK_API_KEY', 'sk-44c7f06f6e8244c681aef8833b7cdb47')
            if api_key:
                self.deepseek_client = initialize_deepseek_client(api_key)
                logger.info("✅ DeepSeek client initialized for Excel analysis")
            else:
                logger.warning("⚠️ No DeepSeek API key found, using rule-based analysis only")
                self.use_deepseek = False
        
        # Pattern recognition for different cell types
        self.input_patterns = [
            r'assumption',
            r'input',
            r'rate',
            r'price',
            r'volume',
            r'growth',
            r'cost',
            r'margin',
            r'percentage',
            r'factor',
            r'multiplier'
        ]
        
        self.output_patterns = [
            r'total',
            r'sum',
            r'result',
            r'profit',
            r'revenue',
            r'npv',
            r'irr',
            r'value',
            r'cash.*flow',
            r'ebitda',
            r'return'
        ]
        
        self.financial_functions = {
            'NPV', 'IRR', 'PV', 'FV', 'PMT', 'RATE', 'NPER',
            'XNPV', 'XIRR', 'MIRR'
        }
        
    async def analyze_excel_model_enhanced(self,
                                           file_id: str,
                                           workbook_data: Dict[str, Any],
                                           ultra_analysis: Optional[Dict[str, Any]] = None,
                                           sheet_name: str = None) -> Tuple[List[CellAnalysis], ModelInsights]:
        """
        Enhanced analysis leveraging Ultra engine's formula analysis and dependency tracking
        
        Args:
            file_id: Excel file identifier
            workbook_data: Parsed Excel workbook data from existing parser
            ultra_analysis: Ultra engine's pre-computation analysis (formulas, dependencies, etc.)
            sheet_name: Specific sheet to analyze (None = analyze all)
            
        Returns:
            Tuple of (cell_analyses, model_insights)
        """
        logger.info(f"🧠 [AI_ANALYSIS_ENHANCED] Starting enhanced Excel analysis with Ultra engine data")
        
        try:
            # Get Ultra engine data if not provided
            if ultra_analysis is None:
                ultra_analysis = await self._get_ultra_engine_analysis(file_id)
                
            # Extract all sheets or focus on specific sheet
            sheets_to_analyze = []
            if sheet_name and sheet_name in workbook_data.get('sheets', {}):
                sheets_to_analyze = [sheet_name]
            else:
                sheets_to_analyze = list(workbook_data.get('sheets', {}).keys())
            
            all_cell_analyses = []
            
            # Analyze each sheet with Ultra engine insights
            for sheet in sheets_to_analyze:
                sheet_data = workbook_data['sheets'][sheet]
                sheet_analyses = await self._analyze_sheet_enhanced(sheet, sheet_data, ultra_analysis, workbook_data)
                all_cell_analyses.extend(sheet_analyses)
            
            # Generate enhanced model-level insights
            model_insights = await self._generate_model_insights_enhanced(all_cell_analyses, ultra_analysis, workbook_data)
            
            logger.info(f"✅ [AI_ANALYSIS_ENHANCED] Completed enhanced analysis: {len(all_cell_analyses)} cells analyzed")
            return all_cell_analyses, model_insights
            
        except Exception as e:
            logger.error(f"❌ [AI_ANALYSIS_ENHANCED] Failed to analyze Excel model: {e}")
            # Fallback to standard analysis
            return await self.analyze_excel_model(workbook_data, sheet_name)
    
    async def analyze_excel_model_deepseek(self,
                                          file_id: str,
                                          workbook_data: Dict[str, Any],
                                          sheet_name: str = None) -> Tuple[List[CellAnalysis], ModelInsights]:
        """
        DeepSeek-powered comprehensive Excel model analysis
        Let DeepSeek do the heavy lifting of understanding the entire model
        """
        logger.info(f"🎯 [DEEPSEEK_ANALYSIS] Starting comprehensive DeepSeek analysis for {file_id}")
        
        try:
            if not self.deepseek_client:
                logger.warning("⚠️ [DEEPSEEK_ANALYSIS] No DeepSeek client available, falling back to enhanced analysis")
                return await self.analyze_excel_model_enhanced(file_id, workbook_data, None, sheet_name)
            
            # Get comprehensive analysis from DeepSeek
            deepseek_analysis = await self.deepseek_client.analyze_complete_excel_model(
                workbook_data, file_id, sheet_name
            )
            
            if not deepseek_analysis:
                logger.warning("⚠️ [DEEPSEEK_ANALYSIS] No response from DeepSeek, falling back to enhanced analysis")
                return await self.analyze_excel_model_enhanced(file_id, workbook_data, None, sheet_name)
            
            logger.info(f"✅ [DEEPSEEK_ANALYSIS] DeepSeek analysis received: {len(deepseek_analysis.get('input_variables', []))} inputs, {len(deepseek_analysis.get('output_targets', []))} outputs")
            
            # Convert DeepSeek analysis to our CellAnalysis format
            cell_analyses = self._convert_deepseek_to_cell_analyses(deepseek_analysis)
            
            # Create model insights from DeepSeek analysis
            model_insights = self._convert_deepseek_to_model_insights(deepseek_analysis)
            
            logger.info(f"✅ [DEEPSEEK_ANALYSIS] Completed DeepSeek analysis: {len(cell_analyses)} cells analyzed")
            return cell_analyses, model_insights
            
        except Exception as e:
            logger.error(f"❌ [DEEPSEEK_ANALYSIS] Failed DeepSeek analysis: {e}")
            # Fallback to enhanced analysis
            return await self.analyze_excel_model_enhanced(file_id, workbook_data, None, sheet_name)
    
    def _convert_deepseek_to_cell_analyses(self, deepseek_analysis: Dict[str, Any]) -> List[CellAnalysis]:
        """Convert DeepSeek analysis results to CellAnalysis objects"""
        cell_analyses = []
        
        # Process input variables
        for input_var in deepseek_analysis.get('input_variables', []):
            cell_analysis = CellAnalysis(
                sheet_name=input_var.get('sheet_name', 'Sheet1'),
                cell_address=input_var['cell_address'],
                cell_type=CellType.INPUT_VARIABLE,
                confidence_score=0.95,  # High confidence from DeepSeek
                description=input_var.get('variable_name', f"Input variable {input_var['cell_address']}"),
                current_value=input_var.get('current_value'),
                business_context=f"Data type: {input_var.get('data_type', 'number')}",
                dependencies=[],
                dependents=input_var.get('referenced_by', [])
            )
            cell_analyses.append(cell_analysis)
        
        # Process output targets
        for output_target in deepseek_analysis.get('output_targets', []):
            cell_analysis = CellAnalysis(
                sheet_name=output_target.get('sheet_name', 'Sheet1'),
                cell_address=output_target['cell_address'],
                cell_type=CellType.OUTPUT_TARGET,
                confidence_score=0.95,  # High confidence from DeepSeek
                description=output_target.get('variable_name', f"Output target {output_target['cell_address']}"),
                current_value=output_target.get('current_value'),
                business_context=f"Formula: {output_target.get('formula', 'N/A')}, Data type: {output_target.get('data_type', 'number')}",
                dependencies=output_target.get('depends_on', []),
                dependents=[]
            )
            cell_analyses.append(cell_analysis)
        
        return cell_analyses
    
    def _convert_deepseek_to_model_insights(self, deepseek_analysis: Dict[str, Any]) -> ModelInsights:
        """Convert DeepSeek analysis to ModelInsights object"""
        
        model_kpis = deepseek_analysis.get('model_kpis', {})
        
        # Extract structure description and create basic insights
        structure_description = model_kpis.get('structure_description', 'Excel model analysis')
        
        # Estimate iterations based on model complexity (number of cells)
        total_cells = model_kpis.get('total_cells', 0)
        if total_cells < 50:
            recommended_iterations = 1000
        elif total_cells < 200:
            recommended_iterations = 5000
        else:
            recommended_iterations = 10000
        
        return ModelInsights(
            business_purpose=structure_description,
            model_type='technical_analysis',
            key_assumptions=[],
            recommended_iterations=recommended_iterations
        )
    
    async def analyze_excel_model(self, 
                                  workbook_data: Dict[str, Any],
                                  sheet_name: str = None) -> Tuple[List[CellAnalysis], ModelInsights]:
        """
        Comprehensive analysis of Excel model structure
        
        Args:
            workbook_data: Parsed Excel workbook data from existing parser
            sheet_name: Specific sheet to analyze (None = analyze all)
            
        Returns:
            Tuple of (cell_analyses, model_insights)
        """
        logger.info(f"🧠 [AI_ANALYSIS] Starting Excel intelligence analysis")
        
        try:
            # Extract all sheets or focus on specific sheet
            sheets_to_analyze = []
            if sheet_name and sheet_name in workbook_data.get('sheets', {}):
                sheets_to_analyze = [sheet_name]
            else:
                sheets_to_analyze = list(workbook_data.get('sheets', {}).keys())
            
            all_cell_analyses = []
            
            # Analyze each sheet
            for sheet in sheets_to_analyze:
                sheet_data = workbook_data['sheets'][sheet]
                sheet_analyses = await self._analyze_sheet(sheet, sheet_data, workbook_data)
                all_cell_analyses.extend(sheet_analyses)
            
            # Generate model-level insights
            model_insights = await self._generate_model_insights(all_cell_analyses, workbook_data)
            
            logger.info(f"✅ [AI_ANALYSIS] Completed analysis: {len(all_cell_analyses)} cells analyzed")
            return all_cell_analyses, model_insights
            
        except Exception as e:
            logger.error(f"❌ [AI_ANALYSIS] Failed to analyze Excel model: {e}")
            # Return empty results rather than failing
            return [], ModelInsights(
                model_type="Unknown",
                complexity_score=0.0,
                key_drivers=[],
                output_variables=[],
                potential_risks=["Analysis failed - manual review recommended"],
                model_quality_issues=["AI analysis encountered errors"],
                recommended_iterations=10000
            )
    
    async def _analyze_sheet(self, 
                           sheet_name: str, 
                           sheet_data: Dict[str, Any],
                           workbook_data: Dict[str, Any]) -> List[CellAnalysis]:
        """Analyze a single sheet to identify cell types and patterns"""
        
        cell_analyses = []
        
        # Get formulas and values from sheet
        formulas = sheet_data.get('formulas', {})
        grid_data = sheet_data.get('data', [])
        
        # Build cell value lookup
        cell_values = {}
        for row_idx, row_data in enumerate(grid_data):
            for col_idx, cell_data in enumerate(row_data):
                if isinstance(cell_data, dict) and cell_data.get('value') is not None:
                    cell_address = f"{chr(65 + col_idx)}{row_idx + 1}"
                    cell_values[cell_address] = cell_data['value']
        
        # Analyze each formula cell
        for cell_coord, formula_str in formulas.items():
            if isinstance(formula_str, str) and formula_str.startswith('='):
                cell_address = self._coord_to_address(cell_coord)
                
                analysis = await self._analyze_single_cell(
                    sheet_name=sheet_name,
                    cell_address=cell_address,
                    formula=formula_str,
                    current_value=cell_values.get(cell_address),
                    all_formulas=formulas,
                    all_values=cell_values,
                    workbook_data=workbook_data
                )
                
                if analysis:
                    cell_analyses.append(analysis)
        
        # Also analyze cells with values but no formulas (potential inputs)
        for cell_address, value in cell_values.items():
            if cell_address not in [self._coord_to_address(coord) for coord in formulas.keys()]:
                # This is a value-only cell - potential input
                analysis = await self._analyze_value_cell(
                    sheet_name=sheet_name,
                    cell_address=cell_address,
                    value=value,
                    context=cell_values
                )
                if analysis:
                    cell_analyses.append(analysis)
        
        logger.debug(f"📊 [AI_ANALYSIS] Sheet {sheet_name}: {len(cell_analyses)} cells analyzed")
        return cell_analyses
    
    async def _analyze_single_cell(self,
                                 sheet_name: str,
                                 cell_address: str,
                                 formula: str,
                                 current_value: Any,
                                 all_formulas: Dict[str, str],
                                 all_values: Dict[str, Any],
                                 workbook_data: Dict[str, Any]) -> Optional[CellAnalysis]:
        """Analyze a single cell with formula"""
        
        # Extract dependencies from formula
        dependencies = self._extract_cell_references(formula)
        
        # Find dependents (cells that reference this cell)
        dependents = self._find_dependents(cell_address, all_formulas)
        
        # Determine cell type based on formula patterns
        cell_type, confidence = self._classify_cell_type(
            formula=formula,
            dependencies=dependencies,
            dependents=dependents,
            cell_address=cell_address
        )
        
        # Generate business context using AI or rules
        business_context = await self._generate_business_context(
            cell_address, formula, cell_type, sheet_name
        )
        
        return CellAnalysis(
            sheet_name=sheet_name,
            cell_address=cell_address,
            cell_type=cell_type,
            confidence_score=confidence,
            description=self._generate_cell_description(cell_type, formula, cell_address),
            formula=formula,
            current_value=current_value,
            dependencies=dependencies,
            dependents=dependents,
            business_context=business_context
        )
    
    async def _analyze_value_cell(self,
                                sheet_name: str,
                                cell_address: str,
                                value: Any,
                                context: Dict[str, Any]) -> Optional[CellAnalysis]:
        """Analyze a cell that contains only a value (potential input variable)"""
        
        # Skip text/string values unless they look like assumptions
        if isinstance(value, str) and not self._is_assumption_label(value):
            return None
        
        # Skip very large/small numbers that are likely intermediate calculations
        if isinstance(value, (int, float)):
            if abs(value) > 1e6 or (value != 0 and abs(value) < 1e-4):
                return None
        
        # Check if nearby cells have assumption-like labels
        assumption_context = self._check_assumption_context(cell_address, context)
        
        # Improved input variable detection
        is_likely_input = (
            assumption_context or 
            self._looks_like_input_value(value) or
            self._has_input_label_nearby(cell_address, context)
        )
        
        if is_likely_input:
            # Generate better description based on context
            description = self._generate_input_description(cell_address, value, assumption_context, context)
            confidence = 0.9 if assumption_context else 0.7 if self._has_input_label_nearby(cell_address, context) else 0.5
            
            return CellAnalysis(
                sheet_name=sheet_name,
                cell_address=cell_address,
                cell_type=CellType.INPUT_VARIABLE,
                confidence_score=confidence,
                description=description,
                current_value=value,
                business_context=assumption_context or self._get_nearby_labels(cell_address, context)
            )
        
        return None
    
    def _classify_cell_type(self,
                          formula: str,
                          dependencies: List[str],
                          dependents: List[str],
                          cell_address: str) -> Tuple[CellType, float]:
        """Classify cell type based on formula analysis"""
        
        formula_upper = formula.upper()
        
        # Check for financial functions (likely outputs)
        for func in self.financial_functions:
            if func in formula_upper:
                return CellType.OUTPUT_TARGET, 0.9
        
        # Check for aggregation functions (likely outputs)
        if any(func in formula_upper for func in ['SUM(', 'SUMIF(', 'SUMIFS(', 'TOTAL']):
            return CellType.OUTPUT_TARGET, 0.8
        
        # Check for complex calculations with many dependencies (intermediate)
        if len(dependencies) > 5:
            return CellType.INTERMEDIATE_CALC, 0.7
        
        # Check for lookup functions
        if any(func in formula_upper for func in ['VLOOKUP(', 'HLOOKUP(', 'INDEX(', 'MATCH(']):
            return CellType.LOOKUP_TABLE, 0.8
        
        # Check for simple arithmetic (could be input or intermediate)
        if len(dependencies) == 0:
            return CellType.CONSTANT, 0.8
        elif len(dependencies) <= 2 and any(op in formula for op in ['+', '-', '*', '/']):
            return CellType.INTERMEDIATE_CALC, 0.6
        
        # Default to intermediate calculation
        return CellType.INTERMEDIATE_CALC, 0.5
    
    async def _generate_business_context(self,
                                       cell_address: str,
                                       formula: str,
                                       cell_type: CellType,
                                       sheet_name: str) -> str:
        """Generate business context description using AI or rules"""
        
        if self.use_deepseek and self.deepseek_client:
            try:
                response = await self.deepseek_client.analyze_excel_context(
                    cell_address, formula, cell_type.value, sheet_name
                )
                if response:
                    return response.strip()
                    
            except Exception as e:
                logger.debug(f"DeepSeek context generation failed: {e}")
        
        # Fallback to rule-based context
        return self._generate_rule_based_context(cell_address, formula, cell_type)
    
    # Removed _call_openai method - now using DeepSeek client directly
    
    def _generate_rule_based_context(self, cell_address: str, formula: str, cell_type: CellType) -> str:
        """Generate context using rule-based logic"""
        
        formula_upper = formula.upper()
        
        if cell_type == CellType.OUTPUT_TARGET:
            if 'NPV' in formula_upper:
                return "Net Present Value calculation - key financial metric"
            elif 'IRR' in formula_upper:
                return "Internal Rate of Return - profitability measure"
            elif 'SUM' in formula_upper:
                return "Aggregated total - likely key output metric"
            else:
                return "Calculated output - potential target for analysis"
        
        elif cell_type == CellType.INPUT_VARIABLE:
            return "Input parameter - candidate for Monte Carlo variation"
        
        elif cell_type == CellType.LOOKUP_TABLE:
            return "Data lookup - references external table or data"
        
        else:
            return "Intermediate calculation supporting the model"
    
    async def _generate_model_insights(self,
                                     cell_analyses: List[CellAnalysis],
                                     workbook_data: Dict[str, Any]) -> ModelInsights:
        """Generate high-level insights about the entire model"""
        
        # Classify model type
        model_type = self._determine_model_type(cell_analyses)
        
        # Calculate complexity
        complexity_score = self._calculate_complexity_score(cell_analyses)
        
        # Identify key drivers (high-confidence input variables)
        key_drivers = [
            analysis.cell_address for analysis in cell_analyses
            if analysis.cell_type == CellType.INPUT_VARIABLE and analysis.confidence_score > 0.7
        ]
        
        # Identify output variables
        output_variables = [
            analysis.cell_address for analysis in cell_analyses
            if analysis.cell_type == CellType.OUTPUT_TARGET and analysis.confidence_score > 0.6
        ]
        
        # Identify potential risks
        potential_risks = self._identify_model_risks(cell_analyses)
        
        # Check model quality
        quality_issues = self._assess_model_quality(cell_analyses)
        
        # Recommend iteration count based on complexity
        recommended_iterations = self._recommend_iterations(complexity_score, len(key_drivers))
        
        return ModelInsights(
            model_type=model_type,
            complexity_score=complexity_score,
            key_drivers=key_drivers,
            output_variables=output_variables,
            potential_risks=potential_risks,
            model_quality_issues=quality_issues,
            recommended_iterations=recommended_iterations
        )
    
    def _determine_model_type(self, cell_analyses: List[CellAnalysis]) -> str:
        """Determine the type of business model based on formula patterns"""
        
        formula_patterns = []
        for analysis in cell_analyses:
            if analysis.formula:
                formula_patterns.append(analysis.formula.upper())
        
        combined_formulas = ' '.join(formula_patterns)
        
        if any(pattern in combined_formulas for pattern in ['NPV', 'IRR', 'DISCOUNT']):
            return "Financial Valuation Model"
        elif any(pattern in combined_formulas for pattern in ['REVENUE', 'COST', 'PROFIT']):
            return "Financial Projection Model"
        elif any(pattern in combined_formulas for pattern in ['BUDGET', 'FORECAST']):
            return "Budget/Forecast Model"
        elif any(pattern in combined_formulas for pattern in ['CASH', 'FLOW']):
            return "Cash Flow Model"
        else:
            return "General Business Model"
    
    def _calculate_complexity_score(self, cell_analyses: List[CellAnalysis]) -> float:
        """Calculate model complexity score (0.0 to 1.0)"""
        
        if not cell_analyses:
            return 0.0
        
        # Factors contributing to complexity
        total_formulas = len([a for a in cell_analyses if a.formula])
        avg_dependencies = np.mean([len(a.dependencies) for a in cell_analyses])
        unique_functions = set()
        
        for analysis in cell_analyses:
            if analysis.formula:
                # Extract function names from formulas
                functions = re.findall(r'([A-Z]+)\(', analysis.formula.upper())
                unique_functions.update(functions)
        
        # Normalize factors
        formula_score = min(total_formulas / 100, 1.0)  # 100+ formulas = max complexity
        dependency_score = min(avg_dependencies / 10, 1.0)  # 10+ avg deps = max complexity  
        function_score = min(len(unique_functions) / 20, 1.0)  # 20+ functions = max complexity
        
        # Weighted average
        complexity = (formula_score * 0.4 + dependency_score * 0.3 + function_score * 0.3)
        return round(complexity, 2)
    
    def _identify_model_risks(self, cell_analyses: List[CellAnalysis]) -> List[str]:
        """Identify potential risks in the model structure"""
        
        risks = []
        
        # Check for circular dependencies
        all_deps = []
        for analysis in cell_analyses:
            for dep in analysis.dependencies:
                all_deps.append((analysis.cell_address, dep))
        
        if len(set(all_deps)) != len(all_deps):
            risks.append("Potential circular dependencies detected")
        
        # Check for models with very few input variables
        input_vars = [a for a in cell_analyses if a.cell_type == CellType.INPUT_VARIABLE]
        if len(input_vars) < 3:
            risks.append("Model has very few input variables - may lack sensitivity")
        
        # Check for models with no clear outputs
        outputs = [a for a in cell_analyses if a.cell_type == CellType.OUTPUT_TARGET]
        if len(outputs) == 0:
            risks.append("No clear output variables identified")
        
        # Check for excessive complexity
        complex_cells = [a for a in cell_analyses if len(a.dependencies) > 10]
        if len(complex_cells) > len(cell_analyses) * 0.3:
            risks.append("High model complexity may impact performance")
        
        return risks
    
    def _assess_model_quality(self, cell_analyses: List[CellAnalysis]) -> List[str]:
        """Assess model quality and identify issues"""
        
        issues = []
        
        # Check confidence scores
        low_confidence = [a for a in cell_analyses if a.confidence_score < 0.5]
        if len(low_confidence) > len(cell_analyses) * 0.3:
            issues.append("Many cells have low classification confidence - manual review recommended")
        
        # Check for missing business context
        no_context = [a for a in cell_analyses if not a.business_context]
        if len(no_context) > len(cell_analyses) * 0.5:
            issues.append("Limited business context available - consider adding cell comments")
        
        return issues
    
    def _recommend_iterations(self, complexity_score: float, num_inputs: int) -> int:
        """Recommend number of Monte Carlo iterations based on model characteristics"""
        
        base_iterations = 10000
        
        # Adjust for complexity
        if complexity_score > 0.7:
            base_iterations = 50000
        elif complexity_score > 0.4:
            base_iterations = 25000
        
        # Adjust for number of inputs
        if num_inputs > 10:
            base_iterations = min(base_iterations * 2, 100000)
        elif num_inputs < 3:
            base_iterations = max(base_iterations // 2, 5000)
        
        return base_iterations
    
    # Utility methods
    def _coord_to_address(self, coord: Tuple[int, int]) -> str:
        """Convert (row, col) coordinates to Excel address like 'A1'"""
        if isinstance(coord, str):
            return coord
        row, col = coord
        return f"{chr(64 + col)}{row}"
    
    def _extract_cell_references(self, formula: str) -> List[str]:
        """Extract cell references from formula"""
        # Simple regex for cell references like A1, B2, etc.
        pattern = r'\b[A-Z]+\d+\b'
        return re.findall(pattern, formula.upper())
    
    def _find_dependents(self, cell_address: str, all_formulas: Dict[str, str]) -> List[str]:
        """Find cells that depend on this cell"""
        dependents = []
        for coord, formula in all_formulas.items():
            if cell_address in self._extract_cell_references(formula):
                dependents.append(self._coord_to_address(coord))
        return dependents
    
    def _is_assumption_label(self, text: str) -> bool:
        """Check if text looks like an assumption label"""
        assumption_keywords = ['assumption', 'input', 'rate', 'factor', 'growth', 'cost']
        return any(keyword in text.lower() for keyword in assumption_keywords)
    
    def _check_assumption_context(self, cell_address: str, context: Dict[str, Any]) -> str:
        """Check if nearby cells suggest this is an assumption"""
        # This is a simplified check - could be enhanced with better spatial analysis
        for addr, value in context.items():
            if isinstance(value, str) and self._is_assumption_label(value):
                return f"Near assumption label: {value}"
        return ""
    
    def _looks_like_input_value(self, value: Any) -> bool:
        """Check if value looks like a typical input parameter"""
        if isinstance(value, (int, float)):
            # Common ranges for business assumptions
            if 0 < value < 1:  # Percentages
                return True
            if 1 < value < 100:  # Rates, factors
                return True
            if 100 <= value <= 10000:  # Prices, costs
                return True
        return False
    
    def _has_input_label_nearby(self, cell_address: str, context: Dict[str, Any]) -> bool:
        """Check if there are input-related labels near this cell"""
        # Get adjacent cell in the same row (left cell for labels)
        col = ord(cell_address[0]) - ord('A')
        row = int(cell_address[1:])
        
        # Check cell to the left (common pattern: label in B, value in C)
        if col > 0:
            label_cell = f"{chr(ord('A') + col - 1)}{row}"
            if label_cell in context:
                label_value = context[label_cell]
                if isinstance(label_value, str) and self._is_input_label(label_value):
                    return True
        
        return False
    
    def _is_input_label(self, text: str) -> bool:
        """Check if text looks like an input variable label"""
        if not isinstance(text, str):
            return False
        
        text_lower = text.lower()
        input_keywords = [
            'price', 'cost', 'rate', 'discount', 'margin', 'percentage', 
            'factor', 'assumption', 'input', 'variable', 'parameter',
            'unit', 'growth', 'inflation', 'tax', 'interest'
        ]
        
        return any(keyword in text_lower for keyword in input_keywords)
    
    def _generate_input_description(self, cell_address: str, value: Any, 
                                  assumption_context: str, context: Dict[str, Any]) -> str:
        """Generate a descriptive name for an input variable"""
        
        # Try to get label from nearby cell
        nearby_label = self._get_nearby_labels(cell_address, context)
        if nearby_label:
            return f"Input parameter: {nearby_label} (value: {value})"
        
        # Fallback based on value type
        if isinstance(value, (int, float)):
            if 0 < value < 1:
                return f"Percentage/Rate parameter: {value*100:.1f}% (cell {cell_address})"
            elif 100 <= value <= 10000:
                return f"Price/Cost parameter: {value} (cell {cell_address})"
            else:
                return f"Numeric parameter: {value} (cell {cell_address})"
        
        return f"Input variable at {cell_address}: {value}"
    
    def _get_nearby_labels(self, cell_address: str, context: Dict[str, Any]) -> str:
        """Get descriptive labels from nearby cells"""
        col = ord(cell_address[0]) - ord('A')
        row = int(cell_address[1:])
        
        # Check cell to the left
        if col > 0:
            label_cell = f"{chr(ord('A') + col - 1)}{row}"
            if label_cell in context:
                label_value = context[label_cell]
                if isinstance(label_value, str) and label_value.strip():
                    return label_value.strip()
        
        return ""
    
    def _generate_cell_description(self, cell_type: CellType, formula: str, cell_address: str) -> str:
        """Generate a human-readable description of the cell"""
        
        type_descriptions = {
            CellType.INPUT_VARIABLE: f"Input variable at {cell_address} - candidate for Monte Carlo simulation",
            CellType.OUTPUT_TARGET: f"Output calculation at {cell_address} - potential target for analysis",
            CellType.INTERMEDIATE_CALC: f"Intermediate calculation at {cell_address}",
            CellType.CONSTANT: f"Constant value at {cell_address}",
            CellType.LOOKUP_TABLE: f"Data lookup at {cell_address}",
            CellType.ASSUMPTION: f"Business assumption at {cell_address}"
        }
        
        return type_descriptions.get(cell_type, f"Cell at {cell_address}")
    
    # Enhanced methods using Ultra engine data
    async def _get_ultra_engine_analysis(self, file_id: str) -> Dict[str, Any]:
        """
        Get Ultra engine's pre-computation analysis data
        
        Returns formulas, dependencies, evaluation order, and other structural insights
        """
        try:
            # Import services here to avoid circular dependencies
            from excel_parser.service import get_formulas_for_file, get_all_parsed_sheets_data
            
            logger.info(f"🔧 [ULTRA_DATA] Fetching Ultra engine analysis for file: {file_id}")
            
            # Get all formulas (per sheet)
            all_formulas = await get_formulas_for_file(file_id)
            
            # Get parsed sheet data
            sheets_data = await get_all_parsed_sheets_data(file_id)
            
            # Build dependency information
            dependency_info = {}
            formula_count = 0
            
            for sheet_name, sheet_formulas in all_formulas.items():
                dependency_info[sheet_name] = {}
                for cell, formula in sheet_formulas.items():
                    if formula and formula.startswith('='):
                        # Extract dependencies from formula
                        dependencies = self._extract_formula_dependencies(formula, sheet_name)
                        dependency_info[sheet_name][cell] = {
                            'formula': formula,
                            'dependencies': dependencies,
                            'is_formula': True
                        }
                        formula_count += 1
            
            # Calculate complexity metrics
            total_cells = sum(len(sheet_data.grid_data) for sheet_data in sheets_data)
            formula_density = formula_count / total_cells if total_cells > 0 else 0
            
            ultra_analysis = {
                'formulas': all_formulas,
                'dependency_info': dependency_info,
                'sheets_data': {sheet.sheet_name: sheet for sheet in sheets_data},
                'metrics': {
                    'total_formulas': formula_count,
                    'total_cells': total_cells,
                    'formula_density': formula_density,
                    'sheets_count': len(all_formulas),
                },
                'file_id': file_id
            }
            
            logger.info(f"✅ [ULTRA_DATA] Ultra analysis completed: {formula_count} formulas across {len(all_formulas)} sheets")
            return ultra_analysis
            
        except Exception as e:
            logger.error(f"❌ [ULTRA_DATA] Failed to get Ultra engine analysis: {e}")
            return {
                'formulas': {},
                'dependency_info': {},
                'sheets_data': {},
                'metrics': {'total_formulas': 0, 'total_cells': 0, 'formula_density': 0},
                'file_id': file_id
            }
    
    def _extract_formula_dependencies(self, formula: str, current_sheet: str) -> List[str]:
        """Extract cell references from a formula"""
        dependencies = []
        
        # Simple regex to find cell references like A1, B2, $A$1, etc.
        cell_pattern = r'\$?[A-Z]+\$?\d+'
        matches = re.findall(cell_pattern, formula)
        
        for match in matches:
            # Clean up the match (remove $ signs)
            clean_match = match.replace('$', '')
            # Add sheet prefix if not cross-sheet reference
            if '!' not in formula or current_sheet in formula:
                dependencies.append(f"{current_sheet}!{clean_match}")
            else:
                dependencies.append(clean_match)
        
        return dependencies
    
    async def _analyze_sheet_enhanced(self, 
                                     sheet_name: str, 
                                     sheet_data: Dict[str, Any],
                                     ultra_analysis: Dict[str, Any],
                                     workbook_data: Dict[str, Any]) -> List[CellAnalysis]:
        """
        Enhanced sheet analysis using Ultra engine insights
        """
        logger.info(f"🔍 [SHEET_ANALYSIS_ENHANCED] Analyzing sheet: {sheet_name}")
        
        cell_analyses = []
        
        # Get Ultra engine data for this sheet
        sheet_formulas = ultra_analysis.get('formulas', {}).get(sheet_name, {})
        sheet_dependencies = ultra_analysis.get('dependency_info', {}).get(sheet_name, {})
        
        logger.info(f"🔍 [ULTRA_INSIGHTS] Found {len(sheet_formulas)} formulas and {len(sheet_dependencies)} dependencies")
        
        # Analyze formula cells with dependency context
        for cell_coord, cell_info in sheet_dependencies.items():
            formula = cell_info.get('formula', '')
            dependencies = cell_info.get('dependencies', [])
            
            # Enhanced classification using dependency patterns
            cell_type = self._classify_cell_enhanced(
                cell_coord, formula, dependencies, sheet_name, ultra_analysis
            )
            
            # Calculate confidence based on Ultra engine insights
            confidence = self._calculate_confidence_enhanced(
                formula, dependencies, cell_type, ultra_analysis
            )
            
            # Generate business context using AI if available
            business_context = await self._generate_business_context(
                cell_coord, formula, dependencies, cell_type
            )
            
            analysis = CellAnalysis(
                sheet_name=sheet_name,
                cell_address=cell_coord,
                cell_type=cell_type,
                confidence_score=confidence,
                description=f"Formula cell with {len(dependencies)} dependencies",
                formula=formula,
                dependencies=dependencies,
                business_context=business_context
            )
            
            cell_analyses.append(analysis)
        
        logger.info(f"✅ [SHEET_ANALYSIS_ENHANCED] Analyzed {len(cell_analyses)} formula cells in {sheet_name}")
        return cell_analyses
    
    def _classify_cell_enhanced(self, 
                               cell_coord: str, 
                               formula: str, 
                               dependencies: List[str],
                               sheet_name: str,
                               ultra_analysis: Dict[str, Any]) -> CellType:
        """
        Enhanced cell classification using Ultra engine dependency insights
        """
        # Count formula complexity metrics
        dependency_count = len(dependencies)
        formula_complexity = len(formula) if formula else 0
        
        # Check for input variable patterns (few or no dependencies)
        if dependency_count <= 2 and self._has_input_patterns(formula):
            return CellType.INPUT_VARIABLE
        
        # Check for output patterns (many dependencies, complex formulas)
        if dependency_count >= 5 and formula_complexity > 50:
            return CellType.OUTPUT_TARGET
        
        # Check for lookup table patterns
        if 'VLOOKUP' in formula or 'HLOOKUP' in formula or 'INDEX' in formula:
            return CellType.LOOKUP_TABLE
        
        # Check for financial functions (likely targets)
        if any(func in formula.upper() for func in self.financial_functions):
            return CellType.OUTPUT_TARGET
        
        # Default to intermediate calculation
        return CellType.INTERMEDIATE_CALC
    
    def _has_input_patterns(self, formula: str) -> bool:
        """Check if formula contains patterns typical of input variables"""
        if not formula:
            return False
        
        # Simple formulas with constants often indicate inputs
        simple_patterns = ['+', '-', '*', '/', '%']
        return any(pattern in formula for pattern in simple_patterns) and len(formula) < 30
    
    def _calculate_confidence_enhanced(self, 
                                      formula: str, 
                                      dependencies: List[str],
                                      cell_type: CellType,
                                      ultra_analysis: Dict[str, Any]) -> float:
        """
        Calculate confidence score using Ultra engine insights
        """
        base_confidence = 0.5
        
        # Boost confidence for well-structured patterns
        if cell_type == CellType.INPUT_VARIABLE and len(dependencies) <= 1:
            base_confidence += 0.3
        
        if cell_type == CellType.OUTPUT_TARGET and len(dependencies) >= 3:
            base_confidence += 0.3
        
        # Formula complexity indicators
        if formula and len(formula) > 20:
            base_confidence += 0.1
        
        # Financial function presence
        if formula and any(func in formula.upper() for func in self.financial_functions):
            base_confidence += 0.2
        
        return min(1.0, base_confidence)
    
    async def _generate_business_context(self,
                                        cell_coord: str,
                                        formula: str,
                                        dependencies: List[str],
                                        cell_type: CellType) -> str:
        """
        Generate business context using AI if available
        """
        if not self.use_deepseek or not self.deepseek_client:
            return f"Formula cell of type {cell_type.value}"
        
        try:
            prompt = f"""
            Analyze this Excel formula cell and provide business context:
            
            Cell: {cell_coord}
            Formula: {formula}
            Dependencies: {', '.join(dependencies[:5])}
            Type: {cell_type.value}
            
            Provide a brief business interpretation of what this cell likely represents.
            """
            
            # Use DeepSeek to generate context (simplified call)
            response = "Formula-based calculation"  # Placeholder - would call DeepSeek API
            return response.strip()
            
        except Exception as e:
            logger.warning(f"⚠️ [BUSINESS_CONTEXT] AI generation failed: {e}")
            return f"Formula cell of type {cell_type.value}"
    
    async def _generate_model_insights_enhanced(self,
                                               cell_analyses: List[CellAnalysis],
                                               ultra_analysis: Dict[str, Any],
                                               workbook_data: Dict[str, Any]) -> ModelInsights:
        """
        Generate enhanced model insights using Ultra engine data
        """
        metrics = ultra_analysis.get('metrics', {})
        
        # Calculate enhanced complexity score
        formula_density = metrics.get('formula_density', 0)
        total_formulas = metrics.get('total_formulas', 0)
        
        complexity_score = min(1.0, formula_density * 2 + (total_formulas / 1000) * 0.5)
        
        # Identify key drivers (input variables with high confidence)
        key_drivers = [
            cell.cell_address for cell in cell_analyses
            if cell.cell_type == CellType.INPUT_VARIABLE and cell.confidence_score > 0.7
        ]
        
        # Identify output variables (targets with high confidence)
        output_variables = [
            cell.cell_address for cell in cell_analyses
            if cell.cell_type == CellType.OUTPUT_TARGET and cell.confidence_score > 0.7
        ]
        
        # Generate model type based on patterns
        model_type = self._determine_model_type(cell_analyses, ultra_analysis)
        
        # Calculate recommended iterations based on complexity
        recommended_iterations = max(1000, min(100000, total_formulas * 100))
        
        return ModelInsights(
            model_type=model_type,
            complexity_score=complexity_score,
            key_drivers=key_drivers[:10],  # Top 10
            output_variables=output_variables[:5],  # Top 5
            potential_risks=["Complex formula dependencies", "Multiple calculation layers"],
            model_quality_issues=[],
            recommended_iterations=recommended_iterations
        )
    
    def _determine_model_type(self, 
                             cell_analyses: List[CellAnalysis],
                             ultra_analysis: Dict[str, Any]) -> str:
        """
        Determine model type based on formula patterns
        """
        # Look for financial functions
        financial_functions_found = set()
        for cell in cell_analyses:
            if cell.formula:
                for func in self.financial_functions:
                    if func in cell.formula.upper():
                        financial_functions_found.add(func)
        
        if financial_functions_found:
            if 'NPV' in financial_functions_found or 'IRR' in financial_functions_found:
                return "Financial Valuation Model"
            elif 'PMT' in financial_functions_found or 'PV' in financial_functions_found:
                return "Financial Planning Model"
            else:
                return "Financial Analysis Model"
        
        # Check for forecasting patterns
        if any('growth' in cell.business_context.lower() for cell in cell_analyses if cell.business_context):
            return "Forecasting Model"
        
        return "Business Model"
