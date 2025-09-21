"""
Variable Suggestion Engine - AI-powered Monte Carlo variable recommendations
Suggests appropriate input variables, distributions, and parameters
"""

import logging
import json
import asyncio
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import re
from scipy import stats
import os
from .deepseek_client import get_deepseek_client

from .excel_intelligence import (
    CellAnalysis, VariableSuggestion, CellType, DistributionType
)

logger = logging.getLogger(__name__)

@dataclass
class DistributionParameters:
    """Parameters for different distribution types"""
    distribution_type: DistributionType
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    most_likely: Optional[float] = None  # For triangular
    mean: Optional[float] = None  # For normal
    std_dev: Optional[float] = None  # For normal
    alpha: Optional[float] = None  # For beta
    beta: Optional[float] = None  # For beta
    confidence_level: float = 0.8  # Confidence in suggestion
    reasoning: str = "Default distribution parameters"  # Explanation for the choice
    risk_impact: str = "medium"  # Risk impact level
    business_rationale: str = "Standard Monte Carlo variable"  # Business justification

@dataclass
class VariableConfiguration:
    """Complete Monte Carlo variable configuration"""
    cell_address: str
    sheet_name: str
    variable_name: str
    distribution: DistributionParameters
    current_value: Any
    business_justification: str
    risk_category: str  # "high", "medium", "low"
    correlation_candidates: List[str]  # Other variables that might correlate

class VariableSuggestionEngine:
    """
    AI-powered engine that suggests appropriate Monte Carlo variables
    based on Excel model analysis and business context
    """
    
    def __init__(self, use_deepseek: bool = True):
        # Re-enable DeepSeek but will rely on timeouts for protection
        self.use_deepseek = use_deepseek
        self.deepseek_client = get_deepseek_client()
        
        if use_deepseek and not self.deepseek_client:
            logger.warning("⚠️ DeepSeek client not initialized, using rule-based suggestions only")
            self.use_deepseek = False
        elif use_deepseek:
            logger.info("✅ Using DeepSeek client for variable suggestions")
        
        # Business context patterns for distribution suggestions
        self.distribution_patterns = {
            'growth_rates': {
                'keywords': ['growth', 'increase', 'rate', 'percent'],
                'suggested_distribution': DistributionType.TRIANGULAR,
                'reasoning': 'Growth rates typically have a most likely value with bounded upside/downside'
            },
            'costs': {
                'keywords': ['cost', 'expense', 'price', 'fee'],
                'suggested_distribution': DistributionType.TRIANGULAR,
                'reasoning': 'Costs usually have a baseline with variation around supply/demand factors'
            },
            'volumes': {
                'keywords': ['volume', 'quantity', 'units', 'sales'],
                'suggested_distribution': DistributionType.NORMAL,
                'reasoning': 'Volumes often follow normal distribution due to market demand patterns'
            },
            'percentages': {
                'keywords': ['percent', 'ratio', 'margin', 'rate'],
                'suggested_distribution': DistributionType.BETA,
                'reasoning': 'Percentages are naturally bounded between 0 and 1'
            },
            'financial_returns': {
                'keywords': ['return', 'yield', 'interest', 'discount'],
                'suggested_distribution': DistributionType.NORMAL,
                'reasoning': 'Financial returns often approximate normal distribution'
            }
        }
        
        # Risk impact categories
        self.high_impact_keywords = [
            'revenue', 'sales', 'price', 'volume', 'margin', 'npv', 'irr'
        ]
        self.medium_impact_keywords = [
            'cost', 'expense', 'rate', 'growth', 'factor'
        ]
    
    async def suggest_monte_carlo_variables(self, 
                                          cell_analyses: List[CellAnalysis],
                                          workbook_data: Dict[str, Any],
                                          max_suggestions: int = 10) -> List[VariableConfiguration]:
        """
        Generate Monte Carlo variable suggestions based on Excel analysis
        
        Args:
            cell_analyses: Results from ExcelIntelligenceAgent
            workbook_data: Original workbook data for context
            max_suggestions: Maximum number of variables to suggest
            
        Returns:
            List of suggested variable configurations
        """
        logger.info(f"🎯 [VAR_SUGGEST] Generating Monte Carlo variable suggestions")
        
        try:
            # Filter to potential input variables
            input_candidates = [
                analysis for analysis in cell_analyses
                if analysis.cell_type in [CellType.INPUT_VARIABLE, CellType.ASSUMPTION]
                and analysis.confidence_score > 0.4
            ]
            
            logger.info(f"📊 [VAR_SUGGEST] Found {len(input_candidates)} input candidates")
            
            # For large models, pre-filter to top candidates to avoid timeout
            if len(input_candidates) > 50:
                logger.info(f"🔧 [VAR_SUGGEST] Large model detected ({len(input_candidates)} candidates), pre-filtering to top 50 by confidence")
                input_candidates = sorted(input_candidates, key=lambda x: x.confidence_score, reverse=True)[:50]
                logger.info(f"📊 [VAR_SUGGEST] Reduced to {len(input_candidates)} high-confidence candidates")
            
            # Score and rank candidates in parallel for faster processing
            import asyncio
            
            async def score_candidate_safe(candidate):
                try:
                    score = await self._score_variable_candidate(candidate, cell_analyses)
                    return (candidate, score)
                except Exception as e:
                    logger.error(f"❌ [VAR_SUGGEST] Failed to score candidate {candidate.cell_address}: {e}")
                    return None
            
            # Process candidates in parallel batches
            logger.info(f"⚡ [VAR_SUGGEST] Processing {len(input_candidates)} candidates in parallel")
            scoring_tasks = [score_candidate_safe(candidate) for candidate in input_candidates]
            
            # Run with a reasonable timeout for the entire scoring phase
            try:
                scored_results = await asyncio.wait_for(
                    asyncio.gather(*scoring_tasks, return_exceptions=True),
                    timeout=30.0  # 30 seconds total for all scoring
                )
                scored_candidates = [result for result in scored_results if result is not None]
            except asyncio.TimeoutError:
                logger.warning(f"⏰ [VAR_SUGGEST] Scoring phase timed out, using partial results")
                scored_candidates = []
            
            # Sort by score (descending)
            scored_candidates.sort(key=lambda x: x[1], reverse=True)
            
            # Generate configurations for top candidates in parallel
            top_candidates = scored_candidates[:max_suggestions]
            logger.info(f"⚡ [VAR_SUGGEST] Generating {len(top_candidates)} configurations in parallel")
            
            async def create_config_safe(candidate_score):
                candidate, score = candidate_score
                try:
                    return await self._create_variable_configuration(
                        candidate, cell_analyses, workbook_data, score
                    )
                except Exception as e:
                    logger.error(f"❌ [VAR_CONFIG] Failed to create configuration for {candidate.cell_address}: {e}")
                    return None
            
            # Run configuration generation in parallel with timeout
            config_tasks = [create_config_safe(cs) for cs in top_candidates]
            try:
                config_results = await asyncio.wait_for(
                    asyncio.gather(*config_tasks, return_exceptions=True),
                    timeout=45.0  # 45 seconds for configuration generation
                )
                suggestions = [config for config in config_results if config is not None]
            except asyncio.TimeoutError:
                logger.warning(f"⏰ [VAR_SUGGEST] Configuration generation timed out")
                suggestions = []
            
            logger.info(f"✅ [VAR_SUGGEST] Generated {len(suggestions)} variable suggestions")
            return suggestions
            
        except Exception as e:
            logger.error(f"❌ [VAR_SUGGEST] Failed to generate suggestions: {e}")
            return []
    
    async def _score_variable_candidate(self, 
                                      candidate: CellAnalysis,
                                      all_analyses: List[CellAnalysis]) -> float:
        """Score a candidate variable based on multiple factors"""
        
        score = 0.0
        
        # Base score from AI confidence
        score += candidate.confidence_score * 30
        
        # Impact score based on number of dependents
        logger.debug(f"🔍 [VAR_SCORE_DEBUG] Calculating impact score for {candidate.cell_address}")
        impact_score = min(len(candidate.dependents) * 5, 25)
        score += impact_score
        
        # Business context score
        logger.debug(f"🔍 [VAR_SCORE_DEBUG] Calculating business context score for {candidate.cell_address}")
        context_score = self._score_business_context(candidate)
        score += context_score
        
        # Value type score (numeric values preferred)
        logger.debug(f"🔍 [VAR_SCORE_DEBUG] Checking value type for {candidate.cell_address}: {type(candidate.current_value)}")
        if isinstance(candidate.current_value, (int, float)):
            score += 15
        
        # Formula complexity score (simpler inputs preferred)
        logger.debug(f"🔍 [VAR_SCORE_DEBUG] Calculating formula complexity for {candidate.cell_address}")
        if candidate.formula:
            complexity = len(self._extract_functions(candidate.formula))
            score += max(10 - complexity * 2, 0)
        else:
            score += 10  # Value-only cells are good inputs
        
        # Risk impact score
        logger.debug(f"🔍 [VAR_SCORE_DEBUG] Assessing risk impact for {candidate.cell_address}")
        risk_category = self._assess_risk_impact(candidate)
        risk_score = {"high": 15, "medium": 10, "low": 5}.get(risk_category, 5)
        score += risk_score
        
        logger.debug(f"🔍 [VAR_SCORE_DEBUG] Final score for {candidate.cell_address}: {min(score, 100)}")
        return min(score, 100)  # Cap at 100
    
    def _score_business_context(self, candidate: CellAnalysis) -> float:
        """Score based on business context keywords"""
        
        context_text = f"{candidate.cell_address} {candidate.description} {candidate.business_context or ''}"
        context_lower = context_text.lower()
        
        score = 0.0
        
        # Check for high-value business terms
        high_value_terms = ['revenue', 'sales', 'price', 'margin', 'growth', 'rate']
        for term in high_value_terms:
            if term in context_lower:
                score += 5
        
        # Check for assumption-like terms
        assumption_terms = ['assumption', 'estimate', 'projection', 'forecast']
        for term in assumption_terms:
            if term in context_lower:
                score += 3
        
        return min(score, 20)
    
    def _assess_risk_impact(self, candidate: CellAnalysis) -> str:
        """Assess the risk impact category of a variable"""
        
        context_lower = f"{candidate.cell_address} {candidate.description}".lower()
        
        # High impact indicators
        if any(keyword in context_lower for keyword in self.high_impact_keywords):
            return "high"
        
        # Medium impact indicators  
        elif any(keyword in context_lower for keyword in self.medium_impact_keywords):
            return "medium"
        
        # Default to low impact
        else:
            return "low"
    
    async def _create_variable_configuration(self,
                                           candidate: CellAnalysis,
                                           all_analyses: List[CellAnalysis],
                                           workbook_data: Dict[str, Any],
                                           score: float) -> Optional[VariableConfiguration]:
        """Create a complete variable configuration"""
        
        try:
            # Generate variable name
            var_name = self._generate_variable_name(candidate)
            
            # Suggest distribution and parameters
            try:
                distribution = await asyncio.wait_for(
                    self._suggest_distribution(candidate, workbook_data),
                    timeout=30.0
                )
            except asyncio.TimeoutError:
                logger.warning(f"⏰ [VAR_CONFIG] DeepSeek distribution suggestion timed out for {candidate.cell_address}")
                # Fallback to simple distribution
                distribution = DistributionParameters(
                    distribution_type=DistributionType.NORMAL,
                    mean=float(candidate.current_value) if isinstance(candidate.current_value, (int, float)) else 100,
                    std_dev=float(candidate.current_value) * 0.1 if isinstance(candidate.current_value, (int, float)) else 10,
                    confidence_level=0.5,
                    reasoning="Timeout fallback - simple normal distribution",
                    risk_impact="medium",
                    business_rationale="DeepSeek timeout - using fallback distribution"
                )
            
            # Generate business justification
            try:
                justification = await asyncio.wait_for(
                    self._generate_business_justification(candidate, distribution),
                    timeout=8.0  # Reduced timeout for faster processing
                )
            except asyncio.TimeoutError:
                logger.warning(f"⏰ [VAR_CONFIG] DeepSeek business justification timed out for {candidate.cell_address}")
                justification = f"Business justification for {candidate.cell_address} - DeepSeek timeout occurred"
            
            # Assess risk category
            risk_category = self._assess_risk_impact(candidate)
            
            # Find correlation candidates
            correlation_candidates = self._find_correlation_candidates(candidate, all_analyses)
            
            return VariableConfiguration(
                cell_address=candidate.cell_address,
                sheet_name=candidate.sheet_name,
                variable_name=var_name,
                distribution=distribution,
                current_value=candidate.current_value,
                business_justification=justification,
                risk_category=risk_category,
                correlation_candidates=correlation_candidates
            )
            
        except Exception as e:
            logger.error(f"❌ [VAR_CONFIG] Failed to create configuration for {candidate.cell_address}: {e}")
            return None
    
    def _generate_variable_name(self, candidate: CellAnalysis) -> str:
        """Generate a human-readable variable name"""
        
        # Start with cell address
        base_name = candidate.cell_address
        
        # Try to extract meaningful name from context
        if candidate.business_context:
            # Look for key business terms
            context_words = re.findall(r'\b[a-zA-Z]+\b', candidate.business_context.lower())
            business_terms = ['revenue', 'cost', 'price', 'growth', 'margin', 'volume', 'rate']
            
            for term in business_terms:
                if term in context_words:
                    return f"{term.title()}_{base_name}"
        
        # Extract from formula if available
        if candidate.formula:
            # Look for meaningful patterns in formula
            if 'growth' in candidate.formula.lower():
                return f"Growth_Rate_{base_name}"
            elif 'cost' in candidate.formula.lower():
                return f"Cost_Factor_{base_name}"
            elif 'price' in candidate.formula.lower():
                return f"Price_{base_name}"
        
        # Default naming
        return f"Variable_{base_name}"
    
    async def _suggest_distribution(self, 
                                  candidate: CellAnalysis,
                                  workbook_data: Dict[str, Any]) -> DistributionParameters:
        """Suggest appropriate probability distribution"""
        
        current_value = candidate.current_value
        context_text = f"{candidate.description} {candidate.business_context or ''}".lower()
        
        # Default to triangular with 20% variation
        suggested_type = DistributionType.TRIANGULAR
        confidence = 0.6
        reasoning = "Default triangular distribution with symmetric variation"
        
        # Pattern-based distribution suggestion
        for pattern_name, pattern_info in self.distribution_patterns.items():
            if any(keyword in context_text for keyword in pattern_info['keywords']):
                suggested_type = pattern_info['suggested_distribution']
                reasoning = pattern_info['reasoning']
                confidence = 0.8
                break
        
        # Generate parameters based on current value and distribution type
        params = self._generate_distribution_parameters(
            suggested_type, current_value, context_text
        )
        
        # Use AI to refine if available (with timeout for large models)
        if self.use_deepseek and self.deepseek_client:
            try:
                logger.info(f"🤖 [DEEPSEEK] Requesting distribution for {candidate.cell_address}")
                import asyncio
                # Add timeout to prevent hanging on large models
                ai_suggestion = await asyncio.wait_for(
                    self._get_ai_distribution_suggestion(candidate, current_value, context_text),
                    timeout=8.0  # 8 second timeout per API call (reduced for faster processing)
                )
                if ai_suggestion:
                    logger.info(f"✅ [DEEPSEEK] Got AI suggestion for {candidate.cell_address}: {ai_suggestion}")
                    params = ai_suggestion
                    confidence = 0.9
                else:
                    logger.warning(f"🤷 [DEEPSEEK] No AI suggestion returned for {candidate.cell_address}")
            except asyncio.TimeoutError:
                logger.warning(f"⏱️ [VAR_SUGGEST] AI suggestion timeout for {candidate.cell_address}")
            except Exception as e:
                logger.error(f"❌ [DEEPSEEK] AI distribution suggestion failed: {e}")
        
        return DistributionParameters(
            distribution_type=suggested_type,
            confidence_level=confidence,
            reasoning=reasoning,
            risk_impact="medium",  # Default, can be overridden later
            business_rationale="Pattern-based distribution suggestion",
            **params
        )
    
    def _generate_distribution_parameters(self,
                                        dist_type: DistributionType,
                                        current_value: Any,
                                        context: str) -> Dict[str, float]:
        """Generate distribution parameters based on type and context"""
        
        if not isinstance(current_value, (int, float)) or current_value == 0:
            # Use default parameters for non-numeric or zero values
            if dist_type == DistributionType.TRIANGULAR:
                return {
                    'min_value': 0.8,
                    'most_likely': 1.0,
                    'max_value': 1.2
                }
            elif dist_type == DistributionType.NORMAL:
                return {
                    'mean': 1.0,
                    'std_dev': 0.1
                }
        
        value = float(current_value)
        
        if dist_type == DistributionType.TRIANGULAR:
            # Determine variation based on context
            if 'volatile' in context or 'uncertain' in context:
                variation = 0.3  # ±30%
            elif 'stable' in context or 'reliable' in context:
                variation = 0.1  # ±10%
            else:
                variation = 0.2  # ±20% default
            
            return {
                'min_value': value * (1 - variation),
                'most_likely': value,
                'max_value': value * (1 + variation)
            }
        
        elif dist_type == DistributionType.NORMAL:
            # Standard deviation as % of mean
            if 'volatile' in context:
                std_pct = 0.15  # 15% std dev
            else:
                std_pct = 0.10  # 10% std dev
                
            return {
                'mean': value,
                'std_dev': value * std_pct
            }
        
        elif dist_type == DistributionType.UNIFORM:
            variation = 0.25  # ±25% for uniform
            return {
                'min_value': value * (1 - variation),
                'max_value': value * (1 + variation)
            }
        
        elif dist_type == DistributionType.BETA:
            # For percentages/ratios
            if 0 <= value <= 1:
                return {
                    'alpha': 2.0,
                    'beta': 2.0,
                    'min_value': 0.0,
                    'max_value': 1.0
                }
            else:
                # Convert to percentage
                return {
                    'alpha': 2.0,
                    'beta': 2.0,
                    'min_value': 0.0,
                    'max_value': 100.0
                }
        
        # Default fallback
        return {
            'min_value': value * 0.8,
            'most_likely': value,
            'max_value': value * 1.2
        }
    
    async def _get_ai_distribution_suggestion(self,
                                            candidate: CellAnalysis,
                                            current_value: Any,
                                            context: str) -> Optional[Dict[str, float]]:
        """Get AI-powered distribution parameter suggestions"""
        
        try:
            # Add timeout to prevent hanging
            import asyncio
            suggestion = await asyncio.wait_for(
                self.deepseek_client.suggest_distribution_parameters(
                    candidate.cell_address,
                    current_value,
                    context,
                    candidate.description
                ),
                timeout=6.0  # 6 second timeout (reduced for faster processing)
            )
            if suggestion:
                return suggestion.get('parameters', {})
                
        except asyncio.TimeoutError:
            logger.warning(f"⏰ [DEEPSEEK] Distribution parameter suggestion timed out for {candidate.cell_address}")
        except Exception as e:
            logger.debug(f"AI parameter suggestion failed: {e}")
        
        return None
    
    # Removed _call_openai method - now using DeepSeek client directly
    
    async def _generate_business_justification(self,
                                             candidate: CellAnalysis,
                                             distribution: DistributionParameters) -> str:
        """Generate business justification for the variable selection"""
        
        if self.use_deepseek and self.deepseek_client:
            try:
                # Add timeout to prevent hanging
                import asyncio
                response = await asyncio.wait_for(
                    self.deepseek_client.generate_business_justification(
                        candidate.cell_address,
                        candidate.current_value,
                        candidate.business_context or "",
                        distribution.distribution_type.value
                    ),
                    timeout=30.0  # 30 second timeout
                )
                if response:
                    return response.strip()
                    
            except asyncio.TimeoutError:
                logger.warning(f"⏰ [DEEPSEEK] Business justification timed out for {candidate.cell_address}")
            except Exception as e:
                logger.debug(f"AI justification generation failed: {e}")
        
        # Fallback to rule-based justification
        return self._generate_rule_based_justification(candidate, distribution)
    
    def _generate_rule_based_justification(self,
                                         candidate: CellAnalysis,
                                         distribution: DistributionParameters) -> str:
        """Generate justification using rule-based logic"""
        
        context_lower = f"{candidate.description} {candidate.business_context or ''}".lower()
        
        if 'revenue' in context_lower or 'sales' in context_lower:
            return "Revenue/sales variables are key drivers of model outcomes and typically exhibit uncertainty."
        elif 'cost' in context_lower:
            return "Cost variables directly impact profitability and often vary due to market conditions."
        elif 'growth' in context_lower or 'rate' in context_lower:
            return "Growth rates and percentages are inherently uncertain and drive model sensitivity."
        elif 'price' in context_lower:
            return "Pricing variables are subject to market forces and competitive dynamics."
        elif len(candidate.dependents) > 3:
            return f"This variable affects {len(candidate.dependents)} other calculations, making it high-impact."
        else:
            return "Variable shows characteristics of an input parameter suitable for Monte Carlo analysis."
    
    def _find_correlation_candidates(self,
                                   candidate: CellAnalysis,
                                   all_analyses: List[CellAnalysis]) -> List[str]:
        """Find other variables that might correlate with this one"""
        
        candidates = []
        context_lower = f"{candidate.description} {candidate.business_context or ''}".lower()
        
        # Look for variables with similar business context
        for analysis in all_analyses:
            if (analysis.cell_address != candidate.cell_address and
                analysis.cell_type in [CellType.INPUT_VARIABLE, CellType.ASSUMPTION]):
                
                other_context = f"{analysis.description} {analysis.business_context or ''}".lower()
                
                # Check for related business concepts
                correlation_patterns = [
                    ('revenue', 'volume'),
                    ('price', 'cost'),
                    ('growth', 'rate'),
                    ('sales', 'marketing'),
                    ('cost', 'expense')
                ]
                
                for pattern1, pattern2 in correlation_patterns:
                    if ((pattern1 in context_lower and pattern2 in other_context) or
                        (pattern2 in context_lower and pattern1 in other_context)):
                        candidates.append(analysis.cell_address)
                        break
        
        return candidates[:5]  # Limit to top 5 candidates
    
    def _extract_functions(self, formula: str) -> List[str]:
        """Extract function names from formula"""
        return re.findall(r'([A-Z]+)\(', formula.upper())

    async def suggest_target_variables(self, 
                                     cell_analyses: List[CellAnalysis],
                                     max_targets: int = 5) -> List[CellAnalysis]:
        """
        Suggest target output variables for Monte Carlo analysis
        
        Args:
            cell_analyses: Results from ExcelIntelligenceAgent
            max_targets: Maximum number of targets to suggest
            
        Returns:
            List of suggested target variables
        """
        logger.info(f"🎯 [TARGET_SUGGEST] Identifying target output variables")
        
        # Filter to potential output targets
        output_candidates = [
            analysis for analysis in cell_analyses
            if analysis.cell_type == CellType.OUTPUT_TARGET
            and analysis.confidence_score > 0.5
        ]
        
        # Score targets based on business importance
        scored_targets = []
        for candidate in output_candidates:
            score = self._score_target_candidate(candidate)
            scored_targets.append((candidate, score))
        
        # Sort by score and return top candidates
        scored_targets.sort(key=lambda x: x[1], reverse=True)
        
        suggested_targets = [candidate for candidate, score in scored_targets[:max_targets]]
        
        logger.info(f"✅ [TARGET_SUGGEST] Suggested {len(suggested_targets)} target variables")
        return suggested_targets
    
    def _score_target_candidate(self, candidate: CellAnalysis) -> float:
        """Score a target variable candidate"""
        
        score = 0.0
        
        # Base confidence score
        score += candidate.confidence_score * 40
        
        # Business importance (based on keywords)
        context_lower = f"{candidate.description} {candidate.business_context or ''}".lower()
        important_terms = ['npv', 'irr', 'profit', 'revenue', 'total', 'sum', 'result']
        for term in important_terms:
            if term in context_lower:
                score += 15
                break
        
        # Formula complexity (more complex = likely important output)
        if candidate.formula:
            complexity = len(self._extract_functions(candidate.formula))
            score += min(complexity * 3, 20)
        
        # Number of dependencies (more inputs = likely important calculation)
        score += min(len(candidate.dependencies) * 2, 20)
        
        return score
    
    async def suggest_monte_carlo_variables_from_deepseek(self, 
                                                         deepseek_analysis: Dict[str, Any],
                                                         workbook_data: Dict[str, Any]) -> List[VariableConfiguration]:
        """
        Create VariableConfiguration objects directly from DeepSeek comprehensive analysis
        No need for rule-based processing - DeepSeek has already done the analysis
        """
        logger.info(f"🎯 [VAR_SUGGEST_DEEPSEEK] Processing DeepSeek analysis with {len(deepseek_analysis.get('input_variables', []))} suggested variables")
        
        try:
            variable_configs = []
            
            for input_var in deepseek_analysis.get('input_variables', []):
                try:
                    # Extract distribution info from DeepSeek analysis
                    distribution_info = input_var.get('distribution', {})
                    dist_type = distribution_info.get('type', 'triangular')
                    parameters = distribution_info.get('parameters', {})
                    
                    # Map DeepSeek distribution types to our enum
                    distribution_mapping = {
                        'triangular': DistributionType.TRIANGULAR,
                        'normal': DistributionType.NORMAL,
                        'uniform': DistributionType.UNIFORM,
                        'beta': DistributionType.BETA,
                        'lognormal': DistributionType.NORMAL  # Map lognormal to normal for now
                    }
                    
                    mapped_dist_type = distribution_mapping.get(dist_type, DistributionType.TRIANGULAR)
                    
                    # Create distribution parameters from DeepSeek suggestions
                    distribution_params = DistributionParameters(
                        distribution_type=mapped_dist_type,
                        min_value=parameters.get('min_value'),
                        max_value=parameters.get('max_value'),
                        most_likely=parameters.get('most_likely'),
                        mean=parameters.get('mean'),
                        std_dev=parameters.get('std_dev'),
                        confidence_level=0.95,  # High confidence from DeepSeek
                        reasoning=distribution_info.get('reasoning', 'AI-suggested distribution'),
                        risk_impact='medium',
                        business_rationale=f"Data type: {input_var.get('data_type', 'number')}"
                    )
                    
                    # Create variable configuration
                    var_config = VariableConfiguration(
                        cell_address=input_var['cell_address'],
                        sheet_name=input_var.get('sheet_name', 'Sheet1'),
                        variable_name=input_var.get('variable_name', f"Variable_{input_var['cell_address']}"),
                        distribution=distribution_params,
                        current_value=input_var.get('current_value'),
                        business_justification=f"Input variable affecting model calculations",
                        risk_category='medium',
                        correlation_candidates=input_var.get('referenced_by', [])
                    )
                    
                    variable_configs.append(var_config)
                    logger.info(f"✅ [VAR_SUGGEST_DEEPSEEK] Created variable config for {input_var['cell_address']}: {input_var.get('variable_name', 'Unnamed')}")
                    
                except Exception as var_error:
                    logger.warning(f"⚠️ [VAR_SUGGEST_DEEPSEEK] Failed to process variable {input_var.get('cell_address', 'unknown')}: {var_error}")
                    continue
            
            logger.info(f"✅ [VAR_SUGGEST_DEEPSEEK] Created {len(variable_configs)} variable configurations from DeepSeek analysis")
            return variable_configs
            
        except Exception as e:
            logger.error(f"❌ [VAR_SUGGEST_DEEPSEEK] Failed to process DeepSeek analysis: {e}")
            return []

    def extract_output_targets_from_deepseek(self, deepseek_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract output target variables from DeepSeek comprehensive analysis
        """
        logger.info(f"🎯 [OUTPUT_TARGETS] Extracting output targets from DeepSeek analysis")
        
        try:
            output_targets = []
            
            for output_target in deepseek_analysis.get('output_targets', []):
                try:
                    target_info = {
                        'cell_address': output_target.get('cell_address'),
                        'variable_name': output_target.get('variable_name'),
                        'target_name': output_target.get('variable_name'),  # Alias for compatibility
                        'current_value': output_target.get('current_value'),
                        'formula': output_target.get('formula'),
                        'sheet_name': output_target.get('sheet_name', 'Sheet1'),
                        'data_type': output_target.get('data_type', 'number'),
                        'depends_on': output_target.get('depends_on', [])
                    }
                    output_targets.append(target_info)
                    logger.debug(f"✅ [OUTPUT_TARGETS] Added target: {target_info['cell_address']} - {target_info['variable_name']}")
                    
                except Exception as e:
                    logger.error(f"❌ [OUTPUT_TARGETS] Error processing target {output_target.get('cell_address', 'unknown')}: {e}")
                    continue
            
            logger.info(f"🎯 [OUTPUT_TARGETS] Extracted {len(output_targets)} output targets")
            return output_targets
            
        except Exception as e:
            logger.error(f"❌ [OUTPUT_TARGETS] Error extracting output targets: {e}")
            return []
