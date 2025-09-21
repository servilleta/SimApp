"""
Results Insight Generator - AI-powered analysis of Monte Carlo simulation results
Generates business-friendly summaries and actionable insights
"""

import logging
import json
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import os
from .deepseek_client import get_deepseek_client
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

class InsightType(Enum):
    RISK_ASSESSMENT = "risk_assessment"
    OPPORTUNITY = "opportunity"
    SENSITIVITY = "sensitivity"
    OUTLIER = "outlier"
    CORRELATION = "correlation"
    TREND = "trend"
    RECOMMENDATION = "recommendation"

class RiskLevel(Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

@dataclass
class StatisticalInsight:
    """Statistical insight derived from simulation results"""
    insight_type: InsightType
    title: str
    description: str
    risk_level: RiskLevel
    confidence: float  # 0.0 to 1.0
    supporting_data: Dict[str, Any]
    business_impact: str
    recommended_action: Optional[str] = None

@dataclass
class ResultsSummary:
    """Comprehensive summary of simulation results"""
    simulation_id: str
    target_variable: str
    total_iterations: int
    success_probability: float  # % of iterations meeting success criteria
    
    # Key statistics
    mean_value: float
    median_value: float
    std_deviation: float
    min_value: float
    max_value: float
    
    # Risk metrics
    percentile_5: float
    percentile_95: float
    value_at_risk_5: float  # 5% VaR
    expected_shortfall_5: float  # 5% ES
    
    # Distribution characteristics
    skewness: float
    kurtosis: float
    distribution_shape: str  # "normal", "skewed_left", "skewed_right", "fat_tailed"
    
    # Business insights
    key_insights: List[StatisticalInsight]
    executive_summary: str
    risk_assessment: str
    opportunities: str
    recommendations: List[str]

class ResultsInsightGenerator:
    """
    AI-powered generator that analyzes Monte Carlo simulation results
    and provides business-friendly insights and recommendations
    """
    
    def __init__(self, use_deepseek: bool = True):
        self.use_deepseek = use_deepseek
        self.deepseek_client = get_deepseek_client()
        
        if use_deepseek and not self.deepseek_client:
            logger.warning("⚠️ DeepSeek client not initialized, using rule-based analysis only")
            self.use_deepseek = False
        elif use_deepseek:
            logger.info("✅ Using DeepSeek client for results analysis")
        
        # Thresholds for different insight types
        self.risk_thresholds = {
            'high_volatility': 0.3,  # CV > 30%
            'negative_skew': -0.5,   # Skewness < -0.5
            'fat_tails': 3.0,        # Kurtosis > 3.0
            'low_success': 0.7       # Success rate < 70%
        }
        
        # Business context templates
        self.insight_templates = {
            InsightType.RISK_ASSESSMENT: "Analysis shows {risk_level} risk with {metric}",
            InsightType.OPPORTUNITY: "Potential upside opportunity identified: {description}",
            InsightType.SENSITIVITY: "Results are {sensitivity} sensitive to {variables}",
            InsightType.OUTLIER: "Unusual pattern detected: {pattern}",
            InsightType.RECOMMENDATION: "Recommended action: {action}"
        }
    
    async def analyze_simulation_results(self,
                                       simulation_id: str,
                                       results_data: np.ndarray,
                                       target_variable: str,
                                       variable_configs: List[Dict[str, Any]] = None,
                                       success_criteria: Optional[Dict[str, Any]] = None) -> ResultsSummary:
        """
        Comprehensive analysis of Monte Carlo simulation results
        
        Args:
            simulation_id: Unique identifier for the simulation
            results_data: Array of simulation results
            target_variable: Name/description of the target variable
            variable_configs: Configuration of input variables used
            success_criteria: Optional criteria for success (e.g., {'min_value': 1000000})
            
        Returns:
            Comprehensive results summary with insights
        """
        logger.info(f"🔍 [RESULTS_AI] Analyzing simulation results for {simulation_id}")
        
        try:
            # Calculate basic statistics
            stats_summary = self._calculate_statistics(results_data)
            
            # Calculate risk metrics
            risk_metrics = self._calculate_risk_metrics(results_data)
            
            # Assess distribution characteristics
            distribution_analysis = self._analyze_distribution(results_data)
            
            # Calculate success probability
            success_prob = self._calculate_success_probability(results_data, success_criteria)
            
            # Generate statistical insights
            insights = await self._generate_insights(
                results_data, stats_summary, risk_metrics, 
                distribution_analysis, variable_configs
            )
            
            # Generate business-friendly summaries
            executive_summary = await self._generate_executive_summary(
                stats_summary, risk_metrics, success_prob, insights
            )
            
            risk_assessment = await self._generate_risk_assessment(
                risk_metrics, distribution_analysis, insights
            )
            
            opportunities = await self._generate_opportunities(
                stats_summary, success_prob, insights
            )
            
            recommendations = await self._generate_recommendations(
                insights, risk_metrics, success_prob
            )
            
            # Combine into comprehensive summary
            summary = ResultsSummary(
                simulation_id=simulation_id,
                target_variable=target_variable,
                total_iterations=len(results_data),
                success_probability=success_prob,
                
                # Statistics
                mean_value=stats_summary['mean'],
                median_value=stats_summary['median'],
                std_deviation=stats_summary['std'],
                min_value=stats_summary['min'],
                max_value=stats_summary['max'],
                
                # Risk metrics
                percentile_5=risk_metrics['p5'],
                percentile_95=risk_metrics['p95'],
                value_at_risk_5=risk_metrics['var_5'],
                expected_shortfall_5=risk_metrics['es_5'],
                
                # Distribution
                skewness=distribution_analysis['skewness'],
                kurtosis=distribution_analysis['kurtosis'],
                distribution_shape=distribution_analysis['shape'],
                
                # Insights
                key_insights=insights,
                executive_summary=executive_summary,
                risk_assessment=risk_assessment,
                opportunities=opportunities,
                recommendations=recommendations
            )
            
            logger.info(f"✅ [RESULTS_AI] Analysis completed with {len(insights)} insights generated")
            return summary
            
        except Exception as e:
            logger.error(f"❌ [RESULTS_AI] Failed to analyze results: {e}")
            # Return minimal summary on failure
            return self._create_fallback_summary(simulation_id, results_data, target_variable)
    
    def _calculate_statistics(self, results: np.ndarray) -> Dict[str, float]:
        """Calculate basic statistical measures"""
        
        return {
            'mean': float(np.mean(results)),
            'median': float(np.median(results)),
            'std': float(np.std(results)),
            'min': float(np.min(results)),
            'max': float(np.max(results)),
            'cv': float(np.std(results) / np.mean(results)) if np.mean(results) != 0 else 0.0,
            'range': float(np.max(results) - np.min(results))
        }
    
    def _calculate_risk_metrics(self, results: np.ndarray) -> Dict[str, float]:
        """Calculate risk-related metrics"""
        
        # Percentiles
        percentiles = np.percentile(results, [5, 10, 25, 50, 75, 90, 95])
        
        # Value at Risk (5th percentile)
        var_5 = percentiles[0]
        
        # Expected Shortfall (mean of worst 5%)
        worst_5_percent = results[results <= var_5]
        es_5 = float(np.mean(worst_5_percent)) if len(worst_5_percent) > 0 else var_5
        
        # Upside/downside deviation
        mean_val = np.mean(results)
        downside_returns = results[results < mean_val] - mean_val
        upside_returns = results[results > mean_val] - mean_val
        
        downside_dev = float(np.sqrt(np.mean(downside_returns**2))) if len(downside_returns) > 0 else 0.0
        upside_dev = float(np.sqrt(np.mean(upside_returns**2))) if len(upside_returns) > 0 else 0.0
        
        return {
            'p5': float(percentiles[0]),
            'p10': float(percentiles[1]),
            'p25': float(percentiles[2]),
            'p50': float(percentiles[3]),
            'p75': float(percentiles[4]),
            'p90': float(percentiles[5]),
            'p95': float(percentiles[6]),
            'var_5': var_5,
            'es_5': es_5,
            'downside_deviation': downside_dev,
            'upside_deviation': upside_dev,
            'upside_downside_ratio': upside_dev / downside_dev if downside_dev > 0 else float('inf')
        }
    
    def _analyze_distribution(self, results: np.ndarray) -> Dict[str, Any]:
        """Analyze distribution characteristics"""
        
        # Calculate skewness and kurtosis
        skewness = float(stats.skew(results))
        kurtosis = float(stats.kurtosis(results))
        
        # Determine distribution shape
        if abs(skewness) < 0.5 and abs(kurtosis) < 0.5:
            shape = "normal"
        elif skewness < -0.5:
            shape = "skewed_left"
        elif skewness > 0.5:
            shape = "skewed_right"
        elif kurtosis > 1.0:
            shape = "fat_tailed"
        else:
            shape = "irregular"
        
        # Test for normality
        try:
            # Shapiro-Wilk test (for smaller samples)
            if len(results) <= 5000:
                _, p_value = stats.shapiro(results[:5000])
                is_normal = p_value > 0.05
            else:
                # D'Agostino's test (for larger samples)
                _, p_value = stats.normaltest(results)
                is_normal = p_value > 0.05
        except:
            is_normal = False
        
        return {
            'skewness': skewness,
            'kurtosis': kurtosis,
            'shape': shape,
            'is_normal': is_normal,
            'outliers_count': self._count_outliers(results)
        }
    
    def _count_outliers(self, results: np.ndarray) -> int:
        """Count statistical outliers using IQR method"""
        
        Q1 = np.percentile(results, 25)
        Q3 = np.percentile(results, 75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = results[(results < lower_bound) | (results > upper_bound)]
        return len(outliers)
    
    def _calculate_success_probability(self, 
                                     results: np.ndarray,
                                     success_criteria: Optional[Dict[str, Any]]) -> float:
        """Calculate probability of meeting success criteria"""
        
        if not success_criteria:
            # Default: probability of positive result
            return float(np.mean(results > 0))
        
        success_count = 0
        total_count = len(results)
        
        for criterion, value in success_criteria.items():
            if criterion == 'min_value':
                success_count = np.sum(results >= value)
            elif criterion == 'max_value':
                success_count = np.sum(results <= value)
            elif criterion == 'target_value':
                # Within 10% of target
                tolerance = value * 0.1
                success_count = np.sum(np.abs(results - value) <= tolerance)
        
        return float(success_count / total_count)
    
    async def _generate_insights(self,
                               results: np.ndarray,
                               stats: Dict[str, float],
                               risk_metrics: Dict[str, float],
                               distribution: Dict[str, Any],
                               variable_configs: List[Dict[str, Any]] = None) -> List[StatisticalInsight]:
        """Generate statistical insights from the analysis"""
        
        insights = []
        
        # Risk assessment insights
        insights.extend(self._analyze_risk_patterns(stats, risk_metrics, distribution))
        
        # Distribution insights
        insights.extend(self._analyze_distribution_patterns(distribution, stats))
        
        # Volatility insights
        insights.extend(self._analyze_volatility_patterns(stats, risk_metrics))
        
        # Outlier insights
        insights.extend(self._analyze_outlier_patterns(results, distribution))
        
        # Generate AI-enhanced insights if available
        if self.use_deepseek and self.deepseek_client:
            ai_insights = await self._generate_ai_insights(stats, risk_metrics, distribution)
            insights.extend(ai_insights)
        
        # Sort by confidence and limit to top insights
        insights.sort(key=lambda x: x.confidence, reverse=True)
        return insights[:10]  # Return top 10 insights
    
    def _analyze_risk_patterns(self,
                             stats: Dict[str, float],
                             risk_metrics: Dict[str, float],
                             distribution: Dict[str, Any]) -> List[StatisticalInsight]:
        """Analyze risk-related patterns"""
        
        insights = []
        
        # High volatility check
        if stats['cv'] > self.risk_thresholds['high_volatility']:
            insights.append(StatisticalInsight(
                insight_type=InsightType.RISK_ASSESSMENT,
                title="High Volatility Detected",
                description=f"Results show high volatility with coefficient of variation of {stats['cv']:.1%}",
                risk_level=RiskLevel.HIGH,
                confidence=0.9,
                supporting_data={'cv': stats['cv'], 'std': stats['std']},
                business_impact="High uncertainty in outcomes requires careful risk management",
                recommended_action="Consider risk mitigation strategies or additional scenario analysis"
            ))
        
        # Negative skew check (tail risk)
        if distribution['skewness'] < self.risk_thresholds['negative_skew']:
            insights.append(StatisticalInsight(
                insight_type=InsightType.RISK_ASSESSMENT,
                title="Downside Tail Risk",
                description=f"Distribution shows negative skew ({distribution['skewness']:.2f}), indicating potential for extreme negative outcomes",
                risk_level=RiskLevel.HIGH,
                confidence=0.8,
                supporting_data={'skewness': distribution['skewness'], 'p5': risk_metrics['p5']},
                business_impact="Asymmetric risk profile with potential for significant losses",
                recommended_action="Focus on downside protection and contingency planning"
            ))
        
        # Fat tails check
        if distribution['kurtosis'] > self.risk_thresholds['fat_tails']:
            insights.append(StatisticalInsight(
                insight_type=InsightType.RISK_ASSESSMENT,
                title="Fat Tail Distribution",
                description=f"Results show fat tails (kurtosis: {distribution['kurtosis']:.2f}), indicating higher probability of extreme events",
                risk_level=RiskLevel.MEDIUM,
                confidence=0.7,
                supporting_data={'kurtosis': distribution['kurtosis']},
                business_impact="Higher than normal probability of extreme outcomes",
                recommended_action="Stress test model with extreme scenarios"
            ))
        
        return insights
    
    def _analyze_distribution_patterns(self,
                                     distribution: Dict[str, Any],
                                     stats: Dict[str, float]) -> List[StatisticalInsight]:
        """Analyze distribution characteristics"""
        
        insights = []
        
        # Distribution shape insights
        if distribution['shape'] == 'skewed_right':
            insights.append(StatisticalInsight(
                insight_type=InsightType.OPPORTUNITY,
                title="Upside Potential Identified",
                description="Distribution shows positive skew, indicating potential for significant upside outcomes",
                risk_level=RiskLevel.LOW,
                confidence=0.8,
                supporting_data={'skewness': distribution['skewness']},
                business_impact="Asymmetric opportunity profile with potential for significant gains",
                recommended_action="Explore strategies to capture upside potential"
            ))
        
        # Normality insights
        if not distribution['is_normal']:
            insights.append(StatisticalInsight(
                insight_type=InsightType.TREND,
                title="Non-Normal Distribution",
                description=f"Results follow a {distribution['shape']} distribution rather than normal",
                risk_level=RiskLevel.MEDIUM,
                confidence=0.7,
                supporting_data={'shape': distribution['shape'], 'is_normal': distribution['is_normal']},
                business_impact="Standard normal distribution assumptions may not apply",
                recommended_action="Use appropriate risk measures for non-normal distributions"
            ))
        
        return insights
    
    def _analyze_volatility_patterns(self,
                                   stats: Dict[str, float],
                                   risk_metrics: Dict[str, float]) -> List[StatisticalInsight]:
        """Analyze volatility and variation patterns"""
        
        insights = []
        
        # Range analysis
        range_ratio = stats['range'] / stats['mean'] if stats['mean'] != 0 else 0
        if range_ratio > 2.0:
            insights.append(StatisticalInsight(
                insight_type=InsightType.SENSITIVITY,
                title="High Sensitivity to Inputs",
                description=f"Results span a wide range ({range_ratio:.1f}x the mean), indicating high sensitivity to input variables",
                risk_level=RiskLevel.MEDIUM,
                confidence=0.8,
                supporting_data={'range_ratio': range_ratio, 'range': stats['range']},
                business_impact="Small changes in inputs can lead to large changes in outcomes",
                recommended_action="Identify and focus on key driver variables for model refinement"
            ))
        
        # Upside/downside analysis
        if risk_metrics['upside_downside_ratio'] > 2.0:
            insights.append(StatisticalInsight(
                insight_type=InsightType.OPPORTUNITY,
                title="Favorable Risk-Return Profile",
                description=f"Upside volatility is {risk_metrics['upside_downside_ratio']:.1f}x higher than downside volatility",
                risk_level=RiskLevel.LOW,
                confidence=0.7,
                supporting_data={'ratio': risk_metrics['upside_downside_ratio']},
                business_impact="Asymmetric volatility favors positive outcomes",
                recommended_action="Consider strategies that benefit from upside volatility"
            ))
        
        return insights
    
    def _analyze_outlier_patterns(self,
                                results: np.ndarray,
                                distribution: Dict[str, Any]) -> List[StatisticalInsight]:
        """Analyze outlier patterns"""
        
        insights = []
        
        outlier_ratio = distribution['outliers_count'] / len(results)
        if outlier_ratio > 0.1:  # More than 10% outliers
            insights.append(StatisticalInsight(
                insight_type=InsightType.OUTLIER,
                title="Significant Outlier Presence",
                description=f"{outlier_ratio:.1%} of results are statistical outliers",
                risk_level=RiskLevel.MEDIUM,
                confidence=0.8,
                supporting_data={'outlier_count': distribution['outliers_count'], 'outlier_ratio': outlier_ratio},
                business_impact="Unusual patterns may indicate model issues or extreme scenarios",
                recommended_action="Investigate outlier causes and validate model assumptions"
            ))
        
        return insights
    
    async def _generate_ai_insights(self,
                                  stats: Dict[str, float],
                                  risk_metrics: Dict[str, float],
                                  distribution: Dict[str, Any]) -> List[StatisticalInsight]:
        """Generate additional insights using AI"""
        
        if not self.use_deepseek or not self.deepseek_client:
            return []
        
        try:
            ai_insights_data = await self.deepseek_client.generate_simulation_insights(
                stats, risk_metrics, distribution
            )
            
            ai_insights = []
            for insight_data in ai_insights_data:
                risk_level = RiskLevel(insight_data.get('risk_level', 'medium'))
                
                ai_insights.append(StatisticalInsight(
                    insight_type=InsightType.RECOMMENDATION,
                    title=insight_data['title'],
                    description=insight_data['description'],
                    risk_level=risk_level,
                    confidence=0.6,  # AI insights get lower confidence
                    supporting_data={},
                    business_impact=insight_data['business_impact']
                ))
            
            return ai_insights
                
        except Exception as e:
            logger.debug(f"AI insight generation failed: {e}")
        
        return []
    
    # Removed _call_openai method - now using DeepSeek client directly
    
    async def _generate_executive_summary(self,
                                        stats: Dict[str, float],
                                        risk_metrics: Dict[str, float],
                                        success_prob: float,
                                        insights: List[StatisticalInsight]) -> str:
        """Generate executive summary"""
        
        if self.use_deepseek and self.deepseek_client:
            try:
                key_insights_text = "; ".join([insight.title for insight in insights[:3]])
                
                response = await self.deepseek_client.generate_executive_summary(
                    stats, risk_metrics, success_prob, [key_insights_text]
                )
                if response:
                    return response.strip()
                    
            except Exception as e:
                logger.debug(f"AI executive summary generation failed: {e}")
        
        # Fallback to template-based summary
        return (f"Monte Carlo analysis shows a mean outcome of {stats['mean']:.2f} with "
                f"{success_prob:.1%} probability of success. Results range from "
                f"{risk_metrics['p5']:.2f} to {risk_metrics['p95']:.2f} with 90% confidence. "
                f"Analysis identified {len(insights)} key insights requiring attention.")
    
    async def _generate_risk_assessment(self,
                                      risk_metrics: Dict[str, float],
                                      distribution: Dict[str, Any],
                                      insights: List[StatisticalInsight]) -> str:
        """Generate risk assessment summary"""
        
        risk_insights = [i for i in insights if i.insight_type == InsightType.RISK_ASSESSMENT]
        high_risk_count = len([i for i in risk_insights if i.risk_level == RiskLevel.HIGH])
        
        if high_risk_count > 0:
            risk_level = "HIGH"
            risk_desc = f"Analysis identified {high_risk_count} high-risk factors"
        elif len(risk_insights) > 2:
            risk_level = "MEDIUM"
            risk_desc = "Multiple risk factors require monitoring"
        else:
            risk_level = "LOW"
            risk_desc = "Risk profile appears manageable"
        
        return (f"Risk Assessment: {risk_level}. {risk_desc}. "
                f"Worst-case scenario (5th percentile): {risk_metrics['p5']:.2f}. "
                f"Expected shortfall: {risk_metrics['es_5']:.2f}.")
    
    async def _generate_opportunities(self,
                                    stats: Dict[str, float],
                                    success_prob: float,
                                    insights: List[StatisticalInsight]) -> str:
        """Generate opportunities summary"""
        
        opportunity_insights = [i for i in insights if i.insight_type == InsightType.OPPORTUNITY]
        
        if opportunity_insights:
            return f"Analysis identified {len(opportunity_insights)} potential opportunities. " + \
                   "; ".join([insight.description for insight in opportunity_insights[:2]])
        else:
            return f"Success probability of {success_prob:.1%} indicates solid execution potential. " + \
                   f"Best-case scenarios (95th percentile) reach {stats['max']:.2f}."
    
    async def _generate_recommendations(self,
                                      insights: List[StatisticalInsight],
                                      risk_metrics: Dict[str, float],
                                      success_prob: float) -> List[str]:
        """Generate actionable recommendations"""
        
        recommendations = []
        
        # Collect specific recommendations from insights
        for insight in insights:
            if insight.recommended_action:
                recommendations.append(insight.recommended_action)
        
        # Add general recommendations based on analysis
        if success_prob < 0.7:
            recommendations.append("Consider adjusting input assumptions to improve success probability")
        
        if risk_metrics['var_5'] < 0:
            recommendations.append("Develop contingency plans for worst-case scenarios")
        
        # Limit to top 5 recommendations
        return recommendations[:5]
    
    def _create_fallback_summary(self,
                               simulation_id: str,
                               results: np.ndarray,
                               target_variable: str) -> ResultsSummary:
        """Create minimal summary when full analysis fails"""
        
        basic_stats = self._calculate_statistics(results)
        
        return ResultsSummary(
            simulation_id=simulation_id,
            target_variable=target_variable,
            total_iterations=len(results),
            success_probability=float(np.mean(results > 0)),
            
            mean_value=basic_stats['mean'],
            median_value=basic_stats['median'],
            std_deviation=basic_stats['std'],
            min_value=basic_stats['min'],
            max_value=basic_stats['max'],
            
            percentile_5=float(np.percentile(results, 5)),
            percentile_95=float(np.percentile(results, 95)),
            value_at_risk_5=float(np.percentile(results, 5)),
            expected_shortfall_5=float(np.mean(results[results <= np.percentile(results, 5)])),
            
            skewness=float(stats.skew(results)),
            kurtosis=float(stats.kurtosis(results)),
            distribution_shape="unknown",
            
            key_insights=[],
            executive_summary="Basic statistical analysis completed. Manual review recommended.",
            risk_assessment="Risk analysis requires manual interpretation.",
            opportunities="Opportunity assessment not available.",
            recommendations=["Review results manually", "Consider re-running analysis"]
        )
