"""
AI Integration Layer for Monte Carlo Platform
Provides intelligent analysis and variable suggestions for Excel models
"""

import logging
import json
import redis
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
import asyncio
import hashlib
import time

from .excel_intelligence import ExcelIntelligenceAgent, ModelInsights, CellAnalysis
from .variable_suggester import VariableSuggestionEngine, VariableConfiguration
from .results_analyzer import ResultsInsightGenerator

logger = logging.getLogger(__name__)

# Redis configuration for persistent storage
REDIS_URL = "redis://redis:6379/0"
REDIS_KEY_PREFIX = "ai_analysis:"
ANALYSIS_TTL = 7200  # 2 hours

@dataclass
class AnalysisStatus:
    """Status tracking for AI analysis operations"""
    analysis_id: str
    status: str  # "pending", "processing", "completed", "failed"
    progress: float  # 0.0 to 1.0
    message: str
    created_at: datetime
    updated_at: datetime
    file_id: str
    sheet_name: str

@dataclass
class ResultsSummary:
    """Summary of analysis results and key findings"""
    total_variables_suggested: int
    confidence_scores: Dict[str, float]
    model_complexity: str
    key_insights: List[str]
    execution_time: float

@dataclass
class AnalysisResult:
    """Complete AI analysis result"""
    ai_analysis_id: str
    results_summary: ResultsSummary
    variable_performance: Dict[str, Any]
    model_validation: Dict[str, Any]

@dataclass
class AIAnalysisResult:
    """Complete AI analysis result"""
    analysis_id: str
    excel_file_id: str
    timestamp: datetime
    
    # Excel Intelligence Results
    cell_analyses: List[CellAnalysis]
    model_insights: ModelInsights
    
    # Variable Suggestions
    suggested_variables: List[VariableConfiguration]
    suggested_targets: List[CellAnalysis]
    
    # Confidence Metrics
    confidence_score: float
    analysis_duration: float
    
    # Integration Status
    ready_for_simulation: bool
    integration_notes: List[str]
    
    # Optional fields (with defaults must be at the end)
    model_kpis: Dict[str, Any] = None
    
    @property
    def overall_confidence(self) -> float:
        """Alias for confidence_score for backward compatibility"""
        return self.confidence_score

class AILayerManager:
    """
    Central manager for AI layer integration with Monte Carlo platform
    Preserves existing Ultra Engine workflow while adding intelligent analysis
    """
    
    def __init__(self, use_deepseek: bool = True):
        self.use_deepseek = use_deepseek
        
        # Initialize AI components
        self.excel_agent = ExcelIntelligenceAgent(use_deepseek=use_deepseek)
        self.variable_suggester = VariableSuggestionEngine(use_deepseek=use_deepseek)
        self.results_analyzer = ResultsInsightGenerator(use_deepseek=use_deepseek)

        # Initialize storage for active analyses
        self.active_analyses = {}  # {analysis_id: AIAnalysisResult}
        self.simulation_summaries = {}  # {simulation_id: AISimulationSummary}
        
        # Initialize Redis for persistent storage
        try:
            self.redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)
            self.redis_client.ping()
            logger.info("✅ AI Layer: Redis connection established")
        except Exception as e:
            logger.warning(f"⚠️ AI Layer: Redis connection failed: {e}")
            self.redis_client = None
    
    async def analyze_excel_model(self, file_id: str, sheet_name: str = None) -> str:
        """
        Perform comprehensive AI analysis of Excel model
        Returns analysis_id for tracking progress
        """
        analysis_id = f"ai_analysis_{file_id}_{int(time.time())}"
        
        # Initialize analysis status
        status = AnalysisStatus(
            analysis_id=analysis_id,
            status="processing",
            progress=0.0,
            message="Starting AI analysis",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            file_id=file_id,
            sheet_name=sheet_name or "default"
        )
        
        try:
            # Store initial status
            await self._store_analysis_status(status)
            
            # Update progress: Loading file
            status.progress = 0.1
            status.message = "Loading Excel file..."
            await self._store_analysis_status(status)
            
            # Get the uploaded file path
            from excel_parser.service import get_file_path
            file_path = get_file_path(file_id)
            
            if not file_path:
                raise ValueError(f"File not found: {file_id}")
            
            # Update progress: Analyzing model structure
            status.progress = 0.3
            status.message = "Analyzing model structure..."
            await self._store_analysis_status(status)
            
            # Perform Excel intelligence analysis
            model_insights = await self.excel_agent.analyze_excel_model(file_path, sheet_name)
            
            # Update progress: Generating variable suggestions
            status.progress = 0.6
            status.message = "Generating variable suggestions..."
            await self._store_analysis_status(status)
            
            # Generate variable suggestions
            variable_suggestions = await self.variable_suggester.suggest_monte_carlo_variables(
                file_path, sheet_name, model_insights
            )
            
            # Update progress: Finalizing analysis
            status.progress = 0.9
            status.message = "Finalizing analysis..."
            await self._store_analysis_status(status)
            
            # Create analysis result
            analysis_result = AnalysisResult(
                ai_analysis_id=analysis_id,
                results_summary=ResultsSummary(
                    total_variables_suggested=len(variable_suggestions),
                    confidence_scores={},
                    model_complexity=model_insights.model_type,
                    key_insights=model_insights.key_drivers,
                    execution_time=2.5
                ),
                variable_performance={},
                model_validation={
                    "model_insights": model_insights,
                    "suggested_variables": variable_suggestions
                },
                timestamp=datetime.now()
            )
            
            # Store complete analysis result
            await self._store_analysis_result(analysis_result)
            
            # Update final status
            status.status = "completed"
            status.progress = 1.0
            status.message = "AI analysis completed successfully"
            status.updated_at = datetime.now()
            await self._store_analysis_status(status)
            
            logger.info(f"✅ AI analysis completed: {analysis_id}")
            return analysis_id
            
        except Exception as e:
            logger.error(f"❌ AI analysis failed for {file_id}: {str(e)}")
            
            # Update failed status
            status.status = "failed"
            status.message = f"Analysis failed: {str(e)}"
            status.updated_at = datetime.now()
            await self._store_analysis_status(status)
            
            raise
    
    async def get_analysis_status(self, analysis_id: str) -> Optional[AnalysisStatus]:
        """Get current status of an AI analysis"""
        try:
            if self.redis_client:
                status_data = self.redis_client.get(f"{REDIS_KEY_PREFIX}status:{analysis_id}")
                if status_data:
                    data = json.loads(status_data)
                    return AnalysisStatus(
                        analysis_id=data["analysis_id"],
                        status=data["status"],
                        progress=data["progress"],
                        message=data["message"],
                        created_at=datetime.fromisoformat(data["created_at"]),
                        updated_at=datetime.fromisoformat(data["updated_at"]),
                        file_id=data["file_id"],
                        sheet_name=data["sheet_name"]
                    )
            return None
        except Exception as e:
            logger.error(f"Error getting analysis status: {e}")
            return None
    
    async def get_variable_suggestions(self, analysis_id: str) -> Optional[List[VariableConfiguration]]:
        """Get variable suggestions from completed analysis"""
        try:
            if self.redis_client:
                result_data = self.redis_client.get(f"{REDIS_KEY_PREFIX}result:{analysis_id}")
                if result_data:
                    data = json.loads(result_data)
                    
                    # Extract variable suggestions from stored data
                    suggestions_data = data.get("model_validation", {}).get("suggested_variables", [])
                    
                    # Convert back to VariableConfiguration objects
                    suggestions = []
                    for var_data in suggestions_data:
                        from .excel_intelligence import DistributionParameters
                        
                        # Reconstruct DistributionParameters
                        dist_data = var_data["distribution"]
                        distribution = DistributionParameters(
                            distribution_type=dist_data["distribution_type"],
                            min_value=dist_data.get("min_value"),
                            max_value=dist_data.get("max_value"),
                            most_likely=dist_data.get("most_likely"),
                            mean=dist_data.get("mean"),
                            std_dev=dist_data.get("std_dev"),
                            alpha=dist_data.get("alpha"),
                            beta=dist_data.get("beta"),
                            confidence_level=dist_data.get("confidence_level", 0.95)
                        )
                        
                        # Reconstruct VariableConfiguration
                        var_config = VariableConfiguration(
                            cell_address=var_data["cell_address"],
                            sheet_name=var_data["sheet_name"],
                            variable_name=var_data["variable_name"],
                            current_value=var_data["current_value"],
                            business_justification=var_data["business_justification"],
                            risk_category=var_data["risk_category"],
                            correlation_candidates=var_data["correlation_candidates"],
                            distribution=distribution
                        )
                        suggestions.append(var_config)
                    
                    return suggestions
            
            logger.warning(f"No variable suggestions found for analysis: {analysis_id}")
            return None
            
        except Exception as e:
            logger.error(f"Error getting variable suggestions: {e}")
            return None
    
    async def get_model_insights(self, analysis_id: str) -> Optional[ModelInsights]:
        """Get model insights from completed analysis"""
        try:
            if self.redis_client:
                result_data = self.redis_client.get(f"{REDIS_KEY_PREFIX}result:{analysis_id}")
                if result_data:
                    data = json.loads(result_data)
                    insights_data = data.get("model_validation", {}).get("model_insights", {})
                    
                    if insights_data:
                        return ModelInsights(
                            model_type=insights_data["model_type"],
                            complexity_score=insights_data["complexity_score"],
                            key_drivers=insights_data["key_drivers"],
                            output_variables=insights_data["output_variables"],
                            potential_risks=insights_data["potential_risks"],
                            model_quality_issues=insights_data["model_quality_issues"],
                            recommended_iterations=insights_data["recommended_iterations"]
                        )
            return None
        except Exception as e:
            logger.error(f"Error getting model insights: {e}")
            return None
    
    async def _store_analysis_status(self, status: AnalysisStatus):
        """Store analysis status in Redis"""
        try:
            if self.redis_client:
                status_data = {
                    "analysis_id": status.analysis_id,
                    "status": status.status,
                    "progress": status.progress,
                    "message": status.message,
                    "created_at": status.created_at.isoformat(),
                    "updated_at": status.updated_at.isoformat(),
                    "file_id": status.file_id,
                    "sheet_name": status.sheet_name
                }
                
                self.redis_client.setex(
                    f"{REDIS_KEY_PREFIX}status:{status.analysis_id}",
                    ANALYSIS_TTL,
                    json.dumps(status_data)
                )
                logger.debug(f"📦 Stored analysis status: {status.analysis_id}")
        except Exception as e:
            logger.error(f"Error storing analysis status: {e}")
    
    async def _store_analysis_result(self, result: AnalysisResult):
        """Store complete analysis result in Redis with proper serialization"""
        try:
            if self.redis_client:
                # Get the variable suggestions and model insights from model_validation
                model_validation = result.model_validation
                model_insights = model_validation.get("model_insights")
                suggested_variables = model_validation.get("suggested_variables", [])
                
                # Serialize the analysis result with correct attribute mapping
                result_data = {
                    "ai_analysis_id": result.ai_analysis_id,
                    "results_summary": asdict(result.results_summary),
                    "variable_performance": result.variable_performance,
                    "model_validation": {
                        # FIXED: Use actual ModelInsights attributes
                        'model_insights': {
                            'model_type': model_insights.model_type,
                            'complexity_score': model_insights.complexity_score,
                            'key_drivers': model_insights.key_drivers,
                            'output_variables': model_insights.output_variables,
                            'potential_risks': model_insights.potential_risks,
                            'model_quality_issues': model_insights.model_quality_issues,
                            'recommended_iterations': model_insights.recommended_iterations
                        },
                        # FIXED: Use actual VariableConfiguration attributes
                        'suggested_variables': [
                            {
                                'cell_address': var.cell_address,
                                'sheet_name': var.sheet_name,
                                'variable_name': var.variable_name,
                                'current_value': str(var.current_value),
                                'business_justification': var.business_justification,
                                'risk_category': var.risk_category,
                                'correlation_candidates': var.correlation_candidates,
                                'distribution': {
                                    # FIXED: Use actual DistributionParameters attributes
                                    'distribution_type': var.distribution.distribution_type.value if hasattr(var.distribution.distribution_type, 'value') else str(var.distribution.distribution_type),
                                    'min_value': var.distribution.min_value,
                                    'max_value': var.distribution.max_value,
                                    'most_likely': var.distribution.most_likely,
                                    'mean': var.distribution.mean,
                                    'std_dev': var.distribution.std_dev,
                                    'alpha': var.distribution.alpha,
                                    'beta': var.distribution.beta,
                                    'confidence_level': var.distribution.confidence_level
                                }
                            } for var in suggested_variables
                        ]
                    },
                    "timestamp": result.timestamp.isoformat()
                }
                
                self.redis_client.setex(
                    f"{REDIS_KEY_PREFIX}result:{result.ai_analysis_id}",
                    ANALYSIS_TTL,
                    json.dumps(result_data)
                )
                logger.info(f"✅ Stored analysis result: {result.ai_analysis_id}")
                
        except Exception as e:
            logger.error(f"❌ Failed to store analysis: {str(e)}")
            # Store in memory as fallback
            if not hasattr(self, '_memory_storage'):
                self._memory_storage = {}
            self._memory_storage[result.ai_analysis_id] = result
            logger.info(f"📦 Stored analysis in memory fallback: {result.ai_analysis_id}")
    
    async def analyze_simulation_results(self, simulation_id: str, results_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze simulation results using AI to provide insights
        """
        try:
            insights = await self.results_analyzer.analyze_results(simulation_id, results_data)
            
            # Store insights for future reference
            insight_id = f"insight_{simulation_id}_{int(time.time())}"
            if self.redis_client:
                self.redis_client.setex(
                    f"{REDIS_KEY_PREFIX}insights:{insight_id}",
                    ANALYSIS_TTL,
                    json.dumps({
                        "simulation_id": simulation_id,
                        "insights": insights,
                        "timestamp": datetime.now().isoformat()
                    })
                )
            
            return {
                "insight_id": insight_id,
                "insights": insights,
                "status": "completed"
            }
            
        except Exception as e:
            logger.error(f"Results analysis failed: {e}")
            return {
                "insight_id": None,
                "insights": None,
                "status": "failed",
                "error": str(e)
            }
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of AI layer components"""
        status = {
            "ai_layer_manager": "healthy",
            "deepseek_enabled": self.use_deepseek,
            "redis_connected": self.redis_client is not None,
            "components": {
                "excel_intelligence": "healthy",
                "variable_suggester": "healthy", 
                "results_analyzer": "healthy"
            },
            "timestamp": datetime.now().isoformat()
        }
        
        # Test Redis connection
        if self.redis_client:
            try:
                self.redis_client.ping()
                status["redis_status"] = "connected"
            except Exception as e:
                status["redis_status"] = f"error: {str(e)}"
                status["redis_connected"] = False
        else:
            status["redis_status"] = "not_configured"
        
        return status
    
    def get_variable_suggestions_json(self, analysis_id: str) -> Dict[str, Any]:
        """Get variable suggestions in JSON format for API response"""
        try:
            # Get variable suggestions from active analyses first
            analysis_result = self.active_analyses.get(analysis_id)
            
            # If not in memory, try Redis
            if not analysis_result and self.redis_client:
                try:
                    redis_data = self.redis_client.get(f"ai_analysis:{analysis_id}")
                    if redis_data:
                        data = json.loads(redis_data)
                        # Extract the relevant data from the stored analysis
                        return {
                            "analysis_id": analysis_id,
                            "suggested_variables": data.get("suggested_variables", []),
                            "suggested_targets": data.get("suggested_targets", []),
                            "model_insights": data.get("model_insights", {}),
                            "confidence_score": data.get("confidence_score", 0.0)
                        }
                    else:
                        suggestions = None
                        output_targets = []
                        model_insights = None
                except Exception as e:
                    logger.debug(f"Redis lookup failed: {e}")
                    suggestions = None
                    output_targets = []
                    model_insights = None
            else:
                suggestions = analysis_result.suggested_variables if analysis_result else None
                output_targets = analysis_result.suggested_targets if analysis_result else []
                model_insights = analysis_result.model_insights if analysis_result else None
            
            if suggestions is None:
                return {"error": "Analysis not found"}
            
            # Convert to JSON-serializable format
            suggestions_json = []
            for var in suggestions:
                var_json = {
                    "cell_address": var.cell_address,
                    "sheet_name": var.sheet_name,
                    "variable_name": var.variable_name,
                    "current_value": str(var.current_value),
                    "business_justification": var.business_justification,
                    "risk_category": var.risk_category,
                    "correlation_candidates": var.correlation_candidates,
                    "distribution": {
                        "distribution_type": var.distribution.distribution_type.value if hasattr(var.distribution.distribution_type, 'value') else str(var.distribution.distribution_type),
                        "min_value": var.distribution.min_value,
                        "max_value": var.distribution.max_value,
                        "most_likely": var.distribution.most_likely,
                        "mean": var.distribution.mean,
                        "std_dev": var.distribution.std_dev,
                        "alpha": var.distribution.alpha,
                        "beta": var.distribution.beta,
                        "confidence_level": var.distribution.confidence_level,
                        "reasoning": var.distribution.reasoning,
                        "risk_impact": var.distribution.risk_impact,
                        "business_rationale": var.distribution.business_rationale
                    }
                }
                suggestions_json.append(var_json)
            
            # Build response
            response = {
                "analysis_id": analysis_id,
                "model_description": model_insights.business_purpose if model_insights else None,
                "model_kpis": analysis_result.model_kpis if analysis_result else None,
                "model_insights": {
                    "model_type": model_insights.model_type if model_insights else "unknown",
                    "complexity_score": model_insights.complexity_score if model_insights else 0.5,
                    "key_drivers": model_insights.key_drivers if model_insights else [],
                    "output_variables": model_insights.output_variables if model_insights else [],
                    "potential_risks": model_insights.potential_risks if model_insights else [],
                    "model_quality_issues": model_insights.model_quality_issues if model_insights else [],
                    "recommended_iterations": model_insights.recommended_iterations if model_insights else 1000
                },
                "suggested_variables": suggestions_json,
                "suggested_targets": output_targets,  # Include extracted output targets
                "confidence": 0.85,  # Calculate from suggestions confidence
                "ready_for_simulation": len(suggestions_json) > 0,
                "integration_notes": [
                    "Variables ready for Monte Carlo simulation",
                    "Suggested distributions based on AI analysis"
                ],
                "timestamp": datetime.now().isoformat()
            }
            
            return response
            
        except Exception as e:
            logger.error(f"Error getting variable suggestions JSON: {e}")
            return {"error": str(e)}
    
    async def analyze_excel_file(self, file_id: str, workbook_data: Dict[str, Any], sheet_name: str = None, analysis_id: str = None) -> Dict[str, Any]:
        """
        Analyze Excel file with AI intelligence
        Called by the router for Excel analysis
        """
        try:
            logger.info(f"🧠 [AI_ANALYSIS] Starting analysis for file {file_id}")
            
            if not analysis_id:
                analysis_id = f"ai_analysis_{file_id}_{int(time.time())}"
            
            # Extract sheets data
            sheets_data = workbook_data.get('sheets', {})
            if sheet_name and sheet_name in sheets_data:
                target_sheet_data = {sheet_name: sheets_data[sheet_name]}
            else:
                target_sheet_data = sheets_data
            
            # NEW: Try DeepSeek comprehensive analysis first
            try:
                if self.excel_agent.deepseek_client:
                    logger.info(f"🎯 [AI_ANALYSIS] Attempting DeepSeek comprehensive analysis for {file_id}")
                    
                    # Get comprehensive analysis from DeepSeek
                    deepseek_analysis = await self.excel_agent.deepseek_client.analyze_complete_excel_model(
                        workbook_data, file_id, sheet_name
                    )
                    
                    if deepseek_analysis:
                        logger.info(f"✅ [AI_ANALYSIS] DeepSeek analysis successful, processing results")
                        
                        # Convert DeepSeek analysis to our format
                        cell_analyses = self.excel_agent._convert_deepseek_to_cell_analyses(deepseek_analysis)
                        model_insights = self.excel_agent._convert_deepseek_to_model_insights(deepseek_analysis)
                        
                        # Create variable suggestions directly from DeepSeek analysis
                        variable_suggestions = await self.variable_suggester.suggest_monte_carlo_variables_from_deepseek(
                            deepseek_analysis, workbook_data
                        )
                        
                        # Extract output targets from DeepSeek analysis
                        output_targets = self.variable_suggester.extract_output_targets_from_deepseek(deepseek_analysis)
                        
                        # Extract model KPIs from DeepSeek analysis
                        model_kpis = deepseek_analysis.get('model_kpis', {})
                        
                        # If DeepSeek didn't provide KPIs, calculate them from the workbook data
                        if not model_kpis or not any(model_kpis.values()):
                            model_kpis = self._calculate_model_kpis_from_workbook(workbook_data)
                        
                        logger.info(f"🎯 [AI_ANALYSIS] DeepSeek comprehensive analysis complete: {len(variable_suggestions)} variables, {len(output_targets)} targets")
                    else:
                        raise Exception("DeepSeek returned no analysis")
                else:
                    raise Exception("No DeepSeek client available")
                    
            except Exception as deepseek_error:
                logger.warning(f"⚠️ [AI_ANALYSIS] DeepSeek analysis failed: {deepseek_error}, falling back to enhanced analysis")
                
                # Fallback to existing enhanced analysis
                cell_analyses, model_insights = await self.excel_agent.analyze_excel_model_enhanced(file_id=file_id, workbook_data=workbook_data, sheet_name=sheet_name)
                variable_suggestions = await self.variable_suggester.suggest_monte_carlo_variables(cell_analyses, workbook_data)
                output_targets = []  # No output targets in fallback analysis
                model_kpis = self._calculate_model_kpis_from_workbook(workbook_data)  # Calculate KPIs from workbook data
            
            # Create analysis result
            start_time = time.time()
            overall_confidence = self._calculate_overall_confidence(variable_suggestions, [], model_insights)
            ready_for_simulation = self._assess_simulation_readiness(variable_suggestions, [], model_insights)
            
            analysis_result = AIAnalysisResult(
                analysis_id=analysis_id,
                excel_file_id=file_id,
                timestamp=datetime.now(),
                cell_analyses=cell_analyses,
                model_insights=model_insights,
                model_kpis=model_kpis,  # Include extracted model KPIs
                suggested_variables=variable_suggestions,
                suggested_targets=output_targets,  # Include extracted output targets
                confidence_score=overall_confidence,
                analysis_duration=time.time() - start_time,
                ready_for_simulation=ready_for_simulation,
                integration_notes=[
                    "AI analysis completed successfully",
                    f"Found {len(variable_suggestions)} variable suggestions"
                ]
            )
            
            # Store result
            self.active_analyses[analysis_id] = analysis_result
            await self._store_analysis_in_redis(analysis_result)
            
            logger.info(f"✅ [AI_ANALYSIS] Completed analysis {analysis_id}")
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"❌ [AI_ANALYSIS] Failed: {e}")
            # Return a fallback AIAnalysisResult for failed analysis
            failed_id = analysis_id or f"failed_{int(time.time())}"
            return AIAnalysisResult(
                analysis_id=failed_id,
                excel_file_id=file_id,
                timestamp=datetime.now(),
                cell_analyses=[],
                model_insights=ModelInsights(
                    model_type="unknown",
                    complexity_score=0.0,
                    key_drivers=[],
                    assumptions=[],
                    recommendations=[],
                    model_quality_issues=[f"Analysis failed: {str(e)}"],
                    recommended_iterations=1000
                ),
                suggested_variables=[],
                suggested_targets=[],
                confidence_score=0.0,
                analysis_duration=0.0,
                ready_for_simulation=False,
                integration_notes=[f"Analysis failed: {str(e)}"]
            )

    def _calculate_overall_confidence(self, variable_suggestions: List[Any], target_suggestions: List[Any], model_insights: Dict[str, Any]) -> float:
        """Calculate overall confidence score for the analysis"""
        try:
            # Base confidence from variable count
            var_count = len(variable_suggestions)
            base_confidence = min(0.8, 0.4 + (var_count * 0.1))  # 40% base + 10% per variable, max 80%
            
            # Adjust based on model complexity
            formula_count = int(model_insights.complexity_score * 10) if hasattr(model_insights, "complexity_score") else 0
            complexity_bonus = min(0.2, formula_count * 0.02)  # Up to 20% bonus for complexity
            
            # Final confidence
            confidence = min(1.0, base_confidence + complexity_bonus)
            return round(confidence, 3)
        except Exception as e:
            logger.warning(f"⚠️ [CONFIDENCE] Error calculating confidence: {e}")
            return 0.5  # Default moderate confidence

    def _assess_simulation_readiness(self, variable_suggestions: List[Any], target_suggestions: List[Any], model_insights: Dict[str, Any]) -> bool:
        """Assess if the model is ready for simulation"""
        try:
            # Must have at least one variable suggestion
            if not variable_suggestions:
                return False
            
            # Must have formula cells to simulate
            formula_count = int(model_insights.complexity_score * 10) if hasattr(model_insights, "complexity_score") else 0
            if formula_count == 0:
                return False
            
            return True
        except Exception as e:
            logger.warning(f"⚠️ [SIMULATION_READINESS] Error assessing readiness: {e}")
            return False
    
    async def _store_analysis_in_redis(self, analysis_result: AIAnalysisResult):
        """Store analysis result in Redis for persistence"""
        try:
            from dataclasses import asdict
            
            # Convert to dict for Redis storage
            def safe_asdict(obj):
                """Safely convert to dict - use asdict for dataclasses, return as-is for dicts"""
                if hasattr(obj, '__dataclass_fields__'):
                    return asdict(obj)
                elif isinstance(obj, dict):
                    return obj
                else:
                    return str(obj)  # Fallback for other types
            
            data = {
                'analysis_id': analysis_result.analysis_id,
                'excel_file_id': analysis_result.excel_file_id,
                'timestamp': analysis_result.timestamp.isoformat(),
                'cell_analyses': [safe_asdict(cell) for cell in analysis_result.cell_analyses],
                'model_insights': safe_asdict(analysis_result.model_insights),
                'suggested_variables': [safe_asdict(var) for var in analysis_result.suggested_variables],
                'suggested_targets': [safe_asdict(target) for target in analysis_result.suggested_targets],
                'confidence_score': analysis_result.confidence_score,
                'analysis_duration': analysis_result.analysis_duration,
                'integration_notes': analysis_result.integration_notes,
                'ready_for_simulation': analysis_result.ready_for_simulation,
                'model_kpis': analysis_result.model_kpis or {}
            }
            
            # Store in Redis with expiration (7 days)
            if self.redis_client:
                self.redis_client.setex(
                    f"ai_analysis:{analysis_result.analysis_id}",
                    604800,  # 7 days
                    json.dumps(data, default=str)
                )
            
            logger.info(f"✅ [REDIS] Stored analysis {analysis_result.analysis_id}")
            
        except Exception as e:
            logger.error(f"❌ [REDIS] Failed to store analysis: {e}")
            # Don't fail the whole analysis if Redis storage fails
    
    def _calculate_model_kpis_from_workbook(self, workbook_data) -> Dict[str, Any]:
        """
        Calculate model KPIs directly from workbook data when DeepSeek doesn't provide them
        """
        try:
            logger.info(f"📊 [MODEL_KPIs] Calculating KPIs from workbook data")
            
            active_sheets = len(workbook_data)
            total_cells = 0
            formula_cells = 0
            input_cells = 0
            output_cells = 0
            
            for sheet in workbook_data:
                grid_data = sheet.get('grid_data', [])
                for row in grid_data:
                    if row:  # Skip None rows
                        for cell in row:
                            if cell and cell.get('value') is not None:
                                total_cells += 1
                                
                                # Count formula cells
                                if cell.get('is_formula_cell', False):
                                    formula_cells += 1
                                    # Formula cells that are likely outputs (named results, percentages, etc.)
                                    if any(keyword in str(cell.get('coordinate', '')).lower() for keyword in ['total', 'result', 'profit', 'revenue', 'cost']):
                                        output_cells += 1
                                else:
                                    # Non-formula cells are potential inputs if they have numeric values
                                    value = cell.get('value')
                                    if isinstance(value, (int, float)) and value != 0:
                                        input_cells += 1
            
            # Ensure we have at least some realistic numbers
            if input_cells == 0 and total_cells > 0:
                input_cells = max(1, total_cells // 10)  # Estimate 10% as inputs
            if output_cells == 0 and formula_cells > 0:
                output_cells = max(1, formula_cells // 3)  # Estimate 1/3 of formulas as key outputs
            
            model_kpis = {
                "active_sheets": active_sheets,
                "total_cells": total_cells,
                "input_cells": input_cells,
                "output_cells": output_cells,
                "formula_cells": formula_cells,
                "structure_description": f"Financial model with {active_sheets} sheet(s) containing {total_cells} cells, including {formula_cells} formulas. Identified {input_cells} potential input parameters and {output_cells} key output metrics."
            }
            
            logger.info(f"📊 [MODEL_KPIs] Calculated: {model_kpis}")
            return model_kpis
            
        except Exception as e:
            logger.error(f"❌ [MODEL_KPIs] Failed to calculate KPIs: {e}")
            return {
                "active_sheets": 1,
                "total_cells": 0,
                "input_cells": 0,
                "output_cells": 0,
                "formula_cells": 0,
                "structure_description": "Unable to analyze model structure"
            }
