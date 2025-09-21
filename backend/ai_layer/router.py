"""
FastAPI Router for AI Layer endpoints
Provides API access to AI analysis capabilities
"""

import logging
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import json

from .ai_integration import AILayerManager
from auth.auth0_dependencies import get_current_active_auth0_user

logger = logging.getLogger(__name__)

# Initialize AI Layer Manager with DeepSeek
ai_manager = AILayerManager(use_deepseek=True)

router = APIRouter(prefix="/ai", tags=["AI Analysis"])

# Request/Response Models
class ExcelAnalysisRequest(BaseModel):
    file_id: str
    sheet_name: Optional[str] = None
    use_deepseek: bool = True

class ExcelAnalysisResponse(BaseModel):
    analysis_id: str
    status: str
    message: str

class VariableSuggestionsResponse(BaseModel):
    analysis_id: str
    model_insights: Dict[str, Any]
    suggested_variables: List[Dict[str, Any]]
    suggested_targets: List[Dict[str, Any]]
    confidence: float
    ready_for_simulation: bool
    integration_notes: List[str]
    timestamp: str

class ResultsAnalysisRequest(BaseModel):
    simulation_id: str
    target_variable: str
    ai_analysis_id: Optional[str] = None

class ResultsAnalysisResponse(BaseModel):
    simulation_id: str
    executive_summary: str
    risk_assessment: str
    opportunities: str
    recommendations: List[str]
    key_insights: List[Dict[str, Any]]
    timestamp: str

@router.post("/analyze-excel", response_model=ExcelAnalysisResponse)
async def analyze_excel_file(
    request: ExcelAnalysisRequest,
    background_tasks: BackgroundTasks,
    current_user=Depends(get_current_active_auth0_user)
):
    """
    Trigger AI analysis of an uploaded Excel file
    Returns immediately while analysis runs in background
    """
    try:
        logger.info(f"🧠 [AI_API] Excel analysis requested for file {request.file_id}")
        
        # Validate file exists (you may want to add actual file validation)
        # For now, we'll assume the file_id is valid from previous upload
        
        # Generate analysis ID once to avoid race conditions
        analysis_id = f"ai_analysis_{request.file_id}_{int(__import__('time').time())}"
        
        # Start background AI analysis
        background_tasks.add_task(
            run_excel_ai_analysis,
            request.file_id,
            request.sheet_name,
            request.use_deepseek,
            analysis_id
        )
        
        return ExcelAnalysisResponse(
            analysis_id=analysis_id,
            status="started",
            message=f"AI analysis started for file {request.file_id}"
        )
        
    except Exception as e:
        logger.error(f"❌ [AI_API] Failed to start Excel analysis: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@router.get("/analysis/{analysis_id}/suggestions", response_model=VariableSuggestionsResponse)
async def get_variable_suggestions(
    analysis_id: str,
    current_user=Depends(get_current_active_auth0_user)
):
    """
    Get AI-generated variable suggestions from Excel analysis
    """
    try:
        logger.info(f"📊 [AI_API] Variable suggestions requested for {analysis_id}")
        
        # Get analysis result
        suggestions_data = ai_manager.get_variable_suggestions_json(analysis_id)
        
        if 'error' in suggestions_data:
            raise HTTPException(status_code=404, detail=suggestions_data['error'])
        
        return VariableSuggestionsResponse(**suggestions_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ [AI_API] Failed to get suggestions: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve suggestions: {str(e)}")

@router.post("/analyze-results", response_model=ResultsAnalysisResponse)
async def analyze_simulation_results(
    request: ResultsAnalysisRequest,
    background_tasks: BackgroundTasks,
    current_user=Depends(get_current_active_auth0_user)
):
    """
    Trigger AI analysis of simulation results
    """
    try:
        logger.info(f"📈 [AI_API] Results analysis requested for simulation {request.simulation_id}")
        
        # Note: In a real implementation, you'd need to retrieve the actual results data
        # For now, this is a placeholder that shows the integration pattern
        
        background_tasks.add_task(
            run_results_ai_analysis,
            request.simulation_id,
            request.target_variable,
            request.ai_analysis_id
        )
        
        return ResultsAnalysisResponse(
            simulation_id=request.simulation_id,
            executive_summary="AI analysis in progress...",
            risk_assessment="Analysis pending",
            opportunities="Analysis pending",
            recommendations=["Check back in a few moments for complete analysis"],
            key_insights=[],
            timestamp=__import__('datetime').datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error(f"❌ [AI_API] Failed to start results analysis: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@router.get("/results/{simulation_id}/summary", response_model=ResultsAnalysisResponse)
async def get_results_summary(
    simulation_id: str,
    current_user=Depends(get_current_active_auth0_user)
):
    """
    Get AI-generated summary of simulation results
    """
    try:
        logger.info(f"📋 [AI_API] Results summary requested for {simulation_id}")
        
        # Get summary data
        summary_data = ai_manager.get_results_summary_json(simulation_id)
        
        if 'error' in summary_data:
            raise HTTPException(status_code=404, detail=summary_data['error'])
        
        # Extract key fields for response
        results_summary = summary_data['results_summary']
        
        return ResultsAnalysisResponse(
            simulation_id=simulation_id,
            executive_summary=results_summary.get('executive_summary', ''),
            risk_assessment=results_summary.get('risk_assessment', ''),
            opportunities=results_summary.get('opportunities', ''),
            recommendations=results_summary.get('recommendations', []),
            key_insights=[
                {
                    'type': insight.get('insight_type', 'unknown'),
                    'title': insight.get('title', ''),
                    'description': insight.get('description', ''),
                    'risk_level': insight.get('risk_level', 'medium'),
                    'confidence': insight.get('confidence', 0.0)
                }
                for insight in results_summary.get('key_insights', [])
            ],
            timestamp=summary_data['timestamp']
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ [AI_API] Failed to get results summary: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve summary: {str(e)}")

@router.get("/health")
async def ai_health_check():
    """
    Health check for AI layer components
    """
    try:
        health_status = {
            'status': 'healthy',
            'ai_manager': 'active',
            'excel_agent': 'ready',
            'variable_suggester': 'ready',
            'results_analyzer': 'ready',
            'deepseek_enabled': ai_manager.use_deepseek,
            'active_analyses': len(ai_manager.active_analyses),
            'simulation_summaries': len(ai_manager.simulation_summaries)
        }
        
        return health_status
        
    except Exception as e:
        logger.error(f"❌ [AI_API] Health check failed: {e}")
        return {
            'status': 'unhealthy',
            'error': str(e)
        }

@router.post("/test-analyze-excel")
async def test_analyze_excel_no_auth(request: ExcelAnalysisRequest):
    """
    TEST ENDPOINT: Analyze Excel file without authentication for debugging
    WARNING: This endpoint bypasses authentication - remove in production!
    """
    try:
        logger.info(f"🧪 [AI_TEST] Test Excel analysis for file: {request.file_id}")
        
        # Start AI analysis using the integrated manager
        analysis_id = ai_manager.start_excel_analysis(
            file_id=request.file_id,
            sheet_name=request.sheet_name,
            use_deepseek=request.use_deepseek
        )
        
        logger.info(f"✅ [AI_TEST] Test analysis started with ID: {analysis_id}")
        
        return ExcelAnalysisResponse(
            analysis_id=analysis_id,
            status="in_progress", 
            message="Test AI analysis started successfully"
        )
        
    except Exception as e:
        logger.error(f"❌ [AI_TEST] Test analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Test analysis failed: {str(e)}")

# Background task functions

async def run_excel_ai_analysis(file_id: str, sheet_name: Optional[str], use_deepseek: bool, analysis_id: str):
    """
    Background task to run Excel AI analysis
    """
    try:
        logger.info(f"🔄 [AI_BACKGROUND] Starting Excel analysis for {file_id}")
        
        # ENHANCED: Get actual workbook data from Excel parser and Ultra engine analysis
        try:
            from excel_parser.service import get_all_parsed_sheets_data
            sheets_data = await get_all_parsed_sheets_data(file_id)
            
            # Convert to workbook_data format expected by AI layer
            workbook_data = {
                'sheets': {},
                'metadata': {'enhanced': True, 'ultra_integration': True}
            }
            
            for sheet in sheets_data:
                workbook_data['sheets'][sheet.sheet_name] = {
                    'formulas': {},  # Will be populated by Ultra analysis
                    'data': sheet.grid_data,
                    'sheet_data': sheet
                }
                
            logger.info(f"✅ [AI_BACKGROUND] Loaded {len(sheets_data)} sheets for enhanced analysis")
            
        except Exception as e:
            logger.warning(f"⚠️ [AI_BACKGROUND] Could not load Excel data: {e}, using fallback")
            # Fallback to placeholder workbook data
            workbook_data = {
                'sheets': {'Sheet1': {'formulas': {}, 'data': []}},
                'metadata': {}
            }
        
        # Run AI analysis
        analysis_result = await ai_manager.analyze_excel_file(
            file_id=file_id,
            workbook_data=workbook_data,
            sheet_name=sheet_name,
            analysis_id=analysis_id
        )
        
        logger.info(f"✅ [AI_BACKGROUND] Excel analysis completed: {analysis_result.analysis_id}")
        
    except Exception as e:
        logger.error(f"❌ [AI_BACKGROUND] Excel analysis failed: {e}")

async def run_results_ai_analysis(simulation_id: str, target_variable: str, ai_analysis_id: Optional[str]):
    """
    Background task to run simulation results AI analysis
    """
    try:
        logger.info(f"🔄 [AI_BACKGROUND] Starting results analysis for {simulation_id}")
        
        # Note: In a real implementation, you'd retrieve the actual simulation results
        # from your existing results storage system.
        
        # Example integration with existing results system:
        # from simulation.results import get_simulation_results
        # results_data = await get_simulation_results(simulation_id)
        
        # Placeholder results data
        import numpy as np
        results_data = np.random.normal(1000, 200, 10000)  # Placeholder simulation results
        
        # Run AI analysis
        summary = await ai_manager.analyze_simulation_results(
            simulation_id=simulation_id,
            results_data=results_data,
            target_variable=target_variable,
            ai_analysis_id=ai_analysis_id
        )
        
        logger.info(f"✅ [AI_BACKGROUND] Results analysis completed: {simulation_id}")
        
    except Exception as e:
        logger.error(f"❌ [AI_BACKGROUND] Results analysis failed: {e}")

# Integration helper endpoints for Ultra Engine

@router.post("/integration/excel-analyzed")
async def notify_excel_analyzed(
    file_id: str,
    workbook_data: Dict[str, Any],
    current_user=Depends(get_current_active_auth0_user)
):
    """
    Integration endpoint called by Excel parser when file is processed
    Automatically triggers AI analysis
    """
    try:
        logger.info(f"🔗 [AI_INTEGRATION] Excel file processed: {file_id}")
        
        # Run AI analysis with actual workbook data
        analysis_result = await ai_manager.analyze_excel_file(
            file_id=file_id,
            workbook_data=workbook_data
        )
        
        return {
            'analysis_id': analysis_result.analysis_id,
            'variables_suggested': len(analysis_result.suggested_variables),
            'targets_suggested': len(analysis_result.suggested_targets),
            'confidence': analysis_result.overall_confidence,
            'ready_for_simulation': analysis_result.ready_for_simulation
        }
        
    except Exception as e:
        logger.error(f"❌ [AI_INTEGRATION] Excel integration failed: {e}")
        raise HTTPException(status_code=500, detail=f"Integration failed: {str(e)}")

@router.post("/integration/simulation-completed")
async def notify_simulation_completed(
    simulation_id: str,
    results_data: List[float],
    target_variable: str,
    variable_configs: Optional[List[Dict[str, Any]]] = None,
    ai_analysis_id: Optional[str] = None,
    current_user=Depends(get_current_active_auth0_user)
):
    """
    Integration endpoint called by Ultra Engine when simulation completes
    Automatically generates AI insights
    """
    try:
        logger.info(f"🔗 [AI_INTEGRATION] Simulation completed: {simulation_id}")
        
        import numpy as np
        results_array = np.array(results_data)
        
        # Run AI analysis with actual results
        summary = await ai_manager.analyze_simulation_results(
            simulation_id=simulation_id,
            results_data=results_array,
            target_variable=target_variable,
            variable_configs=variable_configs,
            ai_analysis_id=ai_analysis_id
        )
        
        return {
            'simulation_id': simulation_id,
            'ai_summary_available': True,
            'executive_summary': summary.results_summary.executive_summary,
            'key_insights_count': len(summary.results_summary.key_insights),
            'recommendations_count': len(summary.results_summary.recommendations)
        }
        
    except Exception as e:
        logger.error(f"❌ [AI_INTEGRATION] Simulation integration failed: {e}")
        raise HTTPException(status_code=500, detail=f"Integration failed: {str(e)}")
