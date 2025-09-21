"""
🧪 BULLETPROOF PROGRESS DTO TESTS
Unit tests for the progress schema transformation and DTO adapter
"""

import pytest
import time
from unittest.mock import patch, MagicMock

# Import the DTO components
from shared.progress_schema import (
    create_progress_dto,
    ProgressDTO,
    PhaseProgress,
    VariableProgress,
    EngineInfo,
    FormulaMetrics,
    determine_current_stage,
    create_phases_from_progress,
    create_variables_from_progress,
    create_engine_info,
    create_formula_metrics
)

class TestProgressDTO:
    """Test the ProgressDTO model and creation"""
    
    def test_progress_dto_creation(self):
        """Test basic ProgressDTO creation"""
        dto = ProgressDTO(
            simulation_id="test-123",
            progress_percentage=50.0,
            current_iteration=50,
            total_iterations=100,
            status="running"
        )
        
        assert dto.simulation_id == "test-123"
        assert dto.progress_percentage == 50.0
        assert dto.overallProgress == 0.0  # Default value
        assert dto.currentStage == "Initializing..."  # Default value
        assert dto.status == "running"

    def test_progress_dto_with_frontend_schema(self):
        """Test ProgressDTO with frontend-expected fields"""
        phases = {
            "simulation": PhaseProgress(stage="Running Simulation", progress=75.0, completed=False)
        }
        variables = {
            "test-sim": VariableProgress(
                name="Target Variable",
                progress=75.0,
                status="running",
                iterations=75,
                totalIterations=100
            )
        }
        
        dto = ProgressDTO(
            simulation_id="test-123",
            progress_percentage=75.0,
            overallProgress=75.0,
            currentStage="Running Monte Carlo Simulation - 75.0%",
            phases=phases,
            variables=variables,
            status="streaming"
        )
        
        assert dto.overallProgress == 75.0
        assert dto.currentStage == "Running Monte Carlo Simulation - 75.0%"
        assert "simulation" in dto.phases
        assert dto.phases["simulation"].progress == 75.0
        assert "test-sim" in dto.variables
        assert dto.variables["test-sim"].iterations == 75

class TestProgressAdapter:
    """Test the progress adapter functions"""
    
    def test_create_progress_dto_basic(self):
        """Test basic DTO creation from raw progress"""
        raw_progress = {
            "progress_percentage": 25.0,
            "current_iteration": 25,
            "total_iterations": 100,
            "status": "streaming",
            "streaming_mode": True,
            "memory_efficient": True
        }
        
        dto = create_progress_dto(
            simulation_id="test-sim-123",
            raw_progress=raw_progress
        )
        
        assert dto.simulation_id == "test-sim-123"
        assert dto.progress_percentage == 25.0
        assert dto.overallProgress == 25.0
        assert dto.status == "streaming"
        assert dto.streaming_mode is True
        assert dto.memory_efficient is True
        assert "test-sim-123" in dto.variables
        assert dto.variables["test-sim-123"].progress == 25.0

    def test_create_progress_dto_with_target_variables(self):
        """Test DTO creation with target variable names"""
        raw_progress = {
            "progress_percentage": 60.0,
            "current_iteration": 60,
            "total_iterations": 100,
            "status": "running"
        }
        
        target_variables = ["Revenue", "Profit Margin"]
        
        dto = create_progress_dto(
            simulation_id="test-sim-456",
            raw_progress=raw_progress,
            target_variables=target_variables
        )
        
        assert dto.variables["test-sim-456"].name == "Revenue"  # Uses first target variable
        assert len(target_variables) == 2  # Metadata includes all variables

    def test_create_progress_dto_completed(self):
        """Test DTO creation for completed simulation"""
        raw_progress = {
            "progress_percentage": 100.0,
            "current_iteration": 100,
            "total_iterations": 100,
            "status": "completed"
        }
        
        dto = create_progress_dto(
            simulation_id="test-sim-completed",
            raw_progress=raw_progress
        )
        
        assert dto.overallProgress == 100.0
        assert dto.currentStage == "Completed"
        assert dto.status == "completed"
        # All phases should be completed
        for phase in dto.phases.values():
            assert phase.completed is True
            assert phase.progress == 100.0

class TestStageMapping:
    """Test stage determination logic"""
    
    def test_determine_current_stage_completed(self):
        """Test stage determination for completed simulation"""
        stage = determine_current_stage(100.0, "completed", "simulation", False)
        assert stage == "Completed"

    def test_determine_current_stage_failed(self):
        """Test stage determination for failed simulation"""
        stage = determine_current_stage(50.0, "failed", "simulation", False)
        assert stage == "Failed"

    def test_determine_current_stage_cancelled(self):
        """Test stage determination for cancelled simulation"""
        stage = determine_current_stage(30.0, "cancelled", "simulation", False)
        assert stage == "Cancelled"

    def test_determine_current_stage_initializing(self):
        """Test stage determination for initializing simulation"""
        stage = determine_current_stage(0.0, "pending", "initialization", False)
        assert stage == "Initializing..."

    def test_determine_current_stage_streaming(self):
        """Test stage determination for streaming simulation"""
        stage = determine_current_stage(45.0, "streaming", "simulation", True)
        assert stage == "Running Monte Carlo Simulation (Streaming Mode) - 45.0%"

    def test_determine_current_stage_regular(self):
        """Test stage determination for regular simulation"""
        stage = determine_current_stage(75.0, "running", "simulation", False)
        assert stage == "Running Monte Carlo Simulation - 75.0%"

class TestPhaseMapping:
    """Test phase creation logic"""
    
    def test_create_phases_initializing(self):
        """Test phase creation for initializing simulation"""
        phases = create_phases_from_progress(0.0, "pending", False)
        
        assert phases["initialization"].progress == 50.0
        assert phases["initialization"].completed is False
        assert phases["simulation"].progress == 0.0
        assert phases["simulation"].completed is False

    def test_create_phases_running(self):
        """Test phase creation for running simulation"""
        phases = create_phases_from_progress(40.0, "streaming", True)
        
        # Earlier phases should be completed
        assert phases["initialization"].completed is True
        assert phases["parsing"].completed is True
        assert phases["smart_analysis"].completed is True
        assert phases["analysis"].completed is True
        
        # Current phase should reflect progress
        assert phases["simulation"].progress == 40.0
        assert phases["simulation"].completed is False

    def test_create_phases_completed(self):
        """Test phase creation for completed simulation"""
        phases = create_phases_from_progress(100.0, "completed", False)
        
        # All phases should be completed
        for phase in phases.values():
            assert phase.progress == 100.0
            assert phase.completed is True

class TestVariableMapping:
    """Test variable creation logic"""
    
    def test_create_variables_basic(self):
        """Test basic variable creation"""
        variables = create_variables_from_progress(
            simulation_id="test-123",
            progress=65.0,
            current_iteration=65,
            total_iterations=100,
            status="running"
        )
        
        assert "test-123" in variables
        var = variables["test-123"]
        assert var.name == "Target Variable"  # Default name
        assert var.progress == 65.0
        assert var.status == "running"
        assert var.iterations == 65
        assert var.totalIterations == 100

    def test_create_variables_with_target_names(self):
        """Test variable creation with target variable names"""
        target_variables = ["Sales Forecast", "Market Share"]
        
        variables = create_variables_from_progress(
            simulation_id="test-456",
            progress=80.0,
            current_iteration=80,
            total_iterations=100,
            status="running",
            target_variables=target_variables
        )
        
        var = variables["test-456"]
        assert var.name == "Sales Forecast"  # Uses first target variable

class TestEngineInfo:
    """Test engine info creation"""
    
    def test_create_engine_info_gpu(self):
        """Test engine info creation for GPU engine"""
        raw_progress = {
            "engine": "WorldClass GPU Engine",
            "engine_type": "GPU",
            "gpu_acceleration": True,
            "streaming_mode": False
        }
        
        engine_info = create_engine_info(raw_progress)
        
        assert engine_info.engine == "WorldClass GPU Engine"
        assert engine_info.engine_type == "GPU"
        assert engine_info.gpu_acceleration is True
        assert engine_info.detected is True

    def test_create_engine_info_streaming(self):
        """Test engine info creation for streaming engine"""
        raw_progress = {
            "streaming_mode": True,
            "memory_efficient": True
        }
        
        engine_info = create_engine_info(raw_progress)
        
        assert engine_info.engine == "WorldClassMonteCarloEngine"  # Default
        assert engine_info.engine_type == "Streaming"  # Inferred from streaming_mode
        assert engine_info.gpu_acceleration is False  # Default
        assert engine_info.detected is True

class TestFormulaMetrics:
    """Test formula metrics creation"""
    
    def test_create_formula_metrics(self):
        """Test formula metrics creation"""
        raw_progress = {
            "total_formulas": 45000,
            "relevant_formulas": 12000,
            "analysis_method": "Smart Dependency Analyzer",
            "cache_hits": 8500,
            "chunks_processed": 15,
            "streaming_mode": True
        }
        
        metrics = create_formula_metrics(raw_progress)
        
        assert metrics.total_formulas == 45000
        assert metrics.relevant_formulas == 12000
        assert metrics.analysis_method == "Smart Dependency Analyzer"
        assert metrics.cache_hits == 8500
        assert metrics.chunks_processed == 15

    def test_create_formula_metrics_defaults(self):
        """Test formula metrics creation with defaults"""
        raw_progress = {
            "streaming_mode": False
        }
        
        metrics = create_formula_metrics(raw_progress)
        
        assert metrics.total_formulas == 0
        assert metrics.relevant_formulas == 0
        assert metrics.analysis_method == "Standard"  # Inferred from non-streaming
        assert metrics.cache_hits == 0
        assert metrics.chunks_processed == 0

class TestIntegration:
    """Integration tests for the complete DTO transformation"""
    
    def test_large_file_simulation_dto(self):
        """Test DTO transformation for large file simulation"""
        raw_progress = {
            "progress_percentage": 35.0,
            "current_iteration": 35,
            "total_iterations": 100,
            "status": "streaming",
            "streaming_mode": True,
            "memory_efficient": True,
            "stage": "simulation",
            "engine": "WorldClass GPU Engine",
            "engine_type": "Streaming",
            "gpu_acceleration": True,
            "total_formulas": 45000,
            "relevant_formulas": 12000,
            "analysis_method": "Smart Dependency Analyzer",
            "cache_hits": 8500,
            "chunks_processed": 15
        }
        
        target_variables = ["Revenue", "Profit", "Market Share"]
        
        dto = create_progress_dto(
            simulation_id="large-file-sim",
            raw_progress=raw_progress,
            target_variables=target_variables
        )
        
        # Verify core transformation
        assert dto.simulation_id == "large-file-sim"
        assert dto.overallProgress == 35.0
        assert dto.streaming_mode is True
        assert dto.memory_efficient is True
        
        # Verify frontend-expected schema
        assert "large-file-sim" in dto.variables
        assert dto.variables["large-file-sim"].name == "Revenue"
        assert dto.variables["large-file-sim"].progress == 35.0
        
        # Verify phases are set correctly
        assert dto.phases["simulation"].progress == 35.0
        assert dto.phases["analysis"].completed is True
        
        # Verify engine info
        assert dto.engineInfo.engine == "WorldClass GPU Engine"
        assert dto.engineInfo.gpu_acceleration is True
        
        # Verify formula metrics
        assert dto.formulaMetrics.total_formulas == 45000
        assert dto.formulaMetrics.analysis_method == "Smart Dependency Analyzer"
        
        # Verify current stage description
        assert "Streaming Mode" in dto.currentStage

    def test_small_file_simulation_dto(self):
        """Test DTO transformation for small file simulation"""
        raw_progress = {
            "progress_percentage": 85.0,
            "current_iteration": 85,
            "total_iterations": 100,
            "status": "running",
            "streaming_mode": False,
            "memory_efficient": False,
            "stage": "simulation",
            "engine": "Standard CPU Engine",
            "engine_type": "CPU",
            "gpu_acceleration": False,
            "total_formulas": 500,
            "analysis_method": "Traditional Analysis"
        }
        
        dto = create_progress_dto(
            simulation_id="small-file-sim",
            raw_progress=raw_progress
        )
        
        # Verify transformation for small file
        assert dto.overallProgress == 85.0
        assert dto.streaming_mode is False
        assert dto.currentStage == "Running Monte Carlo Simulation - 85.0%"
        
        # Verify engine info for CPU
        assert dto.engineInfo.engine == "Standard CPU Engine"
        assert dto.engineInfo.gpu_acceleration is False
        
        # Verify formula metrics for small file
        assert dto.formulaMetrics.total_formulas == 500
        assert dto.formulaMetrics.analysis_method == "Traditional Analysis"

if __name__ == "__main__":
    pytest.main([__file__]) 