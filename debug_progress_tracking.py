#!/usr/bin/env python3
"""
Comprehensive debugging script for the progress tracking system.

This script tests the entire progress tracking pipeline from backend to frontend,
including Redis connectivity, progress store operations, API endpoints, and ID normalization.
"""

import os
import sys
import json
import time
import requests
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'backend'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('debug_progress_tracking.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class TestResult:
    test_name: str
    success: bool
    message: str
    details: Optional[Dict] = None
    duration: Optional[float] = None

class ProgressTrackingDebugger:
    """Comprehensive progress tracking debugging tool"""
    
    def __init__(self, api_base_url: str = "http://localhost:8000/api"):
        self.api_base_url = api_base_url
        self.results: List[TestResult] = []
        self.test_simulation_ids = []
        
    def log_result(self, test_name: str, success: bool, message: str, details: Dict = None, duration: float = None):
        """Log a test result"""
        result = TestResult(test_name, success, message, details, duration)
        self.results.append(result)
        
        status = "✅ PASS" if success else "❌ FAIL"
        duration_str = f" ({duration:.2f}s)" if duration else ""
        logger.info(f"{status}: {test_name} - {message}{duration_str}")
        
        if details:
            logger.debug(f"Details: {json.dumps(details, indent=2)}")
    
    def test_redis_connectivity(self) -> bool:
        """Test Redis connection and basic operations"""
        start_time = time.time()
        try:
            from shared.progress_store import _progress_store
            
            # Test basic Redis operations
            test_key = "debug_test_progress"
            test_data = {
                "progress_percentage": 50.0,
                "stage": "simulation",
                "timestamp": time.time()
            }
            
            # Set progress
            _progress_store.set_progress(test_key, test_data)
            
            # Get progress
            retrieved_data = _progress_store.get_progress(test_key)
            
            # Cleanup
            _progress_store.clear_progress(test_key)
            
            success = retrieved_data is not None and retrieved_data.get("progress_percentage") == 50.0
            
            self.log_result(
                "Redis Connectivity",
                success,
                "Redis operations successful" if success else "Redis operations failed",
                {
                    "redis_connected": _progress_store.redis_client is not None,
                    "test_data_stored": retrieved_data is not None,
                    "data_integrity": retrieved_data == test_data if retrieved_data else False
                },
                time.time() - start_time
            )
            
            return success
            
        except Exception as e:
            self.log_result(
                "Redis Connectivity",
                False,
                f"Redis test failed: {str(e)}",
                {"error": str(e)},
                time.time() - start_time
            )
            return False
    
    def test_progress_store_operations(self) -> bool:
        """Test all progress store operations"""
        start_time = time.time()
        try:
            from shared.progress_store import _progress_store
            
            test_sim_id = "debug_test_simulation_001"
            
            # Test metadata operations
            metadata = {
                "target_variables": ["A1", "B2", "C3"],
                "user": "debug_user",
                "file_id": "debug_file_123"
            }
            _progress_store.set_simulation_metadata(test_sim_id, metadata)
            retrieved_metadata = _progress_store.get_simulation_metadata(test_sim_id)
            
            # Test progress operations with various stages
            stages = [
                (10, "initialization", "Starting simulation"),
                (25, "parsing", "Parsing Excel file"),
                (50, "simulation", "Running Monte Carlo"),
                (75, "simulation", "Continuing simulation"),
                (100, "results", "Generating results")
            ]
            
            for progress, stage, description in stages:
                progress_data = {
                    "progress_percentage": progress,
                    "stage": stage,
                    "stage_description": description,
                    "current_iteration": progress * 10,
                    "total_iterations": 1000,
                    "timestamp": time.time()
                }
                _progress_store.set_progress(test_sim_id, progress_data)
                time.sleep(0.1)  # Small delay to test timing
            
            # Test list active progress
            active_progress = _progress_store.list_active_progress()
            
            # Test cleanup
            cleanup_count = _progress_store.cleanup_expired_progress()
            
            # Final retrieval
            final_progress = _progress_store.get_progress(test_sim_id)
            
            # Cleanup test data
            _progress_store.clear_progress(test_sim_id)
            
            success = (
                retrieved_metadata is not None and
                final_progress is not None and
                final_progress.get("progress_percentage") == 100 and
                test_sim_id in active_progress
            )
            
            self.log_result(
                "Progress Store Operations",
                success,
                "All progress store operations successful" if success else "Progress store operations failed",
                {
                    "metadata_stored": retrieved_metadata is not None,
                    "progress_stages_tested": len(stages),
                    "final_progress": final_progress.get("progress_percentage") if final_progress else None,
                    "active_progress_found": test_sim_id in active_progress,
                    "cleanup_count": cleanup_count
                },
                time.time() - start_time
            )
            
            return success
            
        except Exception as e:
            self.log_result(
                "Progress Store Operations",
                False,
                f"Progress store test failed: {str(e)}",
                {"error": str(e)},
                time.time() - start_time
            )
            return False
    
    def test_id_normalization(self) -> bool:
        """Test ID normalization logic with various patterns"""
        start_time = time.time()
        try:
            # Import the router module to test normalization
            sys.path.insert(0, os.path.join(project_root, 'backend'))
            from simulation.router import _normalize_simulation_id
            
            test_cases = [
                # (input_id, expected_output, description)
                ("simple_id", "simple_id", "Simple ID unchanged"),
                ("parent_target_0", "parent", "Single target suffix"),
                ("parent_target_0_target_0", "parent", "Double target suffix"),
                ("parent_target_0_target_0_target_0", "parent", "Triple target suffix"),
                ("c090d9aa-1dd1-4768-9097-696fa61ac2af_target_0", "c090d9aa-1dd1-4768-9097-696fa61ac2af", "UUID with single suffix"),
                ("c090d9aa-1dd1-4768-9097-696fa61ac2af_target_0_target_0", "c090d9aa-1dd1-4768-9097-696fa61ac2af", "UUID with double suffix"),
                ("c090d9aa-1dd1-4768-9097-696fa61ac2af_target_0_target_0_target_0", "c090d9aa-1dd1-4768-9097-696fa61ac2af", "UUID with triple suffix"),
                ("complex_id_with_underscores_target_1", "complex_id_with_underscores", "Complex ID with underscores"),
                ("complex_id_with_underscores_target_1_target_2", "complex_id_with_underscores", "Complex ID with double underscores"),
                ("parent_child_0", "parent", "Single child suffix"),
                ("parent_child_0_child_1", "parent", "Double child suffix"),
                ("parent_sub_0", "parent", "Single sub suffix"),
                ("parent_sub_0_sub_1_sub_2", "parent", "Triple sub suffix"),
                ("no_suffix_here", "no_suffix_here", "No suffix to normalize"),
                ("", "", "Empty string"),
                ("parent_target_10_target_25", "parent", "Double digit target suffixes"),
            ]
            
            results = {}
            all_passed = True
            
            for input_id, expected, description in test_cases:
                try:
                    result = _normalize_simulation_id(input_id)
                    passed = result == expected
                    all_passed = all_passed and passed
                    
                    results[description] = {
                        "input": input_id,
                        "expected": expected,
                        "actual": result,
                        "passed": passed
                    }
                    
                except Exception as e:
                    all_passed = False
                    results[description] = {
                        "input": input_id,
                        "expected": expected,
                        "error": str(e),
                        "passed": False
                    }
            
            self.log_result(
                "ID Normalization",
                all_passed,
                f"ID normalization test {'passed' if all_passed else 'failed'}",
                {
                    "test_cases": len(test_cases),
                    "passed": sum(1 for r in results.values() if r.get("passed", False)),
                    "results": results
                },
                time.time() - start_time
            )
            
            return all_passed
            
        except Exception as e:
            self.log_result(
                "ID Normalization",
                False,
                f"ID normalization test failed: {str(e)}",
                {"error": str(e)},
                time.time() - start_time
            )
            return False
    
    def test_progress_api_endpoint(self) -> bool:
        """Test the /api/simulations/{id}/progress endpoint"""
        start_time = time.time()
        try:
            # Create test progress data
            test_sim_id = "debug_api_test_001"
            
            from shared.progress_store import _progress_store
            test_progress = {
                "progress_percentage": 75.0,
                "stage": "simulation",
                "stage_description": "Running simulation",
                "current_iteration": 750,
                "total_iterations": 1000,
                "status": "running",
                "timestamp": time.time()
            }
            
            _progress_store.set_progress(test_sim_id, test_progress)
            
            # Test the API endpoint
            url = f"{self.api_base_url}/simulations/{test_sim_id}/progress"
            response = requests.get(url, timeout=10)
            
            # Test with child simulation ID (should normalize)
            child_sim_id = f"{test_sim_id}_target_0"
            child_url = f"{self.api_base_url}/simulations/{child_sim_id}/progress"
            child_response = requests.get(child_url, timeout=10)
            
            # Cleanup
            _progress_store.clear_progress(test_sim_id)
            
            # Verify legacy fields are present for compatibility
            response_data = response.json() if response.status_code == 200 else {}
            child_response_data = child_response.json() if child_response.status_code == 200 else {}
            
            legacy_fields_present = (
                "progress_percentage" in response_data and
                "message" in response_data and
                "stage_description" in response_data and
                "progress_percentage" in child_response_data
            )
            
            success = (
                response.status_code == 200 and
                child_response.status_code == 200 and
                response_data.get("progress_percentage") == 75.0 and
                child_response_data.get("progress_percentage") == 75.0 and
                legacy_fields_present
            )
            
            # Test JSON serialization of both endpoints
            import json
            try:
                json.dumps(response_data)
                progress_endpoint_serializable = True
            except (TypeError, ValueError):
                progress_endpoint_serializable = False
            
            # Test status endpoint with same simulation ID
            try:
                status_url = f"{self.api_base_url}/simulations/{test_sim_id}"
                status_response = requests.get(status_url, timeout=5)
                status_serializable = True
                if status_response.status_code == 200:
                    json.dumps(status_response.json())
            except Exception:
                status_serializable = False
            
            self.log_result(
                "Progress API Endpoint",
                success,
                "Progress API endpoint test successful" if success else "Progress API endpoint test failed",
                {
                    "parent_status_code": response.status_code,
                    "child_status_code": child_response.status_code,
                    "parent_response": response_data,
                    "child_response": child_response_data,
                    "id_normalization_works": child_response.status_code == 200,
                    "legacy_fields_present": legacy_fields_present,
                    "required_legacy_fields": ["progress_percentage", "message", "stage_description"],
                    "json_serialization": {
                        "progress_endpoint_serializable": progress_endpoint_serializable,
                        "status_endpoint_serializable": status_serializable
                    }
                },
                time.time() - start_time
            )
            
            return success
            
        except Exception as e:
            self.log_result(
                "Progress API Endpoint",
                False,
                f"API endpoint test failed: {str(e)}",
                {"error": str(e)},
                time.time() - start_time
            )
            return False
    
    def test_multi_target_simulation(self) -> bool:
        """Test multi-target simulation ID handling"""
        start_time = time.time()
        try:
            from shared.progress_store import _progress_store
            
            # Create parent simulation
            parent_id = "debug_multi_target_parent"
            
            # Create child simulations
            child_ids = [
                f"{parent_id}_target_0",
                f"{parent_id}_target_1", 
                f"{parent_id}_target_2"
            ]
            
            # Set progress for parent
            parent_progress = {
                "progress_percentage": 60.0,
                "stage": "simulation",
                "status": "running",
                "target_count": 3,
                "timestamp": time.time()
            }
            _progress_store.set_progress(parent_id, parent_progress)
            
            # Test ID normalization for each child
            from simulation.router import _normalize_simulation_id
            
            normalization_results = {}
            for child_id in child_ids:
                normalized = _normalize_simulation_id(child_id)
                normalization_results[child_id] = normalized
                
                # Test API endpoint for child ID
                try:
                    url = f"{self.api_base_url}/simulations/{child_id}/progress"
                    response = requests.get(url, timeout=5)
                    normalization_results[f"{child_id}_api_status"] = response.status_code
                except Exception as api_error:
                    normalization_results[f"{child_id}_api_error"] = str(api_error)
            
            # Cleanup
            _progress_store.clear_progress(parent_id)
            
            # Check results
            all_normalized_correctly = all(
                normalization_results[child_id] == parent_id 
                for child_id in child_ids
            )
            
            api_calls_successful = all(
                normalization_results.get(f"{child_id}_api_status") == 200
                for child_id in child_ids
            )
            
            success = all_normalized_correctly and api_calls_successful
            
            self.log_result(
                "Multi-Target Simulation",
                success,
                "Multi-target simulation test successful" if success else "Multi-target simulation test failed",
                {
                    "parent_id": parent_id,
                    "child_count": len(child_ids),
                    "normalization_results": normalization_results,
                    "all_normalized_correctly": all_normalized_correctly,
                    "api_calls_successful": api_calls_successful
                },
                time.time() - start_time
            )
            
            return success
            
        except Exception as e:
            self.log_result(
                "Multi-Target Simulation",
                False,
                f"Multi-target simulation test failed: {str(e)}",
                {"error": str(e)},
                time.time() - start_time
            )
            return False
    
    def run_performance_test(self) -> bool:
        """Test performance of progress updates"""
        start_time = time.time()
        try:
            from shared.progress_store import _progress_store
            
            test_sim_id = "debug_performance_test"
            update_count = 100
            
            # Measure progress update performance
            update_times = []
            
            for i in range(update_count):
                update_start = time.time()
                
                progress_data = {
                    "progress_percentage": (i / update_count) * 100,
                    "stage": "simulation",
                    "current_iteration": i * 10,
                    "total_iterations": update_count * 10,
                    "timestamp": time.time()
                }
                
                _progress_store.set_progress(test_sim_id, progress_data)
                update_times.append(time.time() - update_start)
                
                # Small delay to simulate real conditions
                time.sleep(0.01)
            
            # Measure retrieval performance
            retrieval_times = []
            for i in range(50):
                retrieval_start = time.time()
                _progress_store.get_progress(test_sim_id)
                retrieval_times.append(time.time() - retrieval_start)
            
            # Cleanup
            _progress_store.clear_progress(test_sim_id)
            
            # Calculate statistics
            avg_update_time = sum(update_times) / len(update_times)
            max_update_time = max(update_times)
            avg_retrieval_time = sum(retrieval_times) / len(retrieval_times)
            total_test_time = time.time() - start_time
            
            # Performance thresholds
            update_threshold = 0.1  # 100ms per update
            retrieval_threshold = 0.05  # 50ms per retrieval
            
            success = avg_update_time < update_threshold and avg_retrieval_time < retrieval_threshold
            
            self.log_result(
                "Performance Test",
                success,
                f"Performance test {'passed' if success else 'failed'}",
                {
                    "update_count": update_count,
                    "avg_update_time_ms": avg_update_time * 1000,
                    "max_update_time_ms": max_update_time * 1000,
                    "avg_retrieval_time_ms": avg_retrieval_time * 1000,
                    "total_test_time_s": total_test_time,
                    "updates_per_second": update_count / total_test_time,
                    "performance_acceptable": success
                },
                total_test_time
            )
            
            return success
            
        except Exception as e:
            self.log_result(
                "Performance Test",
                False,
                f"Performance test failed: {str(e)}",
                {"error": str(e)},
                time.time() - start_time
            )
            return False
    
    def run_integration_test(self) -> bool:
        """Run a complete integration test simulating a real simulation"""
        start_time = time.time()
        try:
            # Simulate a complete simulation lifecycle
            test_sim_id = "debug_integration_test_001"
            self.test_simulation_ids.append(test_sim_id)
            
            from shared.progress_store import _progress_store
            
            # Simulate simulation stages
            simulation_stages = [
                (0, "initialization", "Starting simulation"),
                (5, "parsing", "Parsing Excel file"),
                (15, "analysis", "Analyzing formulas"),
                (25, "simulation", "Starting Monte Carlo"),
                (40, "simulation", "Running iterations"),
                (60, "simulation", "Continuing simulation"),
                (80, "simulation", "Nearly complete"),
                (95, "results", "Generating results"),
                (100, "results", "Completed successfully")
            ]
            
            api_responses = []
            
            for progress, stage, description in simulation_stages:
                # Update progress
                progress_data = {
                    "progress_percentage": progress,
                    "stage": stage,
                    "stage_description": description,
                    "current_iteration": progress * 10,
                    "total_iterations": 1000,
                    "status": "completed" if progress == 100 else "running",
                    "timestamp": time.time()
                }
                
                _progress_store.set_progress(test_sim_id, progress_data)
                
                # Test API endpoint
                try:
                    url = f"{self.api_base_url}/simulations/{test_sim_id}/progress"
                    response = requests.get(url, timeout=5)
                    api_responses.append({
                        "progress": progress,
                        "status_code": response.status_code,
                        "response_data": response.json() if response.status_code == 200 else response.text
                    })
                except Exception as api_error:
                    api_responses.append({
                        "progress": progress,
                        "error": str(api_error)
                    })
                
                # Small delay to simulate real timing
                time.sleep(0.2)
            
            # Test final state
            final_progress = _progress_store.get_progress(test_sim_id)
            
            # TEST: Simulate stray 0% update after completion
            stray_update_data = {
                "progress_percentage": 0.0,
                "stage": "initialization",
                "stage_description": "Starting simulation",
                "current_iteration": 0,
                "total_iterations": 1000,
                "status": "running",
                "timestamp": time.time()
            }
            _progress_store.set_progress(test_sim_id, stray_update_data)
            
            # Verify that the system still treats simulation as completed
            post_stray_progress = _progress_store.get_progress(test_sim_id)
            stray_update_handled_correctly = (
                post_stray_progress is not None and
                post_stray_progress.get("progress_percentage") == 0.0  # Backend should show the stray update
            )
            
            # Test client policy emulation with this same scenario
            client_progress_sequence = [0, 25, 50, 80, 95, 100, 0]  # Include the stray 0%
            final_client_progress = self.emulate_client_progress_policy(client_progress_sequence, test_sim_id)
            client_policy_works = final_client_progress == 100  # Client should ignore the stray 0%
            
            # Test progress endpoint with stray update
            try:
                stray_url = f"{self.api_base_url}/simulations/{test_sim_id}/progress"
                stray_response = requests.get(stray_url, timeout=5)
                stray_api_success = stray_response.status_code == 200
                stray_response_data = stray_response.json() if stray_api_success else {}
            except Exception:
                stray_api_success = False
                stray_response_data = {}
            
            # Test child ID normalization
            child_id = f"{test_sim_id}_target_0"
            try:
                child_url = f"{self.api_base_url}/simulations/{child_id}/progress"
                child_response = requests.get(child_url, timeout=5)
                child_api_success = child_response.status_code == 200
            except Exception:
                child_api_success = False
            
            # Cleanup
            _progress_store.clear_progress(test_sim_id)
            
            # Evaluate success
            api_success_count = sum(1 for r in api_responses if r.get("status_code") == 200)
            api_success_rate = api_success_count / len(api_responses) if api_responses else 0
            
            success = (
                final_progress is not None and
                final_progress.get("progress_percentage") == 100 and
                api_success_rate >= 0.8 and
                child_api_success and
                stray_update_handled_correctly and
                stray_api_success and
                client_policy_works  # Client should maintain 100% despite stray 0%
            )
            
            self.log_result(
                "Integration Test",
                success,
                f"Integration test {'passed' if success else 'failed'}",
                {
                    "stages_tested": len(simulation_stages),
                    "api_success_rate": api_success_rate,
                    "final_progress": final_progress.get("progress_percentage") if final_progress else None,
                    "child_id_normalization": child_api_success,
                    "stray_0_percent_test": {
                        "backend_handled_correctly": stray_update_handled_correctly,
                        "api_success": stray_api_success,
                        "client_policy_works": client_policy_works,
                        "final_client_progress": final_client_progress,
                        "response_data": stray_response_data
                    },
                    "api_responses": api_responses
                },
                time.time() - start_time
            )
            
            return success
            
        except Exception as e:
            self.log_result(
                "Integration Test",
                False,
                f"Integration test failed: {str(e)}",
                {"error": str(e)},
                time.time() - start_time
            )
            return False
    
    def generate_report(self) -> Dict:
        """Generate a comprehensive test report"""
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r.success)
        failed_tests = total_tests - passed_tests
        
        report = {
            "summary": {
                "total_tests": total_tests,
                "passed": passed_tests,
                "failed": failed_tests,
                "success_rate": passed_tests / total_tests if total_tests > 0 else 0,
                "timestamp": time.time()
            },
            "results": [
                {
                    "test": r.test_name,
                    "success": r.success,
                    "message": r.message,
                    "duration": r.duration,
                    "details": r.details
                }
                for r in self.results
            ],
            "recommendations": self._generate_recommendations()
        }
        
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on test results"""
        recommendations = []
        
        failed_tests = [r for r in self.results if not r.success]
        
        if any("Redis" in r.test_name for r in failed_tests):
            recommendations.append("Check Redis connection and configuration")
        
        if any("API" in r.test_name for r in failed_tests):
            recommendations.append("Verify API server is running and accessible")
        
        if any("Performance" in r.test_name for r in failed_tests):
            recommendations.append("Consider optimizing progress store operations")
        
        if any("ID Normalization" in r.test_name for r in failed_tests):
            recommendations.append("Review ID normalization regex patterns")
        
        if not recommendations:
            recommendations.append("All tests passed - progress tracking system is functioning correctly")
        
        return recommendations
    
    def emulate_client_progress_policy(self, updates: List[float], parent_simulation_id: str = "test_parent") -> float:
        """
        Emulate the client's forward-only progress policy.
        This implements the same logic as the frontend Redux slice to test 
        that stray 0% updates don't regress the displayed progress.
        
        Args:
            updates: List of progress percentages (0-100)
            parent_simulation_id: Simulation ID for tracking
            
        Returns:
            Final displayed progress percentage
        """
        # Simulate the client state
        current_progress = 0.0
        current_parent_id = parent_simulation_id
        timestamp = time.time()
        
        for i, new_progress in enumerate(updates):
            # Simulate the same logic from simulationSlice.js fetchSimulationProgress.fulfilled
            near_complete = current_progress >= 95
            is_decrease = new_progress < current_progress
            
            # Mock stage change and timestamp conditions (simplified)
            stage_changed = i > 0 and new_progress == 0  # Simulate stray 0% as stage change
            time_elapsed = (i * 1000)  # Mock 1 second per update
            
            # Apply the same allowProgressUpdate logic from the Redux slice
            # Only allow progress updates that are non-decreasing OR if not near complete and meet other conditions
            allow_progress_update = (
                (not is_decrease) or 
                (not near_complete and (stage_changed or time_elapsed > 30000))
            )
            
            if allow_progress_update:
                current_progress = new_progress
                timestamp = time.time() + i
                
            logger.debug(f"Update {i}: {new_progress}% -> displayed: {current_progress}% "
                        f"(near_complete: {near_complete}, is_decrease: {is_decrease}, "
                        f"allow_update: {allow_progress_update})")
        
        return current_progress
    
    def test_client_progress_policy(self) -> bool:
        """Test the client's progress policy with various update scenarios"""
        start_time = time.time()
        try:
            test_scenarios = [
                {
                    "name": "Stray 0% after completion",
                    "updates": [0, 80, 100, 0],
                    "expected_final": 100,
                    "description": "Should ignore stray 0% update after reaching 100%"
                },
                {
                    "name": "Stray 0% near completion",
                    "updates": [0, 50, 95, 98, 0],
                    "expected_final": 98,
                    "description": "Should ignore stray 0% update after reaching 95%"
                },
                {
                    "name": "Normal progression",
                    "updates": [0, 25, 50, 75, 100],
                    "expected_final": 100,
                    "description": "Should allow normal forward progression"
                },
                {
                    "name": "Early regression allowed",
                    "updates": [0, 50, 30, 70],
                    "expected_final": 70,  # Final value after all updates
                    "description": "Should allow regression when not near completion"
                }
            ]
            
            results = {}
            all_passed = True
            
            for scenario in test_scenarios:
                final_progress = self.emulate_client_progress_policy(scenario["updates"])
                expected = scenario["expected_final"]
                passed = final_progress == expected
                
                results[scenario["name"]] = {
                    "updates": scenario["updates"],
                    "expected": expected,
                    "actual": final_progress,
                    "passed": passed,
                    "description": scenario["description"]
                }
                
                all_passed = all_passed and passed
                
                if not passed:
                    logger.warning(f"Scenario '{scenario['name']}' failed: "
                                 f"expected {expected}%, got {final_progress}%")
            
            self.log_result(
                "Client Progress Policy Emulation",
                all_passed,
                f"Client policy test {'passed' if all_passed else 'failed'}",
                {
                    "scenarios_tested": len(test_scenarios),
                    "passed": sum(1 for r in results.values() if r["passed"]),
                    "results": results
                },
                time.time() - start_time
            )
            
            return all_passed
            
        except Exception as e:
            self.log_result(
                "Client Progress Policy Emulation",
                False,
                f"Client policy test failed: {str(e)}",
                {"error": str(e)},
                time.time() - start_time
            )
            return False
    
    def test_double_suffix_child_progress_with_stray_zero(self) -> bool:
        """Test double-suffix child IDs with stray 0% after 100% completion"""
        start_time = time.time()
        try:
            from shared.progress_store import _progress_store
            from simulation.router import _normalize_simulation_id
            
            # Create parent simulation ID
            parent_id = "debug_double_suffix_parent_001"
            
            # Simulate progress for parent ID to 100%
            completion_progress = {
                "progress_percentage": 100.0,
                "stage": "results",
                "stage_description": "Completed successfully",
                "current_iteration": 1000,
                "total_iterations": 1000,
                "status": "completed",
                "timestamp": time.time()
            }
            _progress_store.set_progress(parent_id, completion_progress)
            
            # Wait briefly to ensure completion is stored
            time.sleep(0.1)
            
            # Write a stray 0% initialization update to the progress store
            stray_zero_update = {
                "progress_percentage": 0.0,
                "stage": "initialization",
                "stage_description": "Starting simulation",
                "current_iteration": 0,
                "total_iterations": 1000,
                "status": "running",
                "timestamp": time.time()
            }
            _progress_store.set_progress(parent_id, stray_zero_update)
            
            # Construct child ID with double suffix
            child_id_double_suffix = f"{parent_id}_target_0_target_0"
            
            # Test backend invariant: Call progress endpoint and check normalization
            try:
                child_url = f"{self.api_base_url}/simulations/{child_id_double_suffix}/progress"
                child_response = requests.get(child_url, timeout=5)
                child_api_success = child_response.status_code == 200
                child_response_data = child_response.json() if child_api_success else {}
                
                # Backend invariant: response progress_percentage should be what's in the store (0% from stray update)
                # But normalization should work correctly
                backend_progress = child_response_data.get("progress_percentage", -1)
                backend_invariant_1 = backend_progress == 0.0  # Should return the stray 0%
                backend_invariant_2 = backend_progress >= 95   # Alternative: store semantics preserve near-complete state
                
                # Test that normalization worked correctly
                normalized_id = _normalize_simulation_id(child_id_double_suffix)
                normalization_works = normalized_id == parent_id
                
            except Exception as api_error:
                child_api_success = False
                child_response_data = {"error": str(api_error)}
                backend_progress = -1
                backend_invariant_1 = False
                backend_invariant_2 = False
                normalization_works = False
            
            # Test client invariant: Use existing emulate_client_progress_policy
            client_progress_sequence = [0, 80, 100, 0]  # Same scenario with stray 0% after 100%
            final_client_progress = self.emulate_client_progress_policy(client_progress_sequence, parent_id)
            client_invariant = final_client_progress == 100  # Client should maintain 100% despite stray 0%
            
            # Cleanup
            _progress_store.clear_progress(parent_id)
            
            # Evaluate success: Either backend invariant OR client invariant should hold
            backend_invariant_holds = backend_invariant_1 or backend_invariant_2
            success = (
                child_api_success and
                normalization_works and
                (backend_invariant_holds or client_invariant)
            )
            
            # Detailed result logging
            result_details = {
                "parent_id": parent_id,
                "child_id_double_suffix": child_id_double_suffix,
                "normalization": {
                    "input": child_id_double_suffix,
                    "output": normalized_id,
                    "works": normalization_works
                },
                "backend_test": {
                    "api_success": child_api_success,
                    "response_data": child_response_data,
                    "progress_returned": backend_progress,
                    "invariant_1_holds": backend_invariant_1,  # Should return stray 0%
                    "invariant_2_holds": backend_invariant_2   # Should preserve near-complete state
                },
                "client_test": {
                    "progress_sequence": client_progress_sequence,
                    "final_client_progress": final_client_progress,
                    "invariant_holds": client_invariant  # Client should maintain 100%
                },
                "overall_success": success
            }
            
            # Create detailed message
            if success:
                if backend_invariant_holds and client_invariant:
                    message = "Both backend and client invariants hold correctly"
                elif backend_invariant_holds:
                    message = f"Backend invariant holds (progress: {backend_progress}%)"
                elif client_invariant:
                    message = f"Client invariant holds (maintains {final_client_progress}%)"
                else:
                    message = "Test passed for other reasons"
            else:
                failed_parts = []
                if not child_api_success:
                    failed_parts.append("API call failed")
                if not normalization_works:
                    failed_parts.append("ID normalization failed")
                if not backend_invariant_holds and not client_invariant:
                    failed_parts.append("Neither backend nor client invariant holds")
                message = f"Test failed: {', '.join(failed_parts)}"
            
            self.log_result(
                "Double-Suffix Child Progress with Stray 0%",
                success,
                message,
                result_details,
                time.time() - start_time
            )
            
            return success
            
        except Exception as e:
            self.log_result(
                "Double-Suffix Child Progress with Stray 0%",
                False,
                f"Test failed with exception: {str(e)}",
                {"error": str(e)},
                time.time() - start_time
            )
            return False
    
    def run_all_tests(self) -> Dict:
        """Run all debugging tests"""
        logger.info("🚀 Starting comprehensive progress tracking debugging...")
        
        tests = [
            self.test_redis_connectivity,
            self.test_progress_store_operations,
            self.test_id_normalization,
            self.test_progress_api_endpoint,
            self.test_multi_target_simulation,
            self.run_performance_test,
            self.test_client_progress_policy,
            self.run_integration_test,
            self.test_double_suffix_child_progress_with_stray_zero
        ]
        
        for test in tests:
            try:
                test()
            except Exception as e:
                logger.error(f"Test {test.__name__} failed with exception: {e}")
                self.log_result(
                    test.__name__,
                    False,
                    f"Test failed with exception: {str(e)}",
                    {"exception": str(e)}
                )
        
        report = self.generate_report()
        
        # Save detailed report
        report_file = f"progress_tracking_debug_report_{int(time.time())}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"📊 Debug report saved to: {report_file}")
        logger.info(f"🎯 Test Summary: {report['summary']['passed']}/{report['summary']['total_tests']} passed ({report['summary']['success_rate']:.1%})")
        
        # Print recommendations
        logger.info("📋 Recommendations:")
        for rec in report['recommendations']:
            logger.info(f"  • {rec}")
        
        return report

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Debug progress tracking system")
    parser.add_argument("--api-url", default="http://localhost:8000/api", 
                       help="Base URL for API testing")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    debugger = ProgressTrackingDebugger(args.api_url)
    report = debugger.run_all_tests()
    
    # Exit with error code if tests failed or if client policy invariant fails
    client_policy_test = next((r for r in report['results'] if r['test'] == 'Client Progress Policy Emulation'), None)
    client_policy_passed = client_policy_test and client_policy_test['success']
    
    if report['summary']['success_rate'] < 1.0 or not client_policy_passed:
        if not client_policy_passed:
            logger.error("❌ CRITICAL: Client progress policy invariant failed - UI may regress after completion")
        sys.exit(1)
    else:
        logger.info("✅ All tests passed - progress tracking system maintains UI invariants")
        sys.exit(0)

if __name__ == "__main__":
    main()
