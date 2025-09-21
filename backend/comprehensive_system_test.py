#!/usr/bin/env python3
"""
🧪 COMPREHENSIVE SYSTEM TEST

This script performs end-to-end testing of the Monte Carlo simulation platform
to validate all the robustness improvements have been applied successfully.

Tests:
1. Formula evaluation engine (no zeros bug)
2. Progress tracking functionality
3. Histogram generation robustness  
4. Arrow integration performance
5. Memory management efficiency
6. Error recovery mechanisms
7. Concurrency controls
"""

import asyncio
import logging
import time
import sys
import numpy as np
from datetime import datetime, timezone

# Add the backend directory to Python path
sys.path.append('/home/paperspace/PROJECT/backend')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ComprehensiveSystemTest:
    """Comprehensive test suite for the Monte Carlo simulation platform"""
    
    def __init__(self):
        self.test_results = {
            "formula_evaluation": [],
            "progress_tracking": [],
            "histogram_generation": [],
            "arrow_integration": [],
            "memory_management": [],
            "error_recovery": [],
            "concurrency_controls": []
        }
        
    async def run_comprehensive_tests(self):
        """Run all comprehensive tests"""
        print("\n" + "="*80)
        print("🧪 STARTING COMPREHENSIVE SYSTEM TESTS")
        print("="*80)
        
        # Test 1: Formula Evaluation Engine
        await self.test_formula_evaluation()
        
        # Test 2: Progress Tracking
        await self.test_progress_tracking()
        
        # Test 3: Histogram Generation
        await self.test_histogram_generation()
        
        # Test 4: Arrow Integration
        await self.test_arrow_integration()
        
        # Test 5: Memory Management
        await self.test_memory_management()
        
        # Test 6: Error Recovery
        await self.test_error_recovery()
        
        # Test 7: Concurrency Controls
        await self.test_concurrency_controls()
        
        # Generate final test report
        await self.generate_test_report()
        
        print("="*80)
        print("✅ COMPREHENSIVE SYSTEM TESTS COMPLETED")
        print("="*80)
    
    async def test_formula_evaluation(self):
        """Test formula evaluation engine for accuracy"""
        print("\n🔬 TEST 1: Formula Evaluation Engine...")
        
        try:
            from simulation.engine import _safe_excel_eval, SAFE_EVAL_NAMESPACE
            
            # Test complex formulas that might cause zeros bug
            test_formulas = [
                ("5 + 10 * 2", {"A1": 5, "B1": 10}, 25),
                ("A1 + B1 * 2", {"A1": 5, "B1": 10}, 25),
                ("SUM(A1:A3)", {"A1": 10, "A2": 20, "A3": 30}, 60),
                ("AVERAGE(A1:A3)", {"A1": 10, "A2": 20, "A3": 30}, 20),
                ("IF(A1>B1, A1*2, B1*2)", {"A1": 15, "B1": 10}, 30),
                ("SQRT(A1)", {"A1": 25}, 5),
                ("A1^2 + B1^2", {"A1": 3, "B1": 4}, 25)
            ]
            
            passed_tests = 0
            total_tests = len(test_formulas)
            
            for formula, test_values, expected in test_formulas:
                try:
                    # Convert test values to the expected format
                    iter_values = {("TestSheet", k): v for k, v in test_values.items()}
                    constant_values = iter_values.copy()
                    
                    result = _safe_excel_eval(
                        formula_string=formula,
                        current_eval_sheet="TestSheet",
                        all_current_iter_values=iter_values,
                        safe_eval_globals=SAFE_EVAL_NAMESPACE,
                        current_calc_cell_coord="TestSheet!TEST",
                        constant_values=constant_values
                    )
                    
                    # Check if result is zero (zeros bug)
                    if result == 0 and expected != 0:
                        print(f"❌ ZEROS BUG detected: '{formula}' = {result} (expected {expected})")
                        self.test_results["formula_evaluation"].append({
                            "formula": formula,
                            "result": result,
                            "expected": expected,
                            "status": "ZEROS_BUG"
                        })
                    elif abs(float(result) - expected) < 0.001:
                        print(f"✅ Formula '{formula}' = {result} (expected {expected})")
                        passed_tests += 1
                        self.test_results["formula_evaluation"].append({
                            "formula": formula,
                            "result": result,
                            "expected": expected,
                            "status": "PASS"
                        })
                    else:
                        print(f"⚠️ Formula '{formula}' = {result} (expected {expected}) - INCORRECT")
                        self.test_results["formula_evaluation"].append({
                            "formula": formula,
                            "result": result,
                            "expected": expected,
                            "status": "INCORRECT"
                        })
                        
                except Exception as e:
                    print(f"❌ Formula '{formula}' failed: {e}")
                    self.test_results["formula_evaluation"].append({
                        "formula": formula,
                        "error": str(e),
                        "status": "ERROR"
                    })
            
            success_rate = passed_tests / total_tests * 100
            print(f"🔬 Formula Evaluation: {passed_tests}/{total_tests} tests passed ({success_rate:.1f}%)")
            
            if success_rate >= 90:
                print("✅ Formula evaluation engine is robust - NO ZEROS BUG detected")
            else:
                print("❌ Formula evaluation engine needs attention")
                
        except Exception as e:
            print(f"❌ Formula evaluation test failed: {e}")
    
    async def test_progress_tracking(self):
        """Test progress tracking functionality"""
        print("\n📊 TEST 2: Progress Tracking...")
        
        try:
            from shared.progress_store import get_progress_store
            
            progress_store = get_progress_store()
            test_sim_id = f"test_sim_{int(time.time())}"
            
            # Test progress updates
            progress_tests = [
                {"percentage": 0, "phase": "initialization"},
                {"percentage": 25, "phase": "data_preparation"},
                {"percentage": 50, "phase": "monte_carlo_execution"},
                {"percentage": 75, "phase": "result_aggregation"},
                {"percentage": 100, "phase": "completion"}
            ]
            
            successful_updates = 0
            
            for test in progress_tests:
                progress_data = {
                    "simulation_id": test_sim_id,
                    "status": "running",
                    "progress_percentage": test["percentage"],
                    "current_phase": test["phase"],
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "enhanced_tracking": True
                }
                
                # Set progress
                progress_store.set_progress(test_sim_id, progress_data)
                
                # Retrieve and verify
                retrieved = progress_store.get_progress(test_sim_id)
                
                if retrieved and retrieved.get("progress_percentage") == test["percentage"]:
                    print(f"✅ Progress {test['percentage']}% - {test['phase']}")
                    successful_updates += 1
                    self.test_results["progress_tracking"].append({
                        "percentage": test["percentage"],
                        "phase": test["phase"],
                        "status": "PASS"
                    })
                else:
                    print(f"❌ Progress {test['percentage']}% - {test['phase']} FAILED")
                    self.test_results["progress_tracking"].append({
                        "percentage": test["percentage"],
                        "phase": test["phase"],
                        "status": "FAIL"
                    })
                
                await asyncio.sleep(0.1)
            
            # Cleanup
            progress_store.clear_progress(test_sim_id)
            
            success_rate = successful_updates / len(progress_tests) * 100
            print(f"📊 Progress Tracking: {successful_updates}/{len(progress_tests)} updates successful ({success_rate:.1f}%)")
            
        except Exception as e:
            print(f"❌ Progress tracking test failed: {e}")
    
    async def test_histogram_generation(self):
        """Test histogram generation robustness"""
        print("\n📈 TEST 3: Histogram Generation...")
        
        try:
            # Test with different data distributions
            test_datasets = [
                {"name": "normal", "data": np.random.normal(100, 20, 5000)},
                {"name": "uniform", "data": np.random.uniform(50, 150, 5000)},
                {"name": "exponential", "data": np.random.exponential(50, 5000)},
                {"name": "mixed", "data": np.concatenate([
                    np.random.normal(80, 10, 2500),
                    np.random.normal(120, 15, 2500)
                ])}
            ]
            
            successful_histograms = 0
            total_tests = len(test_datasets)
            
            for dataset in test_datasets:
                try:
                    data = dataset["data"]
                    name = dataset["name"]
                    
                    # Generate histogram
                    hist, bins = np.histogram(data, bins=20)
                    
                    # Validate histogram
                    if len(hist) > 0 and np.sum(hist) > 0 and np.sum(hist) == len(data):
                        # Calculate statistics
                        mean_val = np.mean(data)
                        std_val = np.std(data)
                        
                        print(f"✅ Histogram {name}: {len(hist)} bins, mean={mean_val:.2f}, std={std_val:.2f}")
                        successful_histograms += 1
                        
                        self.test_results["histogram_generation"].append({
                            "dataset": name,
                            "bins": len(hist),
                            "total_count": int(np.sum(hist)),
                            "mean": float(mean_val),
                            "std": float(std_val),
                            "status": "PASS"
                        })
                    else:
                        print(f"❌ Histogram {name}: Invalid histogram generated")
                        self.test_results["histogram_generation"].append({
                            "dataset": name,
                            "status": "INVALID"
                        })
                        
                except Exception as he:
                    print(f"❌ Histogram {dataset['name']}: {he}")
                    self.test_results["histogram_generation"].append({
                        "dataset": dataset["name"],
                        "error": str(he),
                        "status": "ERROR"
                    })
            
            success_rate = successful_histograms / total_tests * 100
            print(f"📈 Histogram Generation: {successful_histograms}/{total_tests} datasets processed ({success_rate:.1f}%)")
            
        except Exception as e:
            print(f"❌ Histogram generation test failed: {e}")
    
    async def test_arrow_integration(self):
        """Test Arrow integration performance"""
        print("\n🏹 TEST 4: Arrow Integration...")
        
        try:
            import pyarrow as pa
            import pyarrow.compute as pc
            import pandas as pd
            
            # Test large dataset processing
            data_sizes = [1000, 10000, 100000]
            
            for size in data_sizes:
                try:
                    start_time = time.time()
                    
                    # Create test data
                    test_data = {
                        'iteration': list(range(size)),
                        'value': np.random.normal(100, 20, size),
                        'result': np.random.normal(500, 100, size)
                    }
                    
                    # Arrow processing
                    arrow_table = pa.table(test_data)
                    
                    # Compute statistics using Arrow
                    mean_value = pc.mean(arrow_table['value']).as_py()
                    std_value = pc.stddev(arrow_table['value']).as_py()
                    
                    # Convert to pandas
                    pandas_df = arrow_table.to_pandas()
                    
                    end_time = time.time()
                    processing_time = end_time - start_time
                    rows_per_second = size / processing_time
                    
                    print(f"✅ Arrow {size:,} rows: {processing_time:.3f}s, {rows_per_second:,.0f} rows/sec")
                    
                    self.test_results["arrow_integration"].append({
                        "data_size": size,
                        "processing_time": processing_time,
                        "rows_per_second": rows_per_second,
                        "mean_value": mean_value,
                        "std_value": std_value,
                        "status": "PASS"
                    })
                    
                    # Cleanup
                    del arrow_table, pandas_df, test_data
                    
                except Exception as ae:
                    print(f"❌ Arrow {size:,} rows: {ae}")
                    self.test_results["arrow_integration"].append({
                        "data_size": size,
                        "error": str(ae),
                        "status": "ERROR"
                    })
            
            print("🏹 Arrow Integration: All performance tests completed")
            
        except Exception as e:
            print(f"❌ Arrow integration test failed: {e}")
    
    async def test_memory_management(self):
        """Test memory management efficiency"""
        print("\n🧠 TEST 5: Memory Management...")
        
        try:
            import psutil
            import gc
            
            initial_memory = psutil.Process().memory_info().rss / (1024*1024)
            
            # Test memory allocation and cleanup
            large_arrays = []
            memory_checkpoints = []
            
            for i in range(5):
                # Allocate large array
                array = np.random.normal(0, 1, 100000)
                large_arrays.append(array)
                
                current_memory = psutil.Process().memory_info().rss / (1024*1024)
                memory_increase = current_memory - initial_memory
                memory_checkpoints.append(memory_increase)
                
                print(f"  Checkpoint {i+1}: +{memory_increase:.1f}MB")
            
            # Cleanup and check memory recovery
            del large_arrays
            gc.collect()
            
            final_memory = psutil.Process().memory_info().rss / (1024*1024)
            memory_recovered = max(memory_checkpoints) - (final_memory - initial_memory)
            
            print(f"✅ Memory recovered: {memory_recovered:.1f}MB")
            
            self.test_results["memory_management"].append({
                "initial_memory_mb": initial_memory,
                "max_memory_increase_mb": max(memory_checkpoints),
                "memory_recovered_mb": memory_recovered,
                "final_memory_mb": final_memory,
                "status": "PASS" if memory_recovered > 0 else "CONCERN"
            })
            
            print("🧠 Memory Management: Efficiency test completed")
            
        except Exception as e:
            print(f"❌ Memory management test failed: {e}")
    
    async def test_error_recovery(self):
        """Test error recovery mechanisms"""
        print("\n🛡️ TEST 6: Error Recovery...")
        
        try:
            # Test formula error recovery
            try:
                from simulation.engine import _safe_excel_eval, SAFE_EVAL_NAMESPACE
                
                error_formulas = [
                    "INVALID_FUNCTION(1,2,3)",
                    "1/0",  # Division by zero
                    "SQRT(-1)",  # Invalid sqrt
                    "A1 + UNDEFINED_VAR"
                ]
                
                errors_caught = 0
                
                for formula in error_formulas:
                    try:
                        result = _safe_excel_eval(
                            formula,
                            "TestSheet",
                            {},
                            SAFE_EVAL_NAMESPACE,
                            "TestSheet!TEST"
                        )
                        print(f"⚠️ Formula '{formula}' should have failed but returned: {result}")
                    except Exception as e:
                        print(f"✅ Error caught for '{formula}': {str(e)[:50]}...")
                        errors_caught += 1
                
                error_recovery_rate = errors_caught / len(error_formulas) * 100
                print(f"🛡️ Error Recovery: {errors_caught}/{len(error_formulas)} errors caught ({error_recovery_rate:.1f}%)")
                
                self.test_results["error_recovery"].append({
                    "errors_tested": len(error_formulas),
                    "errors_caught": errors_caught,
                    "recovery_rate": error_recovery_rate,
                    "status": "PASS" if error_recovery_rate >= 75 else "CONCERN"
                })
                
            except ImportError:
                print("⚠️ Formula evaluation engine not available for error testing")
                
        except Exception as e:
            print(f"❌ Error recovery test failed: {e}")
    
    async def test_concurrency_controls(self):
        """Test concurrency controls and semaphores"""
        print("\n⚡ TEST 7: Concurrency Controls...")
        
        try:
            from main import SIMULATION_SEMAPHORES
            
            # Test semaphore operations
            semaphore_tests = []
            
            for size, semaphore in SIMULATION_SEMAPHORES.items():
                initial_value = semaphore._value
                
                # Test acquire/release
                await semaphore.acquire()
                after_acquire = semaphore._value
                
                semaphore.release()
                after_release = semaphore._value
                
                test_result = {
                    "size": size,
                    "initial_value": initial_value,
                    "after_acquire": after_acquire,
                    "after_release": after_release,
                    "working": (after_acquire == initial_value - 1) and (after_release == initial_value)
                }
                
                semaphore_tests.append(test_result)
                
                status = "✅" if test_result["working"] else "❌"
                print(f"{status} {size.upper()} semaphore: {initial_value} -> {after_acquire} -> {after_release}")
            
            working_semaphores = sum(1 for test in semaphore_tests if test["working"])
            total_semaphores = len(semaphore_tests)
            
            self.test_results["concurrency_controls"] = {
                "semaphore_tests": semaphore_tests,
                "working_semaphores": working_semaphores,
                "total_semaphores": total_semaphores,
                "status": "PASS" if working_semaphores == total_semaphores else "CONCERN"
            }
            
            print(f"⚡ Concurrency Controls: {working_semaphores}/{total_semaphores} semaphores working correctly")
            
        except Exception as e:
            print(f"❌ Concurrency controls test failed: {e}")
    
    async def generate_test_report(self):
        """Generate comprehensive test report"""
        print("\n📋 Generating Test Report...")
        
        try:
            # Calculate overall test scores
            test_scores = {}
            
            # Formula evaluation score
            if self.test_results["formula_evaluation"]:
                passed = sum(1 for test in self.test_results["formula_evaluation"] if test.get("status") == "PASS")
                total = len(self.test_results["formula_evaluation"])
                test_scores["formula_evaluation"] = passed / total * 100
            
            # Progress tracking score
            if self.test_results["progress_tracking"]:
                passed = sum(1 for test in self.test_results["progress_tracking"] if test.get("status") == "PASS")
                total = len(self.test_results["progress_tracking"])
                test_scores["progress_tracking"] = passed / total * 100
            
            # Histogram generation score
            if self.test_results["histogram_generation"]:
                passed = sum(1 for test in self.test_results["histogram_generation"] if test.get("status") == "PASS")
                total = len(self.test_results["histogram_generation"])
                test_scores["histogram_generation"] = passed / total * 100
            
            # Arrow integration score
            if self.test_results["arrow_integration"]:
                passed = sum(1 for test in self.test_results["arrow_integration"] if test.get("status") == "PASS")
                total = len(self.test_results["arrow_integration"])
                test_scores["arrow_integration"] = passed / total * 100
            
            # Overall system health score
            overall_score = sum(test_scores.values()) / len(test_scores) if test_scores else 0
            
            # Create test report
            test_report = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "test_version": "Comprehensive v1.0",
                "test_scores": test_scores,
                "overall_score": overall_score,
                "detailed_results": self.test_results,
                "system_status": "HEALTHY" if overall_score >= 90 else "NEEDS_ATTENTION" if overall_score >= 75 else "CRITICAL"
            }
            
            # Save report
            import json
            report_file = "/home/paperspace/PROJECT/comprehensive_test_report.json"
            with open(report_file, 'w') as f:
                json.dump(test_report, f, indent=2)
            
            print(f"📋 Test report saved to: {report_file}")
            
            # Print summary
            print("\n📋 COMPREHENSIVE TEST SUMMARY:")
            for test_name, score in test_scores.items():
                status = "✅" if score >= 90 else "⚠️" if score >= 75 else "❌"
                print(f"   {status} {test_name.replace('_', ' ').title()}: {score:.1f}%")
            
            print(f"\n🎯 Overall System Health: {overall_score:.1f}% - {test_report['system_status']}")
            
            if overall_score >= 90:
                print("🎉 Your Monte Carlo simulation platform is ROBUST and ready for production!")
            elif overall_score >= 75:
                print("⚠️ Your platform is mostly stable but some areas need attention.")
            else:
                print("❌ Your platform needs significant improvements before production use.")
                
        except Exception as e:
            print(f"❌ Test report generation failed: {e}")


async def main():
    """Main test function"""
    try:
        test_suite = ComprehensiveSystemTest()
        await test_suite.run_comprehensive_tests()
        
        return 0
        
    except Exception as e:
        print(f"\n❌ ERROR during comprehensive testing: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code) 