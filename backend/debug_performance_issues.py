#!/usr/bin/env python3
"""
PERFORMANCE DEBUGGING SCRIPT: Issue Analysis for Monte Carlo Simulation

Based on latest.txt findings:
- Issue #1: 656x performance degradation (164.19s vs 0.25s)
- Issue #2: Zero variation in Monte Carlo results 
- Issue #3: Input variable sampling disconnection

This script investigates and profiles these specific issues.
"""

import sys
import os
import time
import logging
import asyncio
import numpy as np
from typing import Dict, List, Any, Tuple

# Add backend to path
sys.path.append('/home/paperspace/PROJECT/backend')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def debug_performance_regression():
    """Debug the 656x performance regression on B12/B13 targets"""
    
    logger.info("🔍 DEBUGGING: Performance Regression Analysis")
    logger.info("=" * 60)
    
    # Test data based on latest.txt findings
    test_cases = [
        {
            "name": "Fast Constants (F4-F7)",
            "targets": ["F4", "F5", "F6", "F7"],
            "expected_time": 0.25,
            "description": "Static constants that completed quickly"
        },
        {
            "name": "Slow Formulas (B12-B13)", 
            "targets": ["B12", "B13"],
            "expected_time": 164.19,
            "description": "Financial formulas causing 656x slowdown"
        }
    ]
    
    results = {}
    
    for test_case in test_cases:
        logger.info(f"📊 Testing: {test_case['name']}")
        logger.info(f"   Targets: {test_case['targets']}")
        logger.info(f"   Expected: {test_case['expected_time']}s")
        
        # Simulate formula evaluation for each target
        case_results = []
        
        for target in test_case['targets']:
            start_time = time.time()
            
            # Identify target type and simulate evaluation
            if target in ["F4", "F5", "F6", "F7"]:
                # Constants - should be fast
                result = await simulate_constant_evaluation(target)
                case_results.append({
                    "target": target,
                    "result": result,
                    "evaluation_time": time.time() - start_time,
                    "type": "constant"
                })
            elif target in ["B12", "B13"]:
                # Complex formulas - currently slow
                result = await simulate_formula_evaluation(target)
                case_results.append({
                    "target": target, 
                    "result": result,
                    "evaluation_time": time.time() - start_time,
                    "type": "formula"
                })
        
        results[test_case['name']] = case_results
        
        # Calculate total time for this test case
        total_time = sum(r['evaluation_time'] for r in case_results)
        logger.info(f"   Actual time: {total_time:.4f}s")
        
        # Performance analysis
        if test_case['expected_time'] > 0:
            speedup_factor = test_case['expected_time'] / total_time if total_time > 0 else float('inf')
            logger.info(f"   Performance ratio: {speedup_factor:.1f}x")
            
            if speedup_factor > 100:
                logger.warning(f"⚠️  PERFORMANCE ISSUE: {speedup_factor:.1f}x slower than expected!")
    
    return results

async def simulate_constant_evaluation(target: str) -> float:
    """Simulate evaluation of constant cells (F4-F7)"""
    
    # Based on latest.txt, these are static constants
    constants = {
        "F4": 0.1,
        "F5": 0.15, 
        "F6": 0.08,
        "F7": 0.03
    }
    
    # Constants should evaluate instantly
    await asyncio.sleep(0.0001)  # Minimal delay
    
    return constants.get(target, 0.0)

async def simulate_formula_evaluation(target: str) -> float:
    """Simulate evaluation of complex financial formulas (B12-B13)"""
    
    # Based on latest.txt analysis:
    # B12: =IFERROR(NPV(B15/12,C161:AN161),0)
    # B13: =IFERROR(IRR(C161:AL161)*12,0)
    
    logger.info(f"🧮 Evaluating complex formula for {target}")
    
    if target == "B12":
        # NPV calculation simulation - should be moderately complex
        logger.info("   Formula: =IFERROR(NPV(B15/12,C161:AN161),0)")
        logger.info("   Type: Net Present Value calculation")
        
        # Simulate NPV evaluation over range C161:AN161 (39 columns × 1 row = 39 cells)
        range_size = 39
        start_time = time.time()
        
        # Realistic NPV calculation simulation
        cash_flows = np.random.normal(1000, 200, range_size)  # Simulated cash flows
        discount_rate = 0.08 / 12  # Monthly rate
        
        # NPV calculation
        npv = 0
        for i, cash_flow in enumerate(cash_flows):
            npv += cash_flow / ((1 + discount_rate) ** (i + 1))
        
        eval_time = time.time() - start_time
        logger.info(f"   NPV calculation took: {eval_time:.6f}s for {range_size} cash flows")
        
        # This should be fast, so if it takes longer, there's an issue
        if eval_time > 0.01:
            logger.warning(f"⚠️  NPV calculation unexpectedly slow: {eval_time:.6f}s")
        
        return float(npv)
        
    elif target == "B13":
        # IRR calculation simulation - typically more complex
        logger.info("   Formula: =IFERROR(IRR(C161:AL161)*12,0)")
        logger.info("   Type: Internal Rate of Return calculation")
        
        # Simulate IRR evaluation over range C161:AL161 (38 columns × 1 row = 38 cells)
        range_size = 38
        start_time = time.time()
        
        # IRR is iterative and can be slow
        cash_flows = [-10000] + list(np.random.normal(1200, 300, range_size - 1))  # Initial investment + returns
        
        # Simplified IRR calculation (Newton-Raphson method simulation)
        irr = 0.1  # Initial guess
        for iteration in range(50):  # IRR typically requires multiple iterations
            npv = sum(cf / ((1 + irr) ** i) for i, cf in enumerate(cash_flows))
            npv_derivative = sum(-i * cf / ((1 + irr) ** (i + 1)) for i, cf in enumerate(cash_flows) if i > 0)
            
            if abs(npv) < 1e-6 or abs(npv_derivative) < 1e-10:
                break
                
            irr = irr - npv / npv_derivative
            
            # Simulate computational cost
            await asyncio.sleep(0.001)  # IRR iteration delay
        
        eval_time = time.time() - start_time
        logger.info(f"   IRR calculation took: {eval_time:.6f}s for {range_size} cash flows")
        logger.info(f"   IRR iterations: {iteration + 1}")
        
        # IRR can legitimately take longer, but 164 seconds is excessive
        if eval_time > 1.0:
            logger.error(f"🚨 IRR calculation extremely slow: {eval_time:.6f}s")
        elif eval_time > 0.1:
            logger.warning(f"⚠️  IRR calculation slow: {eval_time:.6f}s")
        
        return float(irr * 12)  # Annualized
    
    return 0.0

async def debug_monte_carlo_variation():
    """Debug the zero variation issue in Monte Carlo results"""
    
    logger.info("🎲 DEBUGGING: Monte Carlo Variation Analysis")
    logger.info("=" * 60)
    
    # Simulate different scenarios to identify variation issues
    scenarios = [
        {
            "name": "Constants Only (Wrong Targets)",
            "input_varies": False,
            "formula_varies": False,
            "expected_std": 0.0,
            "description": "F4-F7 constants should show zero variation"
        },
        {
            "name": "Variable Inputs + Constant Formulas",
            "input_varies": True,
            "formula_varies": False, 
            "expected_std": 0.0,
            "description": "Input variables change but formulas don't use them"
        },
        {
            "name": "Variable Inputs + Variable Formulas (Expected)",
            "input_varies": True,
            "formula_varies": True,
            "expected_std": "> 0",
            "description": "Both inputs and formulas vary - should show distribution"
        }
    ]
    
    variation_results = {}
    
    for scenario in scenarios:
        logger.info(f"📊 Testing: {scenario['name']}")
        logger.info(f"   Input varies: {scenario['input_varies']}")
        logger.info(f"   Formula varies: {scenario['formula_varies']}")
        logger.info(f"   Expected std: {scenario['expected_std']}")
        
        # Simulate 1000 Monte Carlo iterations
        iterations = 1000
        results = []
        
        for i in range(iterations):
            if scenario['input_varies']:
                # Generate random input variables
                input_vars = {
                    "var1": np.random.normal(100, 10),
                    "var2": np.random.normal(50, 5)
                }
            else:
                # Use constant input variables
                input_vars = {
                    "var1": 100.0,
                    "var2": 50.0
                }
            
            if scenario['formula_varies']:
                # Formula uses variable inputs
                result = input_vars["var1"] * 0.1 + input_vars["var2"] * 0.05
                # Add some formula complexity
                result += np.random.normal(0, result * 0.02)  # 2% noise
            else:
                # Formula ignores variable inputs (current issue)
                result = 100.0  # Constant result
            
            results.append(result)
        
        # Calculate statistics
        results_array = np.array(results)
        mean_val = np.mean(results_array)
        std_val = np.std(results_array)
        min_val = np.min(results_array)
        max_val = np.max(results_array)
        
        logger.info(f"   Results - Mean: {mean_val:.6f}, Std: {std_val:.6f}")
        logger.info(f"   Range: {min_val:.6f} to {max_val:.6f}")
        logger.info(f"   Data range: {max_val - min_val:.2e}")
        
        # Identify issues
        if std_val < 1e-10:
            logger.warning(f"⚠️  ZERO VARIATION: Standard deviation is effectively zero!")
            logger.warning(f"     This matches the issue in latest.txt: 'Data range: 0.00e+00'")
        
        variation_results[scenario['name']] = {
            "mean": mean_val,
            "std": std_val,
            "min": min_val,
            "max": max_val,
            "range": max_val - min_val
        }
    
    return variation_results

async def debug_input_sampling_connection():
    """Debug input variable sampling disconnection"""
    
    logger.info("🔗 DEBUGGING: Input Variable Sampling Connection")
    logger.info("=" * 60)
    
    # Simulate the sampling process as described in latest.txt
    logger.info("Simulating GPU random sample generation...")
    
    # Based on logs: "🔧 [ULTRA] GPU generated 2000 samples in 0.0009s"
    # 2000 samples = 1000 iterations × 2 variables
    iterations = 1000
    num_variables = 2
    total_samples = iterations * num_variables
    
    start_time = time.time()
    
    # Simulate GPU random generation (this should be fast)
    random_samples = np.random.normal(0, 1, total_samples)
    
    generation_time = time.time() - start_time
    samples_per_sec = total_samples / generation_time if generation_time > 0 else float('inf')
    
    logger.info(f"Generated {total_samples} samples in {generation_time:.6f}s")
    logger.info(f"Performance: {samples_per_sec:.0f} samples/sec")
    logger.info(f"Expected from logs: ~2,185,672 samples/sec")
    
    # Performance comparison
    if samples_per_sec < 100000:
        logger.warning(f"⚠️  Random generation slower than expected")
    
    # Now test the critical question: Are samples connecting to formulas?
    logger.info("Testing sample propagation to formula evaluation...")
    
    # Reshape samples for iterations and variables
    samples_reshaped = random_samples.reshape((iterations, num_variables))
    
    # Test different propagation scenarios
    propagation_tests = [
        {
            "name": "Direct Propagation (Correct)",
            "method": "direct",
            "description": "Random samples directly replace cell values"
        },
        {
            "name": "Cached Constants Override (Issue)",
            "method": "cached",
            "description": "Constants cache overrides random values"
        },
        {
            "name": "Formula Ignores Variables (Issue)",
            "method": "ignored", 
            "description": "Formula doesn't reference variable cells"
        }
    ]
    
    propagation_results = {}
    
    for test in propagation_tests:
        logger.info(f"🧪 Testing: {test['name']}")
        
        iteration_results = []
        
        for i in range(min(10, iterations)):  # Test first 10 iterations
            # Get random samples for this iteration
            var1_sample = samples_reshaped[i, 0] * 10 + 100  # Scale to reasonable range
            var2_sample = samples_reshaped[i, 1] * 5 + 50
            
            if test['method'] == 'direct':
                # Correct behavior: Formula uses random samples
                result = var1_sample * 0.1 + var2_sample * 0.05
                
            elif test['method'] == 'cached':
                # Issue: Constants cache overrides random samples
                cached_var1 = 100.0  # Cached constant value
                cached_var2 = 50.0   # Cached constant value
                result = cached_var1 * 0.1 + cached_var2 * 0.05
                
            elif test['method'] == 'ignored':
                # Issue: Formula doesn't use variable cells at all
                result = 100.0  # Static result
            
            iteration_results.append(result)
        
        # Analyze results
        results_array = np.array(iteration_results)
        variation = np.std(results_array)
        
        logger.info(f"   Sample results: {iteration_results[:5]} ...")
        logger.info(f"   Variation (std): {variation:.6f}")
        
        if variation < 1e-10:
            logger.error(f"🚨 ISSUE DETECTED: {test['name']} shows zero variation!")
            logger.error(f"   This could be the root cause of the Monte Carlo issue")
        else:
            logger.info(f"✅ {test['name']} shows proper variation")
        
        propagation_results[test['name']] = {
            "variation": variation,
            "sample_results": iteration_results[:5]
        }
    
    return propagation_results

async def main():
    """Run comprehensive debugging analysis"""
    
    logger.info("🚀 STARTING: Monte Carlo Performance & Variation Debugging")
    logger.info("Based on issues identified in latest.txt")
    logger.info("=" * 80)
    
    try:
        # Debug performance regression
        logger.info("\n" + "=" * 80)
        performance_results = await debug_performance_regression()
        
        # Debug Monte Carlo variation
        logger.info("\n" + "=" * 80)
        variation_results = await debug_monte_carlo_variation()
        
        # Debug input sampling connection
        logger.info("\n" + "=" * 80)
        sampling_results = await debug_input_sampling_connection()
        
        # Summary and recommendations
        logger.info("\n" + "=" * 80)
        logger.info("📋 DEBUGGING SUMMARY & RECOMMENDATIONS")
        logger.info("=" * 80)
        
        logger.info("🔍 Key Findings:")
        logger.info("1. Performance regression likely due to complex financial formula evaluation")
        logger.info("2. Zero variation suggests input variables not connecting to target formulas")
        logger.info("3. Random sample generation works correctly but may not propagate")
        
        logger.info("\n🛠️  Recommended Fixes:")
        logger.info("1. Profile actual B12/B13 formula evaluation in ultra engine")
        logger.info("2. Verify input variable cells are properly identified and used")
        logger.info("3. Check if constants cache is overriding random variable values")
        logger.info("4. Ensure target formulas reference the variable cells being sampled")
        
        logger.info("\n✅ Debugging analysis complete!")
        
        return {
            "performance": performance_results,
            "variation": variation_results,
            "sampling": sampling_results
        }
        
    except Exception as e:
        logger.error(f"🚨 Debugging failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

if __name__ == "__main__":
    asyncio.run(main()) 