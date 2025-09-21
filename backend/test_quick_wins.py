"""
Test file for SuperEngine Quick Wins improvements
================================================
This file tests:
1. JIT compilation performance
2. New Excel functions
3. New distribution types
"""

import asyncio
import time
import cupy as cp
import numpy as np
from super_engine.engine import SuperEngine
from super_engine.gpu_kernels import KERNEL_LIBRARY

async def test_jit_performance():
    """Test JIT compilation performance improvement."""
    print("\n=== Testing JIT Compilation Performance ===")
    
    iterations = 10000
    
    # Create engines with and without JIT
    engine_no_jit = SuperEngine(iterations=iterations, use_jit=False)
    engine_with_jit = SuperEngine(iterations=iterations, use_jit=True)
    
    # Test formula
    formula = "=A1 * B1 + C1 / D1 - E1"
    
    # Create test data
    mc_configs = [
        type('Config', (), {'sheet_name': 'Sheet1', 'name': 'A1', 'min_value': 100, 'most_likely': 150, 'max_value': 200}),
        type('Config', (), {'sheet_name': 'Sheet1', 'name': 'B1', 'min_value': 0.8, 'most_likely': 1.0, 'max_value': 1.2}),
        type('Config', (), {'sheet_name': 'Sheet1', 'name': 'C1', 'min_value': 50, 'most_likely': 75, 'max_value': 100}),
        type('Config', (), {'sheet_name': 'Sheet1', 'name': 'D1', 'min_value': 2, 'most_likely': 3, 'max_value': 4}),
        type('Config', (), {'sheet_name': 'Sheet1', 'name': 'E1', 'min_value': 10, 'most_likely': 15, 'max_value': 20}),
    ]
    
    ordered_calc_steps = [
        ('Sheet1', 'F1', formula)
    ]
    
    # Test without JIT
    start_time = time.time()
    result_no_jit = await engine_no_jit.run_simulation(
        mc_configs, ordered_calc_steps, 'Sheet1', 'F1', {}, None
    )
    time_no_jit = time.time() - start_time
    
    # Test with JIT
    start_time = time.time()
    result_with_jit = await engine_with_jit.run_simulation(
        mc_configs, ordered_calc_steps, 'Sheet1', 'F1', {}, None
    )
    time_with_jit = time.time() - start_time
    
    print(f"Without JIT: {time_no_jit:.3f}s")
    print(f"With JIT: {time_with_jit:.3f}s")
    print(f"Speedup: {time_no_jit/time_with_jit:.2f}x")
    print(f"Results match: {np.allclose(result_no_jit['statistics']['mean'], result_with_jit['statistics']['mean'])}")

def test_new_excel_functions():
    """Test new Excel functions."""
    print("\n=== Testing New Excel Functions ===")
    
    # Test financial functions
    print("\n--- Financial Functions ---")
    
    # Test PV
    rate = cp.array([0.05])
    nper = cp.array([10])
    pmt = cp.array([-1000])
    pv_result = KERNEL_LIBRARY['PV'](rate, nper, pmt)
    print(f"PV(5%, 10 years, -$1000/year) = ${float(pv_result[0]):,.2f}")
    
    # Test FV
    fv_result = KERNEL_LIBRARY['FV'](rate, nper, pmt)
    print(f"FV(5%, 10 years, -$1000/year) = ${float(fv_result[0]):,.2f}")
    
    # Test PMT
    pv = cp.array([10000])
    fv = cp.array([0])
    pmt_result = KERNEL_LIBRARY['PMT'](rate, nper, pv, fv)
    print(f"PMT(5%, 10 years, $10000 loan) = ${float(pmt_result[0]):,.2f}")
    
    # Test date/time functions
    print("\n--- Date/Time Functions ---")
    
    # Test with some dates
    dates = cp.array([44927, 44958, 44988, 45019])  # Dec 2022 - Feb 2023
    
    years = KERNEL_LIBRARY['YEAR'](dates)
    months = KERNEL_LIBRARY['MONTH'](dates)
    days = KERNEL_LIBRARY['DAY'](dates)
    
    print(f"YEAR: {years}")
    print(f"MONTH: {months}")
    print(f"DAY: {days}")
    
    today = KERNEL_LIBRARY['TODAY']()
    now = KERNEL_LIBRARY['NOW']()
    print(f"TODAY: {float(today[0]):.0f}")
    print(f"NOW: {float(now[0]):.5f}")

def test_new_distributions():
    """Test new distribution types."""
    print("\n=== Testing New Distribution Types ===")
    
    size = 1000
    
    # Test Poisson
    print("\n--- Poisson Distribution ---")
    poisson_samples = KERNEL_LIBRARY['POISSON'](lam=3.0, size=size)
    print(f"Mean: {float(cp.mean(poisson_samples)):.2f} (expected: 3.0)")
    print(f"Variance: {float(cp.var(poisson_samples)):.2f} (expected: 3.0)")
    
    # Test Binomial
    print("\n--- Binomial Distribution ---")
    binomial_samples = KERNEL_LIBRARY['BINOMIAL'](n=10, p=0.3, size=size)
    print(f"Mean: {float(cp.mean(binomial_samples)):.2f} (expected: 3.0)")
    print(f"Variance: {float(cp.var(binomial_samples)):.2f} (expected: 2.1)")
    
    # Test Student's t
    print("\n--- Student's t Distribution ---")
    t_samples = KERNEL_LIBRARY['STUDENT_T'](df=5, size=size)
    print(f"Mean: {float(cp.mean(t_samples)):.2f} (expected: 0.0)")
    print(f"Variance: {float(cp.var(t_samples)):.2f} (expected: ~1.67)")
    
    # Test PERT
    print("\n--- PERT Distribution ---")
    pert_samples = KERNEL_LIBRARY['PERT'](minimum=10, most_likely=20, maximum=40, size=size)
    print(f"Mean: {float(cp.mean(pert_samples)):.2f}")
    print(f"Min: {float(cp.min(pert_samples)):.2f}")
    print(f"Max: {float(cp.max(pert_samples)):.2f}")
    
    # Test Discrete
    print("\n--- Discrete Distribution ---")
    values = cp.array([10, 20, 30, 40, 50])
    probabilities = cp.array([0.1, 0.2, 0.3, 0.3, 0.1])
    discrete_samples = KERNEL_LIBRARY['DISCRETE'](values, probabilities, size=size)
    print(f"Mean: {float(cp.mean(discrete_samples)):.2f} (expected: 30.0)")
    print(f"Unique values: {cp.unique(discrete_samples)}")
    
    # Test Exponential
    print("\n--- Exponential Distribution ---")
    exp_samples = KERNEL_LIBRARY['EXPONENTIAL'](scale=2.0, size=size)
    print(f"Mean: {float(cp.mean(exp_samples)):.2f} (expected: 2.0)")
    print(f"Variance: {float(cp.var(exp_samples)):.2f} (expected: 4.0)")

def test_jit_with_functions():
    """Test JIT compilation with new functions."""
    print("\n=== Testing JIT with New Functions ===")
    
    from super_engine.jit_compiler import JitCompiler
    
    jit = JitCompiler()
    
    # Test data
    sim_data = {
        "A1": cp.array([100.0, 200.0, 300.0, 400.0, 500.0]),
        "B1": cp.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        "C1": cp.array([10.0, 20.0, 30.0, 40.0, 50.0]),
    }
    
    # Test formulas with new functions
    formulas = [
        ("=ABS(A1 - B1)", "ABS function"),
        ("=SQRT(A1)", "SQRT function"),
        ("=MIN(A1, B1, C1)", "MIN function"),
        ("=MAX(A1, B1, C1)", "MAX function"),
        ("=AVERAGE(A1, B1, C1)", "AVERAGE function"),
        ("=IF(A1 > 200, A1 * 2, A1 / 2)", "IF function"),
        ("=SIN(B1) + COS(B1)", "Trigonometric functions"),
        ("=LOG(A1) + EXP(B1)", "LOG and EXP functions"),
    ]
    
    for formula, description in formulas:
        try:
            result = jit.compile_and_run(formula, sim_data)
            print(f"{description}: {formula}")
            print(f"  Result: {result}")
            print(f"  Mean: {float(cp.mean(result)):.2f}")
        except Exception as e:
            print(f"{description}: {formula} - ERROR: {e}")

async def main():
    """Run all tests."""
    print("SuperEngine Quick Wins Test Suite")
    print("=" * 50)
    
    # Test JIT performance
    await test_jit_performance()
    
    # Test new Excel functions
    test_new_excel_functions()
    
    # Test new distributions
    test_new_distributions()
    
    # Test JIT with new functions
    test_jit_with_functions()
    
    print("\n" + "=" * 50)
    print("All tests completed!")

if __name__ == "__main__":
    asyncio.run(main()) 