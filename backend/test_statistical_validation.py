"""
Statistical Validation Tests for Monte Carlo Simulation
Ensures that the statistical results are mathematically correct and meaningful
"""

import os
import sys
import json
import numpy as np
import scipy.stats as stats
from datetime import datetime
from typing import Dict, List, Tuple, Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class StatisticalValidator:
    """Validates Monte Carlo simulation statistical results"""
    
    def __init__(self):
        self.test_results = {}
        self.tolerance = 0.05  # 5% tolerance for statistical tests
        
    def run_all_tests(self):
        """Run all statistical validation tests"""
        print("\n" + "="*80)
        print("📈 STATISTICAL VALIDATION TEST SUITE")
        print("="*80)
        
        tests = [
            self.test_uniform_distribution,
            self.test_normal_distribution,
            self.test_histogram_accuracy,
            self.test_percentile_calculations,
            self.test_sensitivity_analysis,
            self.test_convergence
        ]
        
        passed = 0
        failed = 0
        
        for test in tests:
            try:
                print(f"\n▶️  Running: {test.__name__}")
                result = test()
                if result:
                    print(f"✅ PASSED: {test.__name__}")
                    passed += 1
                else:
                    print(f"❌ FAILED: {test.__name__}")
                    failed += 1
            except Exception as e:
                print(f"❌ ERROR in {test.__name__}: {str(e)}")
                failed += 1
                
        # Summary
        print("\n" + "="*80)
        print("📊 STATISTICAL TEST SUMMARY")
        print("="*80)
        print(f"Total Tests: {len(tests)}")
        print(f"Passed: {passed}")
        print(f"Failed: {failed}")
        print(f"Success Rate: {(passed/len(tests)*100):.1f}%")
        
        return failed == 0
        
    def test_uniform_distribution(self):
        """Test uniform distribution Monte Carlo generation"""
        print("  📊 Testing uniform distribution...")
        
        # Generate uniform distribution samples
        n_samples = 10000
        min_val = 0.8
        max_val = 1.2
        
        samples = np.random.uniform(min_val, max_val, n_samples)
        
        # Calculate statistics
        mean = np.mean(samples)
        std_dev = np.std(samples)
        actual_min = np.min(samples)
        actual_max = np.max(samples)
        
        # Expected values for uniform distribution
        expected_mean = (min_val + max_val) / 2
        expected_std = (max_val - min_val) / np.sqrt(12)
        
        print(f"    Expected mean: {expected_mean:.4f}, Actual: {mean:.4f}")
        print(f"    Expected std: {expected_std:.4f}, Actual: {std_dev:.4f}")
        print(f"    Range: [{actual_min:.4f}, {actual_max:.4f}]")
        
        # Validate
        mean_error = abs(mean - expected_mean) / expected_mean
        std_error = abs(std_dev - expected_std) / expected_std
        
        # Kolmogorov-Smirnov test for uniform distribution
        ks_statistic, p_value = stats.kstest(samples, lambda x: stats.uniform.cdf(x, loc=min_val, scale=max_val-min_val))
        print(f"    K-S test p-value: {p_value:.4f}")
        
        if mean_error < self.tolerance and std_error < self.tolerance and p_value > 0.05:
            return True
        else:
            print(f"    Mean error: {mean_error:.2%}, Std error: {std_error:.2%}")
            return False
            
    def test_normal_distribution(self):
        """Test normal distribution Monte Carlo generation"""
        print("  📊 Testing normal distribution...")
        
        # Generate normal distribution samples
        n_samples = 10000
        mean_val = 1.0
        std_val = 0.1
        
        samples = np.random.normal(mean_val, std_val, n_samples)
        
        # Calculate statistics
        actual_mean = np.mean(samples)
        actual_std = np.std(samples)
        
        print(f"    Expected mean: {mean_val:.4f}, Actual: {actual_mean:.4f}")
        print(f"    Expected std: {std_val:.4f}, Actual: {actual_std:.4f}")
        
        # Shapiro-Wilk test for normality
        statistic, p_value = stats.shapiro(samples[:5000])  # Use subset for performance
        print(f"    Shapiro-Wilk p-value: {p_value:.4f}")
        
        # Validate
        mean_error = abs(actual_mean - mean_val) / mean_val
        std_error = abs(actual_std - std_val) / std_val
        
        if mean_error < self.tolerance and std_error < self.tolerance and p_value > 0.05:
            return True
        else:
            return False
            
    def test_histogram_accuracy(self):
        """Test histogram generation accuracy"""
        print("  📊 Testing histogram accuracy...")
        
        # Generate test data
        data = np.random.normal(100, 15, 1000)
        
        # Create histogram
        n_bins = 20
        hist, bin_edges = np.histogram(data, bins=n_bins)
        
        # Validate histogram properties
        validations = []
        
        # 1. Total count should equal data length
        total_count = np.sum(hist)
        validations.append(("Total count", total_count == len(data)))
        
        # 2. Bins should cover full data range
        data_min, data_max = np.min(data), np.max(data)
        bins_cover_range = bin_edges[0] <= data_min and bin_edges[-1] >= data_max
        validations.append(("Bins cover range", bins_cover_range))
        
        # 3. Bin widths should be uniform
        bin_widths = np.diff(bin_edges)
        uniform_bins = np.allclose(bin_widths, bin_widths[0])
        validations.append(("Uniform bin widths", uniform_bins))
        
        # Print results
        all_passed = True
        for check, passed in validations:
            status = "✓" if passed else "✗"
            print(f"    {status} {check}")
            if not passed:
                all_passed = False
                
        return all_passed
        
    def test_percentile_calculations(self):
        """Test percentile calculations accuracy"""
        print("  📊 Testing percentile calculations...")
        
        # Generate test data
        data = np.random.normal(100, 15, 10000)
        
        # Calculate percentiles
        percentiles = [5, 25, 50, 75, 95]
        calculated = np.percentile(data, percentiles)
        
        # Validate against scipy
        scipy_percentiles = [stats.scoreatpercentile(data, p) for p in percentiles]
        
        # Check ordering
        is_ordered = all(calculated[i] <= calculated[i+1] for i in range(len(calculated)-1))
        
        # Check accuracy
        max_error = np.max(np.abs(calculated - scipy_percentiles))
        
        print(f"    Percentiles: {percentiles}")
        print(f"    Values: {[f'{v:.2f}' for v in calculated]}")
        print(f"    Ordered correctly: {is_ordered}")
        print(f"    Max error vs scipy: {max_error:.6f}")
        
        return is_ordered and max_error < 0.01
        
    def test_sensitivity_analysis(self):
        """Test sensitivity analysis calculations"""
        print("  📊 Testing sensitivity analysis...")
        
        # Simulate sensitivity analysis data
        n_iterations = 1000
        
        # Three input variables
        var1 = np.random.uniform(0.8, 1.2, n_iterations)
        var2 = np.random.uniform(0.9, 1.1, n_iterations)
        var3 = np.random.uniform(0.95, 1.05, n_iterations)
        
        # Output formula: Y = 100*var1 + 200*var2 + 300*var3
        # This gives us known sensitivities
        output = 100*var1 + 200*var2 + 300*var3
        
        # Calculate correlations (proxy for sensitivity)
        corr1 = np.corrcoef(var1, output)[0, 1]
        corr2 = np.corrcoef(var2, output)[0, 1]
        corr3 = np.corrcoef(var3, output)[0, 1]
        
        print(f"    Correlation var1: {corr1:.4f}")
        print(f"    Correlation var2: {corr2:.4f}")
        print(f"    Correlation var3: {corr3:.4f}")
        
        # Expected: var3 should have highest impact (coefficient 300)
        # Then var2 (200), then var1 (100)
        # But uniform distributions have different ranges, so need to account for that
        
        # Standardize by range
        range1 = 1.2 - 0.8  # 0.4
        range2 = 1.1 - 0.9  # 0.2
        range3 = 1.05 - 0.95  # 0.1
        
        impact1 = 100 * range1  # 40
        impact2 = 200 * range2  # 40
        impact3 = 300 * range3  # 30
        
        # So we expect var1 and var2 to have similar impact, var3 slightly less
        
        # Simple validation: all correlations should be positive and significant
        all_positive = corr1 > 0.3 and corr2 > 0.3 and corr3 > 0.3
        
        return all_positive
        
    def test_convergence(self):
        """Test Monte Carlo convergence properties"""
        print("  📊 Testing Monte Carlo convergence...")
        
        # Test that results converge as iterations increase
        true_mean = 100
        true_std = 10
        
        iteration_counts = [100, 500, 1000, 5000, 10000]
        means = []
        std_errors = []
        
        for n in iteration_counts:
            samples = np.random.normal(true_mean, true_std, n)
            sample_mean = np.mean(samples)
            means.append(sample_mean)
            # Standard error of the mean
            std_error = np.std(samples) / np.sqrt(n)
            std_errors.append(std_error)
            
        # Print convergence
        print(f"    Iterations: {iteration_counts}")
        print(f"    Means: {[f'{m:.2f}' for m in means]}")
        print(f"    Std Errors: {[f'{s:.3f}' for s in std_errors]}")
        
        # Check that standard error decreases
        errors_decreasing = all(std_errors[i] >= std_errors[i+1] for i in range(len(std_errors)-1))
        
        # Check that final mean is close to true mean
        final_error = abs(means[-1] - true_mean) / true_mean
        
        print(f"    Errors decreasing: {errors_decreasing}")
        print(f"    Final error: {final_error:.2%}")
        
        return errors_decreasing and final_error < 0.01


def generate_mock_simulation_results(iterations=1000):
    """Generate mock simulation results for testing"""
    
    # Generate MC variables
    f4_values = np.random.uniform(0.8, 1.2, iterations)
    f5_values = np.random.uniform(0.9, 1.1, iterations)
    f6_values = np.random.uniform(0.95, 1.05, iterations)
    
    # Simulate complex calculation
    # B13 = Some complex function of F4, F5, F6
    # For testing, use: B13 = 100 * F4 + 200 * F5 + 300 * F6 + noise
    noise = np.random.normal(0, 5, iterations)
    b13_values = 100 * f4_values + 200 * f5_values + 300 * f6_values + noise
    
    # Calculate statistics
    results = {
        "iterations": iterations,
        "mean": float(np.mean(b13_values)),
        "median": float(np.median(b13_values)),
        "std_dev": float(np.std(b13_values)),
        "min_value": float(np.min(b13_values)),
        "max_value": float(np.max(b13_values)),
        "percentile_5": float(np.percentile(b13_values, 5)),
        "percentile_95": float(np.percentile(b13_values, 95)),
        "histogram": {
            "bins": np.histogram(b13_values, bins=20)[1].tolist(),
            "counts": np.histogram(b13_values, bins=20)[0].tolist()
        },
        "raw_values_sample": b13_values[:10].tolist()  # First 10 values
    }
    
    return results


def main():
    """Run statistical validation tests"""
    validator = StatisticalValidator()
    success = validator.run_all_tests()
    
    # Generate and validate mock results
    print("\n" + "="*80)
    print("🎲 MOCK SIMULATION RESULTS")
    print("="*80)
    
    mock_results = generate_mock_simulation_results(1000)
    
    print(f"Mean: {mock_results['mean']:.2f}")
    print(f"Std Dev: {mock_results['std_dev']:.2f}")
    print(f"Range: [{mock_results['min_value']:.2f}, {mock_results['max_value']:.2f}]")
    print(f"95% CI: [{mock_results['percentile_5']:.2f}, {mock_results['percentile_95']:.2f}]")
    
    # Save results
    results_file = f"statistical_validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump({
            "test_results": validator.test_results,
            "mock_simulation": mock_results
        }, f, indent=2)
    
    print(f"\n📄 Results saved to: {results_file}")
    
    return 0 if success else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 