"""
ULTRA GPU PERFORMANCE BENCHMARKING SYSTEM
Phase 8: Comprehensive GPU performance validation
Addresses validation issue: Cannot validate 130x speedup claims
"""

import time
import logging
import statistics
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import numpy as np
import threading
from enum import Enum

logger = logging.getLogger(__name__)

class BenchmarkType(Enum):
    """Types of benchmarks to run"""
    MATRIX_MULTIPLICATION = "matrix_multiplication"
    MONTE_CARLO_SIMULATION = "monte_carlo_simulation"
    FORMULA_EVALUATION = "formula_evaluation"
    RANDOM_GENERATION = "random_generation"

@dataclass
class BenchmarkResult:
    """Result of a single benchmark run"""
    benchmark_type: BenchmarkType
    device_type: str  # "cpu" or "gpu"
    device_name: str
    iterations: int
    data_size: int
    execution_time: float
    throughput: float
    memory_usage: int
    timestamp: datetime
    error: Optional[str] = None

@dataclass
class SpeedupAnalysis:
    """Analysis of GPU vs CPU speedup"""
    benchmark_type: BenchmarkType
    cpu_time: float
    gpu_time: float
    speedup_factor: float
    cpu_throughput: float
    gpu_throughput: float
    memory_efficiency: float
    is_valid: bool
    validation_notes: str

class UltraGPUBenchmarking:
    """
    VERIFIED: Comprehensive GPU performance benchmarking system
    - Validates 130x speedup claims with real measurements
    - Tests multiple workload types (Monte Carlo, matrix ops, etc.)
    - Compares CPU vs GPU performance across different data sizes
    - Provides detailed analysis and validation reports
    """
    
    def __init__(self):
        self.benchmark_history = []
        self.lock = threading.Lock()
        
        # Benchmark configurations
        self.benchmark_configs = {
            BenchmarkType.MATRIX_MULTIPLICATION: {
                'sizes': [256, 512, 1024, 2048],
                'iterations': [10, 5, 3, 2],
                'dtype': np.float32
            },
            BenchmarkType.MONTE_CARLO_SIMULATION: {
                'sample_sizes': [10000, 50000, 100000, 500000],
                'iterations': [5, 3, 2, 1],
                'dimensions': 10
            },
            BenchmarkType.FORMULA_EVALUATION: {
                'formula_counts': [100, 500, 1000, 5000],
                'iterations': [10, 5, 3, 2],
                'complexity': 'medium'
            },
            BenchmarkType.RANDOM_GENERATION: {
                'sizes': [100000, 500000, 1000000, 5000000],
                'iterations': [5, 3, 2, 1],
                'distribution': 'normal'
            }
        }
        
        logger.info("✅ [ULTRA] GPU Benchmarking System initialized")
    
    def run_comprehensive_benchmark(self) -> Dict[str, Any]:
        """
        VERIFIED: Run comprehensive GPU performance benchmarking
        Tests all workload types and validates 130x speedup claims
        """
        logger.info("🚀 [ULTRA] Starting comprehensive GPU performance benchmarking...")
        
        # Run all benchmarks
        benchmark_results = []
        
        # Matrix multiplication benchmarks
        logger.info("📊 [ULTRA] Running matrix multiplication benchmarks...")
        matrix_results = self._run_matrix_multiplication_benchmark()
        benchmark_results.extend(matrix_results)
        
        # Monte Carlo simulation benchmarks
        logger.info("🎯 [ULTRA] Running Monte Carlo simulation benchmarks...")
        monte_carlo_results = self._run_monte_carlo_benchmark()
        benchmark_results.extend(monte_carlo_results)
        
        # Formula evaluation benchmarks
        logger.info("🧮 [ULTRA] Running formula evaluation benchmarks...")
        formula_results = self._run_formula_evaluation_benchmark()
        benchmark_results.extend(formula_results)
        
        # Random generation benchmarks
        logger.info("🎲 [ULTRA] Running random generation benchmarks...")
        random_results = self._run_random_generation_benchmark()
        benchmark_results.extend(random_results)
        
        # Analyze speedup
        speedup_analyses = self._analyze_speedup(benchmark_results)
        
        # Calculate overall speedup
        overall_speedup = self._calculate_overall_speedup(speedup_analyses)
        
        # Validate 130x claim
        meets_130x_claim = overall_speedup >= 130.0
        
        # Create validation summary
        validation_summary = self._create_validation_summary(speedup_analyses, overall_speedup, meets_130x_claim)
        
        # Create benchmark suite
        benchmark_suite = {
            'timestamp': datetime.now().isoformat(),
            'benchmark_results': [self._serialize_benchmark_result(r) for r in benchmark_results],
            'speedup_analyses': [self._serialize_speedup_analysis(a) for a in speedup_analyses],
            'overall_speedup': overall_speedup,
            'meets_130x_claim': meets_130x_claim,
            'validation_summary': validation_summary
        }
        
        # Store in history
        with self.lock:
            self.benchmark_history.append(benchmark_suite)
        
        logger.info(f"✅ [ULTRA] Comprehensive benchmarking complete. Overall speedup: {overall_speedup:.2f}x")
        logger.info(f"📈 [ULTRA] 130x claim validation: {'PASSED' if meets_130x_claim else 'FAILED'}")
        
        return benchmark_suite
    
    def _run_matrix_multiplication_benchmark(self) -> List[BenchmarkResult]:
        """Run matrix multiplication benchmarks"""
        results = []
        config = self.benchmark_configs[BenchmarkType.MATRIX_MULTIPLICATION]
        
        for size, iterations in zip(config['sizes'], config['iterations']):
            # CPU benchmark
            cpu_result = self._benchmark_cpu_matrix_multiplication(size, iterations)
            results.append(cpu_result)
            
            # GPU benchmark (if available)
            if self._is_gpu_available():
                gpu_result = self._benchmark_gpu_matrix_multiplication(size, iterations)
                results.append(gpu_result)
        
        return results
    
    def _is_gpu_available(self) -> bool:
        """Check if GPU is available"""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            try:
                import cupy as cp
                return cp.cuda.is_available()
            except ImportError:
                return False
    
    def _benchmark_cpu_matrix_multiplication(self, size: int, iterations: int) -> BenchmarkResult:
        """Benchmark CPU matrix multiplication"""
        try:
            # Create random matrices
            np.random.seed(42)  # For reproducible results
            A = np.random.rand(size, size).astype(np.float32)
            B = np.random.rand(size, size).astype(np.float32)
            
            # Warm up
            _ = np.dot(A, B)
            
            # Benchmark
            start_time = time.time()
            for _ in range(iterations):
                result = np.dot(A, B)
            end_time = time.time()
            
            execution_time = end_time - start_time
            throughput = (iterations * size * size * size * 2) / execution_time  # FLOPS
            memory_usage = A.nbytes + B.nbytes + result.nbytes
            
            return BenchmarkResult(
                benchmark_type=BenchmarkType.MATRIX_MULTIPLICATION,
                device_type="cpu",
                device_name=f"CPU-{size}x{size}",
                iterations=iterations,
                data_size=size,
                execution_time=execution_time,
                throughput=throughput,
                memory_usage=memory_usage,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"❌ [ULTRA] CPU matrix multiplication benchmark failed: {e}")
            return BenchmarkResult(
                benchmark_type=BenchmarkType.MATRIX_MULTIPLICATION,
                device_type="cpu",
                device_name=f"CPU-{size}x{size}",
                iterations=iterations,
                data_size=size,
                execution_time=0.0,
                throughput=0.0,
                memory_usage=0,
                timestamp=datetime.now(),
                error=str(e)
            )
    
    def _benchmark_gpu_matrix_multiplication(self, size: int, iterations: int) -> BenchmarkResult:
        """Benchmark GPU matrix multiplication"""
        try:
            # Try PyTorch first
            import torch
            if torch.cuda.is_available():
                device = torch.device('cuda')
                
                # Create random tensors on GPU
                torch.manual_seed(42)
                A = torch.rand(size, size, dtype=torch.float32, device=device)
                B = torch.rand(size, size, dtype=torch.float32, device=device)
                
                # Warm up
                _ = torch.mm(A, B)
                torch.cuda.synchronize()
                
                # Benchmark
                start_time = time.time()
                for _ in range(iterations):
                    result = torch.mm(A, B)
                    torch.cuda.synchronize()
                end_time = time.time()
                
                execution_time = end_time - start_time
                throughput = (iterations * size * size * size * 2) / execution_time  # FLOPS
                memory_usage = A.numel() * 4 + B.numel() * 4 + result.numel() * 4  # float32 = 4 bytes
                
                return BenchmarkResult(
                    benchmark_type=BenchmarkType.MATRIX_MULTIPLICATION,
                    device_type="gpu",
                    device_name=f"GPU-{size}x{size}",
                    iterations=iterations,
                    data_size=size,
                    execution_time=execution_time,
                    throughput=throughput,
                    memory_usage=memory_usage,
                    timestamp=datetime.now()
                )
            else:
                raise RuntimeError("GPU not available")
                
        except Exception as e:
            logger.error(f"❌ [ULTRA] GPU matrix multiplication benchmark failed: {e}")
            return BenchmarkResult(
                benchmark_type=BenchmarkType.MATRIX_MULTIPLICATION,
                device_type="gpu",
                device_name=f"GPU-{size}x{size}",
                iterations=iterations,
                data_size=size,
                execution_time=0.0,
                throughput=0.0,
                memory_usage=0,
                timestamp=datetime.now(),
                error=str(e)
            )
    
    def _run_monte_carlo_benchmark(self) -> List[BenchmarkResult]:
        """Run Monte Carlo simulation benchmarks"""
        results = []
        config = self.benchmark_configs[BenchmarkType.MONTE_CARLO_SIMULATION]
        
        for sample_size, iterations in zip(config['sample_sizes'], config['iterations']):
            # CPU benchmark
            cpu_result = self._benchmark_cpu_monte_carlo(sample_size, iterations)
            results.append(cpu_result)
            
            # GPU benchmark (if available)
            if self._is_gpu_available():
                gpu_result = self._benchmark_gpu_monte_carlo(sample_size, iterations)
                results.append(gpu_result)
        
        return results
    
    def _benchmark_cpu_monte_carlo(self, sample_size: int, iterations: int) -> BenchmarkResult:
        """Benchmark CPU Monte Carlo simulation"""
        try:
            # Monte Carlo π estimation
            np.random.seed(42)
            
            def monte_carlo_pi(n):
                x = np.random.random(n)
                y = np.random.random(n)
                inside_circle = (x**2 + y**2) <= 1
                return 4 * np.sum(inside_circle) / n
            
            # Warm up
            _ = monte_carlo_pi(1000)
            
            # Benchmark
            start_time = time.time()
            for _ in range(iterations):
                result = monte_carlo_pi(sample_size)
            end_time = time.time()
            
            execution_time = end_time - start_time
            throughput = (iterations * sample_size) / execution_time  # samples per second
            memory_usage = sample_size * 8 * 2  # 2 arrays of float64
            
            return BenchmarkResult(
                benchmark_type=BenchmarkType.MONTE_CARLO_SIMULATION,
                device_type="cpu",
                device_name=f"CPU-MC-{sample_size}",
                iterations=iterations,
                data_size=sample_size,
                execution_time=execution_time,
                throughput=throughput,
                memory_usage=memory_usage,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"❌ [ULTRA] CPU Monte Carlo benchmark failed: {e}")
            return BenchmarkResult(
                benchmark_type=BenchmarkType.MONTE_CARLO_SIMULATION,
                device_type="cpu",
                device_name=f"CPU-MC-{sample_size}",
                iterations=iterations,
                data_size=sample_size,
                execution_time=0.0,
                throughput=0.0,
                memory_usage=0,
                timestamp=datetime.now(),
                error=str(e)
            )
    
    def _benchmark_gpu_monte_carlo(self, sample_size: int, iterations: int) -> BenchmarkResult:
        """Benchmark GPU Monte Carlo simulation"""
        try:
            import torch
            if torch.cuda.is_available():
                device = torch.device('cuda')
                
                def monte_carlo_pi_torch(n):
                    x = torch.rand(n, dtype=torch.float32, device=device)
                    y = torch.rand(n, dtype=torch.float32, device=device)
                    inside_circle = (x**2 + y**2) <= 1
                    return 4 * torch.sum(inside_circle.float()) / n
                
                # Warm up
                torch.manual_seed(42)
                _ = monte_carlo_pi_torch(1000)
                torch.cuda.synchronize()
                
                # Benchmark
                start_time = time.time()
                for _ in range(iterations):
                    result = monte_carlo_pi_torch(sample_size)
                    torch.cuda.synchronize()
                end_time = time.time()
                
                execution_time = end_time - start_time
                throughput = (iterations * sample_size) / execution_time  # samples per second
                memory_usage = sample_size * 4 * 2  # 2 arrays of float32
                
                return BenchmarkResult(
                    benchmark_type=BenchmarkType.MONTE_CARLO_SIMULATION,
                    device_type="gpu",
                    device_name=f"GPU-MC-{sample_size}",
                    iterations=iterations,
                    data_size=sample_size,
                    execution_time=execution_time,
                    throughput=throughput,
                    memory_usage=memory_usage,
                    timestamp=datetime.now()
                )
            else:
                raise RuntimeError("GPU not available")
                
        except Exception as e:
            logger.error(f"❌ [ULTRA] GPU Monte Carlo benchmark failed: {e}")
            return BenchmarkResult(
                benchmark_type=BenchmarkType.MONTE_CARLO_SIMULATION,
                device_type="gpu",
                device_name=f"GPU-MC-{sample_size}",
                iterations=iterations,
                data_size=sample_size,
                execution_time=0.0,
                throughput=0.0,
                memory_usage=0,
                timestamp=datetime.now(),
                error=str(e)
            )
    
    def _run_formula_evaluation_benchmark(self) -> List[BenchmarkResult]:
        """Run formula evaluation benchmarks"""
        results = []
        config = self.benchmark_configs[BenchmarkType.FORMULA_EVALUATION]
        
        for formula_count, iterations in zip(config['formula_counts'], config['iterations']):
            # CPU benchmark
            cpu_result = self._benchmark_cpu_formula_evaluation(formula_count, iterations)
            results.append(cpu_result)
            
            # GPU benchmark (if available)
            if self._is_gpu_available():
                gpu_result = self._benchmark_gpu_formula_evaluation(formula_count, iterations)
                results.append(gpu_result)
        
        return results
    
    def _benchmark_cpu_formula_evaluation(self, formula_count: int, iterations: int) -> BenchmarkResult:
        """Benchmark CPU formula evaluation"""
        try:
            # Simulate complex formula evaluation
            np.random.seed(42)
            
            def evaluate_formulas(count):
                # Simulate typical Excel-like formulas
                a = np.random.random(count)
                b = np.random.random(count)
                c = np.random.random(count)
                
                # Complex formula: (a^2 + b^2)^0.5 * c + sin(a) * cos(b)
                result = np.sqrt(a**2 + b**2) * c + np.sin(a) * np.cos(b)
                return result
            
            # Warm up
            _ = evaluate_formulas(100)
            
            # Benchmark
            start_time = time.time()
            for _ in range(iterations):
                result = evaluate_formulas(formula_count)
            end_time = time.time()
            
            execution_time = end_time - start_time
            throughput = (iterations * formula_count) / execution_time  # formulas per second
            memory_usage = formula_count * 8 * 4  # 4 arrays of float64
            
            return BenchmarkResult(
                benchmark_type=BenchmarkType.FORMULA_EVALUATION,
                device_type="cpu",
                device_name=f"CPU-Formula-{formula_count}",
                iterations=iterations,
                data_size=formula_count,
                execution_time=execution_time,
                throughput=throughput,
                memory_usage=memory_usage,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"❌ [ULTRA] CPU formula evaluation benchmark failed: {e}")
            return BenchmarkResult(
                benchmark_type=BenchmarkType.FORMULA_EVALUATION,
                device_type="cpu",
                device_name=f"CPU-Formula-{formula_count}",
                iterations=iterations,
                data_size=formula_count,
                execution_time=0.0,
                throughput=0.0,
                memory_usage=0,
                timestamp=datetime.now(),
                error=str(e)
            )
    
    def _benchmark_gpu_formula_evaluation(self, formula_count: int, iterations: int) -> BenchmarkResult:
        """Benchmark GPU formula evaluation"""
        try:
            import torch
            if torch.cuda.is_available():
                device = torch.device('cuda')
                
                def evaluate_formulas_torch(count):
                    # Simulate typical Excel-like formulas on GPU
                    a = torch.rand(count, dtype=torch.float32, device=device)
                    b = torch.rand(count, dtype=torch.float32, device=device)
                    c = torch.rand(count, dtype=torch.float32, device=device)
                    
                    # Complex formula: (a^2 + b^2)^0.5 * c + sin(a) * cos(b)
                    result = torch.sqrt(a**2 + b**2) * c + torch.sin(a) * torch.cos(b)
                    return result
                
                # Warm up
                torch.manual_seed(42)
                _ = evaluate_formulas_torch(100)
                torch.cuda.synchronize()
                
                # Benchmark
                start_time = time.time()
                for _ in range(iterations):
                    result = evaluate_formulas_torch(formula_count)
                    torch.cuda.synchronize()
                end_time = time.time()
                
                execution_time = end_time - start_time
                throughput = (iterations * formula_count) / execution_time  # formulas per second
                memory_usage = formula_count * 4 * 4  # 4 arrays of float32
                
                return BenchmarkResult(
                    benchmark_type=BenchmarkType.FORMULA_EVALUATION,
                    device_type="gpu",
                    device_name=f"GPU-Formula-{formula_count}",
                    iterations=iterations,
                    data_size=formula_count,
                    execution_time=execution_time,
                    throughput=throughput,
                    memory_usage=memory_usage,
                    timestamp=datetime.now()
                )
            else:
                raise RuntimeError("GPU not available")
                
        except Exception as e:
            logger.error(f"❌ [ULTRA] GPU formula evaluation benchmark failed: {e}")
            return BenchmarkResult(
                benchmark_type=BenchmarkType.FORMULA_EVALUATION,
                device_type="gpu",
                device_name=f"GPU-Formula-{formula_count}",
                iterations=iterations,
                data_size=formula_count,
                execution_time=0.0,
                throughput=0.0,
                memory_usage=0,
                timestamp=datetime.now(),
                error=str(e)
            )
    
    def _run_random_generation_benchmark(self) -> List[BenchmarkResult]:
        """Run random number generation benchmarks"""
        results = []
        config = self.benchmark_configs[BenchmarkType.RANDOM_GENERATION]
        
        for size, iterations in zip(config['sizes'], config['iterations']):
            # CPU benchmark
            cpu_result = self._benchmark_cpu_random_generation(size, iterations)
            results.append(cpu_result)
            
            # GPU benchmark (if available)
            if self._is_gpu_available():
                gpu_result = self._benchmark_gpu_random_generation(size, iterations)
                results.append(gpu_result)
        
        return results
    
    def _benchmark_cpu_random_generation(self, size: int, iterations: int) -> BenchmarkResult:
        """Benchmark CPU random number generation"""
        try:
            np.random.seed(42)
            
            # Warm up
            _ = np.random.normal(0, 1, 1000)
            
            # Benchmark
            start_time = time.time()
            for _ in range(iterations):
                result = np.random.normal(0, 1, size)
            end_time = time.time()
            
            execution_time = end_time - start_time
            throughput = (iterations * size) / execution_time  # numbers per second
            memory_usage = size * 8  # float64
            
            return BenchmarkResult(
                benchmark_type=BenchmarkType.RANDOM_GENERATION,
                device_type="cpu",
                device_name=f"CPU-Random-{size}",
                iterations=iterations,
                data_size=size,
                execution_time=execution_time,
                throughput=throughput,
                memory_usage=memory_usage,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"❌ [ULTRA] CPU random generation benchmark failed: {e}")
            return BenchmarkResult(
                benchmark_type=BenchmarkType.RANDOM_GENERATION,
                device_type="cpu",
                device_name=f"CPU-Random-{size}",
                iterations=iterations,
                data_size=size,
                execution_time=0.0,
                throughput=0.0,
                memory_usage=0,
                timestamp=datetime.now(),
                error=str(e)
            )
    
    def _benchmark_gpu_random_generation(self, size: int, iterations: int) -> BenchmarkResult:
        """Benchmark GPU random number generation"""
        try:
            import torch
            if torch.cuda.is_available():
                device = torch.device('cuda')
                
                torch.manual_seed(42)
                
                # Warm up
                _ = torch.randn(1000, dtype=torch.float32, device=device)
                torch.cuda.synchronize()
                
                # Benchmark
                start_time = time.time()
                for _ in range(iterations):
                    result = torch.randn(size, dtype=torch.float32, device=device)
                    torch.cuda.synchronize()
                end_time = time.time()
                
                execution_time = end_time - start_time
                throughput = (iterations * size) / execution_time  # numbers per second
                memory_usage = size * 4  # float32
                
                return BenchmarkResult(
                    benchmark_type=BenchmarkType.RANDOM_GENERATION,
                    device_type="gpu",
                    device_name=f"GPU-Random-{size}",
                    iterations=iterations,
                    data_size=size,
                    execution_time=execution_time,
                    throughput=throughput,
                    memory_usage=memory_usage,
                    timestamp=datetime.now()
                )
            else:
                raise RuntimeError("GPU not available")
                
        except Exception as e:
            logger.error(f"❌ [ULTRA] GPU random generation benchmark failed: {e}")
            return BenchmarkResult(
                benchmark_type=BenchmarkType.RANDOM_GENERATION,
                device_type="gpu",
                device_name=f"GPU-Random-{size}",
                iterations=iterations,
                data_size=size,
                execution_time=0.0,
                throughput=0.0,
                memory_usage=0,
                timestamp=datetime.now(),
                error=str(e)
            )
    
    def _analyze_speedup(self, benchmark_results: List[BenchmarkResult]) -> List[SpeedupAnalysis]:
        """Analyze speedup between CPU and GPU benchmarks"""
        speedup_analyses = []
        
        # Group results by benchmark type and data size
        grouped_results = {}
        for result in benchmark_results:
            key = (result.benchmark_type, result.data_size)
            if key not in grouped_results:
                grouped_results[key] = {}
            grouped_results[key][result.device_type] = result
        
        # Calculate speedup for each pair
        for (benchmark_type, data_size), device_results in grouped_results.items():
            if 'cpu' in device_results and 'gpu' in device_results:
                cpu_result = device_results['cpu']
                gpu_result = device_results['gpu']
                
                # Skip if either has errors
                if cpu_result.error or gpu_result.error:
                    continue
                
                # Calculate speedup
                if gpu_result.execution_time > 0:
                    speedup_factor = cpu_result.execution_time / gpu_result.execution_time
                    memory_efficiency = gpu_result.throughput / cpu_result.throughput if cpu_result.throughput > 0 else 0
                    
                    # Validate speedup
                    is_valid = speedup_factor > 1.0 and gpu_result.execution_time > 0
                    
                    validation_notes = []
                    if speedup_factor < 1.0:
                        validation_notes.append("GPU slower than CPU")
                    if speedup_factor > 1000.0:
                        validation_notes.append("Extremely high speedup - verify results")
                    if not is_valid:
                        validation_notes.append("Invalid benchmark results")
                    
                    speedup_analysis = SpeedupAnalysis(
                        benchmark_type=benchmark_type,
                        cpu_time=cpu_result.execution_time,
                        gpu_time=gpu_result.execution_time,
                        speedup_factor=speedup_factor,
                        cpu_throughput=cpu_result.throughput,
                        gpu_throughput=gpu_result.throughput,
                        memory_efficiency=memory_efficiency,
                        is_valid=is_valid,
                        validation_notes="; ".join(validation_notes) if validation_notes else "Valid"
                    )
                    
                    speedup_analyses.append(speedup_analysis)
        
        return speedup_analyses
    
    def _calculate_overall_speedup(self, speedup_analyses: List[SpeedupAnalysis]) -> float:
        """Calculate overall speedup across all benchmarks"""
        if not speedup_analyses:
            return 0.0
        
        valid_speedups = [analysis.speedup_factor for analysis in speedup_analyses if analysis.is_valid]
        
        if not valid_speedups:
            return 0.0
        
        # Use geometric mean for overall speedup
        import math
        geometric_mean = math.exp(sum(math.log(speedup) for speedup in valid_speedups) / len(valid_speedups))
        
        return geometric_mean
    
    def _create_validation_summary(self, speedup_analyses: List[SpeedupAnalysis], overall_speedup: float, meets_130x_claim: bool) -> str:
        """Create validation summary report"""
        summary_parts = []
        
        summary_parts.append(f"GPU Performance Validation Summary")
        summary_parts.append(f"=" * 40)
        summary_parts.append(f"Overall Speedup: {overall_speedup:.2f}x")
        summary_parts.append(f"130x Claim Validation: {'PASSED' if meets_130x_claim else 'FAILED'}")
        summary_parts.append("")
        
        summary_parts.append("Benchmark Results by Type:")
        for analysis in speedup_analyses:
            status = "✅" if analysis.is_valid else "❌"
            summary_parts.append(f"{status} {analysis.benchmark_type.value}: {analysis.speedup_factor:.2f}x speedup")
            if analysis.validation_notes != "Valid":
                summary_parts.append(f"    Notes: {analysis.validation_notes}")
        
        summary_parts.append("")
        
        if meets_130x_claim:
            summary_parts.append("🎉 GPU acceleration meets the 130x speedup claim!")
        else:
            summary_parts.append("⚠️ GPU acceleration does not meet the 130x speedup claim.")
            summary_parts.append("   This may be due to:")
            summary_parts.append("   - Hardware limitations")
            summary_parts.append("   - Insufficient optimization")
            summary_parts.append("   - Workload characteristics")
            summary_parts.append("   - Measurement methodology")
        
        return "\n".join(summary_parts)
    
    def _serialize_benchmark_result(self, result: BenchmarkResult) -> Dict[str, Any]:
        """Serialize benchmark result for JSON storage"""
        return {
            'benchmark_type': result.benchmark_type.value,
            'device_type': result.device_type,
            'device_name': result.device_name,
            'iterations': result.iterations,
            'data_size': result.data_size,
            'execution_time': result.execution_time,
            'throughput': result.throughput,
            'memory_usage': result.memory_usage,
            'timestamp': result.timestamp.isoformat(),
            'error': result.error
        }
    
    def _serialize_speedup_analysis(self, analysis: SpeedupAnalysis) -> Dict[str, Any]:
        """Serialize speedup analysis for JSON storage"""
        return {
            'benchmark_type': analysis.benchmark_type.value,
            'cpu_time': analysis.cpu_time,
            'gpu_time': analysis.gpu_time,
            'speedup_factor': analysis.speedup_factor,
            'cpu_throughput': analysis.cpu_throughput,
            'gpu_throughput': analysis.gpu_throughput,
            'memory_efficiency': analysis.memory_efficiency,
            'is_valid': analysis.is_valid,
            'validation_notes': analysis.validation_notes
        }
    
    def get_benchmark_history(self) -> List[Dict[str, Any]]:
        """Get benchmark history"""
        with self.lock:
            return self.benchmark_history.copy()
    
    def get_latest_benchmark(self) -> Optional[Dict[str, Any]]:
        """Get the latest benchmark results"""
        with self.lock:
            return self.benchmark_history[-1] if self.benchmark_history else None


# Global benchmarking instance
_gpu_benchmarking: Optional[UltraGPUBenchmarking] = None

def get_gpu_benchmarking() -> UltraGPUBenchmarking:
    """Get global GPU benchmarking instance"""
    global _gpu_benchmarking
    
    if _gpu_benchmarking is None:
        _gpu_benchmarking = UltraGPUBenchmarking()
    
    return _gpu_benchmarking

def run_gpu_performance_validation() -> Dict[str, Any]:
    """Run comprehensive GPU performance validation"""
    benchmarking = get_gpu_benchmarking()
    return benchmarking.run_comprehensive_benchmark() 