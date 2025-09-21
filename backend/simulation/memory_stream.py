"""
Enterprise Memory Streaming System for Large Excel File Processing
Implements intelligent chunking, memory management, and streaming computation.
"""

import numpy as np
import gc
import psutil
import threading
import time
from typing import Dict, List, Any, Generator, Optional, Callable
import logging
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import queue
from contextlib import contextmanager

logger = logging.getLogger(__name__)

@dataclass
class MemoryStats:
    """Memory usage statistics."""
    total_mb: float
    available_mb: float
    used_mb: float
    usage_percent: float
    process_mb: float

@dataclass
class StreamingChunk:
    """Represents a chunk of data in the streaming pipeline."""
    chunk_id: int
    data: Any
    metadata: Dict[str, Any]
    size_estimate: int

class MemoryMonitor:
    """Real-time memory usage monitoring and alerting."""
    
    def __init__(self, warning_threshold: float = 85.0, critical_threshold: float = 95.0):
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.monitoring = False
        self.monitor_thread = None
        self.callbacks = []
        self.logger = logging.getLogger(__name__)
        
    def start_monitoring(self, interval: float = 1.0) -> None:
        """Start continuous memory monitoring."""
        if self.monitoring:
            return
            
        self.monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop, 
            args=(interval,), 
            daemon=True
        )
        self.monitor_thread.start()
        self.logger.info(f"📊 [MemoryMonitor] Started monitoring (warning: {self.warning_threshold}%, critical: {self.critical_threshold}%)")
        
    def stop_monitoring(self) -> None:
        """Stop memory monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        self.logger.info("📊 [MemoryMonitor] Stopped monitoring")
        
    def add_callback(self, callback: Callable[[MemoryStats, str], None]) -> None:
        """Add callback for memory alerts."""
        self.callbacks.append(callback)
        
    def get_memory_stats(self) -> MemoryStats:
        """Get current memory statistics."""
        system_memory = psutil.virtual_memory()
        process = psutil.Process()
        process_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        return MemoryStats(
            total_mb=system_memory.total / 1024 / 1024,
            available_mb=system_memory.available / 1024 / 1024,
            used_mb=system_memory.used / 1024 / 1024,
            usage_percent=system_memory.percent,
            process_mb=process_memory
        )
        
    def _monitor_loop(self, interval: float) -> None:
        """Main monitoring loop."""
        while self.monitoring:
            try:
                stats = self.get_memory_stats()
                
                # Check thresholds and trigger callbacks
                if stats.usage_percent >= self.critical_threshold:
                    self._trigger_callbacks(stats, "CRITICAL")
                elif stats.usage_percent >= self.warning_threshold:
                    self._trigger_callbacks(stats, "WARNING")
                    
                time.sleep(interval)
            except Exception as e:
                self.logger.error(f"Error in memory monitoring: {e}")
                time.sleep(interval)
                
    def _trigger_callbacks(self, stats: MemoryStats, level: str) -> None:
        """Trigger registered callbacks."""
        for callback in self.callbacks:
            try:
                callback(stats, level)
            except Exception as e:
                self.logger.error(f"Error in memory callback: {e}")

class StreamingDataProcessor:
    """
    Processes large datasets in chunks to manage memory usage.
    """
    
    def __init__(self, 
                 chunk_size: int = 1000,
                 max_memory_mb: float = 2048,
                 enable_compression: bool = True):
        self.chunk_size = chunk_size
        self.max_memory_mb = max_memory_mb
        self.enable_compression = enable_compression
        self.logger = logging.getLogger(__name__)
        self.memory_monitor = MemoryMonitor()
        
        # Performance tracking
        self.processing_stats = {
            'chunks_processed': 0,
            'total_processing_time': 0.0,
            'memory_cleanups': 0,
            'compression_ratio': 1.0,
            'peak_memory_mb': 0.0
        }
        
        # Add memory callback
        self.memory_monitor.add_callback(self._handle_memory_alert)
        
    def stream_process_excel_data(self, 
                                sheet_data: Dict[str, Any],
                                variables: List[Dict[str, Any]],
                                processor_func: Callable[[StreamingChunk], Any]) -> Generator[Any, None, None]:
        """
        Stream process Excel data in chunks with memory management.
        
        Args:
            sheet_data: Excel sheet data
            variables: Input variables
            processor_func: Function to process each chunk
            
        Yields:
            Processed results from each chunk
        """
        start_time = time.time()
        self.memory_monitor.start_monitoring()
        
        try:
            self.logger.info(f"🌊 [Streaming] Starting stream processing with chunk_size={self.chunk_size}")
            
            # Extract data rows
            data_rows = sheet_data.get('data', [])
            total_rows = len(data_rows)
            
            # Process in chunks
            chunk_id = 0
            for i in range(0, total_rows, self.chunk_size):
                chunk_data = data_rows[i:i + self.chunk_size]
                
                # Create chunk with metadata
                chunk = StreamingChunk(
                    chunk_id=chunk_id,
                    data=chunk_data,
                    metadata={
                        'start_row': i,
                        'end_row': min(i + self.chunk_size, total_rows),
                        'total_rows': total_rows,
                        'variables': variables
                    },
                    size_estimate=self._estimate_chunk_size(chunk_data)
                )
                
                # Check memory before processing
                self._check_memory_before_chunk()
                
                # Process chunk
                chunk_start = time.time()
                try:
                    result = processor_func(chunk)
                    yield result
                    
                    chunk_time = time.time() - chunk_start
                    self.processing_stats['chunks_processed'] += 1
                    self.processing_stats['total_processing_time'] += chunk_time
                    
                    self.logger.info(f"⚡ [Streaming] Processed chunk {chunk_id} ({len(chunk_data)} rows) in {chunk_time:.3f}s")
                    
                except Exception as e:
                    self.logger.error(f"Error processing chunk {chunk_id}: {e}")
                    raise
                
                # Memory cleanup after chunk
                self._cleanup_after_chunk()
                chunk_id += 1
                
        finally:
            self.memory_monitor.stop_monitoring()
            total_time = time.time() - start_time
            self.logger.info(f"🎯 [Streaming] Completed stream processing in {total_time:.3f}s")
            self._log_final_stats()
    
    def stream_monte_carlo_iterations(self,
                                    iterations: int,
                                    variables: List[Dict[str, Any]],
                                    formula_processor: Callable[[Dict[str, np.ndarray]], np.ndarray]) -> np.ndarray:
        """
        Stream process Monte Carlo iterations with memory management.
        
        Args:
            iterations: Total number of iterations
            variables: Input variables
            formula_processor: Function to process variable samples
            
        Returns:
            Combined results from all chunks
        """
        self.logger.info(f"🎲 [StreamingMC] Processing {iterations} iterations in chunks of {self.chunk_size}")
        
        results = []
        processed_iterations = 0
        
        while processed_iterations < iterations:
            # Calculate chunk size (may be smaller for last chunk)
            current_chunk_size = min(self.chunk_size, iterations - processed_iterations)
            
            # Generate samples for this chunk
            chunk_samples = self._generate_chunk_samples(variables, current_chunk_size)
            
            # Process chunk
            try:
                chunk_results = formula_processor(chunk_samples)
                if isinstance(chunk_results, np.ndarray):
                    results.extend(chunk_results.flatten())
                else:
                    results.append(chunk_results)
                    
                processed_iterations += current_chunk_size
                
                # Memory cleanup
                del chunk_samples, chunk_results
                if processed_iterations % (self.chunk_size * 5) == 0:  # Cleanup every 5 chunks
                    gc.collect()
                    
            except Exception as e:
                self.logger.error(f"Error in streaming MC iteration: {e}")
                raise
                
        return np.array(results)
    
    def _generate_chunk_samples(self, variables: List[Dict[str, Any]], chunk_size: int) -> Dict[str, np.ndarray]:
        """Generate variable samples for a chunk."""
        chunk_samples = {}
        
        for var in variables:
            var_name = var['name']
            
            if var.get('distribution', 'triangular') == 'triangular':
                chunk_samples[var_name] = np.random.triangular(
                    var['min_value'],
                    var['most_likely'],
                    var['max_value'],
                    size=chunk_size
                )
            elif var.get('distribution') == 'normal':
                mean = var.get('mean', var.get('most_likely', 0))
                std = var.get('std', var.get('std_dev', 1))
                chunk_samples[var_name] = np.random.normal(mean, std, size=chunk_size)
            else:
                # Default to triangular
                chunk_samples[var_name] = np.random.triangular(
                    var['min_value'],
                    var['most_likely'],
                    var['max_value'],
                    size=chunk_size
                )
                
        return chunk_samples
    
    def _estimate_chunk_size(self, chunk_data: Any) -> int:
        """Estimate memory size of chunk data."""
        try:
            if isinstance(chunk_data, list):
                return len(str(chunk_data).encode('utf-8'))
            elif isinstance(chunk_data, np.ndarray):
                return chunk_data.nbytes
            else:
                return len(str(chunk_data).encode('utf-8'))
        except:
            return 1000  # Default estimate
    
    def _check_memory_before_chunk(self) -> None:
        """Check memory usage before processing a chunk."""
        stats = self.memory_monitor.get_memory_stats()
        
        # Update peak memory tracking
        self.processing_stats['peak_memory_mb'] = max(
            self.processing_stats['peak_memory_mb'],
            stats.process_mb
        )
        
        # Force cleanup if memory usage is high
        if stats.usage_percent > 80:
            self.logger.warning(f"⚠️ [Memory] High memory usage ({stats.usage_percent:.1f}%), forcing cleanup")
            self._force_memory_cleanup()
    
    def _cleanup_after_chunk(self) -> None:
        """Cleanup memory after processing a chunk."""
        # Force garbage collection periodically
        if self.processing_stats['chunks_processed'] % 10 == 0:
            gc.collect()
            self.processing_stats['memory_cleanups'] += 1
    
    def _force_memory_cleanup(self) -> None:
        """Force aggressive memory cleanup."""
        gc.collect()  # Full garbage collection
        self.processing_stats['memory_cleanups'] += 1
        
        # Log memory status after cleanup
        stats = self.memory_monitor.get_memory_stats()
        self.logger.info(f"🧹 [Memory] Cleanup completed. Usage: {stats.usage_percent:.1f}%, Process: {stats.process_mb:.1f}MB")
    
    def _handle_memory_alert(self, stats: MemoryStats, level: str) -> None:
        """Handle memory usage alerts."""
        if level == "CRITICAL":
            self.logger.critical(f"🚨 [Memory] CRITICAL: {stats.usage_percent:.1f}% usage!")
            self._force_memory_cleanup()
        elif level == "WARNING":
            self.logger.warning(f"⚠️ [Memory] WARNING: {stats.usage_percent:.1f}% usage")
    
    def _log_final_stats(self) -> None:
        """Log final processing statistics."""
        avg_chunk_time = (
            self.processing_stats['total_processing_time'] / 
            max(1, self.processing_stats['chunks_processed'])
        )
        
        self.logger.info(f"📊 [StreamingStats] Chunks processed: {self.processing_stats['chunks_processed']}")
        self.logger.info(f"📊 [StreamingStats] Average chunk time: {avg_chunk_time:.3f}s")
        self.logger.info(f"📊 [StreamingStats] Memory cleanups: {self.processing_stats['memory_cleanups']}")
        self.logger.info(f"📊 [StreamingStats] Peak memory: {self.processing_stats['peak_memory_mb']:.1f}MB")

class LargeFileOptimizer:
    """
    Optimizes processing strategies for large Excel files.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def analyze_file_characteristics(self, sheet_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze file characteristics to determine optimal processing strategy."""
        start_time = time.time()
        
        # Count elements
        data_rows = sheet_data.get('data', [])
        total_cells = sum(len(row) for row in data_rows)
        formula_cells = 0
        complex_formulas = 0
        
        for row in data_rows:
            for cell in row:
                if isinstance(cell, dict) and cell.get('formula'):
                    formula_cells += 1
                    if self._is_complex_formula(cell['formula']):
                        complex_formulas += 1
        
        # Calculate characteristics
        analysis_time = time.time() - start_time
        
        characteristics = {
            'total_rows': len(data_rows),
            'total_cells': total_cells,
            'formula_cells': formula_cells,
            'complex_formulas': complex_formulas,
            'formula_density': formula_cells / max(1, total_cells),
            'complexity_ratio': complex_formulas / max(1, formula_cells),
            'analysis_time': analysis_time,
            'estimated_memory_mb': self._estimate_memory_requirement(total_cells),
            'recommended_chunk_size': self._recommend_chunk_size(total_cells, formula_cells)
        }
        
        self.logger.info(f"🔍 [FileAnalysis] Analyzed file in {analysis_time:.3f}s")
        self.logger.info(f"📊 [FileAnalysis] {total_cells} cells, {formula_cells} formulas ({characteristics['formula_density']:.1%} density)")
        self.logger.info(f"⚡ [FileAnalysis] Recommended chunk size: {characteristics['recommended_chunk_size']}")
        
        return characteristics
    
    def _is_complex_formula(self, formula: str) -> bool:
        """Determine if a formula is complex."""
        if not formula:
            return False
            
        # Check for complexity indicators
        complex_functions = ['VLOOKUP', 'INDEX', 'MATCH', 'SUMPRODUCT', 'ARRAY']
        nested_level = formula.count('(')
        
        return (
            any(func in formula.upper() for func in complex_functions) or
            nested_level > 3 or
            len(formula) > 100
        )
    
    def _estimate_memory_requirement(self, total_cells: int) -> float:
        """Estimate memory requirement in MB."""
        # Rough estimate: 100 bytes per cell on average
        return (total_cells * 100) / 1024 / 1024
    
    def _recommend_chunk_size(self, total_cells: int, formula_cells: int) -> int:
        """Recommend optimal chunk size based on file characteristics."""
        # Base chunk size
        base_chunk = 1000
        
        # Adjust based on formula density
        formula_ratio = formula_cells / max(1, total_cells)
        if formula_ratio > 0.5:
            base_chunk = 500  # Many formulas = smaller chunks
        elif formula_ratio < 0.1:
            base_chunk = 2000  # Few formulas = larger chunks
        
        # Adjust based on total size
        if total_cells > 100000:
            base_chunk = min(base_chunk, 250)  # Very large files = smaller chunks
        elif total_cells < 10000:
            base_chunk = max(base_chunk, 1500)  # Small files = larger chunks
        
        return base_chunk

@contextmanager
def memory_managed_processing(max_memory_mb: float = 2048):
    """Context manager for memory-managed processing."""
    monitor = MemoryMonitor()
    processor = StreamingDataProcessor(max_memory_mb=max_memory_mb)
    
    monitor.start_monitoring()
    try:
        yield processor
    finally:
        monitor.stop_monitoring()

def create_streaming_processor(chunk_size: int = None, 
                             max_memory_mb: float = None) -> StreamingDataProcessor:
    """Factory function to create an optimally configured streaming processor."""
    # Get system memory for defaults
    system_memory = psutil.virtual_memory()
    available_mb = system_memory.available / 1024 / 1024
    
    if max_memory_mb is None:
        # Use 50% of available memory as default
        max_memory_mb = min(2048, available_mb * 0.5)
    
    if chunk_size is None:
        # Choose chunk size based on available memory
        chunk_size = max(500, min(2000, int(max_memory_mb / 4)))
    
    processor = StreamingDataProcessor(
        chunk_size=chunk_size,
        max_memory_mb=max_memory_mb,
        enable_compression=True
    )
    
    logger.info(f"🌊 Created streaming processor: chunk_size={chunk_size}, max_memory={max_memory_mb:.0f}MB")
    
    return processor 