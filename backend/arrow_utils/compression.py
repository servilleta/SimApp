"""
Arrow Compression Utilities
Handles compression and decompression of Arrow data for efficient storage
"""

import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc
import logging
from typing import Optional, Union, Dict, Any
from io import BytesIO

logger = logging.getLogger(__name__)

class ArrowCompressionUtil:
    """
    Utilities for compressing and decompressing Arrow data
    Supports multiple compression algorithms optimized for different use cases
    """
    
    # Compression algorithms supported
    ALGORITHMS = {
        'snappy': 'snappy',    # Fast compression/decompression
        'gzip': 'gzip',        # Good compression ratio
        'lz4': 'lz4',          # Very fast, moderate compression
        'zstd': 'zstd',        # Excellent compression ratio
        'brotli': 'brotli'     # High compression ratio
    }
    
    def __init__(self, default_algorithm: str = 'snappy'):
        self.default_algorithm = default_algorithm
        self._validate_algorithm(default_algorithm)
    
    def _validate_algorithm(self, algorithm: str):
        """Validate compression algorithm is supported"""
        if algorithm not in self.ALGORITHMS:
            raise ValueError(f"Unsupported compression algorithm: {algorithm}. "
                           f"Supported: {list(self.ALGORITHMS.keys())}")
    
    def compress_table_to_parquet(self, 
                                 table: pa.Table, 
                                 compression: str = None) -> bytes:
        """
        Compress Arrow table to Parquet format in memory
        Returns compressed bytes
        """
        compression = compression or self.default_algorithm
        self._validate_algorithm(compression)
        
        try:
            # Create in-memory buffer
            buffer = BytesIO()
            
            # Write table to Parquet with compression
            pq.write_table(
                table, 
                buffer, 
                compression=compression,
                use_dictionary=True,  # Enable dictionary encoding
                write_statistics=False  # Skip statistics for speed
            )
            
            compressed_bytes = buffer.getvalue()
            buffer.close()
            
            # Log compression stats
            original_size = table.nbytes
            compressed_size = len(compressed_bytes)
            ratio = compressed_size / original_size if original_size > 0 else 0
            
            logger.debug(f"Compressed {original_size:,} bytes to {compressed_size:,} bytes "
                        f"({ratio:.2%} of original) using {compression}")
            
            return compressed_bytes
            
        except Exception as e:
            logger.error(f"Error compressing table to Parquet: {e}")
            raise
    
    def decompress_table_from_parquet(self, compressed_bytes: bytes) -> pa.Table:
        """
        Decompress Parquet bytes back to Arrow table
        """
        try:
            # Create buffer from bytes
            buffer = BytesIO(compressed_bytes)
            
            # Read table from Parquet
            table = pq.read_table(buffer)
            buffer.close()
            
            logger.debug(f"Decompressed {len(compressed_bytes):,} bytes to "
                        f"{table.nbytes:,} bytes ({len(table)} rows)")
            
            return table
            
        except Exception as e:
            logger.error(f"Error decompressing Parquet bytes: {e}")
            raise
    
    def compress_table_to_ipc(self, 
                             table: pa.Table, 
                             compression: str = None) -> bytes:
        """
        Compress Arrow table to IPC (Arrow) format in memory
        More efficient for streaming between Arrow systems
        """
        compression = compression or self.default_algorithm
        self._validate_algorithm(compression)
        
        try:
            # Create in-memory buffer
            buffer = BytesIO()
            
            # Create IPC writer with compression
            with pa.ipc.new_stream(buffer, table.schema, 
                                 options=pa.ipc.IpcWriteOptions(
                                     compression=compression)) as writer:
                writer.write_table(table)
            
            compressed_bytes = buffer.getvalue()
            buffer.close()
            
            # Log compression stats
            original_size = table.nbytes
            compressed_size = len(compressed_bytes)
            ratio = compressed_size / original_size if original_size > 0 else 0
            
            logger.debug(f"IPC compressed {original_size:,} bytes to {compressed_size:,} bytes "
                        f"({ratio:.2%} of original) using {compression}")
            
            return compressed_bytes
            
        except Exception as e:
            logger.error(f"Error compressing table to IPC: {e}")
            raise
    
    def decompress_table_from_ipc(self, compressed_bytes: bytes) -> pa.Table:
        """
        Decompress IPC bytes back to Arrow table
        """
        try:
            # Create buffer from bytes
            buffer = BytesIO(compressed_bytes)
            
            # Read table from IPC stream
            with pa.ipc.open_stream(buffer) as reader:
                table = reader.read_all()
            
            buffer.close()
            
            logger.debug(f"IPC decompressed {len(compressed_bytes):,} bytes to "
                        f"{table.nbytes:,} bytes ({len(table)} rows)")
            
            return table
            
        except Exception as e:
            logger.error(f"Error decompressing IPC bytes: {e}")
            raise
    
    def get_compression_info(self, table: pa.Table) -> Dict[str, Any]:
        """
        Analyze table and recommend optimal compression algorithm
        """
        try:
            table_size = table.nbytes
            num_rows = len(table)
            num_columns = len(table.columns)
            
            # Test compression ratios with different algorithms
            compression_results = {}
            
            for algo_name, algo_code in self.ALGORITHMS.items():
                try:
                    # Test Parquet compression
                    compressed = self.compress_table_to_parquet(table, algo_code)
                    compression_results[algo_name] = {
                        'compressed_size': len(compressed),
                        'ratio': len(compressed) / table_size if table_size > 0 else 0,
                        'format': 'parquet'
                    }
                except Exception as e:
                    logger.warning(f"Failed to test {algo_name} compression: {e}")
                    compression_results[algo_name] = {'error': str(e)}
            
            # Find best algorithm (smallest compressed size)
            best_algo = min(
                (k for k, v in compression_results.items() if 'compressed_size' in v),
                key=lambda k: compression_results[k]['compressed_size'],
                default=self.default_algorithm
            )
            
            return {
                'table_stats': {
                    'original_size': table_size,
                    'num_rows': num_rows,
                    'num_columns': num_columns,
                    'avg_row_size': table_size / num_rows if num_rows > 0 else 0
                },
                'compression_results': compression_results,
                'recommended_algorithm': best_algo,
                'best_ratio': compression_results.get(best_algo, {}).get('ratio', 1.0)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing compression options: {e}")
            return {'error': str(e)}
    
    def compress_with_best_algorithm(self, table: pa.Table) -> tuple[bytes, str, Dict[str, Any]]:
        """
        Compress table using the best algorithm for this specific table
        Returns (compressed_bytes, algorithm_used, compression_info)
        """
        try:
            # Analyze compression options
            info = self.get_compression_info(table)
            
            if 'error' in info:
                # Fallback to default
                algorithm = self.default_algorithm
                compressed = self.compress_table_to_parquet(table, algorithm)
            else:
                algorithm = info['recommended_algorithm']
                compressed = self.compress_table_to_parquet(table, algorithm)
            
            return compressed, algorithm, info
            
        except Exception as e:
            logger.error(f"Error with best algorithm compression: {e}")
            # Final fallback
            algorithm = self.default_algorithm
            compressed = self.compress_table_to_parquet(table, algorithm)
            return compressed, algorithm, {'error': str(e)}

# Global compression utility instance
_global_compressor: Optional[ArrowCompressionUtil] = None

def get_compressor() -> ArrowCompressionUtil:
    """Get global Arrow compression utility instance"""
    global _global_compressor
    if _global_compressor is None:
        _global_compressor = ArrowCompressionUtil()
    return _global_compressor

def compress_table(table: pa.Table, algorithm: str = 'snappy') -> bytes:
    """Convenience function to compress a table"""
    return get_compressor().compress_table_to_parquet(table, algorithm)

def decompress_table(compressed_bytes: bytes) -> pa.Table:
    """Convenience function to decompress a table"""
    return get_compressor().decompress_table_from_parquet(compressed_bytes) 