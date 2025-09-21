"""
Arrow Schema Definitions for Monte Carlo Simulation Platform
Defines standardized schemas for parameters, results, and statistics
"""

import pyarrow as pa

# Input parameters schema for Monte Carlo variables
PARAMETERS_SCHEMA = pa.schema([
    ('cell_id', pa.string()),
    ('formula', pa.string()),
    ('distribution_type', pa.string()),  # normal, uniform, triangular, etc.
    ('param1', pa.float64()),           # mean, min, etc.
    ('param2', pa.float64()),           # std, max, etc.
    ('param3', pa.float64()),           # optional third parameter
    ('correlation_group', pa.string()),
    ('dependencies', pa.list_(pa.string()))
])

# Simulation results schema for streaming results
RESULTS_SCHEMA = pa.schema([
    ('iteration', pa.uint32()),
    ('cell_id', pa.string()),
    ('value', pa.float64()),
    ('timestamp', pa.timestamp('ms'))
])

# Statistics schema for final analysis
STATISTICS_SCHEMA = pa.schema([
    ('cell_id', pa.string()),
    ('mean', pa.float64()),
    ('std_dev', pa.float64()),
    ('min_value', pa.float64()),
    ('max_value', pa.float64()),
    ('percentile_5', pa.float64()),
    ('percentile_25', pa.float64()),
    ('percentile_50', pa.float64()),
    ('percentile_75', pa.float64()),
    ('percentile_95', pa.float64()),
    ('var_95', pa.float64()),          # Value at Risk
    ('cvar_95', pa.float64()),         # Conditional Value at Risk
    ('histogram_bins', pa.list_(pa.float64())),
    ('histogram_counts', pa.list_(pa.uint32()))
])

def validate_parameters_table(table: pa.Table) -> bool:
    """Validate that a table matches the parameters schema"""
    try:
        expected_schema = PARAMETERS_SCHEMA
        return table.schema.equals(expected_schema)
    except Exception:
        return False

def validate_results_table(table: pa.Table) -> bool:
    """Validate that a table matches the results schema"""
    try:
        expected_schema = RESULTS_SCHEMA
        return table.schema.equals(expected_schema)
    except Exception:
        return False

def validate_statistics_table(table: pa.Table) -> bool:
    """Validate that a table matches the statistics schema"""
    try:
        expected_schema = STATISTICS_SCHEMA
        return table.schema.equals(expected_schema)
    except Exception:
        return False

def create_empty_parameters_table() -> pa.Table:
    """Create an empty parameters table with correct schema"""
    return pa.Table.from_arrays([
        pa.array([], type=pa.string()),        # cell_id
        pa.array([], type=pa.string()),        # formula
        pa.array([], type=pa.string()),        # distribution_type
        pa.array([], type=pa.float64()),       # param1
        pa.array([], type=pa.float64()),       # param2
        pa.array([], type=pa.float64()),       # param3
        pa.array([], type=pa.string()),        # correlation_group
        pa.array([], type=pa.list_(pa.string()))  # dependencies
    ], schema=PARAMETERS_SCHEMA)

def create_empty_results_table() -> pa.Table:
    """Create an empty results table with correct schema"""
    return pa.Table.from_arrays([
        pa.array([], type=pa.uint32()),        # iteration
        pa.array([], type=pa.string()),        # cell_id
        pa.array([], type=pa.float64()),       # value
        pa.array([], type=pa.timestamp('ms'))  # timestamp
    ], schema=RESULTS_SCHEMA)

def create_empty_statistics_table() -> pa.Table:
    """Create an empty statistics table with correct schema"""
    return pa.Table.from_arrays([
        pa.array([], type=pa.string()),        # cell_id
        pa.array([], type=pa.float64()),       # mean
        pa.array([], type=pa.float64()),       # std_dev
        pa.array([], type=pa.float64()),       # min_value
        pa.array([], type=pa.float64()),       # max_value
        pa.array([], type=pa.float64()),       # percentile_5
        pa.array([], type=pa.float64()),       # percentile_25
        pa.array([], type=pa.float64()),       # percentile_50
        pa.array([], type=pa.float64()),       # percentile_75
        pa.array([], type=pa.float64()),       # percentile_95
        pa.array([], type=pa.float64()),       # var_95
        pa.array([], type=pa.float64()),       # cvar_95
        pa.array([], type=pa.list_(pa.float64())),  # histogram_bins
        pa.array([], type=pa.list_(pa.uint32()))    # histogram_counts
    ], schema=STATISTICS_SCHEMA) 