"""
Test script for Power Engine VLOOKUP text fix
"""

import asyncio
import logging
from simulation.power_engine import PowerMonteCarloEngine
from simulation.schemas import VariableConfig

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_power_vlookup():
    """Test Power Engine with VLOOKUP text lookup"""
    
    # Create mock data structures
    class MockVariable:
        def __init__(self, name, sheet_name, min_val, most_likely, max_val):
            self.name = name
            self.sheet_name = sheet_name
            self.min_value = min_val
            self.most_likely = most_likely
            self.max_value = max_val
    
    # Test constants including text values
    test_constants = {
        ('Sheet1', 'A8'): 'A',  # Text constant for VLOOKUP
        ('Sheet1', 'A10'): 'A',
        ('Sheet1', 'B10'): 100,
        ('Sheet1', 'A11'): 'B', 
        ('Sheet1', 'B11'): 200,
        ('Sheet1', 'A12'): 'C',
        ('Sheet1', 'B12'): 300,
    }
    
    # Create test variable
    variables = [
        MockVariable('D2', 'Sheet1', 0.8, 1.0, 1.2)  # Variable for simulation
    ]
    
    # Test formula with VLOOKUP
    test_formula = '=VLOOKUP(A8, A10:B12, 2, FALSE)'
    
    # Initialize Power Engine
    engine = PowerMonteCarloEngine(iterations=10)
    
    # Test formula evaluation directly
    logger.info("Testing direct formula evaluation...")
    
    # Create processor
    processor = engine.streaming_processor
    
    # Test values
    test_values = {
        ('Sheet1', 'D2'): 1.0,
    }
    
    # Evaluate formula
    result = processor._evaluate_formula(
        test_formula, 
        'Sheet1', 
        test_values,
        test_constants
    )
    
    logger.info(f"Formula result: {result}")
    logger.info(f"Expected: 100 (lookup of 'A' should return 100)")
    
    if result == 100:
        logger.info("✅ Power Engine VLOOKUP text fix is working!")
    else:
        logger.error(f"❌ Power Engine VLOOKUP text fix FAILED! Got {result} instead of 100")
    
    return result == 100

if __name__ == "__main__":
    success = asyncio.run(test_power_vlookup())
    exit(0 if success else 1) 