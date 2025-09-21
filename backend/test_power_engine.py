import asyncio
import logging
from simulation.power_engine import PowerMonteCarloEngine

logging.basicConfig(level=logging.INFO)

async def test_power_engine():
    """Test the Power engine initialization"""
    try:
        # Create Power engine instance
        power_engine = PowerMonteCarloEngine(iterations=100)
        
        print("✅ Power engine initialized successfully")
        print(f"Config: {power_engine.config}")
        print(f"Components initialized:")
        print(f"  - Sparse detector: {power_engine.sparse_detector is not None}")
        print(f"  - Streaming processor: {power_engine.streaming_processor is not None}")
        print(f"  - Cache manager: {power_engine.cache_manager is not None}")
        
        # Test sparse range detection
        test_formula = "=SUM(A1:A10000)"
        ranges = power_engine.sparse_detector.analyze_sum_range(test_formula)
        print(f"\nSparse range detection test:")
        print(f"  Formula: {test_formula}")
        print(f"  Detected ranges: {ranges}")
        
        return True
        
    except Exception as e:
        print(f"❌ Power engine test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_power_engine())
    exit(0 if success else 1)
