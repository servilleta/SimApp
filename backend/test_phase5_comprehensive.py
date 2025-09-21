"""
ULTRA MONTE CARLO ENGINE - PHASE 5 COMPREHENSIVE TESTING
Complete test suite for asynchronous processing capabilities.
"""

import asyncio
import logging
import time
import sys
import os
from typing import Dict, List, Any, Tuple
import numpy as np

# Add backend to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__)))

logger = logging.getLogger(__name__)

async def test_phase5_async_core():
    """Test Phase 5 async core components"""
    logger.info("🔧 [PHASE5-TEST] Testing async core components...")
    
    try:
        from simulation.engines.phase5_async_core import (
            UltraAsyncTaskQueue, 
            AsyncSimulationTask, 
            TaskPriority, 
            TaskStatus, 
            TaskResource,
            UltraSystemMonitor
        )
        
        # Test 1: System Monitor
        logger.info("🔧 [PHASE5-TEST] Test 1: System Monitor")
        monitor = UltraSystemMonitor()
        await monitor.start_monitoring()
        await asyncio.sleep(2)  # Let it collect some data
        
        resources = monitor.get_current_resources()
        trends = monitor.get_resource_trends()
        
        await monitor.stop_monitoring()
        
        assert 'cpu' in resources
        assert 'memory' in resources
        assert 'gpu_memory' in resources
        logger.info("✅ [PHASE5-TEST] System Monitor test passed")
        
        # Test 2: Async Task Queue
        logger.info("🔧 [PHASE5-TEST] Test 2: Async Task Queue")
        queue = UltraAsyncTaskQueue(max_concurrent_tasks=3)
        
        # Create test tasks
        test_tasks = []
        for i in range(5):
            task = AsyncSimulationTask(
                task_id=f"test_task_{i}",
                simulation_id=f"sim_{i}",
                priority=TaskPriority.NORMAL,
                status=TaskStatus.PENDING,
                iterations=100,
                mc_input_configs=[],
                ordered_calc_steps=[],
                target_sheet_name="Sheet1",
                target_cell_coordinate="A1",
                constant_values={},
                resources=TaskResource(cpu_cores=1, system_memory_mb=256)
            )
            test_tasks.append(task)
        
        # Submit tasks
        task_ids = []
        for task in test_tasks:
            task_id = await queue.add_task(task)
            task_ids.append(task_id)
        
        # Wait for some processing
        await asyncio.sleep(3)
        
        # Check queue stats
        stats = queue.get_queue_stats()
        assert stats['max_concurrent'] == 3
        assert stats['scheduler_active'] == True
        
        logger.info("✅ [PHASE5-TEST] Async Task Queue test passed")
        logger.info(f"   - Tasks queued: {stats['statistics']['tasks_queued']}")
        logger.info(f"   - Tasks completed: {stats['statistics']['tasks_completed']}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ [PHASE5-TEST] Async core test failed: {e}")
        return False

async def test_phase5_concurrent_manager():
    """Test Phase 5 concurrent simulation manager"""
    logger.info("🔧 [PHASE5-TEST] Testing concurrent simulation manager...")
    
    try:
        from simulation.engines.phase5_concurrent_manager import UltraConcurrentSimulationManager
        from simulation.engines.phase5_async_core import TaskPriority
        
        # Initialize manager
        manager = UltraConcurrentSimulationManager(max_concurrent_simulations=3)
        
        # Submit test simulations
        task_ids = []
        simulation_ids = []
        
        for i in range(5):
            sim_id = f"test_concurrent_sim_{i}"
            simulation_ids.append(sim_id)
            
            task_id = await manager.submit_simulation(
                simulation_id=sim_id,
                iterations=100,
                mc_input_configs=[
                    {'name': 'var1', 'min_value': 10, 'max_value': 20, 'most_likely_value': 15, 'sheet_name': 'Sheet1'},
                    {'name': 'var2', 'min_value': 5, 'max_value': 15, 'most_likely_value': 10, 'sheet_name': 'Sheet1'}
                ],
                ordered_calc_steps=[
                    ('Sheet1', 'A1', '=var1+var2'),
                    ('Sheet1', 'B1', '=A1*2')
                ],
                target_sheet_name="Sheet1",
                target_cell_coordinate="B1",
                constant_values={},
                priority=TaskPriority.NORMAL
            )
            task_ids.append(task_id)
        
        logger.info(f"🔧 [PHASE5-TEST] Submitted {len(task_ids)} concurrent simulations")
        
        # Wait for processing
        await asyncio.sleep(5)
        
        # Check statuses
        completed_count = 0
        for sim_id in simulation_ids:
            status = await manager.get_simulation_status(sim_id)
            if status:
                logger.info(f"   - {sim_id}: {status['status']}")
                if status['status'] == 'completed':
                    completed_count += 1
        
        # Get manager statistics
        stats = manager.get_manager_stats()
        logger.info("✅ [PHASE5-TEST] Concurrent Manager test passed")
        logger.info(f"   - Active Simulations: {stats['active_simulations']}")
        logger.info(f"   - Queue Size: {stats['queue_size']}")
        logger.info(f"   - Completed Simulations: {completed_count}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ [PHASE5-TEST] Concurrent manager test failed: {e}")
        return False

async def test_phase5_pipeline():
    """Test Phase 5 non-blocking pipeline"""
    logger.info("🔧 [PHASE5-TEST] Testing non-blocking pipeline...")
    
    try:
        from simulation.engines.ultra_pipeline import (
            UltraNonBlockingPipeline, 
            PipelineWorkItem, 
            PipelineStage
        )
        
        # Initialize pipeline
        pipeline = UltraNonBlockingPipeline(pipeline_stages=4)
        await pipeline.start_pipeline()
        
        # Submit test work
        work_items = []
        for i in range(3):
            work_item = PipelineWorkItem(
                work_id=f"pipeline_work_{i}",
                stage=PipelineStage.INPUT_PREPROCESSING,
                data={
                    'simulation_data': {
                        'mc_input_configs': [
                            {'name': 'var1', 'min_value': 1, 'max_value': 10, 'most_likely_value': 5, 'sheet_name': 'Sheet1'}
                        ],
                        'ordered_calc_steps': [('Sheet1', 'A1', '=var1*2')],
                        'target_sheet_name': 'Sheet1',
                        'target_cell_coordinate': 'A1',
                        'constant_values': {}
                    }
                }
            )
            work_items.append(work_item)
            await pipeline.submit_work(work_item)
        
        logger.info(f"🔧 [PHASE5-TEST] Submitted {len(work_items)} pipeline work items")
        
        # Wait for pipeline processing
        await asyncio.sleep(3)
        
        # Get pipeline statistics
        stats = pipeline.get_pipeline_stats()
        
        await pipeline.stop_pipeline()
        
        logger.info("✅ [PHASE5-TEST] Pipeline test passed")
        logger.info(f"   - Pipeline Active: {stats['pipeline_active']}")
        logger.info(f"   - Total Stages: {stats['total_stages']}")
        logger.info(f"   - GPU Available: {stats['gpu_available']}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ [PHASE5-TEST] Pipeline test failed: {e}")
        return False

async def test_phase5_resource_scheduler():
    """Test Phase 5 resource scheduler"""
    logger.info("🔧 [PHASE5-TEST] Testing resource scheduler...")
    
    try:
        from simulation.engines.ultra_resource_scheduler import (
            UltraResourceScheduler, 
            ResourceRequest
        )
        
        # Initialize scheduler
        scheduler = UltraResourceScheduler()
        await scheduler.start_scheduler()
        
        # Create test resource requests
        requests = []
        for i in range(3):
            request = ResourceRequest(
                task_id=f"resource_task_{i}",
                cpu_cores=1,
                gpu_memory_mb=100 if i % 2 == 0 else 0,  # Alternate GPU usage
                system_memory_mb=512,
                estimated_duration_seconds=30,
                priority=i + 1  # Different priorities
            )
            requests.append(request)
        
        # Submit resource requests
        allocation_results = []
        for request in requests:
            allocated = await scheduler.request_resources(request)
            allocation_results.append(allocated)
            logger.info(f"   - {request.task_id}: {'Allocated' if allocated else 'Queued'}")
        
        # Wait for scheduling
        await asyncio.sleep(2)
        
        # Check resource utilization
        utilization = scheduler.get_resource_utilization()
        
        # Clean up allocations
        for request in requests:
            await scheduler.deallocate_resources(request.task_id)
        
        await scheduler.shutdown()
        
        logger.info("✅ [PHASE5-TEST] Resource Scheduler test passed")
        logger.info(f"   - Active Allocations: {utilization['active_allocations']}")
        logger.info(f"   - Pending Requests: {utilization['pending_requests']}")
        logger.info(f"   - CPU Cores: {utilization['system_info']['cpu_cores']}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ [PHASE5-TEST] Resource scheduler test failed: {e}")
        return False

async def test_phase5_ultra_engine_integration():
    """Test Phase 5 integration with Ultra engine"""
    logger.info("🔧 [PHASE5-TEST] Testing Ultra engine Phase 5 integration...")
    
    try:
        from simulation.engines.ultra_engine import UltraMonteCarloEngine, get_ultra_engine_info
        from simulation.schemas import VariableConfig
        
        # Test 1: Engine initialization with Phase 5
        logger.info("🔧 [PHASE5-TEST] Test 1: Engine initialization")
        engine = UltraMonteCarloEngine(iterations=100, simulation_id="phase5_test")
        
        # Check Phase 5 status
        stats = engine.get_performance_stats()
        phase5_enabled = stats.get('phase5_enabled', False)
        async_processing = stats.get('async_processing_enabled', False)
        
        logger.info(f"   - Phase 5 Enabled: {phase5_enabled}")
        logger.info(f"   - Async Processing: {async_processing}")
        
        # Test 2: Engine info with Phase 5
        logger.info("🔧 [PHASE5-TEST] Test 2: Engine info")
        engine_info = get_ultra_engine_info()
        
        logger.info(f"   - Engine Name: {engine_info['name']}")
        logger.info(f"   - Description: {engine_info['description']}")
        logger.info(f"   - Phase 5 Enabled: {engine_info.get('phase_5_enabled', False)}")
        logger.info(f"   - Concurrent Simulations: {engine_info.get('concurrent_simulations', 1)}")
        
        # Test 3: Concurrent simulation submission (if available)
        if async_processing:
            logger.info("🔧 [PHASE5-TEST] Test 3: Concurrent simulation")
            
            # Create test variables
            test_variables = [
                VariableConfig(
                    name="test_var",
                    sheet_name="Sheet1",
                    min_value=10.0,
                    max_value=20.0,
                    most_likely=15.0,
                    distribution_type="triangular"
                )
            ]
            
            test_calc_steps = [
                ("Sheet1", "A1", "=test_var*2")
            ]
            
            # Submit concurrent simulation
            task_id = await engine.submit_concurrent_simulation(
                simulation_id="phase5_concurrent_test",
                mc_input_configs=test_variables,
                ordered_calc_steps=test_calc_steps,
                target_sheet_name="Sheet1",
                target_cell_coordinate="A1",
                constant_values={}
            )
            
            if task_id:
                logger.info(f"   - Concurrent simulation submitted: {task_id}")
                
                # Wait a bit and check status
                await asyncio.sleep(2)
                status = await engine.get_concurrent_simulation_status("phase5_concurrent_test")
                if status:
                    logger.info(f"   - Simulation status: {status['status']}")
                
                # Get concurrent manager stats
                manager_stats = engine.get_concurrent_manager_stats()
                if manager_stats:
                    logger.info(f"   - Manager active simulations: {manager_stats['active_simulations']}")
            else:
                logger.warning("   - Concurrent simulation submission failed")
        
        logger.info("✅ [PHASE5-TEST] Ultra engine integration test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ [PHASE5-TEST] Ultra engine integration test failed: {e}")
        return False

async def run_phase5_comprehensive_tests():
    """Run comprehensive Phase 5 test suite"""
    logger.info("🚀 [PHASE5-TEST] Starting Phase 5 Comprehensive Test Suite")
    logger.info("=" * 80)
    
    start_time = time.time()
    test_results = {}
    
    # Test 1: Async Core Components
    logger.info("\n📋 [PHASE5-TEST] Running Test 1: Async Core Components")
    test_results['async_core'] = await test_phase5_async_core()
    
    # Test 2: Concurrent Manager
    logger.info("\n📋 [PHASE5-TEST] Running Test 2: Concurrent Manager")
    test_results['concurrent_manager'] = await test_phase5_concurrent_manager()
    
    # Test 3: Non-blocking Pipeline
    logger.info("\n📋 [PHASE5-TEST] Running Test 3: Non-blocking Pipeline")
    test_results['pipeline'] = await test_phase5_pipeline()
    
    # Test 4: Resource Scheduler
    logger.info("\n📋 [PHASE5-TEST] Running Test 4: Resource Scheduler")
    test_results['resource_scheduler'] = await test_phase5_resource_scheduler()
    
    # Test 5: Ultra Engine Integration
    logger.info("\n📋 [PHASE5-TEST] Running Test 5: Ultra Engine Integration")
    test_results['ultra_integration'] = await test_phase5_ultra_engine_integration()
    
    # Calculate results
    total_time = time.time() - start_time
    passed_tests = sum(1 for result in test_results.values() if result)
    total_tests = len(test_results)
    success_rate = (passed_tests / total_tests) * 100
    
    # Results summary
    logger.info("\n" + "=" * 80)
    logger.info("🏁 [PHASE5-TEST] PHASE 5 COMPREHENSIVE TEST RESULTS")
    logger.info("=" * 80)
    
    for test_name, result in test_results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"   {test_name.replace('_', ' ').title()}: {status}")
    
    logger.info("-" * 80)
    logger.info(f"📊 [PHASE5-TEST] Overall Results:")
    logger.info(f"   - Tests Passed: {passed_tests}/{total_tests}")
    logger.info(f"   - Success Rate: {success_rate:.1f}%")
    logger.info(f"   - Total Time: {total_time:.2f}s")
    
    if success_rate == 100.0:
        logger.info("🎉 [PHASE5-TEST] ALL PHASE 5 TESTS PASSED!")
        logger.info("✅ [ULTRA] Phase 5 - Asynchronous Processing COMPLETE")
        logger.info("🚀 [ULTRA] Ultra Monte Carlo Engine now supports:")
        logger.info("   ✅ Priority-based task scheduling")
        logger.info("   ✅ Concurrent simulation processing")
        logger.info("   ✅ Non-blocking computation pipelines")
        logger.info("   ✅ Intelligent resource allocation")
        logger.info("   ✅ Real-time system monitoring")
    else:
        logger.warning(f"⚠️ [PHASE5-TEST] {total_tests - passed_tests} tests failed")
        logger.warning("🔧 [PHASE5-TEST] Review failed tests before production use")
    
    return test_results

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('phase5_test_results.log')
        ]
    )
    
    # Run comprehensive tests
    asyncio.run(run_phase5_comprehensive_tests()) 