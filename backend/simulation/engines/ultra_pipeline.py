"""
ULTRA MONTE CARLO ENGINE - PHASE 5: NON-BLOCKING PIPELINE
Non-blocking pipeline for asynchronous formula evaluation and computation.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum

# GPU imports with fallback
try:
    import cupy as cp
    CUDA_AVAILABLE = True
except ImportError:
    cp = None
    CUDA_AVAILABLE = False

logger = logging.getLogger(__name__)

class PipelineStage(Enum):
    """Pipeline stage types"""
    INPUT_PREPROCESSING = "input_preprocessing"
    FORMULA_EVALUATION = "formula_evaluation"
    GPU_COMPUTATION = "gpu_computation"
    OUTPUT_PROCESSING = "output_processing"

@dataclass
class PipelineWorkItem:
    """Work item that flows through the pipeline"""
    work_id: str
    stage: PipelineStage
    data: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    processing_history: List[Dict[str, Any]] = field(default_factory=list)
    
    def record_stage_completion(self, stage: PipelineStage, processing_time: float):
        """Record completion of a pipeline stage"""
        self.processing_history.append({
            'stage': stage.value,
            'processing_time': processing_time,
            'completed_at': time.time()
        })

class UltraNonBlockingPipeline:
    """Non-blocking pipeline for asynchronous formula evaluation and computation"""
    
    def __init__(self, pipeline_stages: int = 4):
        self.pipeline_stages = pipeline_stages
        self.stage_queues = [asyncio.Queue() for _ in range(pipeline_stages)]
        self.stage_workers = []
        self.pipeline_active = False
        
        # Stage definitions
        self.stage_definitions = [
            PipelineStage.INPUT_PREPROCESSING,
            PipelineStage.FORMULA_EVALUATION,
            PipelineStage.GPU_COMPUTATION,
            PipelineStage.OUTPUT_PROCESSING
        ]
        
        # Performance tracking
        self.stage_stats = {stage: {
            'items_processed': 0,
            'total_processing_time': 0.0,
            'avg_processing_time': 0.0,
            'errors': 0
        } for stage in self.stage_definitions}
        
        self.gpu_available = CUDA_AVAILABLE
        
        logger.info(f"🔧 [ULTRA-PHASE5] Non-blocking Pipeline initialized with {pipeline_stages} stages")
        logger.info(f"   - GPU Available: {self.gpu_available}")
    
    async def start_pipeline(self):
        """Start the non-blocking pipeline"""
        if self.pipeline_active:
            return
        
        self.pipeline_active = True
        
        # Start worker tasks for each stage
        for i in range(self.pipeline_stages):
            stage_type = self.stage_definitions[i] if i < len(self.stage_definitions) else None
            worker = asyncio.create_task(self._stage_worker(i, stage_type))
            self.stage_workers.append(worker)
        
        logger.info(f"🔧 [ULTRA-PHASE5] Pipeline started with {self.pipeline_stages} stages")
    
    async def _stage_worker(self, stage_id: int, stage_type: Optional[PipelineStage]):
        """Worker for a specific pipeline stage"""
        while self.pipeline_active:
            try:
                # Get work from this stage's queue
                work_item = await self.stage_queues[stage_id].get()
                
                if work_item is None:  # Shutdown signal
                    break
                
                # Process the work item
                stage_start_time = time.time()
                
                try:
                    result = await self._process_stage(stage_id, stage_type, work_item)
                    
                    # Record stage completion
                    processing_time = time.time() - stage_start_time
                    if stage_type:
                        work_item.record_stage_completion(stage_type, processing_time)
                        
                        # Update statistics
                        self.stage_stats[stage_type]['items_processed'] += 1
                        self.stage_stats[stage_type]['total_processing_time'] += processing_time
                        self.stage_stats[stage_type]['avg_processing_time'] = (
                            self.stage_stats[stage_type]['total_processing_time'] / 
                            self.stage_stats[stage_type]['items_processed']
                        )
                    
                    # Pass to next stage if not final
                    if stage_id < self.pipeline_stages - 1 and result is not None:
                        await self.stage_queues[stage_id + 1].put(result)
                
                except Exception as e:
                    if stage_type:
                        self.stage_stats[stage_type]['errors'] += 1
                    logger.error(f"🔧 [ULTRA-PHASE5] Stage {stage_id} processing error: {e}")
                
                # Mark task as done
                self.stage_queues[stage_id].task_done()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"🔧 [ULTRA-PHASE5] Stage {stage_id} worker error: {e}")
                await asyncio.sleep(0.1)
    
    async def _process_stage(self, stage_id: int, stage_type: Optional[PipelineStage], work_item: PipelineWorkItem) -> Optional[PipelineWorkItem]:
        """Process work item at specific stage"""
        if not stage_type:
            return work_item
        
        try:
            if stage_type == PipelineStage.INPUT_PREPROCESSING:
                return await self._stage_input_preprocessing(work_item)
            elif stage_type == PipelineStage.FORMULA_EVALUATION:
                return await self._stage_formula_evaluation(work_item)
            elif stage_type == PipelineStage.GPU_COMPUTATION:
                return await self._stage_gpu_computation(work_item)
            elif stage_type == PipelineStage.OUTPUT_PROCESSING:
                return await self._stage_output_processing(work_item)
            else:
                return work_item
        
        except Exception as e:
            logger.error(f"🔧 [ULTRA-PHASE5] Stage {stage_type.value} error: {e}")
            work_item.metadata['error'] = str(e)
            return work_item
    
    async def _stage_input_preprocessing(self, work_item: PipelineWorkItem) -> PipelineWorkItem:
        """Stage 0: Input preprocessing and validation"""
        # Validate input data
        if 'simulation_data' not in work_item.data:
            raise ValueError("Missing simulation_data in work item")
        
        # Preprocess input data
        sim_data = work_item.data['simulation_data']
        
        # Extract key components
        work_item.data['mc_inputs'] = sim_data.get('mc_input_configs', [])
        work_item.data['calc_steps'] = sim_data.get('ordered_calc_steps', [])
        work_item.data['constants'] = sim_data.get('constant_values', {})
        
        # Add preprocessing metadata
        work_item.metadata['input_variables'] = len(work_item.data['mc_inputs'])
        work_item.metadata['formula_count'] = len(work_item.data['calc_steps'])
        
        # Simulate processing time
        await asyncio.sleep(0.001)
        
        return work_item
    
    async def _stage_formula_evaluation(self, work_item: PipelineWorkItem) -> PipelineWorkItem:
        """Stage 1: Formula evaluation and computation"""
        mc_inputs = work_item.data.get('mc_inputs', [])
        
        # Simulate formula evaluation
        evaluation_results = []
        for var_config in mc_inputs:
            # Simple triangular distribution simulation
            min_val = var_config.get('min_value', 0)
            max_val = var_config.get('max_value', 100)
            mode_val = var_config.get('most_likely_value', (min_val + max_val) / 2)
            
            # Simple triangular distribution value
            result_value = (min_val + mode_val + max_val) / 3
            evaluation_results.append(result_value)
        
        work_item.data['evaluation_results'] = evaluation_results
        work_item.metadata['evaluated_formulas'] = len(evaluation_results)
        
        # Simulate processing time
        await asyncio.sleep(0.005)
        
        return work_item
    
    async def _stage_gpu_computation(self, work_item: PipelineWorkItem) -> PipelineWorkItem:
        """Stage 2: GPU computation and acceleration"""
        evaluation_results = work_item.data.get('evaluation_results', [])
        
        if self.gpu_available and len(evaluation_results) > 100:
            # Use GPU for computations
            try:
                # Convert to GPU arrays
                gpu_data = cp.array(evaluation_results)
                
                # Perform GPU computations
                gpu_mean = cp.mean(gpu_data)
                gpu_std = cp.std(gpu_data)
                gpu_min = cp.min(gpu_data)
                gpu_max = cp.max(gpu_data)
                
                # Convert back to CPU
                work_item.data['gpu_stats'] = {
                    'mean': float(gpu_mean),
                    'std': float(gpu_std),
                    'min': float(gpu_min),
                    'max': float(gpu_max)
                }
                
                work_item.metadata['gpu_accelerated'] = True
                
            except Exception as e:
                logger.warning(f"🔧 [ULTRA-PHASE5] GPU computation failed: {e}")
                work_item.metadata['gpu_accelerated'] = False
                # CPU fallback
                import numpy as np
                np_data = np.array(evaluation_results)
                work_item.data['gpu_stats'] = {
                    'mean': float(np.mean(np_data)),
                    'std': float(np.std(np_data)),
                    'min': float(np.min(np_data)),
                    'max': float(np.max(np_data))
                }
        else:
            # CPU computation
            work_item.metadata['gpu_accelerated'] = False
            if evaluation_results:
                import numpy as np
                np_data = np.array(evaluation_results)
                work_item.data['gpu_stats'] = {
                    'mean': float(np.mean(np_data)),
                    'std': float(np.std(np_data)),
                    'min': float(np.min(np_data)),
                    'max': float(np.max(np_data))
                }
            else:
                work_item.data['gpu_stats'] = {
                    'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0
                }
        
        # Simulate processing time
        processing_time = 0.01 if work_item.metadata.get('gpu_accelerated', False) else 0.03
        await asyncio.sleep(processing_time)
        
        return work_item
    
    async def _stage_output_processing(self, work_item: PipelineWorkItem) -> PipelineWorkItem:
        """Stage 3: Output processing and formatting"""
        evaluation_results = work_item.data.get('evaluation_results', [])
        gpu_stats = work_item.data.get('gpu_stats', {})
        
        # Format final output
        work_item.data['final_output'] = {
            'simulation_id': work_item.work_id,
            'results': evaluation_results,
            'statistics': gpu_stats,
            'metadata': work_item.metadata,
            'processing_summary': {
                'total_stages': len(work_item.processing_history),
                'total_processing_time': sum(stage['processing_time'] for stage in work_item.processing_history),
                'gpu_accelerated': work_item.metadata.get('gpu_accelerated', False)
            }
        }
        
        # Simulate processing time
        await asyncio.sleep(0.001)
        
        return work_item
    
    async def submit_work(self, work_item: PipelineWorkItem) -> str:
        """Submit work to the pipeline"""
        if not self.pipeline_active:
            await self.start_pipeline()
        
        await self.stage_queues[0].put(work_item)
        logger.info(f"🔧 [ULTRA-PHASE5] Work submitted to pipeline: {work_item.work_id}")
        return work_item.work_id
    
    async def stop_pipeline(self):
        """Stop the pipeline gracefully"""
        self.pipeline_active = False
        
        # Send shutdown signals
        for queue in self.stage_queues:
            await queue.put(None)
        
        # Wait for workers to finish
        for worker in self.stage_workers:
            await worker
        
        logger.info(f"🔧 [ULTRA-PHASE5] Pipeline stopped")
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get comprehensive pipeline statistics"""
        return {
            'pipeline_active': self.pipeline_active,
            'total_stages': self.pipeline_stages,
            'queue_sizes': [queue.qsize() for queue in self.stage_queues],
            'stage_statistics': self.stage_stats,
            'gpu_available': self.gpu_available,
            'workers_active': len([w for w in self.stage_workers if not w.done()]),
            'pipeline_type': 'UltraNonBlockingPipeline'
        }

# Factory function
def create_non_blocking_pipeline(stages: int = 4) -> UltraNonBlockingPipeline:
    """Create a non-blocking processing pipeline"""
    return UltraNonBlockingPipeline(pipeline_stages=stages) 