"""
B2B API Simulation Worker - EXACT FRONTEND PIPELINE REPLICATION

This module replicates the frontend's _run_ultra_multi_target_simulation function
line by line to ensure identical behavior and fix dependency analysis issues.
"""
import sys
import os
import logging
import time
import asyncio
import uuid as uuid_module
import numpy as np
from typing import Dict, Any, List
from pathlib import Path

# Add project root to Python path for imports
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

logger = logging.getLogger(__name__)

# Ensure logger is configured to output to console for Docker logs
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


def run_isolated_simulation(simulation_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    EXACT REPLICATION OF FRONTEND ULTRA ENGINE PIPELINE
    
    This function replicates the _run_ultra_multi_target_simulation function
    line by line to ensure identical behavior and fix dependency issues.
    
    The frontend pipeline consists of:
    1. Excel File Loading & Validation
    2. Excel Parsing & Formula Extraction  
    3. Formula Dependency Analysis (Multi-target)
    4. Sheet Consistency Checks & Fixes
    5. Excel Constants Loading (Sophisticated)
    6. Multi-Target Ultra Engine Simulation
    7. Results Processing & Statistics
    """
    simulation_id = simulation_config["simulation_id"]
    file_id = simulation_config["file_id"]
    file_path = simulation_config["file_path"]
    variables = simulation_config["variables"]
    output_cells = simulation_config["output_cells"]  # List of strings like ["D10"]
    iterations = simulation_config["iterations"]

    try:
        logger.info(f"🎯 [B2B_ULTRA_REPLICATION] Starting {simulation_id} with {len(output_cells)} targets")
        logger.info(f"🎯 [B2B_ULTRA_REPLICATION] Process ID: {os.getpid()}")

        start_time = time.time()

        # Set up async event loop for the isolated process (REQUIRED FOR ASYNC FUNCTIONS)
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        # Import required modules (exactly like frontend)
        from simulation.schemas import VariableConfig, ConstantConfig
        from simulation.engines.ultra_engine import create_ultra_engine
        from excel_parser.service import get_formulas_for_file, get_constants_for_file
        from simulation.formula_utils import get_evaluation_order

        # ✅ STEP 1: INITIALIZATION (EXACT FRONTEND REPLICATION)
        logger.info(f"📊 [B2B_ULTRA] STEP 1: Excel File Loading")
        
        # Create Ultra engine instance (exactly like frontend)
        ultra_engine = create_ultra_engine(iterations=iterations, simulation_id=simulation_id)
        
        # Convert variables to mc_inputs format (exactly like frontend)
        mc_inputs = []
        mc_input_cells = set()
        
        for var in variables:
            # Convert B2B variable format to VariableConfig
            logger.info(f"🔍 [B2B_DEBUG] Processing variable: {var}")
            logger.info(f"🔍 [B2B_DEBUG] Distribution keys: {list(var['distribution'].keys())}")
            
            # Extract distribution values with detailed debugging
            min_val = var["distribution"].get("min_value", var["distribution"].get("min", 0.0))
            most_likely_val = var["distribution"].get("most_likely", var["distribution"].get("mode", 0.0))
            max_val = var["distribution"].get("max_value", var["distribution"].get("max", 1.0))
            
            logger.info(f"🔍 [B2B_DEBUG] Extracted values: min={min_val}, most_likely={most_likely_val}, max={max_val}")
            
            # Ensure we have valid float values
            if min_val is None:
                min_val = 0.0
            if most_likely_val is None:
                most_likely_val = 0.5
            if max_val is None:
                max_val = 1.0
                
            var_config = VariableConfig(
                name=var["cell"],  # "B5" - this is the cell coordinate
                sheet_name="Sheet1",  # Will be auto-detected later
                min_value=float(min_val),
                most_likely=float(most_likely_val),
                max_value=float(max_val)
            )
            mc_inputs.append(var_config)
            # Build mc_input_cells set (used for dependency analysis)
            mc_input_cells.add(("Sheet1", var["cell"].upper()))

        # Convert outputs to target_cells format (exactly like frontend)
        target_cells = output_cells  # Already a list of strings

        logger.info(f"🎯 [B2B_ULTRA] Variables: {len(mc_inputs)}, Targets: {len(target_cells)}")

        # ✅ STEP 2: EXCEL PARSING PROGRESS (EXACT FRONTEND REPLICATION)
        logger.info(f"📊 [B2B_ULTRA] STEP 2: Excel Parsing")
        
        # Get all formulas from the Excel file (exactly like frontend)
        try:
            all_formulas = loop.run_until_complete(get_formulas_for_file(file_id))
            total_formulas = sum(len(sheet_formulas) for sheet_formulas in all_formulas.values()) if all_formulas else 0
            
            logger.info(f"🎯 [B2B_ULTRA] Loaded {total_formulas} formulas from {len(all_formulas)} sheets")
            
        except Exception as e:
            logger.error(f"🎯 [B2B_ULTRA] Failed to load formulas: {e}")
            all_formulas = {}

        # ✅ STEP 3: FORMULA DEPENDENCY ANALYSIS (EXACT FRONTEND REPLICATION)
        logger.info(f"🔧 [B2B_ULTRA] STEP 3: Formula Dependency Analysis")
        
        # Get ordered calculation steps for ALL target cells (exactly like frontend)
        all_ordered_steps = []
        seen_steps = set()  # Avoid duplicates

        for i, target_cell in enumerate(target_cells):
            logger.info(f"🔧 [B2B_ULTRA] Analyzing dependencies for target {i+1}/{len(target_cells)}: {target_cell}")
            
            # Parse target cell to get sheet and coordinate (exactly like frontend)
            if "!" in target_cell:
                target_sheet, target_coord = target_cell.split("!", 1)
            else:
                # ✅ CRITICAL FIX: Use actual sheet name from Excel file, not hardcoded "Sheet1"
                if all_formulas:
                    # Get the first available sheet name from formulas
                    available_sheets = list(all_formulas.keys())
                    target_sheet = available_sheets[0] if available_sheets else "Sheet1"
                    logger.info(f"🎯 [B2B_ULTRA] Auto-detected sheet: {target_sheet} for target {target_cell}")
                else:
                    target_sheet = "Sheet1"  # Fallback
                target_coord = target_cell

            logger.info(f"🎯 [B2B_ULTRA] Getting dependencies for {target_sheet}!{target_coord}")

            target_steps = get_evaluation_order(
                target_sheet_name=target_sheet,
                target_cell_coord=target_coord,
                all_formulas=all_formulas,
                mc_input_cells=mc_input_cells,
                engine_type='ultra'
            )

            # Add unique steps to the combined list (exactly like frontend)
            for step in target_steps:
                step_key = (step[0], step[1].upper())  # (sheet, cell) as key
                if step_key not in seen_steps:
                    all_ordered_steps.append(step)
                    seen_steps.add(step_key)

            logger.info(f"🎯 [B2B_ULTRA] Added {len(target_steps)} steps for {target_cell} (total unique: {len(all_ordered_steps)})")

        ordered_calc_steps = all_ordered_steps
        logger.info(f"🎯 [B2B_ULTRA] Found {len(ordered_calc_steps)} calculation steps")

        # ✅ SHEET CONSISTENCY DEBUG (EXACT FRONTEND REPLICATION)
        mc_input_sheets = set()
        target_sheets = set()
        calc_step_sheets = set()

        for mc_sheet, mc_cell in mc_input_cells:
            mc_input_sheets.add(mc_sheet)

        for target_cell in target_cells:
            if "!" in target_cell:
                sheet_name = target_cell.split("!", 1)[0]
                target_sheets.add(sheet_name)
            else:
                # Auto-detected sheet logic (from earlier in this function)
                if all_formulas:
                    available_sheets = list(all_formulas.keys())
                    target_sheets.add(available_sheets[0] if available_sheets else "Sheet1")

        for sheet, cell, formula in ordered_calc_steps[:10]:  # Check first 10
            calc_step_sheets.add(sheet)

        logger.info(f"🔍 [B2B_ULTRA] Monte Carlo input sheets: {mc_input_sheets}")
        logger.info(f"🔍 [B2B_ULTRA] Target cell sheets: {target_sheets}")
        logger.info(f"🔍 [B2B_ULTRA] Calculation step sheets (sample): {calc_step_sheets}")

        # ✅ SHEET MISMATCH FIX (EXACT FRONTEND REPLICATION)
        all_sheets = mc_input_sheets | target_sheets | calc_step_sheets
        if len(all_sheets) > 1:
            logger.warning(f"🚨 [B2B_ULTRA] Multiple sheets detected: {all_sheets}")
            logger.warning(f"🚨 [B2B_ULTRA] This could cause Monte Carlo variable lookup failures!")
            
            # 🔧 SHEET MISMATCH FIX: Inject Monte Carlo variables on ALL calculation sheets
            original_mc_inputs = list(mc_input_cells)
            expanded_mc_inputs = set()

            for orig_sheet, orig_cell in original_mc_inputs:
                for calculation_sheet in calc_step_sheets:
                    expanded_mc_inputs.add((calculation_sheet, orig_cell))

            mc_input_cells = expanded_mc_inputs
            logger.info(f"🔧 [B2B_ULTRA] Expanded MC inputs from {len(original_mc_inputs)} to {len(mc_input_cells)} across sheets: {calc_step_sheets}")
        else:
            logger.info(f"✅ [B2B_ULTRA] All operations on single sheet: {all_sheets}")

        # ✅ CRITICAL FIX: Build calculated_cells set (EXACT FRONTEND REPLICATION)
        calculated_cells = set()
        for sheet, cell, _ in ordered_calc_steps:
            # Normalize cell reference (remove $ signs for absolute references)
            normalized_cell = cell.replace('$', '').upper()
            calculated_cells.add((sheet, normalized_cell))

        # ✅ CRITICAL FIX: Load Excel constants properly (EXACT FRONTEND REPLICATION)
        # Get first target sheet for constants loading (primary sheet)
        primary_target_sheet = "Sheet1"  # Default fallback
        if target_cells:
            first_target = target_cells[0]
            if "!" in first_target:
                primary_target_sheet = first_target.split("!", 1)[0]
            elif all_formulas:
                available_sheets = list(all_formulas.keys())
                primary_target_sheet = available_sheets[0] if available_sheets else "Sheet1"

        # Load Excel constants excluding MC input cells (exactly like frontend)
        all_file_constants = loop.run_until_complete(get_constants_for_file(
            file_id, 
            exclude_cells=mc_input_cells, 
            target_sheet=primary_target_sheet
        ))

        # ✅ CRITICAL: Only use constants for cells that are NOT being calculated
        constant_values = {}
        constants_used = 0
        for (sheet, cell), value in all_file_constants.items():
            # Normalize cell reference for comparison
            normalized_cell = cell.replace('$', '').upper()
            if (sheet, normalized_cell) not in calculated_cells:
                constant_values[(sheet, cell)] = value
                constants_used += 1

        logger.info(f"📊 [B2B_ULTRA] Using {constants_used} Excel constants for non-calculated cells")
        logger.info(f"📊 [B2B_ULTRA] Skipping constants for {len(calculated_cells)} cells that will be calculated fresh")

        # ✅ CRITICAL: Run TRUE multi-target simulation (EXACT FRONTEND REPLICATION)
        logger.info(f"⚡ [B2B_ULTRA] Starting Ultra Engine Multi-Target Simulation")
        logger.info(f"⚡ [B2B_ULTRA] - Targets: {target_cells}")
        logger.info(f"⚡ [B2B_ULTRA] - Variables: {len(mc_inputs)}")
        logger.info(f"⚡ [B2B_ULTRA] - Constants: {constants_used}")
        logger.info(f"⚡ [B2B_ULTRA] - Calculation steps: {len(ordered_calc_steps)}")
        logger.info(f"⚡ [B2B_ULTRA] - Iterations: {iterations}")

        # Use the Ultra Engine's multi-target capability (exactly like frontend)
        multi_target_result = loop.run_until_complete(
            ultra_engine.run_multi_target_simulation(
                target_cells=target_cells,
                mc_input_configs=mc_inputs,
                ordered_calc_steps=ordered_calc_steps,
                constant_values=constant_values,
                workbook_path=file_path
            )
        )

        logger.info(f"🎯 [B2B_ULTRA] Simulation completed successfully")
        logger.info(f"🎯 [B2B_ULTRA] Targets calculated: {len(multi_target_result.targets)}")
        logger.info(f"🎯 [B2B_ULTRA] Iterations completed: {multi_target_result.total_iterations}")
        logger.info(f"🎯 [B2B_ULTRA] Correlations calculated: {len(multi_target_result.correlations)}")

        # ✅ RESULTS PROCESSING (ADAPTED FROM FRONTEND)
        execution_time = time.time() - start_time

        # Extract statistics for ALL targets (Fixed: B2B API now supports multiple targets)
        statistics = {}
        histogram_data = {}
        raw_results = {}

        # Process ALL target cells, not just the first one
        for target_cell in target_cells:
            target_name = f"Output_{target_cell}"
            
            if target_cell in multi_target_result.target_results:
                results_array = multi_target_result.target_results[target_cell]
            
                if results_array is not None and len(results_array) > 0:
                    # Calculate statistics manually
                    stats = {
                        "mean": float(np.mean(results_array)),
                        "std_dev": float(np.std(results_array)),
                        "min": float(np.min(results_array)),
                        "max": float(np.max(results_array)),
                        "percentiles": {
                            "5": float(np.percentile(results_array, 5)),
                            "25": float(np.percentile(results_array, 25)),
                            "50": float(np.percentile(results_array, 50)),
                            "75": float(np.percentile(results_array, 75)),
                            "95": float(np.percentile(results_array, 95))
                        }
                    }
                    statistics[target_name] = stats

                    # Generate histogram data
                    hist, bin_edges = np.histogram(results_array, bins=50)
                    histogram_data[target_name] = {
                        "histogram": hist.tolist(),
                        "bin_edges": bin_edges.tolist(),
                        "bin_centers": ((bin_edges[:-1] + bin_edges[1:]) / 2).tolist()
                    }
                    
                    # Handle both numpy arrays and lists
                    if hasattr(results_array, 'tolist'):
                        raw_results[target_name] = results_array.tolist()
                    else:
                        raw_results[target_name] = list(results_array)

        logger.info(f"✅ [B2B_ULTRA] B2B simulation {simulation_id} completed successfully in {execution_time:.2f}s")

        return {
            "status": "completed",
            "execution_time": execution_time,
            "statistics": statistics,
            "histogram_data": histogram_data,
            "raw_results": raw_results,
            "iteration_errors": [],
            "metadata": {
                "iterations": iterations,
                "variables_count": len(mc_inputs),
                "constants_count": constants_used,
                "calculation_steps": len(ordered_calc_steps),
                "output_cells_count": len(target_cells),
                "process_id": os.getpid(),
                "engine_id": f"b2b_ultra_{uuid_module.uuid4().hex[:8]}",
                "frontend_replication": True,
                "multi_target_enabled": True
            }
        }

    except Exception as e:
        logger.error(f"❌ [B2B_ULTRA] Simulation {simulation_config.get('simulation_id', 'unknown')} failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            "status": "failed",
            "error": str(e),
            "execution_time": time.time() - start_time if 'start_time' in locals() else 0,
            "metadata": {
                "process_id": os.getpid(),
                "error_type": type(e).__name__,
                "frontend_replication": True
            }
        }


def update_progress_callback(simulation_id: str, progress_data: Dict[str, Any]) -> None:
    """
    Progress callback for isolated simulations.
    
    Note: This runs in the isolated process, so it can't directly update
    the main application's progress store. Progress updates are handled
    by the monitoring system in start_isolated_simulation.
    """
    # In isolated process, we can only log progress
    logger.info(f"🔄 [ISOLATED_WORKER] Progress {simulation_id}: {progress_data.get('progress_percentage', 0)}%")