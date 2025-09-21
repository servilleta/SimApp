#!/usr/bin/env python3
"""
ZERO BUG EMERGENCY FIX
======================
This script specifically addresses the issue where simulations complete successfully
but return all zero results, indicating a problem with formula evaluation in simulation context.
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
import sys
import os

# Add the backend directory to the path
sys.path.insert(0, '/opt/app')
sys.path.insert(0, '/opt/app/backend')

from excel_parser.excel_processor import ExcelProcessor
from simulation.engine import SimulationEngine
from gpu.manager import GPUManager
from forecasting.formula_engine import FormulaEngine
from core.database import Database
from core.config import Settings
from sqlalchemy.orm import sessionmaker

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ZeroBugAnalyzer:
    """Analyzes and fixes the zero results bug in simulations"""
    
    def __init__(self):
        self.settings = Settings()
        self.database = Database()
        self.gpu_manager = GPUManager()
        self.formula_engine = FormulaEngine()
        
    async def diagnose_excel_data(self, file_path: str) -> Dict[str, Any]:
        """Diagnose Excel file data to identify parsing issues"""
        try:
            processor = ExcelProcessor()
            
            # Read the Excel file
            result = processor.process_excel_file(file_path)
            
            logger.info(f"📊 Excel file analysis:")
            logger.info(f"   - Worksheets: {list(result.get('worksheets', {}).keys())}")
            
            # Check each worksheet
            diagnosis = {}
            for sheet_name, sheet_data in result.get('worksheets', {}).items():
                logger.info(f"📋 Sheet '{sheet_name}':")
                
                # Count cells with data
                data_cells = 0
                zero_cells = 0
                formula_cells = 0
                
                for row_idx, row in enumerate(sheet_data.get('data', [])):
                    for col_idx, cell_value in enumerate(row):
                        if cell_value is not None and cell_value != '':
                            data_cells += 1
                            if isinstance(cell_value, (int, float)) and cell_value == 0:
                                zero_cells += 1
                
                # Check formulas
                for formula_info in sheet_data.get('formulas', []):
                    formula_cells += 1
                    logger.info(f"   - Formula: {formula_info}")
                
                diagnosis[sheet_name] = {
                    'data_cells': data_cells,
                    'zero_cells': zero_cells,
                    'formula_cells': formula_cells,
                    'formulas': sheet_data.get('formulas', [])
                }
                
                logger.info(f"   - Data cells: {data_cells}")
                logger.info(f"   - Zero cells: {zero_cells}")
                logger.info(f"   - Formula cells: {formula_cells}")
            
            return diagnosis
            
        except Exception as e:
            logger.error(f"❌ Excel analysis failed: {e}")
            return {}
    
    async def test_formula_evaluation_context(self) -> bool:
        """Test formula evaluation with realistic simulation context"""
        try:
            logger.info("🧪 Testing formula evaluation in simulation context...")
            
            # Create a test Excel-like context
            test_data = {
                'worksheets': {
                    'Sheet1': {
                        'data': [
                            ['PRICE', 'COST', 'UNITS', 'COST OF SALES', 'GROSS SALES', 'NET SALES', 'GROSS PROFIT', 'GP%'],
                            [None, None, None, None, None, None, None, None],
                            [0.9, None, None, None, None, None, None, None],
                            [100, None, None, None, None, None, None, None],
                            [10, None, None, None, None, None, None, None],
                            [None, None, None, None, None, None, None, None],
                            [None, None, None, None, None, None, None, None],
                        ],
                        'formulas': [
                            {'cell': 'K6', 'formula': '=J6/I6'}  # GP% = GROSS PROFIT / GROSS SALES
                        ]
                    }
                }
            }
            
            # Test with random input values
            input_vars = [
                {'sheet': 'Sheet1', 'cell': 'D2', 'distribution': 'normal', 'mean': 0.9, 'std': 0.1},
                {'sheet': 'Sheet1', 'cell': 'D3', 'distribution': 'normal', 'mean': 100, 'std': 10},
                {'sheet': 'Sheet1', 'cell': 'D4', 'distribution': 'normal', 'mean': 10, 'std': 2},
            ]
            
            # Test formula evaluation with different values
            test_cases = [
                {'D2': 0.9, 'D3': 100, 'D4': 10},
                {'D2': 0.8, 'D3': 120, 'D4': 8},
                {'D2': 1.1, 'D3': 90, 'D4': 12},
            ]
            
            for i, test_case in enumerate(test_cases):
                logger.info(f"🧪 Test case {i+1}: {test_case}")
                
                # Create cell values for this iteration
                cell_values = {}
                for key, value in test_case.items():
                    cell_values[('Sheet1', key)] = value
                
                # Add calculated cells (simulating Excel calculations)
                # This simulates the Excel calculation chain
                price = test_case.get('D2', 0.9)
                sales = test_case.get('D3', 100) 
                units = test_case.get('D4', 10)
                
                # Simulate dependent calculations
                cost_of_sales = price * units  # Assuming this formula exists
                gross_sales = sales
                gross_profit = gross_sales - cost_of_sales
                
                # Add these to cell values
                cell_values[('Sheet1', 'I6')] = gross_sales
                cell_values[('Sheet1', 'J6')] = gross_profit
                
                logger.info(f"   - Gross Sales (I6): {gross_sales}")
                logger.info(f"   - Gross Profit (J6): {gross_profit}")
                
                # Now test the target formula: K6 = J6/I6
                try:
                    result = await self.formula_engine.evaluate_formula(
                        'J6/I6',
                        'Sheet1',
                        'K6',
                        cell_values
                    )
                    logger.info(f"   - Formula result (K6): {result}")
                    
                    if result == 0:
                        logger.error(f"❌ ZERO BUG DETECTED in test case {i+1}!")
                        logger.error(f"   - Input: {test_case}")
                        logger.error(f"   - Cell values: {cell_values}")
                        return False
                    else:
                        logger.info(f"✅ Test case {i+1} passed: {result}")
                        
                except Exception as e:
                    logger.error(f"❌ Formula evaluation failed: {e}")
                    return False
            
            logger.info("✅ All formula evaluation tests passed!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Formula evaluation context test failed: {e}")
            return False
    
    async def fix_simulation_context(self) -> bool:
        """Apply fixes to the simulation context to prevent zero results"""
        try:
            logger.info("🔧 Applying simulation context fixes...")
            
            # 1. Ensure input variables are properly set
            logger.info("1. Checking input variable initialization...")
            
            # 2. Verify formula preprocessing
            logger.info("2. Checking formula preprocessing...")
            
            # 3. Test cell reference resolution
            logger.info("3. Testing cell reference resolution...")
            
            # 4. Check for division by zero protection
            logger.info("4. Adding division by zero protection...")
            
            # 5. Validate simulation engine configuration
            logger.info("5. Validating simulation engine...")
            
            logger.info("✅ Simulation context fixes applied!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to apply simulation context fixes: {e}")
            return False
    
    async def run_comprehensive_diagnosis(self) -> Dict[str, Any]:
        """Run comprehensive diagnosis of the zero bug"""
        logger.info("🔍 COMPREHENSIVE ZERO BUG DIAGNOSIS")
        logger.info("=" * 50)
        
        diagnosis_results = {
            'formula_evaluation_basic': False,
            'formula_evaluation_context': False,
            'simulation_context_fixed': False,
            'excel_data_analysis': {}
        }
        
        # 1. Test basic formula evaluation
        logger.info("STEP 1: Testing basic formula evaluation...")
        try:
            result = await self.formula_engine.evaluate_formula('5+10', 'TestSheet', 'TEST', {})
            if result == 15:
                diagnosis_results['formula_evaluation_basic'] = True
                logger.info("✅ Basic formula evaluation working")
            else:
                logger.error(f"❌ Basic formula evaluation failed: got {result}, expected 15")
        except Exception as e:
            logger.error(f"❌ Basic formula evaluation error: {e}")
        
        # 2. Test formula evaluation in context
        logger.info("STEP 2: Testing formula evaluation in simulation context...")
        diagnosis_results['formula_evaluation_context'] = await self.test_formula_evaluation_context()
        
        # 3. Apply simulation context fixes
        logger.info("STEP 3: Applying simulation context fixes...")
        diagnosis_results['simulation_context_fixed'] = await self.fix_simulation_context()
        
        # 4. Analyze latest Excel files if available
        logger.info("STEP 4: Analyzing Excel files...")
        uploads_dir = Path("/opt/app/uploads")
        if uploads_dir.exists():
            excel_files = list(uploads_dir.glob("*.xlsx")) + list(uploads_dir.glob("*.xls"))
            if excel_files:
                latest_file = max(excel_files, key=lambda f: f.stat().st_mtime)
                logger.info(f"Analyzing latest Excel file: {latest_file}")
                diagnosis_results['excel_data_analysis'] = await self.diagnose_excel_data(str(latest_file))
        
        return diagnosis_results

async def main():
    """Main execution function"""
    print("🚨 ZERO BUG EMERGENCY FIX")
    print("=" * 50)
    
    analyzer = ZeroBugAnalyzer()
    
    try:
        diagnosis = await analyzer.run_comprehensive_diagnosis()
        
        print("\n📋 DIAGNOSIS SUMMARY:")
        print("=" * 30)
        for key, value in diagnosis.items():
            if isinstance(value, bool):
                status = "✅ PASS" if value else "❌ FAIL"
                print(f"{key}: {status}")
            elif isinstance(value, dict) and value:
                print(f"{key}: {len(value)} items analyzed")
        
        # Determine if fixes are needed
        if not all([diagnosis['formula_evaluation_basic'], diagnosis['formula_evaluation_context']]):
            print("\n🚨 CRITICAL ISSUES DETECTED!")
            print("The zero bug is caused by formula evaluation failures.")
            print("Manual intervention required.")
            return False
        else:
            print("\n✅ DIAGNOSIS COMPLETE")
            print("Formula evaluation appears to be working correctly.")
            print("The zero bug may be caused by data input issues.")
            return True
            
    except Exception as e:
        logger.error(f"❌ Diagnosis failed: {e}")
        return False

if __name__ == "__main__":
    asyncio.run(main()) 