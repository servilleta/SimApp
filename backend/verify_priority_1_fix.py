#!/usr/bin/env python3
"""
Priority 1 Fix Verification

This script verifies if our Priority 1 fix is working correctly:
1. Check if Row 107 cells are excluded from constants
2. Check if Row 107 cells are loaded as formulas instead
3. Verify the formulas reference F4, F5, F6 variables
"""

import asyncio
import logging
from pathlib import Path
import sys
import os

# Add the backend directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from excel_parser.service import get_constants_for_file, get_formulas_for_file

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Priority1FixVerifier:
    """Verifies the Priority 1 fix for Monte Carlo variable connection"""
    
    def __init__(self):
        self.file_id = "c9ebace1-dd72-4a9f-92da-62375ee630cd"
        
    async def verify_fix(self):
        """Run complete verification of Priority 1 fix"""
        
        print("🔍 PRIORITY 1 FIX VERIFICATION")
        print("=" * 60)
        
        # Step 1: Check if Row 107 cells are excluded from constants
        await self._verify_constants_exclusion()
        
        # Step 2: Check if Row 107 cells exist as formulas
        await self._verify_formulas_inclusion()
        
        # Step 3: Check formula content
        await self._verify_formula_content()
        
        print("\n🎯 VERIFICATION COMPLETE")
        
    async def _verify_constants_exclusion(self):
        """Verify Row 107 cells are excluded from constants"""
        print("\n📊 STEP 1: Constants Exclusion Verification")
        print("-" * 40)
        
        try:
            constants = await get_constants_for_file(self.file_id)
            
            # Check if any Row 107 cells exist in constants
            row_107_in_constants = []
            for (sheet, cell), value in constants.items():
                if '107' in cell and sheet in ['WIZEMICE Likest', 'WIZEMICE High', 'WIZEMICE Low']:
                    row_107_in_constants.append((sheet, cell, value))
            
            print(f"   Total constants loaded: {len(constants)}")
            print(f"   Row 107 cells in constants: {len(row_107_in_constants)}")
            
            if row_107_in_constants:
                print("   ⚠️  WARNING: Row 107 cells found in constants (should be excluded):")
                for sheet, cell, value in row_107_in_constants[:10]:  # Show first 10
                    print(f"      {sheet}!{cell}: {value}")
                if len(row_107_in_constants) > 10:
                    print(f"      ... and {len(row_107_in_constants) - 10} more")
            else:
                print("   ✅ SUCCESS: No Row 107 cells found in constants (properly excluded)")
                
        except Exception as e:
            print(f"   ❌ ERROR loading constants: {e}")
            
    async def _verify_formulas_inclusion(self):
        """Verify Row 107 cells exist as formulas"""
        print("\n📋 STEP 2: Formulas Inclusion Verification")
        print("-" * 40)
        
        try:
            formulas_dict = await get_formulas_for_file(self.file_id)
            
            # Check if Row 107 cells exist in formulas
            row_107_formulas = []
            total_formulas = 0
            for sheet_name, sheet_formulas in formulas_dict.items():
                total_formulas += len(sheet_formulas)
                for cell, formula in sheet_formulas.items():
                    if '107' in cell and sheet_name in ['WIZEMICE Likest', 'WIZEMICE High', 'WIZEMICE Low']:
                        row_107_formulas.append((sheet_name, cell, formula))
            
            print(f"   Total formulas loaded: {total_formulas}")
            print(f"   Row 107 cells in formulas: {len(row_107_formulas)}")
            
            if row_107_formulas:
                print("   ✅ SUCCESS: Row 107 cells found in formulas:")
                for sheet, cell, formula in row_107_formulas[:10]:  # Show first 10
                    print(f"      {sheet}!{cell}: {formula}")
                if len(row_107_formulas) > 10:
                    print(f"      ... and {len(row_107_formulas) - 10} more")
            else:
                print("   ⚠️  WARNING: No Row 107 cells found in formulas")
                print("      This suggests they may still be constants or don't exist")
                
        except Exception as e:
            print(f"   ❌ ERROR loading formulas: {e}")
            
    async def _verify_formula_content(self):
        """Verify Row 107 formulas reference F4, F5, F6"""
        print("\n🔗 STEP 3: Formula Content Verification")
        print("-" * 40)
        
        try:
            formulas_dict = await get_formulas_for_file(self.file_id)
            
            # Analyze Row 107 formula content
            f_variable_refs = []
            all_row_107 = []
            
            for sheet_name, sheet_formulas in formulas_dict.items():
                for cell, formula in sheet_formulas.items():
                    if '107' in cell and sheet_name in ['WIZEMICE Likest', 'WIZEMICE High', 'WIZEMICE Low']:
                        all_row_107.append((sheet_name, cell, formula))
                        # Check if formula references F4, F5, or F6
                        if any(var in str(formula) for var in ['F4', 'F5', 'F6']):
                            f_variable_refs.append((sheet_name, cell, formula))
            
            print(f"   Row 107 formulas referencing F4/F5/F6: {len(f_variable_refs)}")
            
            if f_variable_refs:
                print("   ✅ SUCCESS: Found F variable references in Row 107:")
                for sheet, cell, formula in f_variable_refs:
                    print(f"      {sheet}!{cell}: {formula}")
            else:
                print("   ⚠️  WARNING: No F4/F5/F6 references found in Row 107 formulas")
                print("      This explains why Monte Carlo variables don't affect cash flows")
                
            # Check what Row 107 formulas actually contain
            if all_row_107:
                print(f"\n   📋 All Row 107 formulas (found {len(all_row_107)} total):")
                for sheet, cell, formula in all_row_107[:10]:  # Show first 10
                    print(f"      {sheet}!{cell}: {formula}")
                if len(all_row_107) > 10:
                    print(f"      ... and {len(all_row_107) - 10} more")
            else:
                print("\n   ⚠️  No Row 107 formulas found at all")
                    
        except Exception as e:
            print(f"   ❌ ERROR analyzing formulas: {e}")

async def main():
    verifier = Priority1FixVerifier()
    await verifier.verify_fix()

if __name__ == "__main__":
    asyncio.run(main()) 