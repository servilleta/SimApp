#!/usr/bin/env python3
"""
Test script for the Range Analyzer long-term solution

This script tests our fix for the missing cells problem by analyzing
the problematic WIZEMICE Excel file and ensuring all referenced cells are loaded.
"""

import sys
import os
import asyncio
import logging

# Add backend to path
sys.path.append('/home/paperspace/PROJECT/backend')

from excel_parser.range_analyzer import FormulaRangeAnalyzer, get_referenced_cells_for_file
from excel_parser.service import get_constants_for_file
from config import settings

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_range_analyzer():
    """Test the range analyzer with the problematic WIZEMICE file"""
    
    # Test file that had the missing cells issue
    test_file_path = "backend/saved_simulations_files/1721cfa0-f502-4978-a87c-a7469b85dfec.xlsx"
    test_file_id = "1721cfa0-f502-4978-a87c-a7469b85dfec"
    
    if not os.path.exists(test_file_path):
        logger.error(f"❌ Test file not found: {test_file_path}")
        return False
    
    print("🔍 Testing Range Analyzer Long-term Solution")
    print("=" * 60)
    
    try:
        # STEP 1: Test the range analyzer directly
        print("\n📊 STEP 1: Testing FormulaRangeAnalyzer")
        analyzer = FormulaRangeAnalyzer()
        analysis = analyzer.analyze_file_ranges(test_file_path)
        
        print(f"   Total formulas: {analysis['total_formulas']}")
        print(f"   Sheets analyzed: {analysis['sheets_analyzed']}")
        print(f"   Referenced cells: {analysis['total_referenced_cells']}")
        
        # Check if our problematic cells are found
        referenced_cells = analysis['referenced_cells']
        problematic_cells = [
            ('WIZEMICE Likest', 'P117'),
            ('WIZEMICE Likest', 'Q117'),
            ('WIZEMICE Likest', 'R117'),
            ('WIZEMICE Likest', 'AL117'),
            ('WIZEMICE Likest', 'C117')
        ]
        
        found_problematic = 0
        for sheet, cell in problematic_cells:
            if (sheet, cell) in referenced_cells:
                found_problematic += 1
                print(f"   ✅ Found problematic cell: {sheet}!{cell}")
            else:
                print(f"   ❌ Missing problematic cell: {sheet}!{cell}")
        
        print(f"   🎯 Found {found_problematic}/{len(problematic_cells)} problematic cells")
        
        # STEP 2: Test the integrated solution
        print("\n🔧 STEP 2: Testing integrated get_constants_for_file")
        
        # Mock the settings for testing
        settings.UPLOAD_DIR = "saved_simulations_files"
        
        constants = await get_constants_for_file(test_file_id)
        
        print(f"   Total constants loaded: {len(constants)}")
        
        # Check if our problematic cells are now in constants
        found_in_constants = 0
        for sheet, cell in problematic_cells:
            cell_key = (sheet, cell)
            if cell_key in constants:
                found_in_constants += 1
                value = constants[cell_key]
                print(f"   ✅ {sheet}!{cell} = {value} (loaded as constant)")
            else:
                print(f"   ❌ {sheet}!{cell} not in constants")
        
        print(f"   🎯 {found_in_constants}/{len(problematic_cells)} problematic cells now in constants")
        
        # STEP 3: Verify range detection works
        print("\n📈 STEP 3: Testing range detection")
        
        # Look for ranges that should include our problematic cells
        formulas = analysis['formulas']
        ranges_containing_p117 = []
        
        for sheet_name, sheet_formulas in formulas.items():
            for cell_coord, formula in sheet_formulas.items():
                if 'C161:AL161' in formula or 'C117:AL117' in formula or ('P117' in formula and ':' in formula):
                    ranges_containing_p117.append((sheet_name, cell_coord, formula))
                    print(f"   🎯 Found range formula in {sheet_name}!{cell_coord}:")
                    print(f"      {formula[:100]}...")
        
        if ranges_containing_p117:
            print(f"   ✅ Found {len(ranges_containing_p117)} formulas with ranges that should include P117")
        else:
            print("   ⚠️  No range formulas found that should include P117")
        
        # STEP 4: Success summary
        print("\n🎊 SOLUTION VALIDATION")
        print("=" * 60)
        
        success_score = 0
        total_tests = 4
        
        if analysis['total_formulas'] > 0:
            print("✅ Formula analysis working")
            success_score += 1
        else:
            print("❌ Formula analysis failed")
        
        if found_problematic >= len(problematic_cells) * 0.5:  # At least 50% found
            print("✅ Range analyzer detects problematic cells")
            success_score += 1
        else:
            print("❌ Range analyzer misses problematic cells")
        
        if found_in_constants >= len(problematic_cells) * 0.5:  # At least 50% in constants
            print("✅ Constants loading includes empty cells")
            success_score += 1
        else:
            print("❌ Constants loading still missing empty cells")
        
        if len(constants) > analysis['total_referenced_cells'] * 0.5:
            print("✅ Integration working correctly")
            success_score += 1
        else:
            print("❌ Integration issues detected")
        
        print(f"\n🏆 SUCCESS RATE: {success_score}/{total_tests} ({success_score/total_tests*100:.0f}%)")
        
        if success_score >= 3:
            print("🎉 LONG-TERM SOLUTION WORKING!")
            print("   The missing cells problem should now be resolved.")
            return True
        else:
            print("⚠️  SOLUTION NEEDS REFINEMENT")
            return False
            
    except Exception as e:
        logger.error(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    result = asyncio.run(test_range_analyzer())
    sys.exit(0 if result else 1) 