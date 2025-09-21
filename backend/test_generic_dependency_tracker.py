#!/usr/bin/env python3
"""
Test Generic Dependency Tracker

This script tests the generic dependency tracker to ensure it properly identifies
cells that should remain as formulas based on Monte Carlo variable dependencies.
"""

import asyncio
import logging
from pathlib import Path
import sys
import os

# Add the backend directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from excel_parser.dependency_tracker import get_monte_carlo_dependent_cells
from excel_parser.service import _find_excel_file

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_dependency_tracker():
    """Test the generic dependency tracker with the current Excel file"""
    
    print("🔍 TESTING GENERIC DEPENDENCY TRACKER")
    print("=" * 60)
    
    # Test with the current file
    file_id = "c9ebace1-dd72-4a9f-92da-62375ee630cd"
    
    # Get file path
    file_path = _find_excel_file(file_id)
    if not file_path:
        print(f"❌ Could not find file path for {file_id}")
        return
    
    print(f"📁 File path: {file_path}")
    
    # Define Monte Carlo variables
    mc_variables = [
        ('WIZEMICE Likest', 'F4'),  # Phase 1 Growth Rate
        ('WIZEMICE Likest', 'F5'),  # Phase 2 Growth Rate
        ('WIZEMICE Likest', 'F6'),  # Phase 3 Growth Rate
    ]
    
    print(f"🎲 Monte Carlo variables: {mc_variables}")
    
    try:
        # Test dependency tracking
        dependent_cells = get_monte_carlo_dependent_cells(file_path, mc_variables)
        
        print(f"\n✅ SUCCESS: Found {len(dependent_cells)} cells dependent on MC variables")
        
        # Group by sheet for better display
        by_sheet = {}
        for sheet, cell in dependent_cells:
            if sheet not in by_sheet:
                by_sheet[sheet] = []
            by_sheet[sheet].append(cell)
        
        # Display results
        for sheet, cells in by_sheet.items():
            print(f"\n📊 Sheet '{sheet}': {len(cells)} dependent cells")
            
            # Group by row for analysis
            by_row = {}
            for cell in cells:
                # Extract row number
                import re
                match = re.search(r'(\d+)', cell)
                if match:
                    row = int(match.group(1))
                    if row not in by_row:
                        by_row[row] = []
                    by_row[row].append(cell)
            
            # Show key rows
            key_rows = sorted(by_row.keys())
            print(f"   Key rows: {key_rows[:10]}{'...' if len(key_rows) > 10 else ''}")
            
            # Check if Row 107 is included (should be for this file)
            if 107 in by_row:
                print(f"   ✅ Row 107: {len(by_row[107])} cells (Growth formulas)")
                print(f"      Sample: {by_row[107][:5]}")
            else:
                print(f"   ⚠️  Row 107: Not found (unexpected for this file)")
            
            # Check cash flow rows
            cash_flow_rows = [r for r in by_row.keys() if r in [161, 160, 162]]  # Around 161
            if cash_flow_rows:
                print(f"   💰 Cash flow rows {cash_flow_rows}: {sum(len(by_row[r]) for r in cash_flow_rows)} cells")
            
        # Verify the dependency chain
        print(f"\n🔗 DEPENDENCY CHAIN VERIFICATION:")
        print(f"   F4, F5, F6 → Row 107 (Growth) → Revenue/Cost Models → Cash Flows → B12/B13")
        
        # Check specific dependencies
        has_row_107 = any(cell[1].endswith('107') for cell in dependent_cells)
        has_cash_flows = any('161' in cell[1] for cell in dependent_cells)
        
        print(f"   ✅ Row 107 formulas: {'Found' if has_row_107 else 'Missing'}")
        print(f"   ✅ Cash flow cells: {'Found' if has_cash_flows else 'Missing'}")
        
        if has_row_107 and has_cash_flows:
            print(f"   🎯 Dependency chain: COMPLETE")
        else:
            print(f"   ⚠️  Dependency chain: INCOMPLETE")
        
    except Exception as e:
        print(f"❌ ERROR testing dependency tracker: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_dependency_tracker()) 