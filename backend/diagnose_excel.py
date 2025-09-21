#!/usr/bin/env python3
"""
Diagnostic tool to analyze Excel files and understand simulation issues.
"""

import asyncio
import sys
import os
import openpyxl
from typing import Dict, List, Tuple, Any
import json

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from excel_parser.service import get_formulas_for_file, get_all_parsed_sheets_data
from simulation.formula_utils import get_evaluation_order

async def diagnose_excel_file(file_id: str):
    """Diagnose issues with an Excel file."""
    print(f"\n🔍 Diagnosing Excel file: {file_id}")
    print("=" * 80)
    
    # 1. Check if file exists
    file_path = f"uploads/{file_id}"
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return
    
    print(f"✅ File found: {file_path}")
    
    # 2. Get all formulas
    try:
        all_formulas = await get_formulas_for_file(file_id)
        print(f"\n📊 Found {sum(len(sheet) for sheet in all_formulas.values())} formulas across {len(all_formulas)} sheets")
        
        # Show formulas by sheet
        for sheet_name, formulas in all_formulas.items():
            print(f"\n  Sheet '{sheet_name}': {len(formulas)} formulas")
            # Show first few formulas
            for cell, formula in list(formulas.items())[:5]:
                print(f"    {cell}: {formula[:50]}{'...' if len(formula) > 50 else ''}")
            if len(formulas) > 5:
                print(f"    ... and {len(formulas) - 5} more")
    except Exception as e:
        print(f"❌ Error getting formulas: {e}")
        all_formulas = {}
    
    # 3. Get all sheet data (including constants)
    try:
        all_sheet_data = await get_all_parsed_sheets_data(file_id)
        print(f"\n📋 Sheet data analysis:")
        
        for sheet_name, sheet_data in all_sheet_data.items():
            print(f"\n  Sheet '{sheet_name}':")
            
            # Count cells with values
            cells_with_values = sum(1 for row in sheet_data for cell in row if cell is not None)
            total_cells = sum(len(row) for row in sheet_data)
            print(f"    Cells with values: {cells_with_values}/{total_cells}")
            
            # Show sample data
            print(f"    Sample data (first 5x5):")
            for i, row in enumerate(sheet_data[:5]):
                row_str = " | ".join(str(cell)[:10] if cell is not None else "    -    " for cell in row[:5])
                print(f"      Row {i+1}: {row_str}")
    except Exception as e:
        print(f"❌ Error getting sheet data: {e}")
        all_sheet_data = {}
    
    # 4. Analyze specific cells
    print("\n🎯 Target cell analysis:")
    target_cells = [
        ("Simple", "E8"), ("Simple", "F8"), ("Simple", "G8"),
        ("Simple", "A8"), ("Simple", "B8"), ("Simple", "C8"),
        ("Simple", "D8")
    ]
    
    # Load workbook for direct cell access
    try:
        wb = openpyxl.load_workbook(file_path, data_only=True)
        
        for sheet_name, cell_coord in target_cells:
            if sheet_name in wb.sheetnames:
                sheet = wb[sheet_name]
                cell = sheet[cell_coord]
                
                # Get formula if exists
                formula = all_formulas.get(sheet_name, {}).get(cell_coord, "No formula")
                
                print(f"\n  {sheet_name}!{cell_coord}:")
                print(f"    Value: {cell.value}")
                print(f"    Formula: {formula}")
                
                # If it's a VLOOKUP, analyze it
                if "VLOOKUP" in str(formula).upper():
                    print(f"    ⚠️  Contains VLOOKUP - checking lookup table...")
                    # Try to extract the range from the formula
                    import re
                    vlookup_match = re.search(r'VLOOKUP\s*\([^,]+,\s*([^,]+)', formula.upper())
                    if vlookup_match:
                        range_ref = vlookup_match.group(1)
                        print(f"    Lookup range: {range_ref}")
    except Exception as e:
        print(f"❌ Error analyzing cells: {e}")
    
    # 5. Check calculation order
    print("\n🔄 Calculation order analysis:")
    try:
        # Flatten formulas for get_evaluation_order
        flat_formulas = {}
        for sheet_name, sheet_formulas in all_formulas.items():
            for cell, formula in sheet_formulas.items():
                flat_formulas[(sheet_name, cell)] = formula
        
        ordered_steps = get_evaluation_order(flat_formulas)
        print(f"  Total calculation steps: {len(ordered_steps)}")
        
        # Show first and last few steps
        print("\n  First 5 steps:")
        for i, (sheet, cell, formula) in enumerate(ordered_steps[:5]):
            print(f"    {i+1}. {sheet}!{cell} = {formula[:50]}{'...' if len(formula) > 50 else ''}")
        
        if len(ordered_steps) > 10:
            print(f"\n  ... {len(ordered_steps) - 10} more steps ...")
            
        print("\n  Last 5 steps:")
        for i, (sheet, cell, formula) in enumerate(ordered_steps[-5:], len(ordered_steps) - 5):
            print(f"    {i+1}. {sheet}!{cell} = {formula[:50]}{'...' if len(formula) > 50 else ''}")
        
        # Check if target cells are in the calculation order
        print("\n  Target cells in calculation order:")
        for sheet_name, cell_coord in target_cells:
            in_order = any(s == sheet_name and c == cell_coord for s, c, _ in ordered_steps)
            print(f"    {sheet_name}!{cell_coord}: {'✅ Yes' if in_order else '❌ No'}")
            
    except Exception as e:
        print(f"❌ Error analyzing calculation order: {e}")
    
    print("\n" + "=" * 80)
    print("Diagnosis complete!")

async def main():
    if len(sys.argv) != 2:
        print("Usage: python diagnose_excel.py <file_id>")
        print("Example: python diagnose_excel.py a00c20dd-1c67-43c4-96c5-b055ffc19c71")
        sys.exit(1)
    
    file_id = sys.argv[1]
    await diagnose_excel_file(file_id)

if __name__ == "__main__":
    asyncio.run(main()) 