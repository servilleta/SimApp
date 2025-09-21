#!/usr/bin/env python3
"""
Analyze intermediate Excel rows 148 and 157 to find amplification source
"""

import openpyxl
import sys

def analyze_intermediate_rows(file_path):
    """Analyze rows 148 and 157 that feed into cash flows"""
    print(f"🔍 ANALYZING INTERMEDIATE ROWS: {file_path}")
    print("=" * 80)
    
    try:
        workbook = openpyxl.load_workbook(file_path, data_only=False)
        main_sheet = workbook['WIZEMICE Likest']
    except Exception as e:
        print(f"❌ Error loading workbook: {e}")
        return
    
    print(f"🎯 Analyzing sheet: {main_sheet.title}")
    print()
    
    # Analyze rows that feed into cash flows
    target_rows = [148, 157, 159]  # Include 159 since C161 references it
    sample_columns = ['C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'AA', 'AB']
    
    for row_num in target_rows:
        print(f"📋 ROW {row_num} ANALYSIS")
        print("-" * 50)
        
        for col in sample_columns:
            cell_ref = f"{col}{row_num}"
            try:
                cell = main_sheet[cell_ref]
                value = cell.value
                formula = None
                
                if hasattr(cell, 'value') and cell.value and isinstance(cell.value, str) and cell.value.startswith('='):
                    formula = cell.value
                elif hasattr(cell, 'formula') and cell.formula:
                    formula = cell.formula
                elif hasattr(cell, '_value') and cell._value and isinstance(cell._value, str) and cell._value.startswith('='):
                    formula = cell._value
                    
                print(f"   {cell_ref}: Value={value}")
                if formula:
                    print(f"       Formula: {formula}")
                    
                    # Check for problematic patterns
                    if formula and any(pattern in formula.upper() for pattern in ['F4', 'F5', 'F6']):
                        print(f"       ⚠️  REFERENCES F VARIABLE")
                    if formula and formula.count('*') > 2:
                        print(f"       ⚠️  MULTIPLE MULTIPLICATIONS ({formula.count('*')} found)")
                    if formula and any(pattern in formula.upper() for pattern in ['POWER', 'EXP', '**']):
                        print(f"       ⚠️  EXPONENTIAL FUNCTION DETECTED")
                    if formula and 'ROW' in formula.upper():
                        print(f"       ⚠️  ROW FUNCTION DETECTED")
                        
                elif value and isinstance(value, (int, float)) and abs(value) > 1000000:
                    print(f"       ⚠️  LARGE VALUE: {value:,.0f}")
                    
            except Exception as e:
                print(f"   {cell_ref}: Error: {e}")
        
        print()
    
    # Now trace back further - check what feeds into row 148 and 157
    print("📋 TRACING DEPENDENCIES")
    print("-" * 50)
    
    # Look for cells that might reference these key rows
    dependency_areas = [
        (120, 147, 'Revenue/Cost calculations'),
        (100, 119, 'Base calculations'),
        (80, 99, 'Input processing')
    ]
    
    for start_row, end_row, description in dependency_areas:
        print(f"\n   🔍 {description} (Rows {start_row}-{end_row}):")
        
        suspicious_cells = []
        for row in range(start_row, min(end_row + 1, start_row + 10)):  # Limit scan
            for col in ['C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R'][:8]:  # Limit columns
                cell_ref = f"{col}{row}"
                try:
                    cell = main_sheet[cell_ref]
                    formula = None
                    
                    if hasattr(cell, 'value') and cell.value and isinstance(cell.value, str) and cell.value.startswith('='):
                        formula = cell.value
                    elif hasattr(cell, 'formula') and cell.formula:
                        formula = cell.formula
                    
                    if formula and any(pattern in formula.upper() for pattern in ['F4', 'F5', 'F6', '*', 'POWER']):
                        suspicious_cells.append((cell_ref, formula, cell.value))
                        
                except Exception:
                    continue
        
        for cell_ref, formula, value in suspicious_cells[:5]:  # Show first 5
            print(f"       {cell_ref}: {formula} = {value}")
            if any(var in formula.upper() for var in ['F4', 'F5', 'F6']):
                print(f"           ✅ REFERENCES F VARIABLE")
            if formula.count('*') > 1:
                print(f"           ⚠️  MULTIPLICATION CHAIN ({formula.count('*')} operations)")
    
    # Check specific cells that might be problematic
    print(f"\n📋 CHECKING SPECIFIC SUSPICIOUS CELLS")
    print("-" * 50)
    
    suspicious_targets = []
    # Generate list of cells to check based on patterns
    for row in [107, 120, 125, 130, 135, 140, 145]:
        for col in ['C', 'F', 'I', 'L', 'O', 'R', 'AB']:
            suspicious_targets.append(f"{col}{row}")
    
    for cell_ref in suspicious_targets:
        try:
            cell = main_sheet[cell_ref]
            formula = None
            value = cell.value
            
            if hasattr(cell, 'value') and cell.value and isinstance(cell.value, str) and cell.value.startswith('='):
                formula = cell.value
            elif hasattr(cell, 'formula') and cell.formula:
                formula = cell.formula
                
            if formula and any(var in formula.upper() for var in ['F4', 'F5', 'F6']):
                print(f"   {cell_ref}: {formula} = {value}")
                if formula.count('*') > 1:
                    print(f"       ⚠️  MULTIPLICATION CHAIN")
                if any(pattern in formula.upper() for pattern in ['POWER', 'EXP']):
                    print(f"       ⚠️  EXPONENTIAL OPERATION")
                    
        except Exception:
            continue
    
    print()
    print("🎯 INTERMEDIATE ANALYSIS COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    excel_file = "/home/paperspace/PROJECT/uploads/2a07dd9d-2048-4ff0-82e1-b6c7d74a6f68_WIZEMICE_FINAL_WORKING_MODEL.xlsx"
    analyze_intermediate_rows(excel_file)
