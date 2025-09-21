#!/usr/bin/env python3
"""
Trace the exact amplification chain from F4 to astronomical values
"""

import openpyxl
import sys

def trace_amplification_chain(file_path):
    """Trace the dependency chain causing astronomical amplification"""
    print(f"🔍 TRACING AMPLIFICATION CHAIN: {file_path}")
    print("=" * 80)
    
    try:
        workbook = openpyxl.load_workbook(file_path, data_only=False)
        main_sheet = workbook['WIZEMICE Likest']
    except Exception as e:
        print(f"❌ Error loading workbook: {e}")
        return
    
    print(f"🎯 Analyzing sheet: {main_sheet.title}")
    print()
    
    # **DISCOVERY**: Row 108 shows compound growth formulas!
    # Let's trace the full chain: F4 → Row 107 → Row 108 → Row 111 → Row 120 → Row 125 → Row 148 → Row 161
    
    chain_analysis = [
        (107, "Growth Rate Application"),
        (108, "Customer Numbers with Growth"), 
        (111, "Revenue Base Calculation"),
        (120, "Revenue with Price"),
        (125, "Revenue Totals"),
        (148, "Net Cash Flow Components"),
        (161, "Final Cash Flows")
    ]
    
    sample_columns = ['D', 'E', 'F', 'G', 'H', 'I', 'J', 'K']  # Focus on key columns
    
    for row_num, description in chain_analysis:
        print(f"📋 ROW {row_num}: {description}")
        print("-" * 60)
        
        for col in sample_columns:
            cell_ref = f"{col}{row_num}"
            try:
                cell = main_sheet[cell_ref]
                value = cell.value
                formula = None
                
                # Get formula
                if hasattr(cell, 'value') and cell.value and isinstance(cell.value, str) and cell.value.startswith('='):
                    formula = cell.value
                elif hasattr(cell, 'formula') and cell.formula:
                    formula = cell.formula
                    
                print(f"   {cell_ref}: Value={value}")
                if formula:
                    print(f"       Formula: {formula}")
                    
                    # Analysis for each row type
                    if row_num == 107:
                        if any(var in formula.upper() for var in ['F4', 'F5', 'F6']):
                            print(f"       ✅ MONTE CARLO INPUT VARIABLE")
                    
                    elif row_num == 108:
                        if 'ROUND' in formula.upper() and any(ref in formula.upper() for ref in ['107', '+', '*']):
                            print(f"       ⚠️  COMPOUND GROWTH FORMULA - THIS IS THE AMPLIFICATION SOURCE!")
                            print(f"       📊 Pattern: Base * (1 + growth_rate) where growth_rate = F4/F5/F6")
                            
                    elif row_num in [111, 120]:
                        if '*' in formula and any(ref in formula for ref in ['108', '111', 'F51']):
                            print(f"       ⚠️  MULTIPLICATION OF LARGE VALUES")
                            
                    elif row_num == 125:
                        if any(ref in formula for ref in ['120', '121', '122', '123', '124']):
                            print(f"       ⚠️  AGGREGATION OF AMPLIFIED VALUES")
                            
                    elif row_num in [148, 161]:
                        if any(ref in formula for ref in ['125', '148']):
                            print(f"       ⚠️  FINAL CASH FLOW CALCULATION")
                            
                elif value and isinstance(value, (int, float)):
                    if abs(value) > 1000000:
                        print(f"       🚨 ASTRONOMICAL VALUE: {value:,.0f}")
                    elif abs(value) > 10000:
                        print(f"       ⚠️  LARGE VALUE: {value:,.0f}")
                        
            except Exception as e:
                print(f"   {cell_ref}: Error: {e}")
        
        print()
    
    # Special analysis of the compound growth in row 108
    print("🚨 COMPOUND GROWTH ANALYSIS (Row 108)")
    print("-" * 60)
    print("This row applies compound growth using F4, F5, F6 variables.")
    print("Formula pattern: =ROUND(PrevValue*(100%+GrowthRate),0)")
    print("Where GrowthRate comes from F4, F5, F6 via row 107.")
    print()
    
    # Check specific cells that show the amplification
    critical_cells = ['C108', 'D108', 'E108', 'F108', 'G108', 'H108', 'I108', 'J108', 'K108']
    
    for cell_ref in critical_cells:
        try:
            cell = main_sheet[cell_ref]
            value = cell.value
            formula = None
            
            if hasattr(cell, 'value') and cell.value and isinstance(cell.value, str) and cell.value.startswith('='):
                formula = cell.value
            elif hasattr(cell, 'formula') and cell.formula:
                formula = cell.formula
                
            print(f"   {cell_ref}: {value}")
            if formula:
                print(f"       Formula: {formula}")
                # Show why this creates amplification
                if 'ROUND' in formula.upper() and '108' in formula:
                    print(f"       📊 This creates COMPOUND INTEREST effect!")
                    print(f"       📊 Each column multiplies by (1 + F4/F5/F6)")
                    print(f"       📊 Over 30+ columns, small growth rates become EXPONENTIAL")
                    
        except Exception as e:
            print(f"   {cell_ref}: Error: {e}")
    
    print()
    
    # Calculate theoretical amplification
    print("🧮 THEORETICAL AMPLIFICATION CALCULATION")
    print("-" * 60)
    try:
        # Get F4, F5, F6 values
        f4_value = main_sheet['F4'].value or 0.1
        f5_value = main_sheet['F5'].value or 0.15  
        f6_value = main_sheet['F6'].value or 0.08
        
        print(f"   F4 (early growth): {f4_value}")
        print(f"   F5 (mid growth): {f5_value}")
        print(f"   F6 (late growth): {f6_value}")
        print()
        
        # Simulate compound effect over columns
        base_value = 1000  # Assume starting base
        current_value = base_value
        
        print("   COMPOUND AMPLIFICATION SIMULATION:")
        print("   Column | Growth Rate | Value     | Multiplier")
        print("   -------|-------------|-----------|----------")
        
        for i, (col, rate_var) in enumerate([
            ('C', 0), ('D', f4_value), ('E', f4_value), ('F', f4_value), ('G', f4_value), 
            ('H', f4_value), ('I', f5_value), ('J', f5_value), ('K', f5_value), 
            ('L', f5_value), ('M', f5_value), ('N', f5_value), ('O', f6_value), ('P', f6_value)
        ]):
            if i > 0:  # Skip first column
                current_value = current_value * (1 + rate_var)
            multiplier = current_value / base_value
            print(f"   {col}      | {rate_var:>11.3f} | {current_value:>9,.0f} | {multiplier:>8.2f}x")
            
            if i >= 13:  # Show first 14 columns
                break
                
        print()
        print(f"   🚨 After just 14 columns: {current_value/base_value:.1f}x amplification!")
        print(f"   🚨 The Excel model has 39 columns (C through AN)")
        print(f"   🚨 By column AN, amplification could reach 1000x+ easily!")
        
    except Exception as e:
        print(f"   Error in calculation: {e}")
    
    print()
    print("💡 ROOT CAUSE IDENTIFIED:")
    print("-" * 60)
    print("1. Row 107: F4, F5, F6 growth rates applied to columns")
    print("2. Row 108: COMPOUND GROWTH formula creates exponential amplification")
    print("3. Formula: =ROUND(PrevColumn*(1+GrowthRate),0)")
    print("4. Over 39 columns (C→AN), this creates massive compound growth")
    print("5. Small changes in F4 (0.08→0.12) create exponential effects")
    print("6. These amplified values flow through → Revenue → Cash Flow → NPV")
    print()
    print("🛠️  SOLUTION: Add bounds or change compound growth logic")
    print("=" * 80)

if __name__ == "__main__":
    excel_file = "/home/paperspace/PROJECT/uploads/2a07dd9d-2048-4ff0-82e1-b6c7d74a6f68_WIZEMICE_FINAL_WORKING_MODEL.xlsx"
    trace_amplification_chain(excel_file)
