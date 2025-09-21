#!/usr/bin/env python3
"""
Check if there's a percentage interpretation issue between Monte Carlo and static scenarios
"""

import openpyxl
import sys

def check_percentage_interpretation(file_path):
    """Check percentage vs decimal interpretation in Excel model"""
    print(f"🔍 CHECKING PERCENTAGE INTERPRETATION: {file_path}")
    print("=" * 80)
    
    try:
        workbook = openpyxl.load_workbook(file_path, data_only=True)  # data_only=True for calculated values
        main_sheet = workbook['WIZEMICE Likest']
    except Exception as e:
        print(f"❌ Error loading workbook: {e}")
        return
    
    print(f"🎯 Analyzing sheet: {main_sheet.title}")
    print()
    
    # Check the input variables F4, F5, F6 values in static scenario
    print("📋 STATIC SCENARIO VALUES (In Excel File)")
    print("-" * 50)
    
    input_vars = ['F4', 'F5', 'F6', 'F7']
    static_values = {}
    
    for var in input_vars:
        cell = main_sheet[var]
        value = cell.value
        static_values[var] = value
        print(f"   {var}: {value}")
        
        # Check if this looks like a percentage (0.0-1.0) or whole number (1-100)
        if isinstance(value, (int, float)):
            if 0 <= value <= 1:
                print(f"       → Interpreted as: {value*100:.1f}% (decimal format)")
            elif 1 < value <= 100:
                print(f"       → Interpreted as: {value:.1f}% (percentage format)")
            else:
                print(f"       → Unusual value: {value}")
    
    print()
    
    # Check what the growth formulas in Row 107 actually reference
    print("📋 ROW 107 GROWTH RATE APPLICATION")
    print("-" * 50)
    
    sample_cols = ['D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P']
    
    for col in sample_cols:
        cell_ref = f"{col}107"
        cell = main_sheet[cell_ref]
        value = cell.value
        
        print(f"   {cell_ref}: {value}")
        
        # Determine which F variable this references and check the percentage interpretation
        if isinstance(value, (int, float)):
            if 0 <= value <= 1:
                print(f"       → Growth rate: {value*100:.1f}% (decimal)")
            elif 1 < value <= 100:
                print(f"       → Growth rate: {value:.1f}% (whole number - PROBLEM!)")
            else:
                print(f"       → Unusual growth rate: {value}")
    
    print()
    
    # Check actual calculated values in Row 108 with static scenario
    print("📋 ROW 108 CUSTOMER CALCULATIONS (Static Scenario)")
    print("-" * 50)
    
    for col in sample_cols[:8]:  # First 8 columns
        cell_ref = f"{col}108"
        cell = main_sheet[cell_ref]
        value = cell.value
        
        print(f"   {cell_ref}: {value:,.0f}" if isinstance(value, (int, float)) else f"   {cell_ref}: {value}")
        
        # Check if growth looks reasonable
        if col != 'D' and isinstance(value, (int, float)) and value > 0:
            # Compare with C108 base
            base_cell = main_sheet['C108']
            base_value = base_cell.value
            
            if isinstance(base_value, (int, float)) and base_value > 0:
                growth_ratio = value / base_value
                print(f"       → Growth from base: {growth_ratio:.2f}x")
                
                if growth_ratio > 10:
                    print(f"       ⚠️  EXCESSIVE GROWTH - Suggests percentage interpretation error!")
                elif growth_ratio > 2:
                    print(f"       ⚠️  HIGH GROWTH - Check if intended")
                else:
                    print(f"       ✅ Reasonable growth")
    
    print()
    
    # Check what Monte Carlo actually sends to the simulation
    print("📋 MONTE CARLO VARIABLE INTERPRETATION ANALYSIS")
    print("-" * 50)
    
    print("🎯 HYPOTHESIS: Monte Carlo sends decimal values (0.10) but Excel expects percentages (10)")
    print()
    print("Static scenario analysis:")
    
    for var in ['F4', 'F5', 'F6']:
        static_val = static_values.get(var, 0)
        print(f"   {var} static value: {static_val}")
        
        if isinstance(static_val, (int, float)):
            if 0 <= static_val <= 1:
                print(f"       → Excel stores as: {static_val} (decimal)")
                print(f"       → Monte Carlo likely sends: {static_val} (decimal)")
                print(f"       → Formula sees: {static_val} (decimal)")
                print(f"       ✅ CONSISTENT - No percentage interpretation issue")
            elif 1 < static_val <= 100:
                print(f"       → Excel stores as: {static_val} (percentage)")  
                print(f"       → Monte Carlo likely sends: {static_val/100} (decimal)")
                print(f"       → Formula sees: {static_val/100} (decimal)")
                print(f"       🚨 PERCENTAGE INTERPRETATION MISMATCH!")
                print(f"       🚨 Excel expects {static_val}% but gets {static_val/100}%")
    
    print()
    
    # Simulate what happens with Monte Carlo values
    print("📋 MONTE CARLO SIMULATION TEST")
    print("-" * 50)
    
    print("Testing what happens when Monte Carlo sends F4=0.12 (12% as decimal):")
    print("vs static Excel scenario with F4=0.10 (10% as decimal)")
    print()
    
    base_customers = static_values.get('F8', 1000)  # Assuming F8 is base customers
    if not isinstance(base_customers, (int, float)):
        base_customers = 1000
    
    # Simulate growth with different interpretations
    static_growth = static_values.get('F4', 0.10)
    monte_carlo_growth_decimal = 0.12  # What Monte Carlo sends
    monte_carlo_growth_percent = 12.0  # If interpreted as percentage
    
    print(f"   Base customers: {base_customers:,.0f}")
    print(f"   Static F4 growth: {static_growth}")
    print(f"   Monte Carlo F4: {monte_carlo_growth_decimal} (decimal) or {monte_carlo_growth_percent} (if misinterpreted)")
    print()
    
    # Calculate first few columns of growth
    print("   Growth simulation:")
    print("   Column | Static Result | MC Decimal | MC Percentage")
    print("   -------|---------------|------------|---------------")
    
    static_val = base_customers
    mc_decimal_val = base_customers  
    mc_percent_val = base_customers
    
    for i, col in enumerate(['C', 'D', 'E', 'F', 'G']):
        if i == 0:
            print(f"   {col}      | {static_val:>13,.0f} | {mc_decimal_val:>10,.0f} | {mc_percent_val:>13,.0f}")
        else:
            static_val = static_val * (1 + static_growth)
            mc_decimal_val = mc_decimal_val * (1 + monte_carlo_growth_decimal)
            mc_percent_val = mc_percent_val * (1 + monte_carlo_growth_percent)
            
            print(f"   {col}      | {static_val:>13,.0f} | {mc_decimal_val:>10,.0f} | {mc_percent_val:>13,.0f}")
            
            if mc_percent_val > 1000000:
                print(f"          |               |            | ⚠️ ASTRONOMICAL!")
                break
    
    print()
    print("🎯 CONCLUSION:")
    if any(isinstance(static_values.get(var), (int, float)) and static_values.get(var, 0) > 1 
           for var in ['F4', 'F5', 'F6']):
        print("🚨 PERCENTAGE INTERPRETATION ERROR LIKELY!")
        print("   Static Excel uses percentage format (10 = 10%)")
        print("   Monte Carlo sends decimal format (0.10 = 10%)")  
        print("   Excel formula treats 0.10 as 0.10% instead of 10%")
    else:
        print("✅ No percentage interpretation issue detected")
        print("   Both static and Monte Carlo use decimal format (0.10 = 10%)")
    
    print("=" * 80)

if __name__ == "__main__":
    excel_file = "/home/paperspace/PROJECT/uploads/2a07dd9d-2048-4ff0-82e1-b6c7d74a6f68_WIZEMICE_FINAL_WORKING_MODEL.xlsx"
    check_percentage_interpretation(excel_file)
