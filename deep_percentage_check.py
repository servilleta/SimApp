#!/usr/bin/env python3
"""
Deep dive into percentage interpretation - checking if Monte Carlo is sending wrong scale values
"""

import openpyxl
import sys

def deep_percentage_analysis():
    """Deep analysis of percentage interpretation"""
    print("🔍 DEEP PERCENTAGE INTERPRETATION ANALYSIS")
    print("=" * 60)
    
    excel_file = "/home/paperspace/PROJECT/uploads/2a07dd9d-2048-4ff0-82e1-b6c7d74a6f68_WIZEMICE_FINAL_WORKING_MODEL.xlsx"
    
    try:
        # Load with formulas (not calculated values)
        workbook_formulas = openpyxl.load_workbook(excel_file, data_only=False)
        # Load with calculated values  
        workbook_values = openpyxl.load_workbook(excel_file, data_only=True)
        
        sheet_formulas = workbook_formulas['WIZEMICE Likest']
        sheet_values = workbook_values['WIZEMICE Likest']
        
        print("📊 DETAILED VARIABLE ANALYSIS")
        print("-" * 40)
        
        # Check F4, F5, F6 in detail
        for var in ['F4', 'F5', 'F6']:
            cell_formula = sheet_formulas[var]
            cell_value = sheet_values[var]
            
            print(f"\n{var} Analysis:")
            print(f"  Raw value: {cell_value.value}")
            print(f"  Data type: {type(cell_value.value)}")
            print(f"  Cell formula: {cell_formula.value}")
            print(f"  Number format: {cell_value.number_format}")
            
            # Check if it's formatted as percentage in Excel
            if cell_value.number_format:
                if '%' in cell_value.number_format:
                    print(f"  ⚠️  FORMATTED AS PERCENTAGE in Excel!")
                    print(f"  ⚠️  Excel shows: {cell_value.value * 100:.1f}%")
                    print(f"  ⚠️  But stores internally as: {cell_value.value}")
                else:
                    print(f"  ✅ Formatted as: {cell_value.number_format}")
        
        print(f"\n📊 MONTE CARLO RANGE ANALYSIS")
        print("-" * 40)
        
        # Let me check what the Monte Carlo simulation configuration actually says
        # The user mentioned the ranges aren't that extreme
        print("User reported Monte Carlo ranges are 'not very big'")
        print("Let's check what ranges might be configured vs what we're assuming:")
        
        print(f"\nAssumed ranges (from our analysis):")
        print(f"  F4: 0.08 to 0.12 (8% to 12%) - decimal format")
        print(f"  F5: 0.12 to 0.20 (12% to 20%) - decimal format") 
        print(f"  F6: 0.05 to 0.10 (5% to 10%) - decimal format")
        
        print(f"\nBUT what if the Monte Carlo UI accepts percentage format?")
        print(f"  User enters: 8% to 12% (in UI)")
        print(f"  Monte Carlo converts to: 8.0 to 12.0 (whole numbers)")
        print(f"  Excel receives: 8.0 instead of 0.08")
        print(f"  Formula becomes: (1 + 8.0) = 9x growth per period!")
        
        print(f"\n🧮 TESTING PERCENTAGE MISINTERPRETATION")
        print("-" * 40)
        
        # Test what happens if Monte Carlo sends whole numbers instead of decimals
        static_f4 = sheet_values['F4'].value  # Should be 0.1
        
        print(f"Static scenario:")
        print(f"  F4 = {static_f4} (Excel internal)")
        print(f"  Growth formula: 1 + {static_f4} = {1 + static_f4}")
        
        # Simulate if Monte Carlo sends percentage as whole number
        monte_carlo_wrong = 12.0  # If Monte Carlo sends 12 instead of 0.12
        monte_carlo_correct = 0.12  # What it should send
        
        print(f"\nIf Monte Carlo sends WRONG format (percentage as whole number):")
        print(f"  F4 = {monte_carlo_wrong}")
        print(f"  Growth formula: 1 + {monte_carlo_wrong} = {1 + monte_carlo_wrong}")
        print(f"  ⚠️  This means 1300% growth per period!")
        
        print(f"\nIf Monte Carlo sends CORRECT format (decimal):")
        print(f"  F4 = {monte_carlo_correct}")
        print(f"  Growth formula: 1 + {monte_carlo_correct} = {1 + monte_carlo_correct}")
        print(f"  ✅ This means 12% growth per period")
        
        # Calculate what astronomical means
        base = 40
        print(f"\nCompound effect over 5 periods (starting with {base}):")
        
        # Correct Monte Carlo
        correct_val = base
        print(f"  CORRECT Monte Carlo (12% = 0.12):")
        for i in range(5):
            if i > 0:
                correct_val *= (1 + monte_carlo_correct)
            print(f"    Period {i+1}: {correct_val:.1f}")
        
        # Wrong Monte Carlo  
        wrong_val = base
        print(f"  WRONG Monte Carlo (12% = 12.0):")
        for i in range(5):
            if i > 0:
                wrong_val *= (1 + monte_carlo_wrong)
            print(f"    Period {i+1}: {wrong_val:,.0f}")
            if wrong_val > 1000000:
                print(f"    🚨 ASTRONOMICAL after just {i+1} periods!")
                break
        
        print(f"\n🎯 HYPOTHESIS TEST")
        print("-" * 40)
        
        # Let's check what the actual simulation cash flows suggest
        simulation_cash_flows = [-283611, -1319950, 4886470, -193394111212515300]
        
        print("From actual simulation, we got cash flows:")
        for i, cf in enumerate(simulation_cash_flows[:4]):
            print(f"  Period {i+1}: {cf:,.0f}")
        
        # The jump from period 3 to 4 is the key indicator
        if len(simulation_cash_flows) >= 4:
            ratio_3_to_4 = abs(simulation_cash_flows[3] / simulation_cash_flows[2])
            print(f"\nRatio Period 4 / Period 3: {ratio_3_to_4:,.0f}x")
            
            if ratio_3_to_4 > 1000000:
                print(f"🚨 This suggests PERCENTAGE MISINTERPRETATION!")
                print(f"   A growth rate of 12.0 (1200%) instead of 0.12 (12%)")
                print(f"   would cause exactly this kind of astronomical jump")
            else:
                print(f"✅ This suggests normal compound growth")
        
        print(f"\n🔍 CHECKING EXCEL CELL FORMATTING")
        print("-" * 40)
        
        # Check how Excel actually formats these cells
        for var in ['F4', 'F5', 'F6']:
            cell = sheet_values[var]
            value = cell.value
            format_code = cell.number_format
            
            print(f"{var}:")
            print(f"  Value: {value}")
            print(f"  Format: {format_code}")
            
            # Check if Excel is expecting percentage format
            if format_code and '%' in format_code:
                print(f"  📊 Excel DISPLAYS this as: {value * 100:.1f}%")
                print(f"  📊 Excel STORES this as: {value}")
                print(f"  🚨 Monte Carlo should send: {value} (decimal)")
                print(f"  🚨 NOT: {value * 100} (percentage)")
            else:
                print(f"  ✅ Excel treats as decimal number")
        
        print(f"\n💡 NEXT STEPS TO VERIFY")
        print("-" * 40)
        print("1. Check Monte Carlo UI - what format do you enter?")
        print("   - Do you enter '12%' or '0.12' or '12'?")
        print("2. Check Monte Carlo backend logs for actual values sent")
        print("3. Check if there's a conversion error in the simulation engine")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    deep_percentage_analysis()
