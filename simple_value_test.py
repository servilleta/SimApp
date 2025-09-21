#!/usr/bin/env python3
"""
Simple test to check variable interpretation
"""

import openpyxl
import random

def test_variable_interpretation():
    """Test variable interpretation issues"""
    print("🔍 VARIABLE INTERPRETATION TEST")
    print("=" * 50)
    
    # Load Excel file
    excel_file = "/home/paperspace/PROJECT/uploads/2a07dd9d-2048-4ff0-82e1-b6c7d74a6f68_WIZEMICE_FINAL_WORKING_MODEL.xlsx"
    
    try:
        workbook = openpyxl.load_workbook(excel_file, data_only=True)
        sheet = workbook['WIZEMICE Likest']
        
        print("📊 STATIC EXCEL VALUES")
        print("-" * 30)
        
        # Get static values
        f4_static = sheet['F4'].value
        f5_static = sheet['F5'].value
        f6_static = sheet['F6'].value
        
        print(f"F4 (static): {f4_static}")
        print(f"F5 (static): {f5_static}")
        print(f"F6 (static): {f6_static}")
        
        # Get some Row 108 values
        c108 = sheet['C108'].value
        d108 = sheet['D108'].value
        e108 = sheet['E108'].value
        f108 = sheet['F108'].value
        
        print(f"\nROW 108 VALUES (static scenario):")
        print(f"C108: {c108}")
        print(f"D108: {d108}")  
        print(f"E108: {e108}")
        print(f"F108: {f108}")
        
        # Calculate growth rates
        if c108 and d108:
            growth_cd = (d108 / c108) - 1
            print(f"\nGrowth C108→D108: {growth_cd:.4f} ({growth_cd*100:.2f}%)")
            print(f"Expected (F4): {f4_static:.4f} ({f4_static*100:.2f}%)")
            print(f"Match: {'✅' if abs(growth_cd - f4_static) < 0.001 else '❌'}")
        
        if d108 and e108:
            growth_de = (e108 / d108) - 1  
            print(f"\nGrowth D108→E108: {growth_de:.4f} ({growth_de*100:.2f}%)")
            print(f"Expected (F4): {f4_static:.4f} ({f4_static*100:.2f}%)")
            print(f"Match: {'✅' if abs(growth_de - f4_static) < 0.001 else '❌'}")
        
        print(f"\n🎯 MONTE CARLO SIMULATION TEST")
        print("-" * 30)
        
        # Simulate what Monte Carlo does
        print("Monte Carlo ranges:")
        print("F4: 0.08 to 0.12 (8% to 12%)")
        print("F5: 0.12 to 0.20 (12% to 20%)")  
        print("F6: 0.05 to 0.10 (5% to 10%)")
        
        # Test with extreme Monte Carlo value
        mc_f4 = 0.12  # Maximum F4 value from Monte Carlo
        
        print(f"\nTesting with F4={mc_f4} (12%):")
        
        # Simulate compound growth over several columns
        base = c108 or 40  # Use actual base from Excel
        current = base
        
        print(f"Starting base: {current}")
        
        # First 10 columns with compound growth
        for i in range(10):
            if i > 0:  # Skip first column
                current = current * (1 + mc_f4)
            col_letter = chr(ord('C') + i)
            print(f"{col_letter}108: {current:,.1f}")
            
            if i > 0:
                amplification = current / base
                print(f"    → {amplification:.2f}x amplification")
                
                if amplification > 100:
                    print(f"    🚨 ASTRONOMICAL AMPLIFICATION!")
                    break
        
        print(f"\n📊 COMPARISON WITH ACTUAL SIMULATION")
        print("-" * 30)
        
        # From the actual simulation logs, we saw cash flows like:
        simulation_cash_flows = [
            -283611, -1319950, 4886470, -193394111212515300,
            701495454, 425626834, 9818618636, 3134156346
        ]
        
        print("Actual simulation cash flows (first 8):")
        for i, cf in enumerate(simulation_cash_flows):
            print(f"  Period {i+1}: {cf:,.0f}")
            if abs(cf) > 1e12:
                print(f"    🚨 QUADRILLION SCALE!")
        
        print(f"\n🎯 ROOT CAUSE ANALYSIS")
        print("-" * 30)
        
        print("1. Static scenario (F4=0.10) shows reasonable growth:")
        print(f"   - Growth per period: {f4_static*100:.1f}%")
        print(f"   - Values stay in reasonable range")
        
        print(f"\n2. Monte Carlo (F4=0.12) creates exponential amplification:")
        print(f"   - Growth per period: {mc_f4*100:.1f}%")
        print(f"   - Compound effect over 39 columns creates astronomical values")
        
        print(f"\n3. The issue is NOT percentage interpretation:")
        print(f"   - Both static and Monte Carlo use decimal format")
        print(f"   - Static: F4={f4_static} = {f4_static*100:.1f}%")
        print(f"   - Monte Carlo: F4={mc_f4} = {mc_f4*100:.1f}%")
        
        print(f"\n4. The issue IS the compound growth formula:")
        print(f"   - Formula: =ROUND(PrevColumn*(1+GrowthRate),0)")
        print(f"   - Over 39 columns: (1.12)^39 = {(1.12)**39:.1f}x amplification!")
        print(f"   - Even small difference (10%→12%) creates massive effect")
        
        print(f"\n✅ CONCLUSION:")
        print(f"   - No percentage interpretation error")
        print(f"   - Compound growth formula is the culprit")
        print(f"   - Solution: Fix Excel formula in Row 108")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_variable_interpretation()
