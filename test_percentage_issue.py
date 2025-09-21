#!/usr/bin/env python3
"""
Test to demonstrate the percentage interpretation issue
"""

def test_percentage_issue():
    """Test what happens with different percentage formats"""
    
    print("🔍 PERCENTAGE INTERPRETATION TEST")
    print("=" * 50)
    
    # Simulate Excel static values (what we see in the file)
    f4_excel_static = 0.1   # 10% stored as 0.1 in Excel
    f5_excel_static = 0.15  # 15% stored as 0.15 in Excel
    f6_excel_static = 0.08  # 8% stored as 0.08 in Excel
    
    print("📊 EXCEL STATIC VALUES (correct):")
    print(f"   F4 = {f4_excel_static} (displays as {f4_excel_static*100:.0f}%)")
    print(f"   F5 = {f5_excel_static} (displays as {f5_excel_static*100:.0f}%)")
    print(f"   F6 = {f6_excel_static} (displays as {f6_excel_static*100:.0f}%)")
    
    # Test case 1: Monte Carlo sends correct decimal format
    print(f"\n✅ CORRECT Monte Carlo (decimal format):")
    mc_f4_correct = 0.12  # 12% as decimal
    mc_f5_correct = 0.18  # 18% as decimal
    mc_f6_correct = 0.09  # 9% as decimal
    
    print(f"   F4 = {mc_f4_correct} → growth factor = {1 + mc_f4_correct}")
    print(f"   F5 = {mc_f5_correct} → growth factor = {1 + mc_f5_correct}")
    print(f"   F6 = {mc_f6_correct} → growth factor = {1 + mc_f6_correct}")
    
    # Test case 2: Monte Carlo sends percentage as whole number (THE BUG)
    print(f"\n🚨 WRONG Monte Carlo (percentage as whole number):")
    mc_f4_wrong = 12.0    # 12% sent as 12.0 instead of 0.12
    mc_f5_wrong = 18.0    # 18% sent as 18.0 instead of 0.18
    mc_f6_wrong = 9.0     # 9% sent as 9.0 instead of 0.09
    
    print(f"   F4 = {mc_f4_wrong} → growth factor = {1 + mc_f4_wrong} (1300% growth!)")
    print(f"   F5 = {mc_f5_wrong} → growth factor = {1 + mc_f5_wrong} (1900% growth!)")
    print(f"   F6 = {mc_f6_wrong} → growth factor = {1 + mc_f6_wrong} (1000% growth!)")
    
    # Demonstrate compound effect
    print(f"\n🧮 COMPOUND EFFECT OVER 5 PERIODS:")
    base_customers = 1000
    
    print(f"   Starting customers: {base_customers}")
    
    # Correct scenario
    customers_correct = base_customers
    print(f"\n   CORRECT (12% = 0.12):")
    for i in range(5):
        if i > 0:
            customers_correct *= (1 + mc_f4_correct)
        print(f"     Period {i+1}: {customers_correct:,.0f} customers")
    
    # Wrong scenario
    customers_wrong = base_customers
    print(f"\n   WRONG (12% = 12.0):")
    for i in range(5):
        if i > 0:
            customers_wrong *= (1 + mc_f4_wrong)
        print(f"     Period {i+1}: {customers_wrong:,.0f} customers")
        if customers_wrong > 1000000:
            print(f"     🚨 ASTRONOMICAL after {i+1} periods!")
            break
    
    print(f"\n💡 CONCLUSION:")
    print(f"   The ratio between wrong and correct after 3 periods:")
    correct_3 = base_customers * (1 + mc_f4_correct) ** 2
    wrong_3 = base_customers * (1 + mc_f4_wrong) ** 2
    ratio = wrong_3 / correct_3
    print(f"   Wrong: {wrong_3:,.0f} / Correct: {correct_3:,.0f} = {ratio:,.0f}x difference")
    
    print(f"\n🎯 THIS EXPLAINS:")
    print(f"   - Why static Excel scenario works (uses 0.1, 0.15, 0.08)")
    print(f"   - Why Monte Carlo gives astronomical results (uses 10.0, 15.0, 8.0)")
    print(f"   - Why sensitivity analysis shows F4 has 95% impact (it IS connected)")
    print(f"   - Why cash flows jump from millions to quadrillions")

if __name__ == "__main__":
    test_percentage_issue()
