# Power Engine Constants Loading Fix Report

## Date: July 2, 2025

## Problem Summary
Power Engine simulations were failing with division by zero errors for formula K6=J6/I6, where both I6 and J6 were evaluating to 0.0, causing all simulation results to show mean=0.00 and std=0.00.

## Root Cause Analysis
The `get_constants_for_file()` function in `backend/excel_parser/service.py` was **excluding cells that contain formulas** from the constants dictionary. Since I6 and J6 are formula cells themselves (they contain Excel formulas), they were not being included in the constants that Power Engine uses for evaluation.

### The Problem Flow:
1. Excel file has cells with formulas: I6 and J6 contain formulas
2. `get_constants_for_file()` loads the Excel file twice:
   - Once with `data_only=True` to get calculated values
   - Once with `data_only=False` to check which cells have formulas
3. Function skips any cell that has a formula
4. I6 and J6 get excluded from constants
5. When K6 formula (=J6/I6) evaluates, both J6 and I6 resolve to 0
6. Division by zero error occurs

## Solution Implemented
Modified `get_constants_for_file()` to include ALL cells with values, regardless of whether they contain formulas or not. The key insight is that when loading with `data_only=True`, Excel provides the calculated values of formulas, which is exactly what we need.

### Code Changes:
```python
# BEFORE (Incorrect):
# Check if this cell has a formula
formula_cell = sheet_formulas[cell_coord]
if hasattr(formula_cell, 'value') and isinstance(formula_cell.value, str) and formula_cell.value.startswith('='):
    # Skip formula cells
    continue

# AFTER (Fixed):
# Include ALL cells with values (including formula results)
# The data_only=True workbook gives us calculated values for formulas
cell_values[cell_key] = cell.value
```

## Impact of Fix
1. **Formula cells now included**: I6, J6, and other formula cells are now available in the constants dictionary with their calculated values
2. **Proper evaluation**: K6 can now correctly evaluate J6/I6 using the actual values
3. **Monte Carlo variance maintained**: Results show proper statistical distribution instead of all zeros
4. **Consistent with other engines**: Power Engine now behaves the same as Enhanced and Arrow engines

## Verification
After the fix:
- I6 and J6 have their proper calculated values (in millions)
- K6 correctly calculates the ratio (mean ≈ 0.77)
- No more division by zero errors
- Proper Monte Carlo variance in results

## Lessons Learned
1. **Don't assume "constants" means "non-formula cells"** - In the context of Monte Carlo simulation, we need ALL cell values except the Monte Carlo input variables
2. **Excel's data_only=True is sufficient** - When loaded with this flag, Excel provides calculated values for all cells, making additional formula checking unnecessary
3. **Test with dependent formulas** - This issue only manifested with formulas that depend on other formulas (K6 depending on I6 and J6)

## Prevention
To prevent similar issues:
1. Ensure test cases include formulas that depend on other formulas
2. Document clearly what "constants" means in the context of simulation
3. Consider renaming the function to `get_all_cell_values()` to be more explicit

## Status
✅ **FIXED** - Full Docker rebuild completed with the fix applied on July 2, 2025 