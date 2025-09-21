"""
AST-based Formula Parser for Power Engine

This module provides a robust formula parser using an Abstract Syntax Tree (AST)
to replace the regex-based dependency extraction. It is the foundation for
advanced formula analysis and GPU function support.
"""
import formulas
import re
from typing import List, Tuple, Union

# Pre-compile regex for performance
CELL_REF_PATTERN = re.compile(r"([a-zA-Z_0-9']+!)?(\$?[A-Z]+\$?[0-9]+)")

def get_dependencies_from_formula(formula_string: str, sheet: str) -> List[Tuple[str, str]]:
    """
    Parses a formula string using an AST and extracts all cell and range dependencies.

    Args:
        formula_string: The Excel formula (e.g., "=IF(A1>B1, SUM(C1:D10), E1)")
        sheet: The name of the sheet where the formula resides.

    Returns:
        A list of unique dependencies as (sheet, cell_reference) tuples.
    """
    if not formula_string.startswith('='):
        return []

    try:
        # The 'formulas' library expects the formula without the leading '='
        model = formulas.Parser().parse(formula_string[1:])
        dependencies = set()

        # Walk the AST to find all cell/range references
        for node in model.walk():
            if isinstance(node, formulas.tokens.operand.Cell):
                # Handle single cell references (e.g., A1, Sheet2!B2)
                cell_sheet = node.sheet if node.sheet else sheet
                dependencies.add((cell_sheet, node.cell.replace('$', '')))

            elif isinstance(node, formulas.tokens.operand.Range):
                # Handle cell ranges (e.g., A1:B10)
                range_sheet = node.sheet if node.sheet else sheet
                start_cell, end_cell = node.start.replace('$', ''), node.end.replace('$', '')
                
                # Expand the range into individual cells
                # Note: This uses an internal helper. A production system might need a more
                # robust range expander utility.
                expanded_range = _expand_range(range_sheet, start_cell, end_cell)
                dependencies.update(expanded_range)

        return list(dependencies)

    except Exception as e:
        # If the AST parser fails, fall back to the simpler regex-based approach
        # This ensures that even with malformed formulas, we get some dependencies.
        # logger.warning(f"AST parsing failed for formula '{formula_string}': {e}. Falling back to regex.")
        return _get_dependencies_regex_fallback(formula_string, sheet)


def _get_dependencies_regex_fallback(formula_string: str, sheet: str) -> List[Tuple[str, str]]:
    """A regex-based fallback for dependency extraction if AST parsing fails."""
    dependencies = set()
    matches = CELL_REF_PATTERN.findall(formula_string.upper())
    for match in matches:
        dep_sheet, cell = match
        dep_sheet = dep_sheet.strip("!'") if dep_sheet else sheet
        dependencies.add((dep_sheet, cell.replace('$', '')))
    return list(dependencies)


def _expand_range(sheet: str, start_cell: str, end_cell: str) -> List[Tuple[str, str]]:
    """
    Expands an Excel range (e.g., "A1:B3") into a list of individual cell references.
    """
    start_col, start_row = _parse_cell_ref(start_cell)
    end_col, end_row = _parse_cell_ref(end_cell)

    # Ensure start/end are correctly ordered
    if start_row > end_row:
        start_row, end_row = end_row, start_row
    if start_col > end_col:
        start_col, end_col = end_col, start_col

    cells = []
    for col in range(start_col, end_col + 1):
        col_letter = _col_idx_to_letter(col)
        for row in range(start_row, end_row + 1):
            cells.append((sheet, f"{col_letter}{row}"))
    return cells


def _parse_cell_ref(cell_ref: str) -> Tuple[int, int]:
    """Parses a cell reference (e.g., 'A1') into 1-based (col, row) integers."""
    match = re.match(r"([A-Z]+)(\d+)", cell_ref.upper())
    if not match:
        raise ValueError(f"Invalid cell reference format: {cell_ref}")

    col_str, row_str = match.groups()
    col_idx = 0
    for char in col_str:
        col_idx = col_idx * 26 + (ord(char) - ord('A')) + 1
    return col_idx, int(row_str)


def _col_idx_to_letter(col_idx: int) -> str:
    """Converts a 1-based column index into an Excel column letter (e.g., 1 -> 'A')."""
    letters = ""
    while col_idx > 0:
        col_idx, remainder = divmod(col_idx - 1, 26)
        letters = chr(65 + remainder) + letters
    return letters 