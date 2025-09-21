from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class CellData(BaseModel):
    value: Any
    formula: Optional[str] = None
    is_formula_cell: bool = False
    coordinate: str # e.g., A1, B2
    # NEW: Original Excel formatting information
    number_format: Optional[str] = None  # Excel number format code (e.g., "0.00", "#,##0", "mm/dd/yyyy")
    display_value: Optional[str] = None  # Formatted display value as it appears in Excel
    data_type: Optional[str] = None      # Excel data type (n=number, s=string, d=date, etc.)
    font_name: Optional[str] = None      # Font family
    font_size: Optional[float] = None    # Font size
    font_bold: Optional[bool] = None     # Bold formatting
    font_italic: Optional[bool] = None   # Italic formatting
    font_color: Optional[str] = None     # Font color (hex)
    fill_color: Optional[str] = None     # Cell background color (hex)
    alignment: Optional[str] = None      # Text alignment (left, center, right)
    # NEW: Border formatting information
    border_top: Optional[str] = None     # Top border style and color
    border_bottom: Optional[str] = None  # Bottom border style and color
    border_left: Optional[str] = None    # Left border style and color
    border_right: Optional[str] = None   # Right border style and color

class SheetData(BaseModel):
    sheet_name: str
    grid_data: List[List[Optional[CellData]]] # List of rows, each row is a list of cells. Optional[CellData] for empty cells.
    # NEW: Column widths from Excel file
    column_widths: Optional[Dict[str, float]] = None  # Maps column letter (A, B, C, etc.) to width in Excel units
    # NEW: Sparse data format for large files (alternative to grid_data)
    sparse_cells: Optional[List[CellData]] = None  # Only non-empty cells with coordinates
    is_sparse_format: bool = False  # Indicates if sparse format is used
    total_rows: Optional[int] = None  # Total number of rows (for sparse format)
    total_cols: Optional[int] = None  # Total number of columns (for sparse format)

class ExcelFileResponse(BaseModel):
    file_id: str
    filename: str
    file_size: int  # File size in bytes
    sheet_names: List[str]  # List of sheet names for quick access
    sheets: List[SheetData] # Changed to return a list of sheets 