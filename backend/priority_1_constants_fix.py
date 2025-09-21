
# CRITICAL FIX for get_constants_for_file() in backend/excel_parser/service.py

async def get_constants_for_file(file_id: str, exclude_cells: Set[Tuple[str, str]] = None) -> Dict[Tuple[str, str], Any]:
    """
    PRIORITY 1 FIX: Get constants but exclude Row 107 Customer Growth cells
    These must be loaded as formulas to maintain F4→Growth dependency chain
    """
    file_path = _find_excel_file(file_id)
    if not file_path or not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"File {file_id} not found")
    
    exclude_cells = exclude_cells or set()
    cell_values = {}
    
    # CRITICAL: Also exclude Row 107 cells that should be formulas
    customer_growth_cells = set()
    for col in range(ord('C'), ord('AL') + 1):  # C through AL
        col_letter = chr(col)
        for sheet in ['WIZEMICE Likest', 'WIZEMICE High', 'WIZEMICE Low']:
            customer_growth_cells.add((sheet, f"{col_letter}107"))
    
    logger.info(f"🔧 [PRIORITY_1_FIX] Excluding {len(customer_growth_cells)} Row 107 cells from constants")
    exclude_cells.update(customer_growth_cells)
    
    # Load workbook with data_only=True to get calculated values
    workbook = openpyxl.load_workbook(file_path, data_only=True)
    
    for sheet_name in workbook.sheetnames:
        sheet = workbook[sheet_name]
        
        for row in sheet.iter_rows():
            for cell in row:
                if cell.value is not None:
                    cell_coord = cell.coordinate
                    cell_key = (sheet_name, cell_coord)
                    
                    # Skip excluded cells (MC inputs + Row 107 formulas)
                    if cell_key in exclude_cells:
                        continue
                    
                    cell_values[cell_key] = cell.value
    
    workbook.close()
    return cell_values
