
# CRITICAL FIX for get_formulas_for_file() in backend/excel_parser/service.py

async def get_formulas_for_file(file_id: str) -> Dict[Tuple[str, str], str]:
    """
    PRIORITY 1 FIX: Ensure Row 107 Customer Growth formulas are loaded
    These cells must be treated as formulas (=F4, =F5, =F6) not constants
    """
    file_path = _find_excel_file(file_id)
    if not file_path:
        raise HTTPException(status_code=404, detail=f"File {file_id} not found")
    
    # Load with data_only=False to get formulas
    workbook = openpyxl.load_workbook(file_path, data_only=False)
    all_formulas = {}
    
    for sheet_name in workbook.sheetnames:
        sheet = workbook[sheet_name]
        
        for row in sheet.iter_rows():
            for cell in row:
                if cell.value is not None and isinstance(cell.value, str) and cell.value.startswith('='):
                    cell_key = (sheet_name, cell.coordinate)
                    all_formulas[cell_key] = cell.value
                    
                    # PRIORITY 1: Log Row 107 formulas specifically
                    if '107' in cell.coordinate:
                        logger.info(f"🔧 [PRIORITY_1_FIX] Found Row 107 formula: {cell_key} = {cell.value}")
    
    workbook.close()
    
    # CRITICAL CHECK: Verify Row 107 formulas exist
    row_107_formulas = {k: v for k, v in all_formulas.items() if '107' in k[1]}
    if len(row_107_formulas) == 0:
        logger.error("🚨 [PRIORITY_1_FIX] CRITICAL: No Row 107 formulas found!")
        logger.error("   This means Excel model doesn't have F4→Growth formulas!")
        logger.error("   User needs to verify Excel model structure.")
    else:
        logger.info(f"✅ [PRIORITY_1_FIX] Found {len(row_107_formulas)} Row 107 formulas")
    
    return all_formulas
