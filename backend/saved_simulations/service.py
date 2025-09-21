import os
import shutil
import uuid
from pathlib import Path
from typing import List, Optional
from sqlalchemy.orm import Session
from fastapi import HTTPException, status

from saved_simulations.models import SavedSimulation
from saved_simulations.schemas import SaveSimulationRequest, SavedSimulationResponse, LoadSimulationResponse
from excel_parser.service import get_parsed_file_info, parse_excel_file, get_all_parsed_sheets_data

# Directory to store saved simulation files
SAVED_SIMULATIONS_DIR = Path("saved_simulations_files")
SAVED_SIMULATIONS_DIR.mkdir(exist_ok=True)

async def save_simulation(
    db: Session, 
    user_id: int, 
    request: SaveSimulationRequest
) -> SavedSimulationResponse:
    """Save a simulation with its Excel file and configuration"""
    
    # Get current file info from excel_parser
    try:
        file_info = await get_parsed_file_info(request.file_id)
    except HTTPException:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"File with ID {request.file_id} not found"
        )
    
    # Check if simulation name already exists for this user
    existing = db.query(SavedSimulation).filter(
        SavedSimulation.user_id == user_id,
        SavedSimulation.name == request.name
    ).first()
    
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Simulation with name '{request.name}' already exists"
        )
    
    # Copy the Excel file to permanent storage
    # Find the original file using the file_id prefix
    original_file_path = None
    upload_dir = Path("uploads")
    
    for file_path in upload_dir.glob(f"{request.file_id}_*"):
        if file_path.suffix in ['.xlsx', '.xls']:
            original_file_path = file_path
            break
    
    if not original_file_path or not original_file_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Original Excel file not found"
        )
    
    # Generate unique filename for saved simulation
    saved_file_id = str(uuid.uuid4())
    saved_file_path = SAVED_SIMULATIONS_DIR / f"{saved_file_id}.xlsx"
    
    # Copy the file
    shutil.copy2(original_file_path, saved_file_path)
    
    # Create database record
    saved_simulation = SavedSimulation(
        name=request.name,
        description=request.description,
        user_id=user_id,
        original_filename=file_info["filename"],
        file_path=str(saved_file_path),
        file_id=request.file_id,
        simulation_config=request.simulation_config,
        simulation_results=request.simulation_results  # Save simulation results
    )
    
    db.add(saved_simulation)
    db.commit()
    db.refresh(saved_simulation)
    
    return SavedSimulationResponse.from_orm(saved_simulation)

async def load_simulation(
    db: Session, 
    user_id: int, 
    simulation_id: int
) -> LoadSimulationResponse:
    """Load a saved simulation and restore its Excel file"""
    
    # Get the saved simulation
    saved_simulation = db.query(SavedSimulation).filter(
        SavedSimulation.id == simulation_id,
        SavedSimulation.user_id == user_id
    ).first()
    
    if not saved_simulation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Saved simulation not found"
        )
    
    # Check if the saved file still exists
    saved_file_path = Path(saved_simulation.file_path)
    if not saved_file_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Saved Excel file not found on disk"
        )
    
    # We'll let parse_excel_file handle the file processing and generate its own file_id
    
    # Re-process the Excel file through the excel_parser
    try:
        # Create a temporary UploadFile-like object to process the restored file
        from fastapi import UploadFile
        
        with open(saved_file_path, "rb") as f:
            file_content = f.read()
        
        # Create a mock UploadFile object
        from io import BytesIO
        file_obj = BytesIO(file_content)
        upload_file = UploadFile(
            file=file_obj,
            filename=saved_simulation.original_filename
        )
        
        # Process the file and get the new file_id
        restored_file_info = await parse_excel_file(upload_file)
        new_file_id = restored_file_info.file_id
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to restore Excel file: {str(e)}"
        )
    
    # Get the complete parsed file info including sheet data
    try:
        # Get basic file info
        basic_info = await get_parsed_file_info(new_file_id)
        
        # Get complete sheet data
        complete_sheets = await get_all_parsed_sheets_data(new_file_id)
        
        # Combine into complete file info
        file_info = {
            "file_id": basic_info["file_id"],
            "filename": basic_info["filename"],
            "status": basic_info["status"],
            "file_size": getattr(restored_file_info, 'file_size', 0),
            "sheet_names": [sheet.sheet_name for sheet in complete_sheets],
            "sheets": [sheet.model_dump() for sheet in complete_sheets]  # Convert to dict for JSON serialization
        }
        
    except Exception as e:
        # If we can't get the complete file info, still return the response but log the error
        print(f"Warning: Could not get complete file info for {new_file_id}: {e}")
        file_info = None
    
    return LoadSimulationResponse(
        id=saved_simulation.id,
        name=saved_simulation.name,
        description=saved_simulation.description,
        original_filename=saved_simulation.original_filename,
        file_id=new_file_id,
        simulation_config=saved_simulation.simulation_config,
        simulation_results=saved_simulation.simulation_results,  # Include saved simulation results
        created_at=saved_simulation.created_at,
        file_info=file_info  # Include the Excel data directly
    )

async def get_user_simulations(
    db: Session, 
    user_id: int
) -> List[SavedSimulationResponse]:
    """Get all saved simulations for a user"""
    
    simulations = db.query(SavedSimulation).filter(
        SavedSimulation.user_id == user_id
    ).order_by(SavedSimulation.created_at.desc()).all()
    
    return [SavedSimulationResponse.from_orm(sim) for sim in simulations]

async def delete_simulation(
    db: Session, 
    user_id: int, 
    simulation_id: int
) -> bool:
    """Delete a saved simulation and its associated file"""
    
    saved_simulation = db.query(SavedSimulation).filter(
        SavedSimulation.id == simulation_id,
        SavedSimulation.user_id == user_id
    ).first()
    
    if not saved_simulation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Saved simulation not found"
        )
    
    # Delete the file from disk
    saved_file_path = Path(saved_simulation.file_path)
    if saved_file_path.exists():
        saved_file_path.unlink()
    
    # Delete from database
    db.delete(saved_simulation)
    db.commit()
    
    return True 