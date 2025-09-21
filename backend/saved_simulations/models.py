from sqlalchemy import Boolean, Column, Integer, String, Text, DateTime, ForeignKey, JSON
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from database import Base

class SavedSimulation(Base):
    __tablename__ = "saved_simulations"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False, index=True)
    description = Column(Text, nullable=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    
    # File information
    original_filename = Column(String, nullable=False)
    file_path = Column(String, nullable=False)  # Path to stored Excel file
    file_id = Column(String, nullable=True)     # Original file_id from excel parser
    
    # Simulation configuration
    simulation_config = Column(JSON, nullable=False)  # Stores input variables, target cells, iterations, etc.
    simulation_results = Column(JSON, nullable=True)  # Stores simulation results (histograms, statistics, etc.)
    
    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationship
    user = relationship("User")

    def __repr__(self):
        return f"<SavedSimulation(name='{self.name}', user_id={self.user_id})>" 