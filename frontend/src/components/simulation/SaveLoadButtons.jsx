import React, { useState } from 'react';
import { useSelector } from 'react-redux';
import SaveSimulationModal from './SaveSimulationModal';
import LoadSimulationModal from './LoadSimulationModal';
import './SaveLoadButtons.css';

const SaveLoadButtons = () => {
  const [showSaveModal, setShowSaveModal] = useState(false);
  const [showLoadModal, setShowLoadModal] = useState(false);
  
  // Check if we have a file loaded and simulation setup
  const { fileInfo } = useSelector(state => state.excel);
  const { inputVariables, resultCells } = useSelector(state => state.simulationSetup);
  
  const canSave = fileInfo && inputVariables.length > 0 && resultCells.length > 0;

  return (
    <>
      <button 
        className="toolbar-btn-compact save-btn"
        onClick={() => setShowSaveModal(true)}
        disabled={!canSave}
        title={!canSave ? "Upload an Excel file and configure simulation to save" : "Save current simulation"}
      >
        <span className="btn-text">💾 Save</span>
      </button>
      
      <button 
        className="toolbar-btn-compact load-btn"
        onClick={() => setShowLoadModal(true)}
        title="Open a saved simulation"
      >
        <span className="btn-text">📂 Open</span>
      </button>

      {/* Save Modal */}
      {showSaveModal && (
        <SaveSimulationModal
          isOpen={showSaveModal}
          onClose={() => setShowSaveModal(false)}
        />
      )}

      {/* Load Modal */}
      {showLoadModal && (
        <LoadSimulationModal
          isOpen={showLoadModal}
          onClose={() => setShowLoadModal(false)}
        />
      )}
    </>
  );
};

export default SaveLoadButtons; 