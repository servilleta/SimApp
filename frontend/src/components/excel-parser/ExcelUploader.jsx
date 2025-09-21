import React, { useState, useCallback, useEffect, memo, lazy, Suspense } from 'react';
import { useDropzone } from 'react-dropzone';
import { useDispatch, useSelector } from 'react-redux';
import { uploadExcel, resetExcelState, selectFileInfo, selectExcelLoading, selectExcelError } from '../../store/excelSlice';
import { runSimulation } from '../../store/simulationSlice';
import { resetSetup } from '../../store/simulationSetupSlice';
import Button from '../common/Button';
import SheetTabs from './SheetTabs';
import LoginModal from '../auth/LoginModal';
import { isAuthenticated } from '../../services/authService';

const ExcelViewWithConfig = lazy(() => import('./ExcelViewWithConfig'));

function ExcelUploaderInner() {
  const dispatch = useDispatch();
  const [uploadedFile, setUploadedFile] = useState(null);
  
  const { fileInfo, isLoading, error } = useSelector(state => state.excel);
  const { inputVariables } = useSelector(state => state.simulationSetup);
  
  const [selectedSheetName, setSelectedSheetName] = useState(null);
  const [dragActive, setDragActive] = useState(false);
  const [showLoginModal, setShowLoginModal] = useState(false);
  const [authChecked, setAuthChecked] = useState(false);

  useEffect(() => {
    if (fileInfo && fileInfo.sheets && fileInfo.sheets.length > 0) {
      if (!selectedSheetName || !(fileInfo.sheets || []).find(s => s.sheet_name === selectedSheetName)){
        setSelectedSheetName(fileInfo.sheets[0].sheet_name);
      }
    } else {
      setSelectedSheetName(null);
    }
  }, [fileInfo, selectedSheetName]);

  // Check authentication on component mount
  useEffect(() => {
    const checkAuth = () => {
      const authenticated = isAuthenticated();
      if (!authenticated) {
        setShowLoginModal(true);
      }
      setAuthChecked(true);
    };
    
    checkAuth();
  }, []);

  const onDrop = useCallback((acceptedFiles) => {
    if (acceptedFiles && acceptedFiles.length > 0) {
      const file = acceptedFiles[0];
      if (file.type !== 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet' && file.type !== 'application/vnd.ms-excel') {
        alert('Invalid file type. Please upload an Excel file (.xlsx, .xls).');
        return;
      }
      setUploadedFile(file);
      setSelectedSheetName(null);
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    multiple: false,
    accept: {
      'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': ['.xlsx'],
      'application/vnd.ms-excel': ['.xls'],
    }
  });

  const handleUpload = () => {
    if (uploadedFile) {
      dispatch(resetExcelState());
      dispatch(uploadExcel(uploadedFile));
    }
  };
  
  const handleReset = () => {
    setUploadedFile(null);
    dispatch(resetExcelState());
  };

  const handleSelectSheet = (sheetName) => {
    setSelectedSheetName(sheetName);
  };

  // Clean modern styles with Braun color system
  const dropzoneStyle = {
    border: isDragActive ? '2px dashed var(--color-braun-orange)' : '2px dashed var(--color-border-light)',
    borderRadius: '12px',
    padding: '48px 32px',
    textAlign: 'center',
    cursor: 'pointer',
    backgroundColor: isDragActive ? 'var(--color-warm-white)' : 'var(--color-white)',
    transition: 'all 0.2s ease',
    position: 'relative',
    overflow: 'hidden',
  };

  const dropzoneContentStyle = {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    gap: '16px',
  };

  const uploadIconStyle = {
    fontSize: '48px',
    color: isDragActive ? 'var(--color-braun-orange)' : 'var(--color-medium-grey)',
    marginBottom: '8px',
  };

  const uploadTextStyle = {
    fontSize: '18px',
    fontWeight: '600',
    color: 'var(--color-charcoal)',
    marginBottom: '8px',
  };

  const uploadSubtextStyle = {
    fontSize: '14px',
    color: 'var(--color-medium-grey)',
  };

  const fileInfoStyle = {
    display: 'flex',
    alignItems: 'center',
    gap: '12px',
    padding: '16px',
    backgroundColor: 'var(--color-warm-white)',
    border: '1px solid var(--color-border-light)',
    borderRadius: '8px',
    marginTop: '20px',
  };

  const loadingStyle = {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    gap: '16px',
    padding: '48px 32px',
    textAlign: 'center',
  };

  const loadingSpinnerStyle = {
    width: '32px',
    height: '32px',
    border: '3px solid var(--color-light-grey)',
    borderTop: '3px solid var(--color-braun-orange)',
    borderRadius: '50%',
    animation: 'spin 1s linear infinite',
  };

  const errorStyle = {
    backgroundColor: 'var(--color-error-bg)',
    border: '1px solid var(--color-error-border)',
    borderRadius: '8px',
    padding: '16px',
    marginBottom: '20px',
  };

  const errorTitleStyle = {
    fontSize: '16px',
    fontWeight: '600',
    color: 'var(--color-error)',
    marginBottom: '8px',
  };

  const errorTextStyle = {
    fontSize: '14px',
    color: 'var(--color-error)',
  };

  const buttonStyle = {
    padding: '12px 24px',
    backgroundColor: uploadedFile ? 'var(--color-braun-orange)' : 'var(--color-medium-grey)',
    color: '#ffffff',
    border: 'none',
    borderRadius: '8px',
    fontSize: '14px',
    fontWeight: '600',
    cursor: uploadedFile ? 'pointer' : 'not-allowed',
    transition: 'all 0.15s ease',
    marginTop: '20px',
  };

  const selectedSheetData = fileInfo && selectedSheetName 
    ? (fileInfo.sheets || []).find(sheet => sheet.sheet_name === selectedSheetName)
    : null;

  const handleLoginSuccess = () => {
    setShowLoginModal(false);
    window.location.reload();
  };

  if (isLoading) {
    return (
      <div style={loadingStyle}>
        <div style={loadingSpinnerStyle}></div>
        <h3 style={{ margin: 0, fontSize: '18px', color: 'var(--color-charcoal)' }}>
          Processing your Excel file...
        </h3>
        <p style={{ margin: 0, fontSize: '14px', color: 'var(--color-medium-grey)' }}>
          This may take a few moments while we analyze your spreadsheet.
        </p>
        
        <LoginModal 
          isOpen={showLoginModal}
          onClose={() => setShowLoginModal(false)}
          onLoginSuccess={handleLoginSuccess}
        />
        
        <style>{`
          @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
          }
        `}</style>
      </div>
    );
  }

  if (fileInfo) {
    return (
      <div>
        {fileInfo.sheets && fileInfo.sheets.length > 0 && (
          <>
            {selectedSheetData ? (
              <div style={{ marginTop: '20px' }}>
                <Suspense fallback={<div>Loading spreadsheet...</div>}>
                  <ExcelViewWithConfig 
                    fileId={fileInfo.file_id} 
                    selectedSheetData={selectedSheetData}
                    fileInfo={fileInfo}
                    onReset={handleReset}
                    sheets={fileInfo.sheets}
                    activeSheetName={selectedSheetName}
                    onSelectSheet={handleSelectSheet}
                  />
                </Suspense>
              </div>
            ) : (
              <div style={{
                padding: '20px',
                textAlign: 'center',
                backgroundColor: '#fef3c7',
                border: '1px solid #fbbf24',
                borderRadius: '8px',
                marginTop: '20px'
              }}>
                <h4 style={{ margin: '0 0 12px 0', color: '#92400e' }}>
                  ⚠️ Waiting for sheet selection...
                </h4>
                <p style={{ margin: '0 0 16px 0', fontSize: '14px', color: '#78350f' }}>
                  Available sheets: {fileInfo.sheets.map(s => s.sheet_name).join(', ')}
                </p>
                <button 
                  onClick={() => handleSelectSheet(fileInfo.sheets[0].sheet_name)}
                  style={{
                    padding: '10px 20px',
                    backgroundColor: '#3b82f6',
                    color: 'white',
                    border: 'none',
                    borderRadius: '6px',
                    cursor: 'pointer',
                    fontSize: '14px',
                    fontWeight: '500'
                  }}
                >
                  Select First Sheet
                </button>
              </div>
            )}
          </>
        )}
        
        <LoginModal 
          isOpen={showLoginModal}
          onClose={() => setShowLoginModal(false)}
          onLoginSuccess={handleLoginSuccess}
        />
      </div>
    );
  }
  
  if (error) {
    return (
      <div>
        <div style={errorStyle}>
          <h4 style={errorTitleStyle}>Upload Error</h4>
          <p style={errorTextStyle}>
            {typeof error === 'string' ? error : error?.message || 'Upload failed'}
          </p>
        </div>
        <button 
          onClick={handleReset}
          style={{
            ...buttonStyle,
            backgroundColor: '#6b7280',
            cursor: 'pointer'
          }}
        >
          Try Again
        </button>
        
        <LoginModal 
          isOpen={showLoginModal}
          onClose={() => setShowLoginModal(false)}
          onLoginSuccess={handleLoginSuccess}
        />
      </div>
    );
  }

  return (
    <div>
      <div {...getRootProps()} style={dropzoneStyle}>
        <input {...getInputProps()} />
        <div style={dropzoneContentStyle}>
          <div style={uploadIconStyle}>📁</div>
          <div>
            <div style={uploadTextStyle}>
              {isDragActive ? 'Drop your Excel file here' : 'Upload Excel File'}
            </div>
            <div style={uploadSubtextStyle}>
              Drag and drop your .xlsx or .xls file, or click to browse
            </div>
          </div>
        </div>
      </div>

      {uploadedFile && (
        <div style={fileInfoStyle}>
          <span style={{ fontSize: '20px' }}>📄</span>
          <div>
            <div style={{ fontSize: '14px', fontWeight: '600', color: 'var(--color-charcoal)' }}>
              {uploadedFile.name}
            </div>
            <div style={{ fontSize: '12px', color: 'var(--color-medium-grey)' }}>
              {(uploadedFile.size / 1024 / 1024).toFixed(2)} MB
            </div>
          </div>
        </div>
      )}

      <button 
        onClick={handleUpload}
        disabled={!uploadedFile}
        style={buttonStyle}
        onMouseEnter={(e) => {
          if (uploadedFile) {
            e.target.style.backgroundColor = 'var(--color-braun-orange-dark)';
          }
        }}
        onMouseLeave={(e) => {
          if (uploadedFile) {
            e.target.style.backgroundColor = 'var(--color-braun-orange)';
          }
        }}
      >
        {uploadedFile ? 'Process File' : 'Select a file to upload'}
      </button>
      
      <LoginModal 
        isOpen={showLoginModal}
        onClose={() => setShowLoginModal(false)}
        onLoginSuccess={handleLoginSuccess}
      />
    </div>
  );
}

const ExcelUploader = memo(ExcelUploaderInner);

export default ExcelUploader; 