import React, { lazy, Suspense } from 'react';
// import ExcelPreview from '../components/excel-parser/ExcelPreview'; // Removed
import { useSelector } from 'react-redux';
// import { Link } from 'react-router-dom'; // Removed if Link is no longer used
// import Button from '../components/common/Button'; // Removed if Button is no longer used
import ErrorBoundary from '../components/common/ErrorBoundary';

// Lazy-load to avoid static circular dependencies during initial bundle evaluation
const ExcelUploader = lazy(() => import('../components/excel-parser/ExcelUploader'));

const UploadPage = () => {
  // const { fileInfo, loading } = useSelector((state) => state.excel); // No longer needed here if only ExcelUploader uses it

  return (
    <div>
      {/* REMOVED: <h2>Upload and Preview Excel File</h2> */}
      {/* REMOVED: <p>Upload your Excel model to begin the simulation process. Supported formats: .xlsx, .xls.</p> */}
      
      <div style={{ /* marginBottom: '2rem', */ padding: '1rem', background: '#fff', borderRadius: '8px', boxShadow: '0 2px 4px rgba(0,0,0,0.05)' }}>
        <Suspense fallback={<div>Loading Excel Uploader...</div>}>
          <ErrorBoundary>
            <ExcelUploader />
          </ErrorBoundary>
        </Suspense>
      </div>

      {/* {loading !== 'pending' && fileInfo && ( Removed this entire block
        <div style={{ marginTop: '2rem', padding: '1rem', background: '#fff', borderRadius: '8px', boxShadow: '0 2px 4px rgba(0,0,0,0.05)' }}>
          <ExcelPreview />
          <div style={{marginTop: '1rem'}}>
            <Link to="/configure">
              <Button variant='primary'>Proceed to Configure Simulation</Button>
            </Link>
          </div>
        </div>
      )} */}
      
    </div>
  );
};

export default UploadPage; 