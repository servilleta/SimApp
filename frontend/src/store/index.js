import { configureStore } from '@reduxjs/toolkit';
import authSlice from './authSlice';
import excelSlice from './excelSlice';
import simulationSetupSlice from './simulationSetupSlice';
import simulationSlice from './simulationSlice';
import resultsSlice from './resultsSlice';

export const store = configureStore({
  reducer: {
    auth: authSlice,
    excel: excelSlice,
    simulationSetup: simulationSetupSlice,
    simulation: simulationSlice,
    results: resultsSlice,
  },
  middleware: (getDefaultMiddleware) =>
    getDefaultMiddleware({
      serializableCheck: {
        ignoredActions: ['persist/PERSIST'],
      },
    }),
});

// Expose store to window for debugging in development and PDF export
if (import.meta.env.DEV || window.location.search.includes('pdf_export')) {
  window.store = store;
}

// Always expose for PDF export functionality
window.__REDUX_STORE__ = store;

export default store; 