import { createSlice } from '@reduxjs/toolkit';

const initialState = {
  data: null,
  loading: false,
  error: null,
};

const resultsSlice = createSlice({
  name: 'results',
  initialState,
  reducers: {
    setResults: (state, action) => {
      state.data = action.payload;
      state.error = null;
    },
    setLoading: (state, action) => {
      state.loading = action.payload;
    },
    setError: (state, action) => {
      state.error = action.payload;
      state.loading = false;
    },
    clearResults: (state) => {
      state.data = null;
      state.error = null;
      state.loading = false;
    },
  },
});

export const { setResults, setLoading, setError, clearResults } = resultsSlice.actions;
export default resultsSlice.reducer; 