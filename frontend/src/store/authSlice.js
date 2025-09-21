import { createSlice, createAsyncThunk } from '@reduxjs/toolkit';
import * as authService from '../services/authService';

const TOKEN_KEY = 'authToken';

// Demo credentials for automatic login - loaded from environment
const DEMO_CREDENTIALS = {
  username: import.meta.env.VITE_DEMO_USERNAME || '',
  password: import.meta.env.VITE_DEMO_PASSWORD || ''
};

// Async thunk for user login
export const loginUser = createAsyncThunk(
  'auth/loginUser',
  async ({ username, password }, { rejectWithValue, dispatch }) => {
    try {
      const data = await authService.login(username, password);
      // data includes { access_token, token_type }
      
      // After successful login, fetch user details
      const user = await authService.getCurrentUser();
      return { token: data.access_token, user }; 
    } catch (error) {
      return rejectWithValue(error.response?.data || { message: 'Login failed' });
    }
  }
);

// Async thunk for demo auto-login
export const autoLoginDemo = createAsyncThunk(
  'auth/autoLoginDemo',
  async (_, { rejectWithValue, dispatch }) => {
    try {
      console.log('🔄 Attempting demo auto-login...');
      const data = await authService.login(DEMO_CREDENTIALS.username, DEMO_CREDENTIALS.password);
      
      // After successful login, fetch user details
      const user = await authService.getCurrentUser();
      console.log('✅ Demo auto-login successful');
      return { token: data.access_token, user }; 
    } catch (error) {
      console.error('❌ Demo auto-login failed:', error.message);
      return rejectWithValue(error.response?.data || { message: 'Auto-login failed' });
    }
  }
);

// Async thunk for user registration
export const registerUser = createAsyncThunk(
  'auth/registerUser',
  async ({ username, email, password }, { rejectWithValue }) => {
    try {
      await authService.register(username, email, password);
      return { message: 'Registration successful' };
    } catch (error) {
      return rejectWithValue(error.response?.data || { message: 'Registration failed' });
    }
  }
);

// Async thunk to fetch current user details if a token exists
export const fetchCurrentUser = createAsyncThunk(
  'auth/fetchCurrentUser',
  async (_, { rejectWithValue, dispatch }) => {
    const token = authService.getToken();
    
    if (!token) {
      dispatch(logoutUser());
      return rejectWithValue('No token found');
    }

    try {
      const user = await authService.getCurrentUser();
      if (!user) {
        dispatch(logoutUser());
        return rejectWithValue('Authentication failed');
      }
      return { user, token };
    } catch (error) {
      // Don't clear auth data here, let the interceptor handle it
      dispatch(logoutUser());
      return rejectWithValue(error.message);
    }
  }
);

const initialState = {
  user: null,
  token: authService.getToken() || null,
  isAuthenticated: !!authService.getToken(),
  isLoading: false,
  error: null,
};

const authSlice = createSlice({
  name: 'auth',
  initialState,
  reducers: {
    logoutUser: (state) => {
      authService.logout();
      state.user = null;
      state.token = null;
      state.isAuthenticated = false;
      state.error = null;
      state.isLoading = false;
    },
    clearAuthError: (state) => {
      state.error = null;
    },
    // Handle external logout events (from API interceptors)
    handleExternalLogout: (state) => {
      console.log('🔐 External logout triggered - clearing auth state');
      state.user = null;
      state.token = null;
      state.isAuthenticated = false;
      state.error = 'Session expired';
      state.isLoading = false;
    },
    // Auth0 specific actions
    setLoading: (state, action) => {
      state.isLoading = action.payload;
    },
    loginSuccess: (state, action) => {
      state.isLoading = false;
      state.isAuthenticated = true;
      state.token = action.payload.token;
      state.user = action.payload.user;
      state.error = null;
    }
  },
  extraReducers: (builder) => {
    builder
      // Login
      .addCase(loginUser.pending, (state) => {
        state.isLoading = true;
        state.error = null;
      })
      .addCase(loginUser.fulfilled, (state, action) => {
        state.isLoading = false;
        state.isAuthenticated = true;
        state.token = action.payload.token;
        state.user = action.payload.user;
        state.error = null;
      })
      .addCase(loginUser.rejected, (state, action) => {
        state.isLoading = false;
        state.error = action.payload;
        state.isAuthenticated = false;
        state.user = null;
        state.token = null;
      })
      // Auto-login Demo
      .addCase(autoLoginDemo.pending, (state) => {
        state.isLoading = true;
        state.error = null;
      })
      .addCase(autoLoginDemo.fulfilled, (state, action) => {
        state.isLoading = false;
        state.isAuthenticated = true;
        state.token = action.payload.token;
        state.user = action.payload.user;
        state.error = null;
      })
      .addCase(autoLoginDemo.rejected, (state, action) => {
        state.isLoading = false;
        state.error = action.payload;
        state.isAuthenticated = false;
        state.user = null;
        state.token = null;
      })
      // Register
      .addCase(registerUser.pending, (state) => {
        state.isLoading = true;
        state.error = null;
      })
      .addCase(registerUser.fulfilled, (state) => {
        state.isLoading = false;
        state.error = null;
      })
      .addCase(registerUser.rejected, (state, action) => {
        state.isLoading = false;
        state.error = action.payload;
      })
      // Fetch Current User
      .addCase(fetchCurrentUser.pending, (state) => {
        state.isLoading = true;
        state.error = null;
      })
      .addCase(fetchCurrentUser.fulfilled, (state, action) => {
        state.isLoading = false;
        state.error = null;
        state.token = action.payload.token;
        state.user = action.payload.user;
        state.isAuthenticated = true;
      })
      .addCase(fetchCurrentUser.rejected, (state, action) => {
        state.isLoading = false;
        state.error = action.payload || 'Failed to fetch user';
        state.token = null;
        state.user = null;
        state.isAuthenticated = false;
      });
  },
});

export const { logoutUser, clearAuthError, handleExternalLogout, setLoading, loginSuccess } = authSlice.actions;

// Setup global event listener for external logout events
let listenerSetup = false;
export const setupAuthEventListeners = (dispatch) => {
  if (listenerSetup) return;
  
  window.addEventListener('auth:logout', (event) => {
    console.log('🎧 Auth logout event received:', event.detail);
    dispatch(handleExternalLogout());
  });
  
  listenerSetup = true;
};

export default authSlice.reducer; 