import React, { Suspense, useEffect, lazy, useState } from 'react';
import { 
  BrowserRouter as Router, 
  Routes, 
  Route, 
  Navigate,
  useLocation
} from 'react-router-dom';
import { useAuth0 } from '@auth0/auth0-react';
import { Provider, useDispatch, useSelector } from 'react-redux';
import { store } from './store';
import MainLayout from './components/layout/MainLayout';
import ProtectedRoute from './components/auth/ProtectedRoute';
import AdminRoute from './components/auth/AdminRoute';
import Auth0ProviderWrapper from './components/auth/Auth0Provider';
import { fetchCurrentUser, setupAuthEventListeners } from './store/authSlice';
import { neumorphicStyles } from './components/common/NeumorphicStyles';
import { clearSimulation } from './store/simulationSlice';
import CookieBanner from './components/common/CookieBanner';
import ErrorBoundary from './components/common/ErrorBoundary';
import { initializeConsoleProtection } from './utils/consoleProtection';
import { securityConfig } from './utils/securityConfig';
import CacheManager from './components/cache/CacheManager';
import { ToastContainer } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';

// Placeholder for a loading component
const Loading = () => <div style={{ padding: '20px', textAlign: 'center' }}>Loading...</div>;

// Lazy loaded components
const LandingPage = lazy(() => import('./pages/LandingPage'));
const HomePage = lazy(() => import('./pages/HomePage'));
const DashboardPage = lazy(() => import('./pages/DashboardPage'));
const UserDashboardPage = lazy(() => import('./pages/UserDashboardPage'));
const UploadPage = lazy(() => import('./pages/UploadPage'));
const SimulatePage = lazy(() => import('./pages/SimulatePage'));
const ResultsPage = lazy(() => import('./pages/ResultsPage'));
const ConfigurePage = lazy(() => import('./pages/ConfigurePage'));
const PrivateLaunchPage = lazy(() => import('./pages/PrivateLaunchPage'));
const GetStartedPage = lazy(() => import('./pages/GetStartedPage'));
const RegisterPage = lazy(() => import('./pages/RegisterPage'));
const LoginPage = lazy(() => import('./pages/LoginPage'));
const CallbackPage = lazy(() => import('./pages/CallbackPage'));
const AdminUsersPage = lazy(() => import('./pages/AdminUsersPage'));
const AdminActiveSimulationsPage = lazy(() => import('./pages/AdminActiveSimulationsPage'));
const AdminLogsPage = lazy(() => import('./pages/AdminLogsPage'));
const AdminMonitoringPage = lazy(() => import('./pages/AdminMonitoringPage'));
const AdminSupportPage = lazy(() => import('./pages/AdminSupportPage'));
const InvoicingPage = lazy(() => import('./pages/InvoicingPage'));
const APIDocumentationPage = lazy(() => import('./pages/APIDocumentationPage'));
const UserAccountPage = lazy(() => import('./pages/UserAccountPage'));
const WebhooksPage = lazy(() => import('./pages/WebhooksPage'));
const WebhookManagementPage = lazy(() => import('./pages/WebhookManagementPage'));
const CookiePolicyPage = lazy(() => import('./pages/CookiePolicyPage'));
const PrivacyPolicyPage = lazy(() => import('./pages/PrivacyPolicyPage'));
const TermsPage = lazy(() => import('./pages/TermsPage'));
const OpenSourceLicensesPage = lazy(() => import('./pages/OpenSourceLicensesPage'));
const AcceptableUsePage = lazy(() => import('./pages/AcceptableUsePage'));
const FeaturesPage = lazy(() => import('./pages/FeaturesPage'));
const PricingPage = lazy(() => import('./pages/PricingPage'));
const AboutPage = lazy(() => import('./pages/AboutPage'));
const ContactPage = lazy(() => import('./pages/ContactPage'));
const SimulationReportPage = lazy(() => import('./pages/SimulationReportPage'));
const PrintView = lazy(() => import('./components/simulation/PrintView'));
const APITestPage = lazy(() => import('./pages/APITestPage'));

// ENHANCED: Component to handle route changes and clear simulations
function RouteChangeHandler() {
  const location = useLocation();
  const dispatch = useDispatch();
  
  useEffect(() => {
    // DON'T auto-clear simulations on navigation - preserve running simulations
    // Users can manually clear results if needed using the "Clear & Retry" button
    const simulationPages = ['/simulate', '/results'];
    const currentPath = location.pathname;
    
    // Log navigation but don't clear running simulations
    if (!simulationPages.some(page => currentPath.startsWith(page))) {
      console.log('[RouteChangeHandler] 🧭 Navigating away from simulation pages - preserving running simulations');
      // dispatch(clearSimulation()); // REMOVED - don't auto-clear
    }
  }, [location.pathname, dispatch]);
  
  return null; // This component doesn't render anything
}

function AppContent() {
  const dispatch = useDispatch();
  const { isAuthenticated: auth0IsAuthenticated, isLoading: auth0Loading } = useAuth0();
  const reduxIsAuthenticated = useSelector(state => state.auth.isAuthenticated);
  const reduxAuthLoading = useSelector(state => state.auth.isLoading);
  
  // State to track redirect attempts and prevent loops
  const [redirectAttempts, setRedirectAttempts] = useState(0);
  const [lastRedirectTime, setLastRedirectTime] = useState(0);
  
  // Use Auth0 as the primary source of truth, with Redux as fallback
  const isAuthenticated = auth0IsAuthenticated || reduxIsAuthenticated;
  const isLoading = auth0Loading || reduxAuthLoading;

  useEffect(() => {
    // Initialize security protection
    if (securityConfig.enableConsoleProtection) {
      initializeConsoleProtection();
      console.log('🛡️ Security protection initialized');
    }
    
    // Setup authentication event listeners
    setupAuthEventListeners(dispatch);
    
    // Try to load current user on app start
    const token = localStorage.getItem('authToken');
    if (token && !auth0IsAuthenticated) {
      dispatch(fetchCurrentUser());
    }
  }, [dispatch, auth0IsAuthenticated]);

  // Add redirect loop protection
  useEffect(() => {
    const currentTime = Date.now();
    const currentPath = window.location.pathname;
    
    if (isAuthenticated && currentPath === '/') {
      // Check if we recently redirected
      if (currentTime - lastRedirectTime < 2000) {
        console.warn('🔄 Redirect loop protection: too many recent redirects');
        setRedirectAttempts(prev => prev + 1);
        return;
      }
      
      if (redirectAttempts >= 3) {
        console.warn('🔄 Redirect loop protection: maximum attempts reached');
        return;
      }
      
      setLastRedirectTime(currentTime);
    }
    
    // Reset attempts after 10 seconds of no redirects
    if (currentTime - lastRedirectTime > 10000) {
      setRedirectAttempts(0);
    }
  }, [isAuthenticated, redirectAttempts, lastRedirectTime]);

  console.log('🔥 APP.JSX RENDERING - Auth state:', {
    auth0IsAuthenticated,
    reduxIsAuthenticated,
    isAuthenticated,
    isLoading,
    redirectAttempts,
    pathname: window.location.pathname
  });

  // Show loading while checking authentication (skip for print-view)
  const isPrintView = window.location.pathname === '/print-view';
  if (isLoading && !isPrintView) {
    return <Loading />;
  }

  return (
    <div style={{ background: neumorphicStyles.colors.background, minHeight: '100vh' }}>
      <CookieBanner />
      <RouteChangeHandler />
      <Routes>
        {/* Public routes */}
        <Route path="/" element={
          isAuthenticated && redirectAttempts < 3 ? (
            <Navigate to="/my-dashboard" replace />
          ) : (
            <Suspense fallback={<Loading />}><LandingPage /></Suspense>
          )
        } />
        <Route path="/login" element={<Suspense fallback={<Loading />}><LoginPage /></Suspense>} />
        <Route path="/register" element={<Suspense fallback={<Loading />}><RegisterPage /></Suspense>} />
        <Route path="/get-started" element={<Suspense fallback={<Loading />}><GetStartedPage /></Suspense>} />
        <Route path="/private-launch" element={<Suspense fallback={<Loading />}><PrivateLaunchPage /></Suspense>} />
        <Route path="/callback" element={<Suspense fallback={<Loading />}><CallbackPage /></Suspense>} />
        <Route path="/privacy" element={<Suspense fallback={<Loading />}><PrivacyPolicyPage /></Suspense>} />
        <Route path="/cookie-policy" element={<Suspense fallback={<Loading />}><CookiePolicyPage /></Suspense>} />
        <Route path="/terms" element={<Suspense fallback={<Loading />}><TermsPage /></Suspense>} />
        <Route path="/open-source-licenses" element={<Suspense fallback={<Loading />}><OpenSourceLicensesPage /></Suspense>} />
        <Route path="/acceptable-use" element={<Suspense fallback={<Loading />}><AcceptableUsePage /></Suspense>} />
        <Route path="/features" element={<Suspense fallback={<Loading />}><FeaturesPage /></Suspense>} />
        <Route path="/pricing" element={<Suspense fallback={<Loading />}><PricingPage /></Suspense>} />
        <Route path="/about" element={<Suspense fallback={<Loading />}><AboutPage /></Suspense>} />
        <Route path="/contact" element={<Suspense fallback={<Loading />}><ContactPage /></Suspense>} />
        
        {/* Print view route - accessible without auth for PDF export */}
        <Route path="/print-view" element={
          <Suspense fallback={<Loading />}>
            <PrintView />
          </Suspense>
        } />
        
        {/* Protected routes */}
        <Route 
          path="/*" 
          element={
            <ProtectedRoute>
              <MainLayout />
            </ProtectedRoute>
          }
        >
          <Route path="dashboard" element={<Suspense fallback={<Loading />}><DashboardPage /></Suspense>} />
          <Route path="my-dashboard" element={<Suspense fallback={<Loading />}><UserDashboardPage /></Suspense>} />
          <Route path="upload" element={<Suspense fallback={<Loading />}><UploadPage /></Suspense>} />
          <Route path="simulate" element={<Suspense fallback={<Loading />}><SimulatePage /></Suspense>} />
          <Route path="results" element={<Suspense fallback={<Loading />}><ResultsPage /></Suspense>} />
          <Route path="configure" element={<Suspense fallback={<Loading />}><ConfigurePage /></Suspense>} />
          <Route path="simulation-report/:simulationId" element={<Suspense fallback={<Loading />}><SimulationReportPage /></Suspense>} />
          <Route path="account" element={<Suspense fallback={<Loading />}><UserAccountPage /></Suspense>} />
          <Route path="api-docs" element={<Suspense fallback={<Loading />}><APIDocumentationPage /></Suspense>} />
          <Route path="api-test" element={<AdminRoute><Suspense fallback={<Loading />}><APITestPage /></Suspense></AdminRoute>} />
          <Route path="webhooks" element={<Suspense fallback={<Loading />}><WebhookManagementPage /></Suspense>} />
          <Route path="admin/users" element={<AdminRoute><Suspense fallback={<Loading />}><AdminUsersPage /></Suspense></AdminRoute>} />
          <Route path="admin/simulations" element={<AdminRoute><Suspense fallback={<Loading />}><AdminActiveSimulationsPage /></Suspense></AdminRoute>} />
          <Route path="admin/logs" element={<AdminRoute><Suspense fallback={<Loading />}><AdminLogsPage /></Suspense></AdminRoute>} />
          <Route path="admin/monitoring" element={<AdminRoute><Suspense fallback={<Loading />}><AdminMonitoringPage /></Suspense></AdminRoute>} />
          <Route path="admin/support" element={<AdminRoute><Suspense fallback={<Loading />}><AdminSupportPage /></Suspense></AdminRoute>} />
          <Route path="admin/invoicing" element={<AdminRoute><Suspense fallback={<Loading />}><InvoicingPage /></Suspense></AdminRoute>} />
        </Route>
      </Routes>
    </div>
  );
}

// Component to conditionally render Auth0Provider based on route
function ConditionalAuth0Wrapper({ children }) {
  const location = window.location;
  const isPrintView = location.pathname === '/print-view';
  
  // Skip Auth0Provider for print-view route to prevent headless browser issues
  if (isPrintView) {
    console.log('🖨️ [PRINT_VIEW] Skipping Auth0Provider for headless browser compatibility');
    return children;
  }
  
  return <Auth0ProviderWrapper>{children}</Auth0ProviderWrapper>;
}

function App() {
  return (
    <Provider store={store}>
      <Router>
        <ConditionalAuth0Wrapper>
          <ErrorBoundary fallback={<div style={{ padding: '2rem', textAlign: 'center', color: '#dc2626' }}>Something went wrong. Please reload the page.</div>}>
            <AppContent />
            <CacheManager />
          </ErrorBoundary>
        </ConditionalAuth0Wrapper>
      </Router>
      
      {/* Toast notifications for PDF export and other features */}
      <ToastContainer
        position="top-right"
        autoClose={5000}
        hideProgressBar={false}
        newestOnTop={false}
        closeOnClick
        rtl={false}
        pauseOnFocusLoss
        draggable
        pauseOnHover
        theme="light"
      />
    </Provider>
  );
}

export default App;