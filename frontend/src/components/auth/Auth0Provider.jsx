import React, { useEffect } from 'react';
import { Auth0Provider, useAuth0 } from '@auth0/auth0-react';
import { useDispatch } from 'react-redux';
import { logoutUser, setLoading, loginSuccess } from '../../store/authSlice';

// Auth0 configuration - using environment variables for security
const auth0Config = {
  domain: import.meta.env.VITE_AUTH0_DOMAIN,
  clientId: import.meta.env.VITE_AUTH0_CLIENT_ID,
  authorizationParams: {
    redirect_uri: `${window.location.origin}/callback`,
    audience: import.meta.env.VITE_AUTH0_AUDIENCE,
    scope: 'openid profile email offline_access'
  },
  cacheLocation: 'localstorage',
  useRefreshTokens: true,
  // Add timeout configurations to prevent long delays
  httpTimeoutInSeconds: 10, // 10 second HTTP timeout
  advancedOptions: {
    defaultScope: 'openid profile email offline_access'
  }
};

// Component to handle Auth0 state synchronization with Redux
function Auth0StateSync() {
  const { isAuthenticated, isLoading, user, getAccessTokenSilently, logout: auth0Logout } = useAuth0();
  const dispatch = useDispatch();
  
  // Make Auth0 client globally accessible for token refresh
  useEffect(() => {
    if (isAuthenticated) {
      window.auth0Client = { getAccessTokenSilently };
    } else {
      delete window.auth0Client;
    }
  }, [isAuthenticated, getAccessTokenSilently]);

  useEffect(() => {
    const syncAuthState = async () => {
      dispatch(setLoading(isLoading));

      if (isAuthenticated && user) {
        try {
          // Get the access token
          const token = await getAccessTokenSilently();
          
          // Store token in localStorage for API calls
          localStorage.setItem('authToken', token);
          
          // Fetch actual user profile from backend (includes admin status)
          try {
            // Use the correct API base URL to avoid CORS issues during Auth0 redirect
            const apiBaseUrl = import.meta.env.VITE_API_URL || 'http://localhost:8000/api';
            const profileUrl = `${apiBaseUrl}/auth0/profile`;
            
            const response = await fetch(profileUrl, {
              headers: {
                'Authorization': `Bearer ${token}`
              }
            });
            
            if (response.ok) {
              const backendUser = await response.json();
              
              // Use backend user data (includes proper admin status)
              const userInfo = {
                id: backendUser.id,
                username: backendUser.username,
                email: backendUser.email,
                full_name: backendUser.full_name,
                is_admin: backendUser.is_admin, // This comes from the database
                auth0_user_id: backendUser.auth0_user_id
              };
              
              // Check if user selected a plan during registration
              const selectedPlan = localStorage.getItem('selectedPlan');
              if (selectedPlan && selectedPlan !== 'free') {
                console.log('🎯 User selected plan during registration:', selectedPlan);
                
                try {
                  // Create Stripe checkout session for the selected plan
                  const checkoutResponse = await fetch(`${apiBaseUrl}/billing/create-checkout-session`, {
                    method: 'POST',
                    headers: {
                      'Content-Type': 'application/json',
                      'Authorization': `Bearer ${token}`
                    },
                    body: JSON.stringify({
                      plan: selectedPlan,
                      success_url: `${window.location.origin}/dashboard?checkout=success`,
                      cancel_url: `${window.location.origin}/dashboard?checkout=cancelled`
                    })
                  });
                  
                  if (checkoutResponse.ok) {
                    const { checkout_url } = await checkoutResponse.json();
                    console.log('🎯 Redirecting to Stripe checkout for plan:', selectedPlan);
                    
                    // Clear the selected plan from localStorage
                    localStorage.removeItem('selectedPlan');
                    
                    // Redirect to Stripe checkout
                    window.location.href = checkout_url;
                    return; // Don't continue with normal login flow
                  } else {
                    console.warn('⚠️ Failed to create Stripe checkout session');
                  }
                } catch (error) {
                  console.warn('⚠️ Error creating Stripe checkout:', error);
                }
                
                // Clear the selected plan even if checkout failed
                localStorage.removeItem('selectedPlan');
              }
              
              // Update Redux store with backend user data
              dispatch(loginSuccess({ user: userInfo, token }));
              
              console.log('🔐 Auth0 user authenticated with backend profile:', userInfo);
            } else {
              throw new Error('Failed to fetch backend user profile');
            }
          } catch (profileError) {
            console.error('Error fetching backend user profile:', profileError);
            
            // Fallback to Auth0 user data if backend profile fails
            const userInfo = {
              id: user.sub,
              username: user.nickname || user.name || user.email,
              email: user.email,
              full_name: user.name,
              is_admin: false, // Default to false if backend profile unavailable
              picture: user.picture
            };
            
            dispatch(loginSuccess({ user: userInfo, token }));
            console.log('🔐 Auth0 user authenticated with fallback profile:', userInfo);
          }
          
        } catch (error) {
          console.error('Error getting access token:', error);
          dispatch(logoutUser());
        }
      } else if (!isAuthenticated && !isLoading) {
        // User is not authenticated
        localStorage.removeItem('authToken');
        dispatch(logoutUser());
      }
    };

    syncAuthState();
  }, [isAuthenticated, isLoading, user, getAccessTokenSilently, dispatch]);

  // Handle logout
  useEffect(() => {
    const handleLogout = () => {
      localStorage.removeItem('authToken');
      auth0Logout({
        logoutParams: {
          returnTo: window.location.origin
        }
      });
    };

    // Listen for logout events
    window.addEventListener('auth0-logout', handleLogout);
    
    return () => {
      window.removeEventListener('auth0-logout', handleLogout);
    };
  }, [auth0Logout]);

  return null; // This component doesn't render anything
}

// Main Auth0 Provider wrapper
export default function Auth0ProviderWrapper({ children }) {
  return (
    <Auth0Provider {...auth0Config}>
      <Auth0StateSync />
      {children}
    </Auth0Provider>
  );
} 