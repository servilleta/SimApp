import React, { useState, useEffect, useRef } from 'react';
import { NavLink, useNavigate } from 'react-router-dom';
import { useSelector, useDispatch } from 'react-redux';
import { logoutUser } from '../../store/authSlice';
import { resetExcelState, setFileInfo } from '../../store/excelSlice';
import { resetSetup, setSimulationSetup, clearRestoringFlag } from '../../store/simulationSetupSlice';
import { clearSimulation } from '../../store/simulationSlice';
import logoFull from '../../assets/images/noBgColor.png';
import logoSymbol from '../../assets/images/symbol_color.svg';
import apiClient from '../../services/api';
import './Sidebar.css';

console.log('🚀 Sidebar.jsx MODULE LOADED!');
console.log('🔥🔥🔥 [CACHE DEBUG] THIS IS A NEW VERSION OF THE FILE! Timestamp:', new Date().toISOString());

const Sidebar = ({ collapsed, setCollapsed }) => {
  const dispatch = useDispatch();
  const navigate = useNavigate();
  const { isAuthenticated, user } = useSelector((state) => state.auth);
  
  // Debug logging for admin status and refresh user data
  useEffect(() => {
    if (user) {
      console.log('🔍 [SIDEBAR_DEBUG] Current user object:', user);
      console.log('🔍 [SIDEBAR_DEBUG] User is_admin status:', user.is_admin);
      console.log('🔍 [SIDEBAR_DEBUG] User username:', user.username);
      console.log('🔍 [SIDEBAR_DEBUG] User email:', user.email);
      console.log('🔍 [SIDEBAR_DEBUG] User keys:', Object.keys(user));
      
      // Force refresh user data if this appears to be Matias but without admin status
      if ((user.username === 'matias redard' || user.email === 'mredard@gmail.com') && !user.is_admin) {
        console.log('🔄 [ADMIN_FIX] Matias detected without admin status - refreshing user data...');
        dispatch({ type: 'auth/fetchCurrentUser' });
      }
    } else {
      console.log('🔍 [SIDEBAR_DEBUG] No user object found');
    }
  }, [user, dispatch]);
  const [recentSimulations, setRecentSimulations] = useState([]);
  const [loadingRecent, setLoadingRecent] = useState(false);
  const [loadingSimulationId, setLoadingSimulationId] = useState(null);
  const [showUserDropdown, setShowUserDropdown] = useState(false);
  const dropdownRef = useRef(null);

  // Close dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (event) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target)) {
        setShowUserDropdown(false);
      }
    };

    if (showUserDropdown) {
      document.addEventListener('mousedown', handleClickOutside);
    }

    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, [showUserDropdown]);

  const handleLogout = () => {
    // Clear Redux state
    dispatch(logoutUser());
    // Clear localStorage
    localStorage.removeItem('authToken');
    // Trigger Auth0 logout
    window.dispatchEvent(new CustomEvent('auth0-logout'));
    setShowUserDropdown(false);
  };

  const handleUserClick = () => {
    setShowUserDropdown(!showUserDropdown);
  };

  const handleAccountClick = () => {
    navigate('/account');
    setShowUserDropdown(false);
  };

  const handleUploadClick = () => {
    // Clear any existing state to ensure fresh upload screen
    dispatch(resetExcelState());
    dispatch(resetSetup());
    dispatch(clearSimulation());
    
    navigate('/simulate');
  };

  const fetchRecentSimulations = async () => {
    if (!isAuthenticated) return;
    
    setLoadingRecent(true);
    try {
      console.log('🔍 [DEBUG] Making API call to /simulation/history?limit=5');
      const response = await apiClient.get('/simulation/history?limit=5');
      const simulationHistory = response.data || [];
      
      console.log('🔍 [DEBUG] API Response received:', response);
      console.log('🔍 [DEBUG] Simulation history data:', simulationHistory);
      console.log('🔍 [DEBUG] Array length:', simulationHistory.length);
      
      // Only use simulation data from the database - no localStorage mixing
      const recentSimulations = simulationHistory.map(sim => ({ 
        ...sim, 
        type: 'simulation' 
      }));
      
      console.log('🔍 [DEBUG] Processed recent simulations:', recentSimulations);
      setRecentSimulations(recentSimulations);
      
    } catch (error) {
      console.error('Failed to fetch recent simulations:', error);
      setRecentSimulations([]);
    } finally {
      setLoadingRecent(false);
    }
  };

  const handleSimulationClick = async (simulation) => {
    console.log('🔍 [CLICK_DEBUG] handleSimulationClick called with:', simulation);
    console.log('🔍 [CLICK_DEBUG] simulation.simulation_id:', simulation.simulation_id);
    
    // Only handle actual simulations from the database
    if (simulation.simulation_id) {
      // Set loading state for this specific simulation
      setLoadingSimulationId(simulation.simulation_id);
      
      // Clear only simulation results to prevent stale data (preserve setup variables)
      dispatch({ 
        type: 'simulation/clearSimulation'
      });
      console.log('🧹 Cleared previous simulation results to prevent stale data');
      
      // Fallback timeout to clear loading state (safety net)
      const timeoutId = setTimeout(() => {
        console.log('⚠️ Loading timeout reached, clearing loading state');
        setLoadingSimulationId(null);
      }, 10000); // 10 second timeout
      
      try {
        // For actual simulations, load full simulation data and restore complete state
        console.log('🔄 Loading full simulation state for:', simulation.simulation_id);
        
        // Fetch complete simulation data from backend
        const response = await apiClient.get(`/simulations/${simulation.simulation_id}?v=999&t=${Date.now()}`);
        const simulationData = response.data;
        console.log('📥 Loaded simulation data:', simulationData);
        console.log('🔍 [DEBUG] file_id in response:', simulationData.file_id);
        console.log('🔍 [DEBUG] All keys in response:', Object.keys(simulationData));
        console.log('🔍 [DEBUG] variables_config:', simulationData.variables_config);
        console.log('🔍 [DEBUG] multi_target_result:', simulationData.multi_target_result);
        
        // Check if we have the needed configuration
        if (!simulationData.file_id) {
          console.warn('⚠️ No file_id in simulation data, falling back to results view');
          navigate(`/simulation-report/${simulation.simulation_id}`);
          return;
        }
        
        // Fetch Excel file data
        const excelResponse = await apiClient.get(`/excel-parser/files/${simulationData.file_id}`);
        const excelData = excelResponse.data;
        console.log('📊 Loaded Excel data:', excelData);
        
        // Prepare file info object
        const fileInfoToRestore = {
          file_id: simulationData.file_id,
          filename: simulationData.original_filename || 'Simulation File',
          file_size: excelData.file_size || 0,
          sheet_names: excelData.sheet_names || (excelData.sheets || []).map(s => s.sheet_name),
          sheets: excelData.sheets || [],
          upload_timestamp: new Date().toISOString()
        };
        
        // Restore Excel file info to Redux store
        dispatch(setFileInfo(fileInfoToRestore));
        
        // Convert backend variables_config to frontend format
        const inputVariables = (simulationData.variables_config || []).map(variable => ({
          name: variable.name || variable.cell,
          sheetName: variable.sheet_name || 'Sheet1',
          min_value: variable.min_value,
          most_likely: variable.most_likely || variable.likely,
          max_value: variable.max_value,
          distribution: variable.distribution || 'uniform'
        }));
        
        // Extract result cells from target_cell or multi_target results
        const resultCells = [];
        if (simulationData.target_cell) {
          resultCells.push({
            name: simulationData.target_cell,
            sheetName: excelData.sheets?.[0]?.sheet_name || 'Sheet1'
          });
        }
        
        // Also extract from multi_target_result if available
        if (simulationData.multi_target_result && simulationData.multi_target_result.targets) {
          simulationData.multi_target_result.targets.forEach(target => {
            if (!resultCells.some(cell => cell.name === target)) {
              resultCells.push({
                name: target,
                sheetName: excelData.sheets?.[0]?.sheet_name || 'Sheet1'
              });
            }
          });
        }
        
        // Prepare simulation config
        const simulationConfig = {
          inputVariables: inputVariables,
          resultCells: resultCells,
          iterations: simulationData.iterations_requested || 1000,
          currentSheetName: excelData.sheets?.[0]?.sheet_name || 'Sheet1'
        };
        console.log('⚙️ Restoring simulation config:', simulationConfig);
        
        // Restore simulation configuration
        console.log('🔧 [REDUX] About to dispatch setSimulationSetup with:', simulationConfig);
        dispatch(setSimulationSetup(simulationConfig));
        console.log('🔧 [REDUX] setSimulationSetup dispatched');
        
        // Restore simulation results if available
        console.log('📊 Checking for simulation results to restore...');
        console.log('📊 multi_target_result:', simulationData.multi_target_result);
        console.log('📊 mean:', simulationData.mean, 'histogram:', !!simulationData.histogram);
        console.log('📊 target_cell:', simulationData.target_cell, 'target_name:', simulationData.target_name);
        
        if (simulationData.multi_target_result) {
          console.log('📊 Restoring multi-target simulation results');
          const { restoreSimulationResults } = await import('../../store/simulationSlice');
          
          // Convert simulation data to the format expected by restoreSimulationResults
          const resultsToRestore = {
            multipleResults: Object.keys(simulationData.multi_target_result.statistics || {}).map((targetName, index) => {
              const stats = simulationData.multi_target_result.statistics[targetName];
              return {
                simulation_id: `restored_${Date.now()}_${index}`,
                temp_id: `restored_temp_${Date.now()}_${index}`,
                status: 'completed',
                target_name: targetName,
                result_cell_coordinate: targetName,
                isRestored: true,
                results: {
                  mean: stats.mean,
                  median: stats.median,
                  std_dev: stats.std,
                  min_value: stats.min,
                  max_value: stats.max,
                  percentiles: stats.percentiles || {},
                  histogram: stats.histogram || { bins: [], values: [] },
                  iterations_run: simulationData.multi_target_result.total_iterations || simulationData.iterations_run,
                  raw_values: simulationData.multi_target_result.target_results?.[targetName] || [],
                  sensitivity_analysis: simulationData.multi_target_result.sensitivity_data?.[targetName] || []
                }
              };
            })
          };
          
          dispatch(restoreSimulationResults(resultsToRestore));
        } else if (simulationData.mean !== null && simulationData.histogram) {
          console.log('📊 Restoring single-target simulation results from individual fields');
          const { restoreSimulationResults } = await import('../../store/simulationSlice');
          
          // Create results from individual database fields
          const resultsToRestore = {
            multipleResults: [{
              simulation_id: `restored_${Date.now()}_0`,
              temp_id: `restored_temp_${Date.now()}_0`,
              status: 'completed',
              target_name: simulationData.target_name || simulationData.target_cell || 'Target',
              result_cell_coordinate: simulationData.target_cell || 'Unknown',
              isRestored: true,
              results: {
                mean: simulationData.mean,
                median: simulationData.median,
                std_dev: simulationData.std,
                min_value: simulationData.min,
                max_value: simulationData.max,
                percentiles: {},
                histogram: simulationData.histogram,
                iterations_run: simulationData.iterations_run,
                errors: [],
                sensitivity_analysis: []
              }
            }]
          };
          
          dispatch(restoreSimulationResults(resultsToRestore));
        } else {
          console.log('📊 No simulation results data found to restore');
        }
        
        // Navigate to simulate page with restored state
        console.log('🧭 Navigating to /simulate with restored state');
        
        // Navigate immediately
        navigate('/simulate');
        
        // Don't clear loading state immediately - let the page components handle it
        // Clear restoration flag after a much longer delay to ensure grid initialization doesn't clear variables
        setTimeout(() => {
          dispatch(clearRestoringFlag());
          console.log('🔧 [REDUX] Cleared restoration flag - normal initialization can now proceed');
        }, 10000); // Increased to 10 seconds to ensure ExcelViewWithConfig and grid are fully loaded
        
        // Extended loading state - clear after grid and charts are loaded
        setTimeout(() => {
          setLoadingSimulationId(null);
          console.log('🔧 [LOADING] Extended loading cleared after page components loaded');
        }, 8000); // 8 seconds to allow grid and charts to fully load
        
      } catch (error) {
        console.error('Failed to load simulation state:', error);
        // Fallback to results view if loading fails
        navigate(`/simulation-report/${simulation.simulation_id}`);
      } finally {
        // Clear timeout but don't clear loading state immediately - let extended loading handle it
        clearTimeout(timeoutId);
        console.log('✅ Navigation completed for simulation:', simulation.simulation_id);
      }
    }
  };

  // Fetch recent simulations when component mounts and user is authenticated
  useEffect(() => {
    console.log('🔍 [DEBUG] useEffect triggered - isAuthenticated:', isAuthenticated);
    if (isAuthenticated) {
      console.log('🔍 [DEBUG] User is authenticated, calling fetchRecentSimulations');
      fetchRecentSimulations();
    } else {
      console.log('🔍 [DEBUG] User not authenticated, skipping API call');
    }
  }, [isAuthenticated]);
  
  // Also poll for updates periodically as a fallback
  useEffect(() => {
    if (!isAuthenticated) return;
    
    // Poll every 30 seconds for updates
    const pollInterval = setInterval(() => {
      console.log('🔄 Sidebar: Periodic refresh of recent simulations');
      fetchRecentSimulations();
    }, 30000);
    
    return () => clearInterval(pollInterval);
  }, [isAuthenticated]);

  // Listen for simulation completion and file upload events to refresh the list
  useEffect(() => {
    const handleSimulationCompleted = (event) => {
      console.log('🔄 Sidebar: Simulation completed event received, detail:', event.detail);
      console.log('🔄 Sidebar: Refreshing recent list due to simulation completion');
      
      // Add a small delay to allow database persistence to complete before refreshing
      setTimeout(() => {
        console.log('🔄 Sidebar: Delayed refresh after simulation completion');
        fetchRecentSimulations();
      }, 2000); // 2 second delay
    };

    const handleFileUploaded = (event) => {
      console.log('🔄 Sidebar: Excel file uploaded, refreshing recent list:', event.detail?.filename);
      
      // File uploads will only show in recent list after a simulation is completed
      // No localStorage storage - everything comes from database
      fetchRecentSimulations();
    };

    // Debug: Log all custom events
    const debugEventHandler = (eventName) => (event) => {
      console.log(`🔍 [DEBUG] Custom event received: ${eventName}`, event.detail);
    };
    
    window.addEventListener('simulation-completed', handleSimulationCompleted);
    window.addEventListener('excel-file-uploaded', handleFileUploaded);
    
    // Add debug listeners
    window.addEventListener('simulation-completed', debugEventHandler('simulation-completed'));
    
    return () => {
      window.removeEventListener('simulation-completed', handleSimulationCompleted);
      window.removeEventListener('excel-file-uploaded', handleFileUploaded);
    };
  }, []);

  const sidebarStyle = {
    width: collapsed ? '80px' : '280px',
    backgroundColor: 'var(--color-white)',
    borderRight: '1px solid var(--color-border-light)',
    display: 'flex',
    flexDirection: 'column',
    height: '100vh',
    position: 'relative',
    transition: 'width 0.2s ease',
    zIndex: 100,
  };

  const headerStyle = {
    padding: collapsed ? '20px 12px' : '20px 24px',
    borderBottom: '1px solid var(--color-border-light)',
    display: 'flex',
    alignItems: 'center',
    justifyContent: collapsed ? 'center' : 'space-between',
  };

  const logoStyle = {
    display: 'flex',
    alignItems: 'center',
    textDecoration: 'none',
    transition: 'all 0.2s ease',
  };

  const logoImageStyle = {
    height: collapsed ? '40px' : '56px',
    width: 'auto',
    objectFit: 'contain',
  };

  const navSectionStyle = {
    padding: collapsed ? '16px 12px' : '16px 24px',
    flex: 1,
    overflowY: 'auto',
  };

  const navItemStyle = {
    display: 'flex',
    alignItems: 'center',
    padding: collapsed ? '12px 0' : '12px 16px',
    borderRadius: '8px',
    textDecoration: 'none',
    color: 'var(--color-text-secondary)',
    fontSize: '14px',
    fontWeight: '500',
    marginBottom: '4px',
    transition: 'all 0.15s ease',
    justifyContent: collapsed ? 'center' : 'flex-start',
  };

  const activeNavItemStyle = {
    ...navItemStyle,
    backgroundColor: 'var(--color-warm-white)',
    color: 'var(--color-charcoal)',
  };

  const iconStyle = {
    fontSize: '18px',
    marginRight: collapsed ? '0' : '12px',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    width: '20px',
    color: 'inherit',
    fontWeight: '400',
  };

  const sectionTitleStyle = {
    fontSize: '12px',
    fontWeight: '600',
    color: 'var(--color-text-tertiary)',
    textTransform: 'uppercase',
    letterSpacing: '0.05em',
    marginBottom: '12px',
    marginTop: '24px',
    display: collapsed ? 'none' : 'block',
  };

  const userSectionStyle = {
    padding: collapsed ? '16px 12px' : '16px 24px',
    borderTop: '1px solid var(--color-border-light)',
    marginTop: 'auto',
  };

  const userInfoStyle = {
    display: collapsed ? 'none' : 'flex',
    alignItems: 'center',
    padding: '12px 0',
    marginBottom: '12px',
  };

  const userAvatarStyle = {
    width: '32px',
    height: '32px',
    borderRadius: '50%',
    backgroundColor: 'var(--color-braun-orange)',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    color: 'white',
    fontSize: '14px',
    fontWeight: '600',
    marginRight: '12px',
  };

  const userNameStyle = {
    fontSize: '14px',
    fontWeight: '600',
    color: 'var(--color-charcoal)',
    marginBottom: '2px',
  };

  const userEmailStyle = {
    fontSize: '12px',
    color: 'var(--color-text-secondary)',
  };

  const collapseButtonStyle = {
    position: 'absolute',
    top: '24px',
    right: '-12px',
    width: '24px',
    height: '24px',
    borderRadius: '50%',
    backgroundColor: 'var(--color-white)',
    border: '1px solid var(--color-border-light)',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    cursor: 'pointer',
    fontSize: '12px',
    color: 'var(--color-text-secondary)',
    zIndex: 101,
    transition: 'all 0.15s ease',
    boxShadow: 'var(--shadow-xs)',
  };

  const logoutButtonStyle = {
    ...navItemStyle,
    color: 'var(--color-error)',
    backgroundColor: 'transparent',
    border: 'none',
    cursor: 'pointer',
    width: '100%',
    textAlign: 'left',
    marginTop: '8px',
  };

  const userDropdownContainerStyle = {
    position: 'relative',
    width: '100%',
  };

  const userClickableStyle = {
    ...userInfoStyle,
    cursor: 'pointer',
    borderRadius: '8px',
    transition: 'all 0.15s ease',
    margin: '0 0 12px 0',
    padding: '12px',
    ':hover': {
      backgroundColor: 'var(--color-warm-white)',
    }
  };

  const dropdownMenuStyle = {
    position: 'absolute',
    bottom: '100%',
    left: '0',
    right: '0',
    marginBottom: '8px',
    background: 'var(--color-white)',
    borderRadius: '8px',
    boxShadow: 'var(--shadow-lg)',
    border: '1px solid var(--color-border-light)',
    overflow: 'hidden',
    zIndex: 1000,
  };

  const dropdownItemStyle = {
    padding: '12px 16px',
    cursor: 'pointer',
    borderBottom: '1px solid var(--color-border-light)',
    transition: 'all 0.15s ease',
    display: 'flex',
    alignItems: 'center',
    gap: '12px',
    color: 'var(--color-text-primary)',
    fontSize: '14px',
    fontWeight: '500',
    backgroundColor: 'transparent',
    border: 'none',
    width: '100%',
    textAlign: 'left',
  };

  const dropdownItemHoverStyle = {
    backgroundColor: 'var(--color-warm-white)',
  };

  const uploadButtonStyle = {
    display: 'flex',
    alignItems: 'center',
    padding: collapsed ? '12px 0' : '12px 16px',
    cursor: 'pointer',
    fontSize: '16px',
    fontWeight: 'bold',
    color: 'var(--color-braun-orange)',
    marginBottom: '16px',
    transition: 'all 0.15s ease',
    backgroundColor: 'transparent',
    border: 'none',
    borderRadius: '8px',
    width: '100%',
    textAlign: 'left',
    justifyContent: collapsed ? 'center' : 'flex-start',
  };

  const uploadIconStyle = {
    width: '32px',
    height: '32px',
    borderRadius: '50%',
    backgroundColor: 'var(--color-braun-orange)',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    color: 'white',
    fontSize: '16px',
    fontWeight: '600',
    marginRight: collapsed ? '0' : '12px',
  };

  const recentSectionStyle = {
    padding: collapsed ? '0' : '0 24px',
    marginTop: '16px',
  };

  const recentTitleStyle = {
    fontSize: '12px',
    fontWeight: '600',
    color: 'var(--color-text-tertiary)',
    textTransform: 'uppercase',
    letterSpacing: '0.05em',
    marginBottom: '12px',
    display: collapsed ? 'none' : 'block',
  };

  const recentItemStyle = {
    display: collapsed ? 'none' : 'flex',
    flexDirection: 'column',
    padding: '8px 12px',
    borderRadius: '8px',
    cursor: 'pointer',
    marginBottom: '4px',
    transition: 'all 0.15s ease',
    backgroundColor: 'transparent',
    border: 'none',
    width: '100%',
    textAlign: 'left',
  };

  const recentItemNameStyle = {
    fontSize: '13px',
    fontWeight: '500',
    color: 'var(--color-charcoal)',
    marginBottom: '2px',
    whiteSpace: 'nowrap',
    overflow: 'hidden',
    textOverflow: 'ellipsis',
  };

  const recentItemDetailsStyle = {
    fontSize: '11px',
    color: 'var(--color-text-secondary)',
    whiteSpace: 'nowrap',
    overflow: 'hidden',
    textOverflow: 'ellipsis',
  };

  const getInitials = (name) => {
    if (!name) return 'U';
    return name.split(' ').map(n => n[0]).join('').toUpperCase();
  };

  const formatSimulationDate = (dateString) => {
    if (!dateString) return '';
    
    try {
      const date = new Date(dateString);
      const now = new Date();
      const diffMs = now - date;
      const diffMins = Math.floor(diffMs / (1000 * 60));
      const diffHours = Math.floor(diffMs / (1000 * 60 * 60));
      const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24));
      
      if (diffMins < 1) return 'Just now';
      if (diffMins < 60) return `${diffMins}m ago`;
      if (diffHours < 24) return `${diffHours}h ago`;
      if (diffDays < 7) return `${diffDays}d ago`;
      
      // For older dates, show month/day
      return date.toLocaleDateString('en-US', { 
        month: 'short', 
        day: 'numeric',
        year: date.getFullYear() !== now.getFullYear() ? 'numeric' : undefined
      });
    } catch (error) {
      return '';
    }
  };

  console.log('🔍 Sidebar: About to return JSX');

  const sidebarContent = (
    <aside style={sidebarStyle}>
      <button
        onClick={() => setCollapsed(!collapsed)}
        style={collapseButtonStyle}
        onMouseEnter={(e) => {
          e.target.style.backgroundColor = 'var(--color-warm-white)';
          e.target.style.borderColor = 'var(--color-medium-grey)';
        }}
        onMouseLeave={(e) => {
          e.target.style.backgroundColor = 'var(--color-white)';
          e.target.style.borderColor = 'var(--color-border-light)';
        }}
      >
        {collapsed ? '→' : '←'}
      </button>

      <div style={headerStyle}>
        <NavLink to="/" style={logoStyle}>
          <img 
            src={collapsed ? logoSymbol : logoFull} 
            alt="SimApp Logo" 
            style={logoImageStyle}
          />
        </NavLink>
      </div>

      <nav style={navSectionStyle}>
        {isAuthenticated && (
          <>
            {/* Upload Button */}
            <button
              onClick={handleUploadClick}
              style={uploadButtonStyle}
              onMouseEnter={(e) => {
                e.target.style.backgroundColor = 'var(--color-warm-white)';
              }}
              onMouseLeave={(e) => {
                e.target.style.backgroundColor = 'transparent';
              }}
            >
              <div style={uploadIconStyle}>+</div>
              {!collapsed && 'New Sim'}
            </button>

            {/* Recent Simulations */}
            <div style={recentSectionStyle}>
              <div style={recentTitleStyle}>Recent</div>
              {loadingRecent ? (
                <div style={{
                  ...recentItemDetailsStyle,
                  padding: '8px 12px',
                  display: collapsed ? 'none' : 'block'
                }}>
                  Loading...
                </div>
              ) : recentSimulations.length > 0 ? (
                recentSimulations.map((simulation, index) => {
                  const isLoading = loadingSimulationId === simulation.simulation_id;
                  return (
                    <button
                      key={simulation.simulation_id || index}
                      onClick={() => handleSimulationClick(simulation)}
                      style={recentItemStyle}
                      className={isLoading ? 'recent-item-loading' : ''}
                      disabled={isLoading}
                      onMouseEnter={(e) => {
                        if (!isLoading) {
                          e.target.style.backgroundColor = 'var(--color-warm-white)';
                        }
                      }}
                      onMouseLeave={(e) => {
                        if (!isLoading) {
                          e.target.style.backgroundColor = 'transparent';
                        }
                      }}
                    >
                      <div className="recent-item-content">
                        <div 
                          style={recentItemNameStyle}
                          className={isLoading ? 'recent-item-name' : ''}
                        >
                          {simulation.file_name || 'Untitled Simulation'}
                        </div>
                        <div style={recentItemDetailsStyle}>
                          {isLoading 
                            ? 'Loading simulation...'
                            : simulation.type === 'upload' 
                              ? `Recently uploaded • ${formatSimulationDate(simulation.created_at)}`
                              : `${simulation.status} • ${simulation.iterations_requested || simulation.iterations_run || 0} iterations • ${formatSimulationDate(simulation.created_at)}`
                          }
                        </div>
                      </div>
                    </button>
                  );
                })
              ) : (
                <div style={{
                  ...recentItemDetailsStyle,
                  padding: '8px 12px',
                  display: collapsed ? 'none' : 'block'
                }}>
                  No recent simulations
                </div>
              )}
            </div>


            <div style={sectionTitleStyle}>API & Integration</div>
            
            <NavLink 
              to="/api-docs" 
              style={({ isActive }) => isActive ? activeNavItemStyle : navItemStyle}
              onMouseEnter={(e) => !e.target.className.includes('active') && (e.target.style.backgroundColor = 'var(--color-warm-white)')}
              onMouseLeave={(e) => !e.target.className.includes('active') && (e.target.style.backgroundColor = 'transparent')}
            >
              <span style={iconStyle}>🔗</span>
              {!collapsed && 'API Documentation'}
            </NavLink>


            {/* API Test Environment - Admin Only */}
            {user && user.is_admin && (
              <NavLink 
                to="/api-test" 
                style={({ isActive }) => isActive ? activeNavItemStyle : navItemStyle}
                onMouseEnter={(e) => !e.target.className.includes('active') && (e.target.style.backgroundColor = 'var(--color-warm-white)')}
                onMouseLeave={(e) => !e.target.className.includes('active') && (e.target.style.backgroundColor = 'transparent')}
              >
                <span style={iconStyle}>🧪</span>
                {!collapsed && 'API Test Environment'}
              </NavLink>
            )}

            <NavLink 
              to="/webhooks" 
              style={({ isActive }) => isActive ? activeNavItemStyle : navItemStyle}
              onMouseEnter={(e) => !e.target.className.includes('active') && (e.target.style.backgroundColor = 'var(--color-warm-white)')}
              onMouseLeave={(e) => !e.target.className.includes('active') && (e.target.style.backgroundColor = 'transparent')}
            >
              <span style={iconStyle}>📡</span>
              {!collapsed && 'Webhooks'}
            </NavLink>

            {(user?.is_admin || (user && (user.username === 'matias redard' || user.email === 'mredard@gmail.com'))) && (
              <>
                <div style={sectionTitleStyle}>Admin</div>
                
                <NavLink 
                  to="/admin/users" 
                  style={({ isActive }) => isActive ? activeNavItemStyle : navItemStyle}
                  onMouseEnter={(e) => !e.target.className.includes('active') && (e.target.style.backgroundColor = 'var(--color-warm-white)')}
                  onMouseLeave={(e) => !e.target.className.includes('active') && (e.target.style.backgroundColor = 'transparent')}
                >
                  <span style={iconStyle}>◉</span>
                  {!collapsed && 'Users'}
                </NavLink>

                <NavLink 
                  to="/admin/simulations" 
                  style={({ isActive }) => isActive ? activeNavItemStyle : navItemStyle}
                  onMouseEnter={(e) => !e.target.className.includes('active') && (e.target.style.backgroundColor = 'var(--color-warm-white)')}
                  onMouseLeave={(e) => !e.target.className.includes('active') && (e.target.style.backgroundColor = 'transparent')}
                >
                  <span style={iconStyle}>⚡</span>
                  {!collapsed && 'Active Simulations'}
                </NavLink>

                <NavLink 
                  to="/admin/logs" 
                  style={({ isActive }) => isActive ? activeNavItemStyle : navItemStyle}
                  onMouseEnter={(e) => !e.target.className.includes('active') && (e.target.style.backgroundColor = 'var(--color-warm-white)')}
                  onMouseLeave={(e) => !e.target.className.includes('active') && (e.target.style.backgroundColor = 'transparent')}
                >
                  <span style={iconStyle}>≡</span>
                  {!collapsed && 'Logs'}
                </NavLink>

                <NavLink 
                  to="/admin/monitoring" 
                  style={({ isActive }) => isActive ? activeNavItemStyle : navItemStyle}
                  onMouseEnter={(e) => !e.target.className.includes('active') && (e.target.style.backgroundColor = 'var(--color-warm-white)')}
                  onMouseLeave={(e) => !e.target.className.includes('active') && (e.target.style.backgroundColor = 'transparent')}
                >
                  <span style={iconStyle}>🔍</span>
                  {!collapsed && 'Monitoring'}
                </NavLink>


                <NavLink 
                  to="/admin/support" 
                  style={({ isActive }) => isActive ? activeNavItemStyle : navItemStyle}
                  onMouseEnter={(e) => !e.target.className.includes('active') && (e.target.style.backgroundColor = 'var(--color-warm-white)')}
                  onMouseLeave={(e) => !e.target.className.includes('active') && (e.target.style.backgroundColor = 'transparent')}
                >
                  <span style={iconStyle}>🎯</span>
                  {!collapsed && 'Support'}
                </NavLink>

                <NavLink 
                  to="/admin/invoicing" 
                  style={({ isActive }) => isActive ? activeNavItemStyle : navItemStyle}
                  onMouseEnter={(e) => !e.target.className.includes('active') && (e.target.style.backgroundColor = 'var(--color-warm-white)')}
                  onMouseLeave={(e) => !e.target.className.includes('active') && (e.target.style.backgroundColor = 'transparent')}
                >
                  <span style={iconStyle}>💰</span>
                  {!collapsed && 'Invoicing'}
                </NavLink>
              </>
            )}
          </>
        )}

        {!isAuthenticated && (
          <>
            <div style={sectionTitleStyle}>Get Started</div>
            
            <NavLink 
              to="/login" 
              style={({ isActive }) => isActive ? activeNavItemStyle : navItemStyle}
              onMouseEnter={(e) => !e.target.className.includes('active') && (e.target.style.backgroundColor = 'var(--color-warm-white)')}
              onMouseLeave={(e) => !e.target.className.includes('active') && (e.target.style.backgroundColor = 'transparent')}
            >
              <span style={iconStyle}>⚿</span>
              {!collapsed && 'Login'}
            </NavLink>

            <NavLink 
              to="/register" 
              style={({ isActive }) => isActive ? activeNavItemStyle : navItemStyle}
              onMouseEnter={(e) => !e.target.className.includes('active') && (e.target.style.backgroundColor = 'var(--color-warm-white)')}
              onMouseLeave={(e) => !e.target.className.includes('active') && (e.target.style.backgroundColor = 'transparent')}
            >
              <span style={iconStyle}>✎</span>
              {!collapsed && 'Register'}
            </NavLink>
          </>
        )}
      </nav>

      {isAuthenticated && user && (
        <div style={userSectionStyle}>
          <div style={userDropdownContainerStyle} ref={dropdownRef}>
            {/* Dropdown Menu */}
            {showUserDropdown && !collapsed && (
              <div style={dropdownMenuStyle}>
                <button 
                  onClick={handleAccountClick}
                  style={dropdownItemStyle}
                  onMouseEnter={(e) => Object.assign(e.target.style, dropdownItemHoverStyle)}
                  onMouseLeave={(e) => Object.assign(e.target.style, dropdownItemStyle)}
                >
                  <span>⚙️</span>
                  <span>Account Settings</span>
                </button>
                
                <button 
                  onClick={handleLogout}
                  style={{...dropdownItemStyle, borderBottom: 'none', color: 'var(--color-error)'}}
                  onMouseEnter={(e) => Object.assign(e.target.style, {...dropdownItemHoverStyle, color: 'var(--color-error)'})}
                  onMouseLeave={(e) => Object.assign(e.target.style, {...dropdownItemStyle, borderBottom: 'none', color: 'var(--color-error)'})}
                >
                  <span>🚪</span>
                  <span>Logout</span>
                </button>
              </div>
            )}

            {/* Clickable User Info */}
            <div 
              style={userClickableStyle}
              onClick={handleUserClick}
              onMouseEnter={(e) => {
                if (!collapsed) {
                  e.target.style.backgroundColor = 'var(--color-warm-white)';
                }
              }}
              onMouseLeave={(e) => {
                if (!collapsed) {
                  e.target.style.backgroundColor = 'transparent';
                }
              }}
            >
              <div style={userAvatarStyle}>
                {getInitials(user.full_name || user.email)}
              </div>
              {!collapsed && (
                <div style={{ flex: 1 }}>
                  <div style={userNameStyle}>
                    {user.full_name || user.email}
                  </div>
                  <div style={userEmailStyle}>
                    {user.email || 'User'}
                    {user.is_admin && (
                      <span style={{ 
                        marginLeft: '0.5rem', 
                        fontSize: '10px', 
                        backgroundColor: 'var(--color-braun-orange)', 
                        color: 'white', 
                        padding: '2px 6px', 
                        borderRadius: '9999px',
                        fontWeight: 'bold'
                      }}>
                        ADMIN
                      </span>
                    )}
                  </div>
                </div>
              )}
              {!collapsed && (
                <div style={{ 
                  fontSize: '12px', 
                  color: 'var(--color-text-secondary)',
                  transform: showUserDropdown ? 'rotate(180deg)' : 'rotate(0deg)',
                  transition: 'transform 0.2s ease'
                }}>
                  ▼
                </div>
              )}
            </div>

            {/* Collapsed Mode - Simple Logout */}
            {collapsed && (
              <button 
                onClick={handleLogout}
                style={{
                  ...userAvatarStyle,
                  cursor: 'pointer',
                  border: 'none',
                  fontSize: '12px'
                }}
                title="Logout"
              >
                ↪
              </button>
            )}
          </div>
        </div>
      )}
    </aside>
  );

  console.log('🔍 Sidebar: Returning JSX content');
  return sidebarContent;
};

console.log('🚀 Sidebar: Component definition complete, exporting...');

export default Sidebar; 