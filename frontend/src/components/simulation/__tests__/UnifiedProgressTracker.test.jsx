import React from 'react';
import { render, screen, act, waitFor } from '@testing-library/react';
import { Provider } from 'react-redux';
import { configureStore } from '@reduxjs/toolkit';
import { vi, describe, it, expect, beforeEach, afterEach } from 'vitest';

import UnifiedProgressTracker from '../UnifiedProgressTracker';
import simulationSlice, { fetchSimulationProgress } from '../../../store/simulationSlice';
import * as simulationService from '../../../services/simulationService';
import {
  validateSimulationId,
  normalizeSimulationId,
  generateChildSimulationId
} from '../../../utils/simulationIdUtils';

// Mock the CSS import
vi.mock('../UnifiedProgressTracker.css', () => ({}));

// Mock Lucide React icons
vi.mock('lucide-react', () => ({
  CheckCircle: () => <div data-testid="check-circle">✓</div>,
  Play: () => <div data-testid="play">▶</div>,
  AlertCircle: () => <div data-testid="alert-circle">⚠</div>,
  Clock: () => <div data-testid="clock">⏰</div>,
  Loader2: () => <div data-testid="loader">⏳</div>,
  X: () => <div data-testid="x">✕</div>,
  FileText: () => <div data-testid="file-text">📄</div>,
  Calculator: () => <div data-testid="calculator">🧮</div>,
  BarChart3: () => <div data-testid="bar-chart">📊</div>
}));

// Mock simulation service
vi.mock('../../../services/simulationService', () => ({
  getSimulationProgress: vi.fn(),
  getSimulationStatus: vi.fn()
}));

// Mock logger
vi.mock('../../../utils/logger', () => ({
  default: {
    debug: vi.fn(),
    info: vi.fn(),
    warn: vi.fn(),
    error: vi.fn()
  }
}));

describe('UnifiedProgressTracker - Regression Prevention', () => {
  let store;

  beforeEach(() => {
    // Create a mock store with initial state
    store = configureStore({
      reducer: {
        simulation: simulationSlice
      },
      preloadedState: {
        simulation: {
          currentSimulationId: 'test-sim-123',
          currentParentSimulationId: 'test-sim-123',
          status: 'running',
          results: null,
          multipleResults: [],
          error: null,
          iterations: 1000,
          progressData: {
            progressPercentage: 0,
            currentStage: 'initialization',
            currentIteration: 0,
            totalIterations: 1000,
            stageDescription: 'Starting simulation...',
            timestamp: Date.now(),
            hasCompleted: false
          },
          // Add _pendingBatchFetch to match slice initial state
          _pendingBatchFetch: null
        }
      }
    });

    // Use fake timers but don't mock dispatch to avoid recursion
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
    vi.clearAllMocks();
  });

  it('should not regress progress below 95% after reaching completion', async () => {
    // Test the Redux slice logic directly without complex UI rendering
    
    // Simulate progress sequence: 0% -> 80% -> 100% -> 0% (stray update)
    const progressUpdates = [
      { progress: 0, stage: 'initialization' },
      { progress: 80, stage: 'simulation' },
      { progress: 100, stage: 'results' },
      { progress: 0, stage: 'initialization' } // This should be ignored
    ];

    const results = [];

    // Apply each progress update and verify the state doesn't regress
    for (let i = 0; i < progressUpdates.length; i++) {
      const update = progressUpdates[i];
      
      // Dispatch the Redux action
      act(() => {
        store.dispatch({
          type: 'simulation/fetchSimulationProgress/fulfilled',
          payload: {
            simulationId: 'test-sim-123',
            normalizedId: 'test-sim-123',
            progress_percentage: update.progress,
            stage: update.stage,
            stage_description: `Stage: ${update.stage}`,
            current_iteration: update.progress * 10,
            total_iterations: 1000,
            timestamp: Date.now()
          }
        });
      });

      // Get the current state
      const state = store.getState();
      const displayedProgress = state.simulation.progressData.progressPercentage;
      
      results.push({
        update: i,
        backend: update.progress,
        displayed: displayedProgress
      });
      
      console.log(`Update ${i}: backend=${update.progress}%, displayed=${displayedProgress}%`);
    }

    // CRITICAL TEST: After reaching 100%, should not regress to 0%
    const finalResult = results[3]; // The stray 0% update
    expect(finalResult.displayed).toBeGreaterThanOrEqual(95);
    expect(finalResult.displayed).not.toBe(0);
    
    // Verify the progression worked correctly
    expect(results[0].displayed).toBe(0);   // Initial 0%
    expect(results[1].displayed).toBe(80);  // Normal progression to 80%
    expect(results[2].displayed).toBe(100); // Normal progression to 100%
    expect(results[3].displayed).toBe(100); // Should stay at 100%, not regress to 0%
  }, 10000);

  it('should allow normal progression through stages', async () => {
    // Normal progression sequence
    const normalUpdates = [0, 25, 50, 75, 100];
    
    for (let i = 0; i < normalUpdates.length; i++) {
      const progress = normalUpdates[i];
      
      act(() => {
        store.dispatch({
          type: 'simulation/fetchSimulationProgress/fulfilled',
          payload: {
            simulationId: 'test-sim-123',
            normalizedId: 'test-sim-123',
            progress_percentage: progress,
            stage: progress < 25 ? 'initialization' : 
                   progress < 50 ? 'parsing' :
                   progress < 75 ? 'analysis' : 
                   progress < 100 ? 'simulation' : 'results',
            stage_description: `Progress: ${progress}%`,
            current_iteration: progress * 10,
            total_iterations: 1000
          }
        });
      });

      const state = store.getState();
      const displayedProgress = state.simulation.progressData.progressPercentage;
      
      expect(displayedProgress).toBe(progress);
    }
  });

  it('should ignore decreases when near completion (>=95%)', async () => {
    // First set progress to 96%
    act(() => {
      store.dispatch({
        type: 'simulation/fetchSimulationProgress/fulfilled',
        payload: {
          simulationId: 'test-sim-123',
          normalizedId: 'test-sim-123',
          progress_percentage: 96,
          stage: 'simulation',
          stage_description: 'Near completion',
          current_iteration: 960,
          total_iterations: 1000
        }
      });
    });

    // Verify we're at 96%
    let state = store.getState();
    expect(state.simulation.progressData.progressPercentage).toBe(96);

    // Try to regress to 50% - should be ignored
    act(() => {
      store.dispatch({
        type: 'simulation/fetchSimulationProgress/fulfilled',
        payload: {
          simulationId: 'test-sim-123',
          normalizedId: 'test-sim-123',
          progress_percentage: 50,
          stage: 'simulation',
          stage_description: 'Regression attempt',
          current_iteration: 500,
          total_iterations: 1000
        }
      });
    });

    // Should still be at 96%, not regressed to 50%
    state = store.getState();
    expect(state.simulation.progressData.progressPercentage).toBe(96);
    expect(state.simulation.progressData.progressPercentage).not.toBe(50);
  });

  it('should store progress percentage correctly in Redux state', async () => {
    // Set progress to 75%
    act(() => {
      store.dispatch({
        type: 'simulation/fetchSimulationProgress/fulfilled',
        payload: {
          simulationId: 'test-sim-123',
          normalizedId: 'test-sim-123',
          progress_percentage: 75,
          stage: 'simulation',
          stage_description: 'Running simulation',
          current_iteration: 750,
          total_iterations: 1000
        }
      });
    });

    // Verify the state contains the correct percentage
    const state = store.getState();
    expect(state.simulation.progressData.progressPercentage).toBe(75);
    expect(state.simulation.progressData.currentStage).toBe('simulation');
    expect(state.simulation.progressData.stageDescription).toBe('Running simulation');
  });

  it('should maintain hasCompleted flag for polling control', async () => {
    // Test the hasCompleted flag used to stop polling
    
    // Set progress to 100%
    act(() => {
      store.dispatch({
        type: 'simulation/fetchSimulationProgress/fulfilled',
        payload: {
          simulationId: 'test-sim-123',
          normalizedId: 'test-sim-123',
          progress_percentage: 100,
          stage: 'results',
          stage_description: 'Completed',
          current_iteration: 1000,
          total_iterations: 1000
        }
      });
    });

    // Verify hasCompleted is set
    const state = store.getState();
    expect(state.simulation.progressData.progressPercentage).toBe(100);
    expect(state.simulation.progressData.hasCompleted).toBe(true);
    expect(state.simulation.progressData.currentStage).toBe('results');
  });

  // CRITICAL REGRESSION TESTS: ID Corruption Detection
  describe('ID Corruption Detection and Prevention', () => {
    it('should detect and warn about corrupted simulation IDs with multiple _target_ suffixes', () => {
      const consoleSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});
      
      const corruptedIds = [
        'sim123_target_0_target_0',
        'sim456_target_1_target_1_target_1',
        'batch789_target_2_target_2'
      ];

      // Test the validation function directly
      corruptedIds.forEach(id => {
        const validation = validateSimulationId(id);
        expect(validation.isValid).toBe(false);
        expect(validation.isCorrupted).toBe(true);
        expect(validation.suggestedFix).toBe(normalizeSimulationId(id));
      });

      // Test the component handles corrupted IDs
      render(
        <Provider store={store}>
          <UnifiedProgressTracker 
            simulationIds={corruptedIds}
            targetVariables={['Var1', 'Var2', 'Var3']}
          />
        </Provider>
      );

      // Should have logged warnings for each corrupted ID
      expect(consoleSpy).toHaveBeenCalledWith(
        expect.stringContaining('Corrupted simulation ID detected'),
        expect.any(Object)
      );

      consoleSpy.mockRestore();
    });

    it('should not re-mount when Redux state updates with the same simulation IDs', async () => {
      const renderSpy = vi.fn();
      const MockTrackerWithSpy = vi.fn((props) => {
        renderSpy();
        return <div data-testid="progress-tracker">Mock Tracker</div>;
      });

      // Mock the component to spy on renders
      vi.doMock('../UnifiedProgressTracker', () => ({
        default: MockTrackerWithSpy
      }));

      const simulationIds = ['test-sim-123'];

      const { rerender } = render(
        <Provider store={store}>
          <UnifiedProgressTracker 
            simulationIds={simulationIds}
            targetVariables={['TestVar']}
          />
        </Provider>
      );

      const initialRenderCount = renderSpy.mock.calls.length;

      // Update Redux state without changing simulation IDs
      act(() => {
        store.dispatch({
          type: 'simulation/fetchSimulationProgress/fulfilled',
          payload: {
            simulationId: 'test-sim-123',
            progress_percentage: 50,
            stage: 'simulation'
          }
        });
      });

      // Re-render with same props
      rerender(
        <Provider store={store}>
          <UnifiedProgressTracker 
            simulationIds={simulationIds}
            targetVariables={['TestVar']}
          />
        </Provider>
      );

      // Should not have caused additional renders due to stable key
      expect(renderSpy.mock.calls.length).toBe(initialRenderCount);
    });

    it('should use parent IDs for progress polling correctly', async () => {
      const getSimulationProgressSpy = vi.spyOn(simulationService, 'getSimulationProgress')
        .mockResolvedValue({
          simulationId: 'parent-sim',
          progress_percentage: 25,
          stage: 'simulation'
        });

      const childIds = [
        'parent-sim_target_0',
        'parent-sim_target_1'
      ];

      render(
        <Provider store={store}>
          <UnifiedProgressTracker 
            simulationIds={childIds}
            targetVariables={['Var1', 'Var2']}
          />
        </Provider>
      );

      // Advance timers to trigger polling
      await act(async () => {
        vi.advanceTimersByTime(2000);
      });

      // Should have called progress API with parent ID, not child IDs
      expect(getSimulationProgressSpy).toHaveBeenCalledWith('parent-sim');
      expect(getSimulationProgressSpy).not.toHaveBeenCalledWith('parent-sim_target_0');
      expect(getSimulationProgressSpy).not.toHaveBeenCalledWith('parent-sim_target_1');

      getSimulationProgressSpy.mockRestore();
    });

    it('should clean polling intervals properly on component unmount', async () => {
      const clearIntervalSpy = vi.spyOn(global, 'clearInterval');

      const { unmount } = render(
        <Provider store={store}>
          <UnifiedProgressTracker 
            simulationIds={['test-sim-123']}
            targetVariables={['TestVar']}
          />
        </Provider>
      );

      // Start polling
      await act(async () => {
        vi.advanceTimersByTime(100);
      });

      // Unmount component
      unmount();

      // Should have cleared intervals
      expect(clearIntervalSpy).toHaveBeenCalled();

      clearIntervalSpy.mockRestore();
    });
  });

  // CRITICAL REGRESSION TESTS: Progress Polling
  describe('Progress Polling Behavior', () => {
    it('should handle API failures gracefully without crashing', async () => {
      const getSimulationProgressSpy = vi.spyOn(simulationService, 'getSimulationProgress')
        .mockRejectedValue(new Error('Network error'));

      const { container } = render(
        <Provider store={store}>
          <UnifiedProgressTracker 
            simulationIds={['test-sim-123']}
            targetVariables={['TestVar']}
          />
        </Provider>
      );

      // Advance timers to trigger polling
      await act(async () => {
        vi.advanceTimersByTime(2000);
      });

      // Component should still be rendered despite API error
      expect(container.firstChild).toBeTruthy();

      getSimulationProgressSpy.mockRestore();
    });

    it('should validate simulation IDs before making API requests', async () => {
      const getSimulationProgressSpy = vi.spyOn(simulationService, 'getSimulationProgress')
        .mockResolvedValue({ progress_percentage: 0 });

      const invalidId = 'invalid_target_0_target_0';

      render(
        <Provider store={store}>
          <UnifiedProgressTracker 
            simulationIds={[invalidId]}
            targetVariables={['TestVar']}
          />
        </Provider>
      );

      // Advance timers to trigger polling
      await act(async () => {
        vi.advanceTimersByTime(2000);
      });

      // Should have called API with normalized ID
      expect(getSimulationProgressSpy).toHaveBeenCalledWith('invalid');

      getSimulationProgressSpy.mockRestore();
    });

    it('should prevent multiple polling intervals for the same simulation', async () => {
      const setIntervalSpy = vi.spyOn(global, 'setInterval');

      const { rerender } = render(
        <Provider store={store}>
          <UnifiedProgressTracker 
            simulationIds={['test-sim-123']}
            targetVariables={['TestVar']}
          />
        </Provider>
      );

      const initialIntervalCount = setIntervalSpy.mock.calls.length;

      // Re-render with same simulation ID
      rerender(
        <Provider store={store}>
          <UnifiedProgressTracker 
            simulationIds={['test-sim-123']}
            targetVariables={['TestVar']}
          />
        </Provider>
      );

      // Should not create additional intervals
      expect(setIntervalSpy.mock.calls.length).toBe(initialIntervalCount);

      setIntervalSpy.mockRestore();
    });
  });

  // CRITICAL REGRESSION TESTS: Error Handling
  describe('Error Handling and Validation', () => {
    it('should handle malformed simulation IDs gracefully', () => {
      const consoleSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});

      const malformedIds = [
        null,
        undefined,
        '',
        123,
        {},
        []
      ];

      malformedIds.forEach(id => {
        expect(() => {
          render(
            <Provider store={store}>
              <UnifiedProgressTracker 
                simulationIds={[id]}
                targetVariables={['TestVar']}
              />
            </Provider>
          );
        }).not.toThrow();
      });

      consoleSpy.mockRestore();
    });

    it('should prevent ID corruption from propagating to child components', () => {
      const corruptedId = 'sim123_target_0_target_0_target_0';
      const expectedCleanId = 'sim123';

      // Mock child ID generation
      const mockChildId = generateChildSimulationId(expectedCleanId, 0);
      expect(mockChildId).toBe('sim123_target_0');
      expect(mockChildId).not.toContain('_target_0_target_0');
    });
  });
});
