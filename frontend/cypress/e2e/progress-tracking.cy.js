/**
 * Progress Tracking Test
 * 
 * Tests the new robust polling architecture for simulation progress.
 * Verifies that progress bars show real progress from 0% to 100%.
 */

describe('Progress Tracking', () => {
  beforeEach(() => {
    // Visit the main page
    cy.visit('/')
    
    // Mock authentication
    cy.window().then((win) => {
      win.localStorage.setItem('authToken', 'test-token')
    })
  })

  it('should show progress bar when simulation starts', () => {
    // Mock the simulation API response
    cy.intercept('POST', '/api/simulations/run', {
      statusCode: 202,
      body: {
        simulation_id: 'test-sim-123',
        status: 'running',
        message: 'Simulation started'
      }
    }).as('startSimulation')

    // Mock progress endpoint
    cy.intercept('GET', '/api/simulations/progress/test-sim-123', {
      statusCode: 200,
      body: {
        status: 'running',
        progress_percentage: 25,
        current_iteration: 250,
        total_iterations: 1000,
        message: 'Running Monte Carlo simulation...',
        engine: 'ultra'
      }
    }).as('getProgress')

    // Start a simulation
    cy.get('[data-testid="upload-file"]').attachFile('sample.xlsx')
    cy.get('[data-testid="run-simulation"]').click()

    // Wait for simulation to start
    cy.wait('@startSimulation')

    // Verify progress bar is visible
    cy.get('[data-testid="progress-bar"]').should('be.visible')
    cy.get('[data-testid="progress-percentage"]').should('contain', '25%')
  })

  it('should update progress continuously', () => {
    // Mock simulation start
    cy.intercept('POST', '/api/simulations/run', {
      statusCode: 202,
      body: {
        simulation_id: 'test-sim-456',
        status: 'running'
      }
    }).as('startSimulation')

    // Mock progress updates
    cy.intercept('GET', '/api/simulations/progress/test-sim-456', (req) => {
      // Simulate progress updates
      const progress = Math.floor(Math.random() * 100)
      req.reply({
        statusCode: 200,
        body: {
          status: 'running',
          progress_percentage: progress,
          current_iteration: progress * 10,
          total_iterations: 1000,
          message: `Running Monte Carlo simulation... ${progress}%`
        }
      })
    }).as('getProgress')

    // Start simulation
    cy.get('[data-testid="upload-file"]').attachFile('sample.xlsx')
    cy.get('[data-testid="run-simulation"]').click()
    cy.wait('@startSimulation')

    // Verify progress updates
    cy.get('[data-testid="progress-percentage"]').should('not.contain', '0%')
    
    // Wait for progress to update
    cy.wait('@getProgress')
    cy.get('[data-testid="progress-bar"]').should('be.visible')
  })

  it('should show completion when simulation finishes', () => {
    // Mock simulation start
    cy.intercept('POST', '/api/simulations/run', {
      statusCode: 202,
      body: {
        simulation_id: 'test-sim-789',
        status: 'running'
      }
    }).as('startSimulation')

    // Mock completion
    cy.intercept('GET', '/api/simulations/progress/test-sim-789', {
      statusCode: 200,
      body: {
        status: 'completed',
        progress_percentage: 100,
        current_iteration: 1000,
        total_iterations: 1000,
        message: 'Simulation completed successfully',
        results: {
          mean: 150.5,
          std_dev: 25.3,
          min: 100.2,
          max: 200.8
        }
      }
    }).as('getProgress')

    // Start simulation
    cy.get('[data-testid="upload-file"]').attachFile('sample.xlsx')
    cy.get('[data-testid="run-simulation"]').click()
    cy.wait('@startSimulation')

    // Verify completion
    cy.wait('@getProgress')
    cy.get('[data-testid="progress-percentage"]').should('contain', '100%')
    cy.get('[data-testid="simulation-complete"]').should('be.visible')
  })

  it('should show error when simulation fails', () => {
    // Mock simulation start
    cy.intercept('POST', '/api/simulations/run', {
      statusCode: 202,
      body: {
        simulation_id: 'test-sim-error',
        status: 'running'
      }
    }).as('startSimulation')

    // Mock failure
    cy.intercept('GET', '/api/simulations/progress/test-sim-error', {
      statusCode: 200,
      body: {
        status: 'failed',
        progress_percentage: 0,
        message: 'GPU validation failed: statistical properties incorrect',
        error: 'GPU validation failed'
      }
    }).as('getProgress')

    // Start simulation
    cy.get('[data-testid="upload-file"]').attachFile('sample.xlsx')
    cy.get('[data-testid="run-simulation"]').click()
    cy.wait('@startSimulation')

    // Verify error display
    cy.wait('@getProgress')
    cy.get('[data-testid="simulation-error"]').should('be.visible')
    cy.get('[data-testid="error-message"]').should('contain', 'GPU validation failed')
  })
}) 