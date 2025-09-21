import { getToken } from './authService';

// AI service endpoints include /api prefix, so we need base URL without /api
const API_BASE_URL = import.meta.env.VITE_API_URL 
  ? import.meta.env.VITE_API_URL.replace('/api', '') 
  : 'http://localhost:8000';

class AIService {
  constructor() {
    this.cache = new Map();
  }

  async makeRequest(endpoint, options = {}) {
    try {
      const token = await getToken();
      const response = await fetch(`${API_BASE_URL}${endpoint}`, {
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json',
          ...options.headers,
        },
        ...options,
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ error: 'Network error' }));
        const error = new Error(errorData.detail || errorData.error || `HTTP ${response.status}`);
        error.status = response.status;
        error.isNotFound = response.status === 404;
        throw error;
      }

      return await response.json();
    } catch (error) {
      // Don't log 404 errors for polling endpoints - they're expected during analysis
      if (error.isNotFound && endpoint.includes('/suggestions')) {
        error.isExpectedPollingError = true;
      } else {
        console.error(`AI Service Error (${endpoint}):`, error);
      }
      throw error;
    }
  }

  /**
   * Analyze Excel file and get AI insights
   */
  async analyzeExcel(fileId, sheetName = null) {
    console.log('🤖 [AI] Starting Excel analysis:', { fileId, sheetName });
    
    const cacheKey = `excel-analysis-${fileId}-${sheetName || 'default'}`;
    if (this.cache.has(cacheKey)) {
      console.log('🤖 [AI] Returning cached Excel analysis');
      return this.cache.get(cacheKey);
    }

    const response = await this.makeRequest('/api/ai/analyze-excel', {
      method: 'POST',
      body: JSON.stringify({
        file_id: fileId,
        sheet_name: sheetName,
      }),
    });

    this.cache.set(cacheKey, response);
    console.log('🤖 [AI] Excel analysis completed:', response);
    return response;
  }

  /**
   * Get AI variable suggestions for an analysis
   */
  async getVariableSuggestions(analysisId) {
    console.log('🤖 [AI] Getting variable suggestions for analysis:', analysisId);
    
    const cacheKey = `suggestions-${analysisId}`;
    if (this.cache.has(cacheKey)) {
      console.log('🤖 [AI] Returning cached variable suggestions');
      return this.cache.get(cacheKey);
    }

    const response = await this.makeRequest(`/api/ai/analysis/${analysisId}/suggestions`);
    
    this.cache.set(cacheKey, response);
    console.log('🤖 [AI] Variable suggestions received:', response);
    return response;
  }

  /**
   * Analyze simulation results and get AI insights
   */
  async analyzeResults(simulationId, includeExecutiveSummary = true) {
    console.log('🤖 [AI] Starting results analysis:', { simulationId, includeExecutiveSummary });
    
    const cacheKey = `results-analysis-${simulationId}-${includeExecutiveSummary}`;
    if (this.cache.has(cacheKey)) {
      console.log('🤖 [AI] Returning cached results analysis');
      return this.cache.get(cacheKey);
    }

    const response = await this.makeRequest('/api/ai/analyze-results', {
      method: 'POST',
      body: JSON.stringify({
        simulation_id: simulationId,
        include_executive_summary: includeExecutiveSummary,
      }),
    });

    this.cache.set(cacheKey, response);
    console.log('🤖 [AI] Results analysis completed:', response);
    return response;
  }

  /**
   * Get AI summary for analyzed results
   */
  async getResultsSummary(analysisId) {
    console.log('🤖 [AI] Getting results summary for analysis:', analysisId);
    
    const cacheKey = `summary-${analysisId}`;
    if (this.cache.has(cacheKey)) {
      console.log('🤖 [AI] Returning cached results summary');
      return this.cache.get(cacheKey);
    }

    const response = await this.makeRequest(`/api/ai/results/${analysisId}/summary`);
    
    this.cache.set(cacheKey, response);
    console.log('🤖 [AI] Results summary received:', response);
    return response;
  }

  /**
   * Check AI service health
   */
  async checkHealth() {
    return await this.makeRequest('/api/ai/health');
  }

  /**
   * Clear cache
   */
  clearCache() {
    this.cache.clear();
    console.log('🤖 [AI] Cache cleared');
  }

  /**
   * Get cache size for debugging
   */
  getCacheInfo() {
    return {
      size: this.cache.size,
      keys: Array.from(this.cache.keys()),
    };
  }
}

// Export singleton instance
export default new AIService();
