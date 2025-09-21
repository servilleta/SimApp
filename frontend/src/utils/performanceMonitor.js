/**
 * Phase 26: Performance monitoring utilities for WebSocket connection timing
 */

export const logPerformanceMetrics = () => {
  try {
    // Get all performance marks and measures
    const marks = performance.getEntriesByType('mark');
    const measures = performance.getEntriesByType('measure');
    
    // Filter Phase 26 WebSocket metrics
    const phase26Marks = marks.filter(mark => mark.name.startsWith('phase26-websocket'));
    const phase26Measures = measures.filter(measure => measure.name.startsWith('phase26-websocket'));
    
    if (phase26Marks.length > 0) {
      console.log('🚀 [PHASE26] Performance Metrics:');
      console.log('📊 Marks:', phase26Marks.map(mark => ({
        name: mark.name,
        timestamp: mark.startTime,
        timeFromStart: mark.startTime - (phase26Marks[0]?.startTime || 0)
      })));
      
      console.log('📊 Measures:', phase26Measures.map(measure => ({
        name: measure.name,
        duration: measure.duration
      })));
      
      // Calculate key timings
      const totalMeasure = phase26Measures.find(m => m.name.includes('total'));
      const importMeasure = phase26Measures.find(m => m.name.includes('import'));
      
      if (totalMeasure) {
        console.log(`🚀 [PHASE26] Total WebSocket connection time: ${totalMeasure.duration.toFixed(2)}ms`);
        
        if (totalMeasure.duration < 500) {
          console.log('✅ [PHASE26] SUCCESS: WebSocket connection under 500ms target!');
        } else {
          console.warn('⚠️ [PHASE26] WARNING: WebSocket connection exceeded 500ms target');
        }
      }
      
      if (importMeasure) {
        console.log(`🚀 [PHASE26] WebSocket service import time: ${importMeasure.duration.toFixed(2)}ms`);
      }
    }
  } catch (error) {
    console.error('🚀 [PHASE26] Performance monitoring error:', error);
  }
};

export const clearPerformanceMetrics = () => {
  try {
    // Clear all Phase 26 performance entries
    const entries = performance.getEntries();
    entries.forEach(entry => {
      if (entry.name.startsWith('phase26-websocket')) {
        // Note: Individual entry removal not supported in all browsers
        // performance.clearMarks() and performance.clearMeasures() clear all
      }
    });
    
    console.log('🚀 [PHASE26] Performance metrics cleared');
  } catch (error) {
    console.error('🚀 [PHASE26] Performance clear error:', error);
  }
};

// Auto-log performance metrics after 10 seconds to capture full simulation startup
export const schedulePerformanceReport = () => {
  setTimeout(() => {
    logPerformanceMetrics();
  }, 10000);
};