/**
 * Utility functions for consistent simulation ID handling across the frontend.
 * Ensures compatibility with backend ID normalization logic.
 */

// Constants for ID patterns and validation rules
export const ID_PATTERNS = {
  TARGET_SUFFIX: /_target_\d+/g,
  SINGLE_TARGET_SUFFIX: /_target_\d+$/,
  MULTIPLE_TARGET_SUFFIX: /(_target_\d+){2,}/,
};

/**
 * Normalizes simulation ID by removing _target_ suffixes using the same
 * regex pattern as the backend (simulation/router.py)
 * @param {string} id - The simulation ID to normalize
 * @returns {string} - The normalized parent ID
 */
export function normalizeSimulationId(id) {
  if (!id || typeof id !== 'string') {
    return '';
  }
  
  // Remove all _target_X suffixes to get the parent ID
  // This matches the backend regex pattern: re.sub(r'_target_\d+', '', simulation_id)
  return id.replace(ID_PATTERNS.TARGET_SUFFIX, '');
}

/**
 * Extracts the parent simulation ID from any child ID
 * @param {string} id - The simulation ID (parent or child)
 * @returns {string} - The parent simulation ID
 */
export function getParentId(id) {
  return normalizeSimulationId(id);
}

/**
 * Detects if an ID is a child simulation (contains _target_ suffix)
 * @param {string} id - The simulation ID to check
 * @returns {boolean} - True if this is a child simulation ID
 */
export function isChildSimulationId(id) {
  if (!id || typeof id !== 'string') {
    return false;
  }
  
  return ID_PATTERNS.SINGLE_TARGET_SUFFIX.test(id);
}

/**
 * Validates simulation ID for corruption (multiple _target_ suffixes)
 * @param {string} id - The simulation ID to validate
 * @returns {object} - Validation result with isValid boolean and error message
 */
export function validateSimulationId(id) {
  if (!id || typeof id !== 'string') {
    return {
      isValid: false,
      error: 'Simulation ID must be a non-empty string',
      isCorrupted: false
    };
  }
  
  // Check for multiple _target_ suffixes which indicates corruption
  const hasMultipleTargets = ID_PATTERNS.MULTIPLE_TARGET_SUFFIX.test(id);
  
  if (hasMultipleTargets) {
    return {
      isValid: false,
      error: `Simulation ID appears corrupted with multiple _target_ suffixes: ${id}`,
      isCorrupted: true,
      suggestedFix: normalizeSimulationId(id)
    };
  }
  
  return {
    isValid: true,
    error: null,
    isCorrupted: false
  };
}

/**
 * Generates a clean child simulation ID from a parent ID
 * @param {string} parentId - The parent simulation ID
 * @param {number} targetIndex - The target index for the child simulation
 * @returns {string} - The clean child simulation ID
 */
export function generateChildSimulationId(parentId, targetIndex) {
  if (!parentId || typeof parentId !== 'string') {
    throw new Error('Parent ID must be a non-empty string');
  }
  
  if (typeof targetIndex !== 'number' || targetIndex < 0) {
    throw new Error('Target index must be a non-negative number');
  }
  
  // Ensure we start with a clean parent ID
  const cleanParentId = normalizeSimulationId(parentId);
  return `${cleanParentId}_target_${targetIndex}`;
}

/**
 * Parses a child simulation ID to extract child information
 * @param {string} id - The simulation ID to parse
 * @returns {object|null} - Object with {isChild: boolean, index: number} or null if invalid
 */
export function parseChildIndex(id) {
  if (!id || typeof id !== 'string') {
    return null;
  }
  
  const match = id.match(ID_PATTERNS.SINGLE_TARGET_SUFFIX);
  if (!match) {
    return { isChild: false, index: null };
  }
  
  // Extract the numeric index from the match
  const indexMatch = id.match(/_target_(\d+)$/);
  if (indexMatch) {
    return {
      isChild: true,
      index: parseInt(indexMatch[1], 10)
    };
  }
  
  return { isChild: true, index: null };
}

/**
 * Deduplicates simulation IDs by converting to parent IDs
 * @param {string[]} ids - Array of simulation IDs (may include duplicates)
 * @param {object} options - Options object
 * @param {boolean} options.logDropped - Whether to log dropped entries in dev builds
 * @returns {string[]} - Array of unique parent simulation IDs
 */
export function deduplicateSimulationIds(ids, options = {}) {
  if (!Array.isArray(ids)) {
    return [];
  }
  
  const { logDropped = false } = options;
  const droppedEntries = [];
  
  const validIds = ids.filter(id => {
    const isValid = id && typeof id === 'string';
    if (!isValid && logDropped) {
      droppedEntries.push(id);
    }
    return isValid;
  });
  
  const parentIds = validIds.map(id => normalizeSimulationId(id));
  const uniqueParentIds = [...new Set(parentIds)];
  
  // Log dropped entries in development builds
  if (logDropped && droppedEntries.length > 0 && import.meta.env.DEV) {
    console.warn(
      `[deduplicateSimulationIds] Dropped ${droppedEntries.length} invalid entries:`,
      droppedEntries
    );
  }
  
  return uniqueParentIds;
}

/**
 * Logs validation warnings for corrupted simulation IDs
 * @param {string} id - The simulation ID to check
 * @param {string} context - Context information for debugging
 */
export function logIdValidationWarning(id, context = '') {
  const validation = validateSimulationId(id);
  
  if (!validation.isValid && validation.isCorrupted) {
    console.warn(
      `🔧 Simulation ID Corruption Detected: ${validation.error}`,
      {
        corruptedId: id,
        suggestedFix: validation.suggestedFix,
        context,
        stackTrace: new Error().stack
      }
    );
  }
}
