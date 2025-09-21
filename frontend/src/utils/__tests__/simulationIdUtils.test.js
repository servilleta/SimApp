import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import {
  normalizeSimulationId,
  getParentId,
  isChildSimulationId,
  validateSimulationId,
  generateChildSimulationId,
  parseChildIndex,
  deduplicateSimulationIds,
  logIdValidationWarning,
  ID_PATTERNS
} from '../simulationIdUtils';

describe('simulationIdUtils', () => {
  let consoleSpy;

  beforeEach(() => {
    consoleSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});
  });

  afterEach(() => {
    consoleSpy.mockRestore();
  });

  describe('normalizeSimulationId', () => {
    it('should remove single _target_ suffix', () => {
      expect(normalizeSimulationId('sim123_target_0')).toBe('sim123');
      expect(normalizeSimulationId('batch456_target_1')).toBe('batch456');
      expect(normalizeSimulationId('test_simulation_target_5')).toBe('test_simulation');
    });

    it('should remove multiple _target_ suffixes (corruption cases)', () => {
      expect(normalizeSimulationId('sim123_target_0_target_0')).toBe('sim123');
      expect(normalizeSimulationId('batch456_target_1_target_1_target_1')).toBe('batch456');
      expect(normalizeSimulationId('test_target_0_target_0_target_0_target_0')).toBe('test');
    });

    it('should handle IDs without _target_ suffixes', () => {
      expect(normalizeSimulationId('sim123')).toBe('sim123');
      expect(normalizeSimulationId('batch456')).toBe('batch456');
      expect(normalizeSimulationId('')).toBe('');
    });

    it('should handle edge cases gracefully', () => {
      expect(normalizeSimulationId(null)).toBe('');
      expect(normalizeSimulationId(undefined)).toBe('');
      expect(normalizeSimulationId('')).toBe('');
      expect(normalizeSimulationId(123)).toBe('');
    });

    it('should match backend regex pattern behavior', () => {
      // Test cases that should match backend's re.sub(r'_target_\d+', '', simulation_id)
      const testCases = [
        { input: 'sim_target_0', expected: 'sim' },
        { input: 'test_target_123', expected: 'test' },
        { input: 'batch_target_0_target_1', expected: 'batch' },
        { input: 'complex_name_target_999', expected: 'complex_name' },
        { input: 'no_target_suffix', expected: 'no_target_suffix' }
      ];

      testCases.forEach(({ input, expected }) => {
        expect(normalizeSimulationId(input)).toBe(expected);
      });
    });
  });

  describe('getParentId', () => {
    it('should return parent ID for child simulation IDs', () => {
      expect(getParentId('sim123_target_0')).toBe('sim123');
      expect(getParentId('batch456_target_2')).toBe('batch456');
    });

    it('should return the same ID for parent simulation IDs', () => {
      expect(getParentId('sim123')).toBe('sim123');
      expect(getParentId('batch456')).toBe('batch456');
    });

    it('should handle corrupted IDs', () => {
      expect(getParentId('sim123_target_0_target_0')).toBe('sim123');
    });
  });

  describe('isChildSimulationId', () => {
    it('should correctly identify child simulation IDs', () => {
      expect(isChildSimulationId('sim123_target_0')).toBe(true);
      expect(isChildSimulationId('batch456_target_1')).toBe(true);
      expect(isChildSimulationId('test_simulation_target_99')).toBe(true);
    });

    it('should correctly identify parent simulation IDs', () => {
      expect(isChildSimulationId('sim123')).toBe(false);
      expect(isChildSimulationId('batch456')).toBe(false);
      expect(isChildSimulationId('test_simulation')).toBe(false);
    });

    it('should handle corrupted IDs (multiple suffixes)', () => {
      // These are technically child IDs, but corrupted
      expect(isChildSimulationId('sim123_target_0_target_0')).toBe(true);
    });

    it('should handle edge cases', () => {
      expect(isChildSimulationId(null)).toBe(false);
      expect(isChildSimulationId(undefined)).toBe(false);
      expect(isChildSimulationId('')).toBe(false);
      expect(isChildSimulationId(123)).toBe(false);
    });
  });

  describe('validateSimulationId', () => {
    it('should validate clean simulation IDs as valid', () => {
      const parentId = 'sim123';
      const childId = 'sim123_target_0';

      expect(validateSimulationId(parentId)).toEqual({
        isValid: true,
        error: null,
        isCorrupted: false
      });

      expect(validateSimulationId(childId)).toEqual({
        isValid: true,
        error: null,
        isCorrupted: false
      });
    });

    it('should detect corrupted simulation IDs', () => {
      const corruptedIds = [
        'sim123_target_0_target_0',
        'batch456_target_1_target_1_target_1',
        'test_target_0_target_0'
      ];

      corruptedIds.forEach(id => {
        const validation = validateSimulationId(id);
        expect(validation.isValid).toBe(false);
        expect(validation.isCorrupted).toBe(true);
        expect(validation.error).toContain('multiple _target_ suffixes');
        expect(validation.suggestedFix).toBe(normalizeSimulationId(id));
      });
    });

    it('should validate non-string inputs as invalid', () => {
      const invalidInputs = [null, undefined, '', 123, {}, []];

      invalidInputs.forEach(input => {
        const validation = validateSimulationId(input);
        expect(validation.isValid).toBe(false);
        expect(validation.isCorrupted).toBe(false);
        expect(validation.error).toBe('Simulation ID must be a non-empty string');
      });
    });
  });

  describe('generateChildSimulationId', () => {
    it('should generate clean child simulation IDs', () => {
      expect(generateChildSimulationId('sim123', 0)).toBe('sim123_target_0');
      expect(generateChildSimulationId('batch456', 1)).toBe('batch456_target_1');
      expect(generateChildSimulationId('test', 99)).toBe('test_target_99');
    });

    it('should normalize parent ID before generating child ID', () => {
      // Even if parent ID is corrupted, should generate clean child ID
      expect(generateChildSimulationId('sim123_target_0_target_0', 1)).toBe('sim123_target_1');
    });

    it('should throw errors for invalid inputs', () => {
      expect(() => generateChildSimulationId('', 0)).toThrow('Parent ID must be a non-empty string');
      expect(() => generateChildSimulationId(null, 0)).toThrow('Parent ID must be a non-empty string');
      expect(() => generateChildSimulationId('sim123', -1)).toThrow('Target index must be a non-negative number');
      expect(() => generateChildSimulationId('sim123', 'invalid')).toThrow('Target index must be a non-negative number');
    });
  });

  describe('parseChildIndex', () => {
    it('should correctly parse child simulation IDs', () => {
      expect(parseChildIndex('sim123_target_0')).toEqual({ isChild: true, index: 0 });
      expect(parseChildIndex('batch456_target_1')).toEqual({ isChild: true, index: 1 });
      expect(parseChildIndex('test_simulation_target_99')).toEqual({ isChild: true, index: 99 });
    });

    it('should correctly identify parent simulation IDs', () => {
      expect(parseChildIndex('sim123')).toEqual({ isChild: false, index: null });
      expect(parseChildIndex('batch456')).toEqual({ isChild: false, index: null });
      expect(parseChildIndex('test_simulation')).toEqual({ isChild: false, index: null });
    });

    it('should handle corrupted IDs', () => {
      // Corrupted IDs should still be detected as child but may have issues with index parsing
      const result = parseChildIndex('sim123_target_0_target_0');
      expect(result.isChild).toBe(true);
      // Index parsing may vary for corrupted IDs
    });

    it('should handle edge cases', () => {
      expect(parseChildIndex(null)).toBeNull();
      expect(parseChildIndex(undefined)).toBeNull();
      expect(parseChildIndex('')).toBeNull();
      expect(parseChildIndex(123)).toBeNull();
    });
  });

  describe('deduplicateSimulationIds', () => {
    it('should remove duplicate parent IDs', () => {
      const input = [
        'sim123',
        'sim123_target_0',
        'sim123_target_1',
        'batch456',
        'batch456_target_0'
      ];

      const result = deduplicateSimulationIds(input);
      expect(result).toEqual(['sim123', 'batch456']);
    });

    it('should handle corrupted IDs by normalizing them', () => {
      const input = [
        'sim123_target_0_target_0',
        'sim123_target_1_target_1',
        'sim123'
      ];

      const result = deduplicateSimulationIds(input);
      expect(result).toEqual(['sim123']);
    });

    it('should handle empty and invalid inputs', () => {
      expect(deduplicateSimulationIds([])).toEqual([]);
      expect(deduplicateSimulationIds(null)).toEqual([]);
      expect(deduplicateSimulationIds(undefined)).toEqual([]);
      expect(deduplicateSimulationIds([null, undefined, '', 'sim123'])).toEqual(['sim123']);
    });

    it('should preserve order of first occurrence', () => {
      const input = [
        'batch456_target_0',
        'sim123_target_0',
        'batch456_target_1',
        'sim123'
      ];

      const result = deduplicateSimulationIds(input);
      expect(result).toEqual(['batch456', 'sim123']);
    });
  });

  describe('logIdValidationWarning', () => {
    it('should log warnings for corrupted simulation IDs', () => {
      const corruptedId = 'sim123_target_0_target_0';
      const context = 'test context';

      logIdValidationWarning(corruptedId, context);

      expect(consoleSpy).toHaveBeenCalledWith(
        expect.stringContaining('Simulation ID Corruption Detected'),
        expect.objectContaining({
          corruptedId,
          suggestedFix: 'sim123',
          context,
          stackTrace: expect.any(String)
        })
      );
    });

    it('should not log warnings for valid simulation IDs', () => {
      const validIds = ['sim123', 'sim123_target_0', 'batch456'];

      validIds.forEach(id => {
        logIdValidationWarning(id, 'test context');
      });

      expect(consoleSpy).not.toHaveBeenCalled();
    });

    it('should include context in warning messages', () => {
      const corruptedId = 'sim123_target_0_target_0';
      const context = 'UnifiedProgressTracker polling';

      logIdValidationWarning(corruptedId, context);

      expect(consoleSpy).toHaveBeenCalledWith(
        expect.any(String),
        expect.objectContaining({
          context
        })
      );
    });
  });

  describe('ID_PATTERNS constants', () => {
    it('should have correct regex patterns', () => {
      expect(ID_PATTERNS.TARGET_SUFFIX).toBeInstanceOf(RegExp);
      expect(ID_PATTERNS.SINGLE_TARGET_SUFFIX).toBeInstanceOf(RegExp);
      expect(ID_PATTERNS.MULTIPLE_TARGET_SUFFIX).toBeInstanceOf(RegExp);
    });

    it('should match expected patterns', () => {
      // TARGET_SUFFIX should match all _target_X patterns
      expect('sim123_target_0'.match(ID_PATTERNS.TARGET_SUFFIX)).toBeTruthy();
      expect('sim123_target_0_target_1'.match(ID_PATTERNS.TARGET_SUFFIX)).toBeTruthy();

      // SINGLE_TARGET_SUFFIX should match only single suffix at end
      expect('sim123_target_0'.match(ID_PATTERNS.SINGLE_TARGET_SUFFIX)).toBeTruthy();
      expect('sim123_target_0_target_1'.match(ID_PATTERNS.SINGLE_TARGET_SUFFIX)).toBeTruthy();

      // MULTIPLE_TARGET_SUFFIX should match corruption cases
      expect('sim123_target_0_target_0'.match(ID_PATTERNS.MULTIPLE_TARGET_SUFFIX)).toBeTruthy();
      expect('sim123_target_0'.match(ID_PATTERNS.MULTIPLE_TARGET_SUFFIX)).toBeFalsy();
    });
  });

  describe('Performance and edge cases', () => {
    it('should handle large numbers of IDs efficiently', () => {
      const largeIdArray = [];
      for (let i = 0; i < 1000; i++) {
        largeIdArray.push(`sim${i % 10}_target_${i}`);
      }

      const start = performance.now();
      const result = deduplicateSimulationIds(largeIdArray);
      const end = performance.now();

      expect(result.length).toBe(10); // Should deduplicate to 10 unique parent IDs
      expect(end - start).toBeLessThan(50); // Should complete in less than 50ms
    });

    it('should handle very long simulation IDs', () => {
      const longId = 'a'.repeat(1000) + '_target_0_target_0';
      const validation = validateSimulationId(longId);

      expect(validation.isValid).toBe(false);
      expect(validation.isCorrupted).toBe(true);
      expect(validation.suggestedFix).toBe('a'.repeat(1000));
    });

    it('should handle special characters in simulation IDs', () => {
      const specialIds = [
        'sim-123_target_0',
        'sim.456_target_1',
        'sim_test-batch.789_target_0_target_0'
      ];

      specialIds.forEach(id => {
        expect(() => normalizeSimulationId(id)).not.toThrow();
        expect(() => validateSimulationId(id)).not.toThrow();
      });
    });
  });

  describe('Consistency with backend behavior', () => {
    it('should match backend normalization for known test cases', () => {
      // These test cases should match the behavior of the backend's
      // re.sub(r'_target_\d+', '', simulation_id) pattern
      const backendTestCases = [
        { input: 'sim123_target_0', expected: 'sim123' },
        { input: 'batch_456_target_1', expected: 'batch_456' },
        { input: 'test_simulation_target_999', expected: 'test_simulation' },
        { input: 'complex-name.test_target_0_target_1', expected: 'complex-name.test' },
        { input: 'no_target_here', expected: 'no_target_here' }
      ];

      backendTestCases.forEach(({ input, expected }) => {
        expect(normalizeSimulationId(input)).toBe(expected);
      });
    });

    it('should handle Unicode characters correctly', () => {
      const unicodeId = 'tëst_ñame_target_0';
      expect(normalizeSimulationId(unicodeId)).toBe('tëst_ñame');
    });
  });
});
