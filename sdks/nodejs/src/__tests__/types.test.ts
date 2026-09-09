import { describe, it, expect } from 'vitest';
import { Decision, Route, RetrieverType } from '../types';

describe('TypeScript Enum Types', () => {
  it('should have correct Decision enum values', () => {
    expect(Decision.ANSWERED).toBe('ANSWERED');
    expect(Decision.ABSTAINED).toBe('ABSTAINED');
    expect(Decision.EVIDENCE_ONLY).toBe('EVIDENCE_ONLY');
  });

  it('should have correct Route enum values', () => {
    expect(Route.FAST).toBe('FAST');
    expect(Route.STANDARD).toBe('STANDARD');
    expect(Route.SLOW).toBe('SLOW');
    expect(Route.CACHED).toBe('CACHED');
  });

  it('should have correct RetrieverType enum values', () => {
    expect(RetrieverType.VECTOR_INDEX).toBe('VECTOR_INDEX');
    expect(RetrieverType.BM25).toBe('BM25');
    expect(RetrieverType.GRAPH).toBe('GRAPH');
    expect(RetrieverType.TREE).toBe('TREE');
  });
});
