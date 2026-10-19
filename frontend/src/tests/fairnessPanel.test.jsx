import { describe, it, expect, vi, beforeEach } from 'vitest';
import React from 'react';
import { render, screen, waitFor } from '@testing-library/react';
import FairnessPanel from '../components/FairnessPanel';

describe('FairnessPanel Component Tests', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it('renders proxy attribution banner and metrics without crashing', async () => {
    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({
        metadata: {
          proxy_dataset: "Blodgett et al. (2016) TwitterAAE Lexicon Proxy & Borkan et al. Identity Lexicon",
          audit_type: "Demographic & Dialect Parity Audit (Proxy Estimates)"
        },
        dialect_parity: {
          aave_proxy: { baseline_fpr: 10.0, context_aware_fpr: 0.0, samples_count: 10 },
          sae_baseline: { baseline_fpr: 0.0, context_aware_fpr: 0.0, samples_count: 10 },
          fpr_gap_baseline: 10.0,
          fpr_gap_context_aware: 0.0,
          within_tolerance: true,
          disparate_impact_ratio: 1.0
        },
        identity_parity: {
          baseline_fpr: 20.0,
          context_aware_fpr: 0.0,
          samples_count: 10
        }
      })
    });

    render(<FairnessPanel backendOnline={true} />);

    expect(screen.getByText(/Fairness & Demographic Parity Audit/i)).toBeDefined();
    expect(screen.getByText(/Blodgett et al\. AAVE Markers/i)).toBeDefined();

    await waitFor(() => {
      expect(screen.getByText(/PASSED \(≤ 5\.0%\)/i)).toBeDefined();
    });
  });
});
