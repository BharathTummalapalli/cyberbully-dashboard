import { describe, it, expect, vi, beforeEach } from 'vitest';
import React from 'react';
import { render, screen, waitFor } from '@testing-library/react';
import ReviewQueue from '../components/ReviewQueue';
import { AuthProvider } from '../context/AuthContext';

describe('ReviewQueue Component Tests', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it('renders review queue header and uncertain band policy description', async () => {
    // Mock fetch for /api/queue
    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => [
        {
          id: 'rq-1',
          prediction_id: 'pred-101',
          escalation_reason: 'Confidence 58.0% in uncertain band [40%, 70%]',
          status: 'pending',
          escalated_at: new Date().toISOString(),
          prediction: {
            id: 'pred-101',
            text: 'I do not think your presentation was completely garbage.',
            author: '@tester',
            target_user: '@developer',
            toxicity_score: 0.78,
            context_delta: -0.20,
            final_score: 0.58
          }
        }
      ]
    });

    render(
      <AuthProvider>
        <ReviewQueue backendOnline={true} />
      </AuthProvider>
    );

    expect(screen.getByText(/Human-in-the-Loop Review Queue/i)).toBeDefined();
    expect(screen.getByText(/0.40 ≤ S_fused ≤ 0.70/i)).toBeDefined();

    await waitFor(() => {
      expect(screen.getByText(/I do not think your presentation was completely garbage/i)).toBeDefined();
      expect(screen.getByText(/APPROVE AS SAFE/i)).toBeDefined();
      expect(screen.getByText(/HIDE AS TOXIC/i)).toBeDefined();
    });
  });
});
