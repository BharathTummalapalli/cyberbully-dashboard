import React, { useState, useEffect } from 'react';
import { AlertOctagon, CheckCircle2, EyeOff, MessageSquare, ShieldAlert, Sparkles, RefreshCw, UserCheck, HelpCircle } from 'lucide-react';
import { useAuth } from '../context/AuthContext';

export default function ReviewQueue({ backendOnline }) {
  const { role, token } = useAuth();
  const [queueItems, setQueueItems] = useState([]);
  const [loading, setLoading] = useState(true);
  const [resolvingId, setResolvingId] = useState(null);
  const [notes, setNotes] = useState({});

  const canModerate = role === 'admin' || role === 'moderator';

  const fetchQueue = async () => {
    setLoading(true);
    try {
      const res = await fetch('/api/queue?status=pending');
      if (res.ok) {
        const data = await res.json();
        setQueueItems(data);
      }
    } catch (err) {
      console.error('Failed to fetch review queue:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchQueue();
  }, [backendOnline]);

  const handleResolve = async (predictionId, action) => {
    if (!canModerate) return;
    setResolvingId(predictionId);
    try {
      const itemNote = notes[predictionId] || '';
      const res = await fetch(`/api/queue/${predictionId}/resolve`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify({ action, notes: itemNote })
      });
      if (res.ok) {
        // Remove resolved item from queue state instantly
        setQueueItems(prev => prev.filter(item => item.prediction_id !== predictionId));
      }
    } catch (err) {
      console.error('Failed to resolve queue item:', err);
    } finally {
      setResolvingId(null);
    }
  };

  return (
    <div className="flex-1 overflow-y-auto bg-slate-900 bg-grid p-8">
      {/* Header */}
      <div className="flex items-center justify-between mb-8">
        <div className="flex items-center space-x-3">
          <div className="bg-amber-500/10 p-2.5 rounded-xl border border-amber-500/20 glow-orange">
            <ShieldAlert className="h-6 w-6 text-amber-400" />
          </div>
          <div>
            <h1 className="text-2xl font-bold tracking-tight text-white font-display">
              Human-in-the-Loop Review Queue
            </h1>
            <p className="text-xs text-slate-400">
              Auto-escalated predictions within the uncertain confidence band [40.0% – 70.0%].
            </p>
          </div>
        </div>

        <button
          onClick={fetchQueue}
          disabled={loading}
          className="flex items-center space-x-2 px-4 py-2.5 rounded-xl text-xs font-bold bg-slate-800 text-slate-200 border border-slate-700 hover:bg-slate-750 transition-all cursor-pointer"
        >
          <RefreshCw className={`h-3.5 w-3.5 ${loading ? 'animate-spin' : ''}`} />
          <span>REFRESH QUEUE</span>
        </button>
      </div>

      {/* Role Notice */}
      {!canModerate && (
        <div className="mb-6 p-4 rounded-2xl bg-violet-950/20 border border-violet-500/20 flex items-center justify-between">
          <div className="flex items-center space-x-2.5 text-xs text-violet-300">
            <UserCheck className="h-4 w-4 text-violet-400 shrink-0" />
            <span>You are viewing the queue in read-only mode. Log in as a Moderator or Admin to submit binding resolutions.</span>
          </div>
        </div>
      )}

      {/* Queue Stats Bar */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
        <div className="glass-panel p-5 rounded-2xl border border-slate-800">
          <span className="text-xs font-semibold text-slate-400">Pending Escalations</span>
          <div className="text-3xl font-extrabold text-amber-400 mt-1 font-display">{queueItems.length}</div>
          <span className="text-[10px] text-slate-500 mt-1 block">Awaiting human consensus</span>
        </div>
        <div className="glass-panel p-5 rounded-2xl border border-slate-800">
          <span className="text-xs font-semibold text-slate-400">Escalation Policy Band</span>
          <div className="text-xl font-extrabold text-white mt-2 font-mono">0.40 ≤ S_fused ≤ 0.70</div>
          <span className="text-[10px] text-violet-400 mt-1 block">Additive Clamped Fusion</span>
        </div>
        <div className="glass-panel p-5 rounded-2xl border border-slate-800">
          <span className="text-xs font-semibold text-slate-400">Target Resolution</span>
          <div className="text-xl font-extrabold text-emerald-400 mt-2 font-display">Active Attribution</div>
          <span className="text-[10px] text-slate-500 mt-1 block">Direct personal vs ambient check</span>
        </div>
      </div>

      {/* Queue Items List */}
      <div className="space-y-5">
        {loading ? (
          <div className="py-20 text-center text-xs text-slate-500 italic">
            Scanning database for escalated review queue items...
          </div>
        ) : queueItems.length === 0 ? (
          <div className="glass-panel py-20 text-center rounded-2xl border border-slate-800 text-slate-500 text-sm">
            <CheckCircle2 className="h-8 w-8 text-emerald-500/50 mx-auto mb-3" />
            <p className="font-semibold text-slate-300">Review Queue is completely clear!</p>
            <span className="text-xs text-slate-500">All borderline cases have been resolved. Test custom text in the Sandbox to trigger new escalations.</span>
          </div>
        ) : (
          queueItems.map(item => {
            const pred = item.prediction || {};
            const isResolving = resolvingId === item.prediction_id;

            return (
              <div key={item.id} className="glass-panel p-6 rounded-2xl border border-slate-800 relative overflow-hidden">
                {/* Top info badge */}
                <div className="flex flex-col md:flex-row md:items-center justify-between gap-3 border-b border-slate-900 pb-4 mb-4">
                  <div className="flex items-center space-x-3">
                    <span className="px-2.5 py-1 rounded-lg text-[10px] font-black uppercase bg-amber-500/10 text-amber-400 border border-amber-500/25 tracking-wider">
                      ESCALATED CASE
                    </span>
                    <span className="text-xs font-mono font-bold text-slate-400">
                      ID: {item.prediction_id}
                    </span>
                    <span className="text-[11px] text-slate-500">
                      {item.escalated_at ? new Date(item.escalated_at).toLocaleString() : ''}
                    </span>
                  </div>

                  {/* Additive Score Breakdown Badges */}
                  <div className="flex items-center space-x-2 text-xs">
                    <span className="px-2 py-0.5 rounded bg-slate-900 border border-slate-800 font-mono text-[11px] text-slate-300">
                      P_toxic: <strong className="text-rose-400">{Math.round((pred.toxicity_score || 0) * 100)}%</strong>
                    </span>
                    <span className="text-slate-600">+</span>
                    <span className="px-2 py-0.5 rounded bg-slate-900 border border-slate-800 font-mono text-[11px] text-slate-300">
                      Δ_context: <strong className="text-violet-400">{(pred.context_delta || 0) >= 0 ? '+' : ''}{(pred.context_delta || 0).toFixed(2)}</strong>
                    </span>
                    <span className="text-slate-600">=</span>
                    <span className="px-2.5 py-0.5 rounded bg-amber-500/10 border border-amber-500/25 font-mono text-xs font-extrabold text-amber-400">
                      S_fused: {Math.round((pred.final_score || 0) * 100)}%
                    </span>
                  </div>
                </div>

                {/* Main Post Text Content */}
                <div className="mb-4">
                  <div className="flex items-center space-x-2 text-xs text-slate-400 mb-1.5 font-medium">
                    <span className="text-white font-bold">{pred.author || '@anonymous'}</span>
                    <span>directing towards</span>
                    <span className="text-violet-400 font-bold">{pred.target_user || '@anonymous'}</span>
                  </div>
                  <blockquote className="p-4 rounded-xl bg-slate-950/70 border border-slate-850 text-slate-100 text-sm font-medium leading-relaxed">
                    "{pred.text}"
                  </blockquote>
                </div>

                {/* Escalation Context Explanation */}
                <div className="p-3 rounded-xl bg-slate-950/40 border border-slate-900 mb-5 flex items-start space-x-2 text-xs text-slate-400">
                  <HelpCircle className="h-4 w-4 text-violet-400 shrink-0 mt-0.5" />
                  <span><strong>Escalation Reason:</strong> {item.escalation_reason}</span>
                </div>

                {/* Resolution Controls */}
                <div className="flex flex-col md:flex-row items-center justify-between gap-4 border-t border-slate-900/80 pt-4">
                  <div className="flex-1 w-full">
                    <input
                      type="text"
                      placeholder="Optional moderator notes / decision rationale..."
                      value={notes[item.prediction_id] || ''}
                      onChange={(e) => setNotes({ ...notes, [item.prediction_id]: e.target.value })}
                      disabled={!canModerate || isResolving}
                      className="w-full bg-slate-950/80 border border-slate-850 rounded-xl px-3.5 py-2 text-xs text-slate-200 focus:outline-none focus:border-violet-500 transition-all font-mono"
                    />
                  </div>

                  <div className="flex space-x-3 shrink-0">
                    <button
                      onClick={() => handleResolve(item.prediction_id, 'Approved')}
                      disabled={!canModerate || isResolving}
                      className="flex items-center space-x-1.5 px-4 py-2 rounded-xl text-xs font-bold bg-emerald-500/10 text-emerald-400 border border-emerald-500/25 hover:bg-emerald-500 hover:text-slate-950 transition-all cursor-pointer disabled:opacity-40 shadow-sm"
                    >
                      <CheckCircle2 className="h-3.5 w-3.5" />
                      <span>APPROVE AS SAFE</span>
                    </button>

                    <button
                      onClick={() => handleResolve(item.prediction_id, 'Hidden')}
                      disabled={!canModerate || isResolving}
                      className="flex items-center space-x-1.5 px-4 py-2 rounded-xl text-xs font-bold bg-rose-500/10 text-rose-400 border border-rose-500/25 hover:bg-rose-500 hover:text-slate-950 transition-all cursor-pointer disabled:opacity-40 shadow-sm"
                    >
                      <EyeOff className="h-3.5 w-3.5" />
                      <span>HIDE AS TOXIC</span>
                    </button>
                  </div>
                </div>
              </div>
            );
          })
        )}
      </div>
    </div>
  );
}
