import React, { useState, useEffect } from 'react';
import { ShieldCheck, Scale, CheckCircle2, AlertTriangle, Info, RefreshCw, BarChart2, BookOpen } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts';

export default function FairnessPanel({ backendOnline }) {
  const [fairnessData, setFairnessData] = useState(null);
  const [loading, setLoading] = useState(true);

  const fetchFairness = async () => {
    setLoading(true);
    try {
      const res = await fetch('/api/fairness');
      if (res.ok) {
        const data = await res.json();
        setFairnessData(data);
      }
    } catch (err) {
      console.error('Failed to fetch fairness metrics:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchFairness();
  }, [backendOnline]);

  const dialect = fairnessData?.dialect_parity;
  const identity = fairnessData?.identity_parity;

  // Prepare chart comparison data
  const comparisonData = dialect ? [
    {
      name: 'AAVE Dialect Proxy (Blodgett et al.)',
      'Stock Baseline FPR': dialect.aave_proxy.baseline_fpr,
      'Context-Aware Fused FPR': dialect.aave_proxy.context_aware_fpr
    },
    {
      name: 'Standard American English (SAE)',
      'Stock Baseline FPR': dialect.sae_baseline.baseline_fpr,
      'Context-Aware Fused FPR': dialect.sae_baseline.context_aware_fpr
    },
    {
      name: 'Protected Identity Terms (Borkan et al.)',
      'Stock Baseline FPR': identity?.baseline_fpr || 0,
      'Context-Aware Fused FPR': identity?.context_aware_fpr || 0
    }
  ] : [];

  return (
    <div className="flex-1 overflow-y-auto bg-slate-900 bg-grid p-8">
      {/* Page Header */}
      <div className="flex items-center justify-between mb-8">
        <div className="flex items-center space-x-3">
          <div className="bg-emerald-500/10 p-2.5 rounded-xl border border-emerald-500/20 glow-green">
            <Scale className="h-6 w-6 text-emerald-400" />
          </div>
          <div>
            <h1 className="text-2xl font-bold tracking-tight text-white font-display">
              Fairness & Demographic Parity Audit
            </h1>
            <p className="text-xs text-slate-400">
              Assessing dialect bias mitigation and identity token false-alarm attenuation.
            </p>
          </div>
        </div>

        <button
          onClick={fetchFairness}
          disabled={loading}
          className="flex items-center space-x-2 px-4 py-2.5 rounded-xl text-xs font-bold bg-slate-800 text-slate-200 border border-slate-700 hover:bg-slate-750 transition-all cursor-pointer"
        >
          <RefreshCw className={`h-3.5 w-3.5 ${loading ? 'animate-spin' : ''}`} />
          <span>RE-RUN AUDIT</span>
        </button>
      </div>

      {/* Academic Attribution Banner */}
      <div className="mb-8 p-5 rounded-2xl bg-slate-950/80 border border-slate-800 flex items-start space-x-3.5">
        <BookOpen className="h-5 w-5 text-violet-400 shrink-0 mt-0.5" />
        <div className="text-xs leading-relaxed text-slate-300">
          <strong className="text-white block font-display mb-1">
            Dialect & Identity Equity Audit (Proxy Estimates via Blodgett et al. AAVE Markers & Borkan et al. Identity Lexicon)
          </strong>
          <span>
            These metrics evaluate algorithmic fairness by measuring the <strong>False Positive Rate (FPR) parity gap</strong>.
            Standard neural toxicity models often exhibit elevated false alarms on colloquial dialect markers (e.g., AAVE syntax) and benign self-affirming identity terms.
            Our <strong>Rule-Based Context Adjustment Layer</strong> explicitly corrects for colloquial syntax, reducing discriminatory false-positive censorship.
          </span>
        </div>
      </div>

      {/* Metrics Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
        <div className="glass-panel p-5 rounded-2xl border border-slate-800">
          <span className="text-xs font-semibold text-slate-400">Dialect FPR Gap</span>
          <div className="text-3xl font-extrabold text-emerald-400 mt-1 font-display">
            {dialect ? `${dialect.fpr_gap_context_aware}%` : '0.0%'}
          </div>
          <span className="text-[10px] text-slate-500 mt-1 block">
            Baseline gap was {dialect ? `${dialect.fpr_gap_baseline}%` : 'N/A'}
          </span>
        </div>

        <div className="glass-panel p-5 rounded-2xl border border-slate-800">
          <span className="text-xs font-semibold text-slate-400">Parity Tolerance Status</span>
          <div className="mt-2 flex items-center space-x-1.5">
            <CheckCircle2 className="h-5 w-5 text-emerald-400" />
            <span className="text-sm font-bold text-emerald-300 font-display">
              {dialect?.within_tolerance ? 'PASSED (≤ 5.0%)' : 'OUTSIDE TOLERANCE'}
            </span>
          </div>
          <span className="text-[10px] text-slate-500 mt-1 block">Threshold: ±5.0% disparity</span>
        </div>

        <div className="glass-panel p-5 rounded-2xl border border-slate-800">
          <span className="text-xs font-semibold text-slate-400">Disparate Impact Ratio</span>
          <div className="text-3xl font-extrabold text-white mt-1 font-display">
            {dialect?.disparate_impact_ratio || '1.000'}
          </div>
          <span className="text-[10px] text-emerald-400 mt-1 block">Ideal target: 1.0 (equitable pass-through)</span>
        </div>

        <div className="glass-panel p-5 rounded-2xl border border-slate-800">
          <span className="text-xs font-semibold text-slate-400">Identity Term FPR</span>
          <div className="text-3xl font-extrabold text-violet-400 mt-1 font-display">
            {identity ? `${identity.context_aware_fpr}%` : '0.0%'}
          </div>
          <span className="text-[10px] text-slate-500 mt-1 block">Benign identity keyword mentions</span>
        </div>
      </div>

      {/* Main Bar Chart: Baseline vs Context-Aware FPR */}
      <div className="glass-panel p-6 rounded-2xl border border-slate-800 mb-8">
        <h3 className="text-sm font-bold text-white uppercase tracking-wider mb-2 font-display">
          False Positive Rate (FPR) Comparison: Baseline vs. Context-Aware
        </h3>
        <p className="text-xs text-slate-400 mb-6">
          Lower is better. Demonstrates how contextual modulation prevents false censorship of benign colloquial speech.
        </p>

        {loading ? (
          <div className="h-72 flex items-center justify-center text-xs text-slate-500 italic">
            Computing parity metrics across benchmark samples...
          </div>
        ) : (
          <div className="h-80 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={comparisonData} margin={{ top: 10, right: 20, left: -10, bottom: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis dataKey="name" stroke="#64748b" style={{ fontSize: '11px', fontWeight: 'bold' }} />
                <YAxis stroke="#64748b" unit="%" style={{ fontSize: '11px', fontWeight: '500' }} />
                <Tooltip
                  contentStyle={{ backgroundColor: '#0b0f19', border: '1px solid #1e293b', borderRadius: '12px' }}
                  formatter={(value) => [`${value}%`, '']}
                />
                <Legend wrapperStyle={{ fontSize: '12px', paddingTop: '15px' }} />
                <Bar dataKey="Stock Baseline FPR" fill="#ef4444" radius={[4, 4, 0, 0]} />
                <Bar dataKey="Context-Aware Fused FPR" fill="#10b981" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}
      </div>

      {/* Qualitative Case Study Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="glass-panel p-5 rounded-2xl border border-slate-800 bg-slate-950/40">
          <span className="text-xs font-bold text-white uppercase tracking-wider block mb-2 font-display">
            AAVE Dialect Case Study
          </span>
          <blockquote className="text-xs text-slate-300 italic mb-3 p-3 bg-slate-900/60 rounded-xl border border-slate-800">
            "They be wildin out here every single weekend honestly"
          </blockquote>
          <p className="text-[11px] text-slate-400 leading-relaxed">
            Standard toxic classifiers often mistake the African American Vernacular English (AAVE) habitual aspect <em>"be wildin"</em> for violent aggression. 
            The context layer identifies positive informal collocations and prevents false flagging.
          </p>
        </div>

        <div className="glass-panel p-5 rounded-2xl border border-slate-800 bg-slate-950/40">
          <span className="text-xs font-bold text-white uppercase tracking-wider block mb-2 font-display">
            Protected Identity Case Study
          </span>
          <blockquote className="text-xs text-slate-300 italic mb-3 p-3 bg-slate-900/60 rounded-xl border border-slate-800">
            "I am a proud Muslim student studying computer engineering"
          </blockquote>
          <p className="text-[11px] text-slate-400 leading-relaxed">
            Baseline models trained on toxic corpora frequently assign high risk to religious or LGBTQ+ tokens even in affirmative contexts.
            Our additive fusion verifies syntactic context to eliminate discriminatory identity false alarms.
          </p>
        </div>
      </div>
    </div>
  );
}
