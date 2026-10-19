import React, { useState, useEffect, useRef } from 'react';
import { 
  Play, Pause, ChevronDown, ChevronUp, Check, EyeOff, AlertOctagon, 
  HelpCircle, Send, ShieldAlert, Sparkles, Filter, Radio, ArrowRight, Activity 
} from 'lucide-react';

export default function Dashboard({ 
  posts, 
  setPosts, 
  handleAction, 
  backendOnline, 
  triggerNextPost, 
  setActiveTab,
  refreshAnalytics 
}) {
  // Simulator configurations
  const [isPlaying, setIsPlaying] = useState(true);
  const [speed, setSpeed] = useState(3);
  const [filter, setFilter] = useState('All');
  const [wsConnected, setWsConnected] = useState(false);
  
  // Custom text test box state
  const [customText, setCustomText] = useState('');
  const [isTesting, setIsTesting] = useState(false);
  
  // Track open explanation panels
  const [expandedPostId, setExpandedPostId] = useState(null);

  const wsRef = useRef(null);

  // WebSocket Live Stream Connection
  useEffect(() => {
    if (!backendOnline) return;

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    // In dev mode with Vite proxy or direct port mapping
    const wsUrl = window.location.port === '5173'
      ? 'ws://127.0.0.1:8000/ws/feed'
      : `${protocol}//${window.location.host}/ws/feed`;

    let ws;
    try {
      ws = new WebSocket(wsUrl);
      wsRef.current = ws;

      ws.onopen = () => {
        setWsConnected(true);
        if (isPlaying) {
          ws.send(JSON.stringify({ action: 'start', interval: speed }));
        }
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          if (data.id && data.text) {
            setPosts(prev => [data, ...prev]);
            if (refreshAnalytics) refreshAnalytics();
          }
        } catch (e) {
          // ignore non-event control frames
        }
      };

      ws.onclose = () => setWsConnected(false);
      ws.onerror = () => setWsConnected(false);
    } catch (e) {
      setWsConnected(false);
    }

    return () => {
      if (ws) ws.close();
    };
  }, [backendOnline]);

  // Sync stream state with WebSocket or HTTP fallback
  useEffect(() => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ 
        action: isPlaying ? 'start' : 'stop', 
        interval: speed 
      }));
    } else if (!wsConnected && isPlaying && backendOnline) {
      // HTTP Polling Fallback if WebSocket is offline
      const interval = setInterval(() => {
        triggerNextPost();
      }, speed * 1000);
      return () => clearInterval(interval);
    }
  }, [isPlaying, speed, wsConnected, backendOnline]);

  // Handle custom manual text analysis
  const handleTestText = async (e) => {
    e.preventDefault();
    if (!customText.trim() || isTesting) return;
    
    setIsTesting(true);
    try {
      const res = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text: customText,
          user_id: 'user_tester',
          username: '@guest_tester'
        })
      });
      if (res.ok) {
        const data = await res.json();
        setPosts(prev => [data, ...prev]);
        setCustomText('');
        setExpandedPostId(data.id);
        if (refreshAnalytics) refreshAnalytics();
      }
    } catch (err) {
      console.error("Failed to analyze text:", err);
    } finally {
      setIsTesting(false);
    }
  };

  // Helper to highlight words using LIME weights
  const renderHighlightedText = (text, highlightedWords = [], limeWeights = {}) => {
    if (!highlightedWords || highlightedWords.length === 0) return <span>{text}</span>;

    const tokens = text.split(/(\s+)/);
    return tokens.map((token, idx) => {
      const cleanToken = token.toLowerCase().replace(/[^\w]/g, '');
      const weight = limeWeights[cleanToken];
      
      if (highlightedWords.includes(cleanToken) && weight > 0) {
        return (
          <span 
            key={idx} 
            className="px-1.5 py-0.5 rounded font-semibold text-rose-300 border border-rose-500/20 bg-rose-950/40 glow-red animate-pulse cursor-help"
            title={`LIME score: +${Math.round(weight * 100)}%`}
          >
            {token}
          </span>
        );
      } else if (weight < -0.05) {
        return (
          <span 
            key={idx} 
            className="px-1.5 py-0.5 rounded font-semibold text-emerald-300 border border-emerald-500/20 bg-emerald-950/40 glow-green cursor-help"
            title={`LIME score: ${Math.round(weight * 100)}%`}
          >
            {token}
          </span>
        );
      }
      return <span key={idx}>{token}</span>;
    });
  };

  // Live aggregates
  const totalScanned = posts.length;
  const cyberbullyingCount = posts.filter(p => p.prediction === 'Cyberbullying').length;
  const rate = totalScanned > 0 ? ((cyberbullyingCount / totalScanned) * 100).toFixed(1) : 0;
  const uncertainCount = posts.filter(p => p.prediction === 'Uncertain' || p.escalate_to_human).length;
  const pendingCount = posts.filter(p => p.moderator_action === 'Pending').length;

  const filteredPosts = posts.filter(post => {
    if (filter === 'All') return true;
    if (filter === 'Uncertain') return post.prediction === 'Uncertain' || post.escalate_to_human;
    if (filter === 'Pending') return post.moderator_action === 'Pending';
    if (filter === 'Approved') return post.moderator_action === 'Approved';
    if (filter === 'Hidden') return post.moderator_action === 'Hidden';
    return true;
  });

  return (
    <div className="flex-1 overflow-y-auto bg-slate-900 bg-grid p-8">
      {/* Upper Statistics Widget */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
        <div className="glass-panel p-5 rounded-2xl border border-slate-800">
          <span className="text-xs font-semibold text-slate-400">Total Scanned Posts</span>
          <div className="text-3xl font-extrabold text-white mt-1 font-display">{totalScanned}</div>
          <div className="flex items-center space-x-1.5 mt-2">
            <Radio className={`h-2.5 w-2.5 ${wsConnected ? 'text-emerald-400 animate-ping' : 'text-slate-500'}`} />
            <span className="text-[10px] text-slate-400 font-mono">
              {wsConnected ? 'STREAMING VIA WEBSOCKET' : 'HTTP POLLING FALLBACK'}
            </span>
          </div>
        </div>

        <div className="glass-panel p-5 rounded-2xl border border-slate-800">
          <span className="text-xs font-semibold text-slate-400">Cyberbullying Confirmed</span>
          <div className="text-3xl font-extrabold text-rose-500 mt-1 font-display">{cyberbullyingCount}</div>
          <span className="text-[10px] text-rose-400 mt-2 block font-medium">Toxicity Rate: {rate}%</span>
        </div>

        <div className="glass-panel p-5 rounded-2xl border border-slate-800">
          <span className="text-xs font-semibold text-slate-400">Uncertain / Escalated</span>
          <div className="text-3xl font-extrabold text-amber-400 mt-1 font-display">{uncertainCount}</div>
          <span className="text-[10px] text-amber-400/80 mt-2 block font-medium">Confidence in [40%, 70%] band</span>
        </div>

        <div className="glass-panel p-5 rounded-2xl border border-slate-800">
          <span className="text-xs font-semibold text-slate-400">Pending Human Action</span>
          <div className="text-3xl font-extrabold text-violet-400 mt-1 font-display">{pendingCount}</div>
          <span className="text-[10px] text-slate-400 mt-2 block font-medium">Awaiting moderation log</span>
        </div>
      </div>

      {/* Simulator Controller & Sandbox Tester */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 mb-8">
        {/* Live Feed Control */}
        <div className="glass-panel p-6 rounded-2xl border border-slate-800/80 lg:col-span-1 flex flex-col justify-between">
          <div>
            <div className="flex items-center space-x-2.5 mb-2">
              <span className="relative flex h-2 w-2">
                <span className={`animate-ping absolute inline-flex h-full w-full rounded-full opacity-75 ${isPlaying ? 'bg-violet-400' : 'bg-slate-500'}`}></span>
                <span className={`relative inline-flex rounded-full h-2 w-2 ${isPlaying ? 'bg-violet-500' : 'bg-slate-600'}`}></span>
              </span>
              <h2 className="text-md font-bold tracking-tight text-white font-display">Ingestion Stream Pipeline</h2>
            </div>
            <p className="text-xs text-slate-400 mb-4">
              Continuous asynchronous message stream feeding the Additive Fusion Engine.
            </p>
          </div>

          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <button
                onClick={() => setIsPlaying(!isPlaying)}
                className={`flex items-center space-x-2 px-4 py-2.5 rounded-xl text-xs font-bold transition-all duration-200 cursor-pointer ${
                  isPlaying 
                    ? 'bg-amber-600/10 text-amber-500 border border-amber-500/20 hover:bg-amber-600/20' 
                    : 'bg-violet-600 text-white hover:bg-violet-500 glow-purple'
                }`}
              >
                {isPlaying ? (
                  <>
                    <Pause className="h-4 w-4" /> <span>PAUSE STREAM</span>
                  </>
                ) : (
                  <>
                    <Play className="h-4 w-4 animate-pulse" /> <span>RESUME STREAM</span>
                  </>
                )}
              </button>
              
              <button
                onClick={triggerNextPost}
                disabled={!backendOnline}
                className="flex items-center space-x-1.5 px-3.5 py-2.5 rounded-xl text-xs font-bold bg-slate-800 text-slate-200 border border-slate-700 hover:bg-slate-750 transition-all cursor-pointer"
              >
                <Sparkles className="h-3.5 w-3.5" />
                <span>STEP NEXT</span>
              </button>
            </div>

            <div className="pt-2">
              <div className="flex justify-between text-xs text-slate-400 mb-1">
                <span>Stream Ingestion Interval</span>
                <span className="font-bold text-violet-400 font-mono">{speed} seconds</span>
              </div>
              <input
                type="range"
                min="1"
                max="8"
                value={speed}
                onChange={(e) => setSpeed(Number(e.target.value))}
                className="w-full h-1 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-violet-600"
              />
            </div>
          </div>
        </div>

        {/* Text Toxicity Testing Tool */}
        <div className="glass-panel p-6 rounded-2xl border border-slate-800/80 lg:col-span-2">
          <div className="flex items-center space-x-2.5 mb-2">
            <ShieldAlert className="h-5 w-5 text-violet-500" />
            <h2 className="text-md font-bold tracking-tight text-white font-display">
              Sandbox Text Toxicity & Fusion Analyzer
            </h2>
          </div>
          <p className="text-xs text-slate-400 mb-4">
            Test phrases with negations, friendly slang, or sarcasm to inspect the Additive Clamped Fusion formula in real time.
          </p>
          
          <form onSubmit={handleTestText} className="flex gap-3">
            <input
              type="text"
              value={customText}
              onChange={(e) => setCustomText(e.target.value)}
              placeholder="E.g., You are not stupid, don't worry! OR Great speech, said no one ever."
              className="flex-1 bg-slate-950/80 border border-slate-800 rounded-xl px-4 py-3 text-sm focus:outline-none focus:border-violet-500 text-slate-200 transition-all font-mono text-xs"
              disabled={!backendOnline || isTesting}
            />
            <button
              type="submit"
              disabled={!backendOnline || isTesting || !customText.trim()}
              className="flex items-center space-x-2 px-5 py-3 rounded-xl bg-violet-600 text-white font-bold text-xs hover:bg-violet-500 transition-all disabled:opacity-40 cursor-pointer shadow-lg shadow-violet-950/20"
            >
              <span>{isTesting ? 'ANALYZING...' : 'RUN FUSION'}</span>
              <Send className="h-3.5 w-3.5" />
            </button>
          </form>
        </div>
      </div>

      {/* Main Stream Controls (Filters) */}
      <div className="flex items-center justify-between border-b border-slate-800 pb-4 mb-6">
        <div className="flex items-center space-x-2">
          <Filter className="h-4 w-4 text-violet-400" />
          <span className="text-sm font-bold text-white font-display">Streaming Moderation Feed</span>
        </div>
        <div className="flex bg-slate-950 p-1.5 rounded-xl border border-slate-800 text-xs">
          {['All', 'Uncertain', 'Pending', 'Approved', 'Hidden'].map(f => (
            <button
              key={f}
              onClick={() => setFilter(f)}
              className={`px-4 py-2 rounded-lg font-semibold transition-all duration-200 cursor-pointer ${
                filter === f
                  ? 'bg-slate-800 text-white shadow-sm'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              {f}
            </button>
          ))}
        </div>
      </div>

      {/* Posts Stream Feed */}
      <div className="space-y-4">
        {filteredPosts.length === 0 ? (
          <div className="glass-panel text-center py-16 rounded-2xl border border-slate-800 text-slate-500 text-sm">
            No posts found matching filter "{filter}". Start the stream or test custom text.
          </div>
        ) : (
          filteredPosts.map((post) => {
            const isBullying = post.prediction === 'Cyberbullying';
            const isUncertain = post.prediction === 'Uncertain' || post.escalate_to_human;
            const isSafe = post.prediction === 'Safe';
            const isExpanded = expandedPostId === post.id;
            
            return (
              <div 
                key={post.id} 
                className={`glass-panel-interactive rounded-2xl overflow-hidden border ${
                  isBullying 
                    ? 'border-rose-900/20 hover:border-rose-900/40' 
                    : isUncertain
                      ? 'border-amber-900/20 hover:border-amber-900/40'
                      : 'border-emerald-900/20 hover:border-emerald-900/40'
                }`}
              >
                {/* Header Section */}
                <div className="p-5 flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-slate-900 bg-slate-950/50">
                  <div className="flex items-center space-x-3">
                    <div className="h-9 w-9 rounded-xl bg-slate-800 flex items-center justify-center font-bold text-violet-400 font-display border border-slate-700">
                      {(post.author || post.username || 'AN').slice(1, 3).toUpperCase()}
                    </div>
                    <div>
                      <div className="flex items-center space-x-2">
                        <span className="font-bold text-sm text-slate-100 font-display">
                          {post.author || post.username || '@anonymous'}
                        </span>
                        <span className="text-slate-600">→</span>
                        <span className="text-xs font-bold text-violet-400 font-display">
                          {post.target_user || '@anonymous'}
                        </span>
                      </div>
                      <span className="text-[10px] text-slate-500 font-medium">
                        {post.timestamp ? new Date(post.timestamp).toLocaleTimeString() : ''}
                      </span>
                    </div>
                  </div>

                  {/* Additive Clamped Fusion Badge & Assessment */}
                  <div className="flex flex-wrap items-center gap-3">
                    {/* Additive Breakdown Indicator */}
                    <div className="px-3 py-1 rounded-xl bg-slate-900 border border-slate-800 flex items-center space-x-2 text-[11px] font-mono">
                      <span className="text-slate-400">P_tox:</span>
                      <strong className="text-rose-400">{Math.round((post.p_toxic || post.toxicity_score || 0) * 100)}%</strong>
                      <span className="text-slate-600">+</span>
                      <span className="text-slate-400">Δ_ctx:</span>
                      <strong className="text-violet-400">{(post.total_context_delta || post.context_delta || 0) >= 0 ? '+' : ''}{(post.total_context_delta || post.context_delta || 0).toFixed(2)}</strong>
                      <span className="text-slate-600">=</span>
                      <span className="text-white font-bold">{Math.round(post.confidence || (post.final_score * 100) || 0)}%</span>
                    </div>

                    {/* Result Badge */}
                    <span className={`flex items-center space-x-1.5 px-3 py-1 rounded-full text-xs font-bold border tracking-wide font-display uppercase ${
                      isBullying
                        ? 'bg-rose-500/10 text-rose-400 border-rose-500/25 glow-red'
                        : isUncertain
                          ? 'bg-amber-500/10 text-amber-400 border-amber-500/25 glow-orange'
                          : 'bg-emerald-500/10 text-emerald-400 border-emerald-500/25 glow-green'
                    }`}>
                      {isBullying ? <AlertOctagon className="h-3.5 w-3.5" /> : isUncertain ? <ShieldAlert className="h-3.5 w-3.5" /> : <Check className="h-3.5 w-3.5" />}
                      <span>{post.prediction}</span>
                    </span>
                  </div>
                </div>

                {/* Content Section */}
                <div className="p-6">
                  <blockquote className="text-sm text-slate-200 leading-relaxed font-medium mb-4">
                    {renderHighlightedText(post.text, post.highlighted_words, post.lime_weights)}
                  </blockquote>
                  
                  {/* Escalation Notice Banner */}
                  {isUncertain && (
                    <div className="mb-4 p-3 rounded-xl bg-amber-500/10 border border-amber-500/20 flex items-center justify-between">
                      <div className="flex items-center space-x-2 text-xs text-amber-300">
                        <ShieldAlert className="h-4 w-4 text-amber-400 shrink-0" />
                        <span>
                          <strong>Confidence in uncertain band (40%–70%):</strong> Auto-escalated to human review queue.
                        </span>
                      </div>
                      {setActiveTab && (
                        <button
                          onClick={() => setActiveTab('queue')}
                          className="flex items-center space-x-1 text-xs font-bold text-amber-400 hover:text-white transition-all cursor-pointer shrink-0 ml-3"
                        >
                          <span>Open in Review Queue</span>
                          <ArrowRight className="h-3.5 w-3.5" />
                        </button>
                      )}
                    </div>
                  )}

                  {/* Expand / Collapse Explainability Button */}
                  <div className="flex items-center justify-between pt-1">
                    <button
                      onClick={() => setExpandedPostId(isExpanded ? null : post.id)}
                      className="flex items-center space-x-1.5 text-slate-400 hover:text-slate-200 text-xs font-semibold transition-all cursor-pointer"
                    >
                      <HelpCircle className="h-4 w-4 text-violet-400" />
                      <span>Inspect Score Fusion & LIME Weights</span>
                      {isExpanded ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
                    </button>

                    {/* Action buttons */}
                    <div className="flex gap-2">
                      <button
                        onClick={() => handleAction(post.id, 'Approved')}
                        className={`flex items-center space-x-1.5 px-3 py-1.5 rounded-xl text-xs font-bold border transition-all cursor-pointer ${
                          post.moderator_action === 'Approved'
                            ? 'bg-emerald-500 text-slate-950 border-emerald-500'
                            : 'bg-slate-950 text-slate-300 border-slate-800 hover:text-emerald-400'
                        }`}
                      >
                        <Check className="h-3 w-3" />
                        <span>APPROVE</span>
                      </button>
                      <button
                        onClick={() => handleAction(post.id, 'Hidden')}
                        className={`flex items-center space-x-1.5 px-3 py-1.5 rounded-xl text-xs font-bold border transition-all cursor-pointer ${
                          post.moderator_action === 'Hidden'
                            ? 'bg-rose-500 text-slate-950 border-rose-500'
                            : 'bg-slate-950 text-slate-300 border-slate-800 hover:text-rose-400'
                        }`}
                      >
                        <EyeOff className="h-3 w-3" />
                        <span>HIDE</span>
                      </button>
                    </div>
                  </div>
                </div>

                {/* Collapsible Explainability & Fusion Panel */}
                {isExpanded && (
                  <div className="bg-slate-950/70 border-t border-slate-900 p-6 transition-all duration-300">
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                      {/* Additive Clamped Equation Details */}
                      <div className="space-y-3">
                        <span className="text-[11px] font-bold text-slate-400 uppercase tracking-widest block font-display">
                          Additive Clamped Fusion Breakdown
                        </span>
                        
                        <div className="p-3 bg-slate-900/60 rounded-xl border border-slate-800 font-mono text-xs text-slate-200">
                          {post.formula || `clamp(P_tox + sum(deltas), 0.0, 1.0) = ${post.confidence}%`}
                        </div>

                        {/* Individual Delta Badges */}
                        <div className="grid grid-cols-2 gap-2 text-[11px] font-mono">
                          <div className="p-2 rounded bg-slate-900 border border-slate-800">
                            <span className="text-slate-500 block">Δ_negation</span>
                            <span className="text-slate-200 font-bold">{post.delta_neg !== undefined ? `${post.delta_neg > 0 ? '+' : ''}${post.delta_neg}` : '0.00'}</span>
                          </div>
                          <div className="p-2 rounded bg-slate-900 border border-slate-800">
                            <span className="text-slate-500 block">Δ_slang</span>
                            <span className="text-slate-200 font-bold">{post.delta_slang !== undefined ? `${post.delta_slang > 0 ? '+' : ''}${post.delta_slang}` : '0.00'}</span>
                          </div>
                          <div className="p-2 rounded bg-slate-900 border border-slate-800">
                            <span className="text-slate-500 block">Δ_sarcasm</span>
                            <span className="text-slate-200 font-bold">{post.delta_sarcasm !== undefined ? `${post.delta_sarcasm > 0 ? '+' : ''}${post.delta_sarcasm}` : '0.00'}</span>
                          </div>
                          <div className="p-2 rounded bg-slate-900 border border-slate-800">
                            <span className="text-slate-500 block">Δ_target</span>
                            <span className="text-slate-200 font-bold">{post.delta_target !== undefined ? `${post.delta_target > 0 ? '+' : ''}${post.delta_target}` : '0.00'}</span>
                          </div>
                        </div>
                      </div>

                      {/* Natural Language Linguistic Context Explanations */}
                      <div className="p-4 rounded-xl bg-slate-900/50 border border-slate-850 flex flex-col justify-center">
                        <span className="text-[11px] font-bold text-slate-400 uppercase tracking-widest block mb-2 font-display">
                          Context Interpretation Log
                        </span>
                        <ul className="space-y-2 text-xs text-slate-300">
                          {post.explanations && post.explanations.length > 0 ? (
                            post.explanations.map((exp, i) => (
                              <li key={i} className="flex items-start space-x-2">
                                <span className="h-1.5 w-1.5 rounded-full bg-violet-400 mt-1.5 shrink-0" />
                                <span>{exp}</span>
                              </li>
                            ))
                          ) : (
                            <li className="text-slate-500 italic">No contextual adjustments triggered.</li>
                          )}
                        </ul>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            );
          })
        )}
      </div>
    </div>
  );
}
