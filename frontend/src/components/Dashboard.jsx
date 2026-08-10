import React, { useState, useEffect } from 'react';
import { Play, Pause, ChevronDown, ChevronUp, Check, EyeOff, AlertOctagon, HelpCircle, Send, ShieldAlert, Sparkles, Filter } from 'lucide-react';

export default function Dashboard({ posts, setPosts, handleAction, backendOnline, triggerNextPost }) {
  // Simulator configurations
  const [isPlaying, setIsPlaying] = useState(true);
  const [speed, setSpeed] = useState(3); // polling interval in seconds
  const [filter, setFilter] = useState('All');
  
  // Custom text test box state
  const [customText, setCustomText] = useState('');
  const [isTesting, setIsTesting] = useState(false);
  
  // Track open explanation panels
  const [expandedPostId, setExpandedPostId] = useState(null);

  // Poll simulator endpoints when playing
  useEffect(() => {
    if (!isPlaying || !backendOnline) return;

    const interval = setInterval(() => {
      triggerNextPost();
    }, speed * 1000);

    return () => clearInterval(interval);
  }, [isPlaying, speed, backendOnline]);

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
        // Prepend custom analyzed post to dashboard logs list
        setPosts(prev => [data, ...prev]);
        setCustomText('');
        setExpandedPostId(data.id); // auto-expand to show explanation
      }
    } catch (err) {
      console.error("Failed to analyze text:", err);
    } finally {
      setIsTesting(false);
    }
  };

  // Helper to highlight words using LIME weights
  const renderHighlightedText = (text, highlightedWords, limeWeights = {}) => {
    if (!highlightedWords || highlightedWords.length === 0) return <span>{text}</span>;

    // Split text into words and punctuation
    const tokens = text.split(/(\s+)/);
    
    return tokens.map((token, idx) => {
      // Strip punctuation to match keywords
      const cleanToken = token.toLowerCase().replace(/[^\w]/g, '');
      const weight = limeWeights[cleanToken];
      
      if (highlightedWords.includes(cleanToken) && weight > 0) {
        // Red highlight for cyberbullying trigger words
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
        // Green highlight for strong positive safe words
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

  // Compute live aggregates from posts list
  const totalScanned = posts.length;
  const cyberbullyingCount = posts.filter(p => p.prediction === 'Cyberbullying').length;
  const rate = totalScanned > 0 ? ((cyberbullyingCount / totalScanned) * 100).toFixed(1) : 0;
  const pendingCount = posts.filter(p => p.moderator_action === 'Pending').length;

  // Filter posts list
  const filteredPosts = posts.filter(post => {
    if (filter === 'All') return true;
    if (filter === 'Pending') return post.moderator_action === 'Pending';
    if (filter === 'Approved') return post.moderator_action === 'Approved';
    if (filter === 'Hidden') return post.moderator_action === 'Hidden';
    if (filter === 'Review') return post.moderator_action === 'Review';
    return true;
  });

  return (
    <div className="flex-1 overflow-y-auto bg-slate-900 bg-grid p-8">
      {/* Upper Statistics Widget */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
        <div className="glass-panel p-5 rounded-2xl border border-slate-800">
          <span className="text-xs font-semibold text-slate-400">Total Scanned Posts</span>
          <div className="text-3xl font-extrabold text-white mt-1 font-display">{totalScanned}</div>
          <span className="text-[10px] text-violet-400 mt-2 block font-medium">Updated live via simulation</span>
        </div>
        <div className="glass-panel p-5 rounded-2xl border border-slate-800">
          <span className="text-xs font-semibold text-slate-400">Cyberbullying Detected</span>
          <div className="text-3xl font-extrabold text-rose-500 mt-1 font-display">{cyberbullyingCount}</div>
          <span className="text-[10px] text-rose-400 mt-2 block font-medium">Toxicity Rate: {rate}%</span>
        </div>
        <div className="glass-panel p-5 rounded-2xl border border-slate-800">
          <span className="text-xs font-semibold text-slate-400">Pending Actions</span>
          <div className="text-3xl font-extrabold text-amber-500 mt-1 font-display">{pendingCount}</div>
          <span className="text-[10px] text-amber-400 mt-2 block font-medium">Requires moderator decisions</span>
        </div>
        <div className="glass-panel p-5 rounded-2xl border border-slate-800">
          <span className="text-xs font-semibold text-slate-400">Action Rate</span>
          <div className="text-3xl font-extrabold text-emerald-400 mt-1 font-display">
            {totalScanned > 0 ? (((totalScanned - pendingCount) / totalScanned) * 100).toFixed(0) : 0}%
          </div>
          <span className="text-[10px] text-emerald-400 mt-2 block font-medium">Decisions logged successfully</span>
        </div>
      </div>

      {/* Simulator Controller & Tester */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 mb-8">
        {/* Live Feed Control */}
        <div className="glass-panel p-6 rounded-2xl border border-slate-800/80 lg:col-span-1 flex flex-col justify-between">
          <div>
            <div className="flex items-center space-x-2.5 mb-2">
              <span className="relative flex h-2 w-2">
                <span className={`animate-ping absolute inline-flex h-full w-full rounded-full opacity-75 ${isPlaying ? 'bg-violet-400' : 'bg-slate-500'}`}></span>
                <span className={`relative inline-flex rounded-full h-2 w-2 ${isPlaying ? 'bg-violet-500' : 'bg-slate-600'}`}></span>
              </span>
              <h2 className="text-md font-bold tracking-tight text-white font-display">Live Feed Simulator</h2>
            </div>
            <p className="text-xs text-slate-400 mb-4">Simulates a social platform stream harvesting real-time tweets.</p>
          </div>

          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <button
                onClick={() => setIsPlaying(!isPlaying)}
                className={`flex items-center space-x-2 px-4 py-2.5 rounded-xl text-xs font-bold transition-all duration-200 ${
                  isPlaying 
                    ? 'bg-amber-600/10 text-amber-500 border border-amber-500/20 hover:bg-amber-600/20' 
                    : 'bg-violet-600 text-white hover:bg-violet-500 glow-purple'
                }`}
              >
                {isPlaying ? (
                  <>
                    <Pause className="h-4 w-4" /> <span>PAUSE FEED</span>
                  </>
                ) : (
                  <>
                    <Play className="h-4 w-4 animate-pulse" /> <span>RESUME FEED</span>
                  </>
                )}
              </button>
              
              <button
                onClick={triggerNextPost}
                disabled={!backendOnline}
                className="flex items-center space-x-1.5 px-3.5 py-2.5 rounded-xl text-xs font-bold bg-slate-800 text-slate-200 border border-slate-700 hover:bg-slate-750 transition-all cursor-pointer"
              >
                <Sparkles className="h-3.5 w-3.5" />
                <span>SCAN NEXT</span>
              </button>
            </div>

            <div className="pt-2">
              <div className="flex justify-between text-xs text-slate-400 mb-1">
                <span>Simulation Interval</span>
                <span className="font-bold text-violet-400">{speed} seconds</span>
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
            <h2 className="text-md font-bold tracking-tight text-white font-display">Sandbox Text Toxicity Analyzer</h2>
          </div>
          <p className="text-xs text-slate-400 mb-4">Input custom phrases to inspect classification accuracy and review tokenized LIME word weight scores.</p>
          
          <form onSubmit={handleTestText} className="flex gap-3">
            <input
              type="text"
              value={customText}
              onChange={(e) => setCustomText(e.target.value)}
              placeholder="E.g., Stop writing ugly garbage, you look stupid!"
              className="flex-1 bg-slate-950/80 border border-slate-800/80 rounded-xl px-4 py-3 text-sm focus:outline-none focus:border-violet-500 text-slate-200 transition-all"
              disabled={!backendOnline || isTesting}
            />
            <button
              type="submit"
              disabled={!backendOnline || isTesting || !customText.trim()}
              className="flex items-center space-x-2 px-5 py-3 rounded-xl bg-violet-600 text-white font-bold text-xs hover:bg-violet-500 transition-all disabled:opacity-40 disabled:cursor-not-allowed cursor-pointer shadow-lg shadow-violet-950/20"
            >
              <span>{isTesting ? 'ANALYZING...' : 'RUN AI'}</span>
              <Send className="h-3.5 w-3.5" />
            </button>
          </form>
        </div>
      </div>

      {/* Main Dashboard Panel Controls (Filters) */}
      <div className="flex items-center justify-between border-b border-slate-800 pb-4 mb-6">
        <div className="flex items-center space-x-2">
          <Filter className="h-4 w-4 text-violet-400" />
          <span className="text-sm font-bold text-white font-display">Moderation Queue</span>
        </div>
        <div className="flex bg-slate-950 p-1.5 rounded-xl border border-slate-800 text-xs">
          {['All', 'Pending', 'Approved', 'Hidden', 'Review'].map(f => (
            <button
              key={f}
              onClick={() => setFilter(f)}
              className={`px-4 py-2 rounded-lg font-semibold transition-all duration-200 ${
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

      {/* Posts Live Stream Feed */}
      <div className="space-y-4">
        {filteredPosts.length === 0 ? (
          <div className="glass-panel text-center py-16 rounded-2xl border border-slate-800 text-slate-500 text-sm">
            No posts found matching the filter "{filter}". Start the simulator feed or enter custom text.
          </div>
        ) : (
          filteredPosts.map((post) => {
            const isBullying = post.prediction === 'Cyberbullying';
            const isActionTaken = post.moderator_action !== 'Pending';
            const isExpanded = expandedPostId === post.id;
            
            return (
              <div 
                key={post.id} 
                className={`glass-panel-interactive rounded-2xl overflow-hidden border ${
                  isBullying 
                    ? 'border-rose-900/10 hover:border-rose-900/30' 
                    : 'border-emerald-900/10 hover:border-emerald-900/30'
                }`}
              >
                {/* Header Section */}
                <div className="p-6 flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-slate-900 bg-slate-950/40">
                  <div className="flex items-center space-x-3">
                    <div className="h-9 w-9 rounded-xl bg-slate-800 flex items-center justify-center font-bold text-violet-400 font-display border border-slate-700">
                      {post.user_id ? post.user_id.slice(-2).toUpperCase() : 'AN'}
                    </div>
                    <div>
                      <div className="flex items-center space-x-2">
                        <span className="font-bold text-sm text-slate-100 font-display">{post.username || '@anonymous'}</span>
                        <span className="px-2 py-0.5 rounded-full text-[9px] font-extrabold bg-slate-900 border border-slate-800 text-slate-400 tracking-wider">
                          {post.user_id || 'user_guest'}
                        </span>
                      </div>
                      <span className="text-[10px] text-slate-500 font-medium">
                        {new Date(post.timestamp).toLocaleTimeString()} • {new Date(post.timestamp).toLocaleDateString()}
                      </span>
                    </div>
                  </div>

                  {/* AI Prediction Badges */}
                  <div className="flex items-center space-x-4">
                    <div className="flex items-center space-x-2.5">
                      <span className="text-xs text-slate-400 font-semibold">AI Assessment:</span>
                      <span className={`flex items-center space-x-1.5 px-3 py-1 rounded-full text-xs font-bold border tracking-wide ${
                        isBullying
                          ? 'bg-rose-500/10 text-rose-400 border-rose-500/25 glow-red'
                          : 'bg-emerald-500/10 text-emerald-400 border-emerald-500/25 glow-green'
                      }`}>
                        {isBullying ? <AlertOctagon className="h-3.5 w-3.5" /> : <Check className="h-3.5 w-3.5" />}
                        <span className="font-display uppercase tracking-wider">{post.prediction}</span>
                      </span>
                    </div>

                    {/* Confidence Score Bar */}
                    <div className="flex items-center space-x-2 w-32 md:w-40">
                      <div className="flex-1 h-2 bg-slate-800 rounded-full overflow-hidden">
                        <div 
                          className={`h-full rounded-full transition-all duration-500 ${isBullying ? 'bg-rose-500 glow-red' : 'bg-emerald-500 glow-green'}`} 
                          style={{ width: `${post.confidence}%` }}
                        />
                      </div>
                      <span className={`text-xs font-bold font-display ${isBullying ? 'text-rose-400' : 'text-emerald-400'}`}>
                        {Math.round(post.confidence)}%
                      </span>
                    </div>
                  </div>
                </div>

                {/* Content Section */}
                <div className="p-6">
                  <blockquote className="text-md text-slate-200 leading-relaxed font-medium mb-4">
                    {renderHighlightedText(post.text, post.highlighted_words, post.lime_weights)}
                  </blockquote>
                  
                  {/* Action Status Badge (If Decided) */}
                  {isActionTaken && (
                    <div className="mb-4">
                      <span className={`inline-flex items-center space-x-1.5 px-2.5 py-1 rounded-lg text-[10px] font-extrabold uppercase border ${
                        post.moderator_action === 'Approved' 
                          ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20' 
                          : post.moderator_action === 'Hidden' 
                            ? 'bg-rose-500/10 text-rose-400 border-rose-500/20'
                            : 'bg-amber-500/10 text-amber-400 border-amber-500/20'
                      }`}>
                        <span>Moderator Action: {post.moderator_action}</span>
                      </span>
                    </div>
                  )}

                  {/* Expand / Collapse Button for Explainability Section */}
                  <div className="flex items-center justify-between">
                    <button
                      onClick={() => setExpandedPostId(isExpanded ? null : post.id)}
                      className="flex items-center space-x-1.5 text-slate-400 hover:text-slate-200 text-xs font-semibold transition-all"
                    >
                      <HelpCircle className="h-4 w-4 text-violet-400" />
                      <span>Why was this flagged? View AI LIME Explainability</span>
                      {isExpanded ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
                    </button>

                    {/* Moderator Action Buttons */}
                    <div className="flex gap-2">
                      <button
                        onClick={() => handleAction(post.id, 'Approved')}
                        className={`flex items-center space-x-1.5 px-3 py-2 rounded-xl text-xs font-bold border transition-all ${
                          post.moderator_action === 'Approved'
                            ? 'bg-emerald-500 text-slate-950 border-emerald-500'
                            : 'bg-slate-950 text-slate-300 border-slate-800 hover:bg-emerald-950/15 hover:text-emerald-400 hover:border-emerald-500/20'
                        }`}
                      >
                        <Check className="h-3.5 w-3.5" />
                        <span>APPROVE</span>
                      </button>
                      <button
                        onClick={() => handleAction(post.id, 'Hidden')}
                        className={`flex items-center space-x-1.5 px-3 py-2 rounded-xl text-xs font-bold border transition-all ${
                          post.moderator_action === 'Hidden'
                            ? 'bg-rose-500 text-slate-950 border-rose-500'
                            : 'bg-slate-950 text-slate-300 border-slate-800 hover:bg-rose-950/15 hover:text-rose-400 hover:border-rose-500/20'
                        }`}
                      >
                        <EyeOff className="h-3.5 w-3.5" />
                        <span>HIDE</span>
                      </button>
                      <button
                        onClick={() => handleAction(post.id, 'Review')}
                        className={`flex items-center space-x-1.5 px-3 py-2 rounded-xl text-xs font-bold border transition-all ${
                          post.moderator_action === 'Review'
                            ? 'bg-amber-500 text-slate-950 border-amber-500'
                            : 'bg-slate-950 text-slate-300 border-slate-800 hover:bg-amber-950/15 hover:text-amber-400 hover:border-amber-500/20'
                        }`}
                      >
                        <span>REVIEW</span>
                      </button>
                    </div>
                  </div>
                </div>

                {/* Collapsible LIME Explainability Panel */}
                {isExpanded && (
                  <div className="bg-slate-950/60 border-t border-slate-900 p-6 transition-all duration-300">
                    <h3 className="text-xs font-bold text-white uppercase tracking-wider mb-4 flex items-center space-x-2">
                      <span className="h-1.5 w-1.5 rounded-full bg-violet-400"></span>
                      <span>LIME Feature Importance Analysis</span>
                    </h3>
                    
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                      {/* Bar charts of LIME word attributions */}
                      <div className="space-y-3.5">
                        <span className="text-[11px] font-bold text-slate-400 uppercase tracking-widest block mb-2">Word Attribution Strengths</span>
                        {Object.entries(post.lime_weights || {}).length === 0 ? (
                          <div className="text-xs text-slate-500">No feature weights generated.</div>
                        ) : (
                          Object.entries(post.lime_weights)
                            .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
                            .slice(0, 5)
                            .map(([word, weight]) => {
                              const pct = Math.abs(weight) * 100;
                              const isPositive = weight > 0;
                              return (
                                <div key={word} className="flex items-center text-xs">
                                  <span className="w-24 font-mono font-bold text-slate-300 truncate">{word}</span>
                                  <div className="flex-1 flex items-center justify-end relative h-5 bg-slate-900/60 border border-slate-800 rounded overflow-hidden">
                                    {/* Bar representation */}
                                    <div 
                                      className={`h-full rounded-sm absolute opacity-80 ${isPositive ? 'right-[50%] bg-rose-500/60 glow-red' : 'left-[50%] bg-emerald-500/60 glow-green'}`}
                                      style={{ 
                                        width: `${pct / 2}%`,
                                      }}
                                    />
                                    {/* Centered zero marker line */}
                                    <div className="absolute left-[50%] h-full w-[1px] bg-slate-700" />
                                    
                                    {/* Value label */}
                                    <span className={`px-2 z-10 font-bold font-mono text-[10px] ${isPositive ? 'text-rose-400' : 'text-emerald-400'}`}>
                                      {isPositive ? '+' : ''}{Math.round(weight * 100)}%
                                    </span>
                                  </div>
                                </div>
                              );
                            })
                        )}
                        <div className="flex justify-between text-[9px] text-slate-500 px-1 border-t border-slate-900 pt-2 font-mono">
                          <span>Supports "Safe"</span>
                          <span>Neutral (0.0)</span>
                          <span>Supports "Bullying"</span>
                        </div>
                      </div>

                      {/* Bulleted natural explanations */}
                      <div className="glass-panel p-4.5 rounded-xl border border-slate-900 bg-slate-950 flex flex-col justify-center">
                        <span className="text-[11px] font-bold text-slate-400 uppercase tracking-widest block mb-3">Interpretation Logs</span>
                        <ul className="space-y-2.5 text-xs">
                          {post.explanations && post.explanations.map((exp, idx) => (
                            <li key={idx} className="flex items-start space-x-2 text-slate-300">
                              <span className={`h-1.5 w-1.5 rounded-full mt-1.5 shrink-0 ${exp.includes('Safe') ? 'bg-emerald-500 glow-green' : 'bg-rose-500 glow-red'}`} />
                              <span>{exp}</span>
                            </li>
                          ))}
                          {(!post.explanations || post.explanations.length === 0) && (
                            <li className="text-slate-500 italic">No explanations available.</li>
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
