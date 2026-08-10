import React from 'react';
import { Shield, LayoutDashboard, BarChart3, Share2, ToggleLeft, ToggleRight, Radio, ServerCrash } from 'lucide-react';

export default function Sidebar({ activeTab, setActiveTab, systemStatus, toggleAiMode, backendOnline }) {
  const menuItems = [
    { id: 'dashboard', label: 'Live Feed', icon: LayoutDashboard },
    { id: 'analytics', label: 'Analytics Panel', icon: BarChart3 },
    { id: 'network', label: 'Bullying Graph', icon: Share2 },
  ];

  return (
    <aside className="w-80 border-r border-slate-800 bg-slate-950 flex flex-col h-screen sticky top-0">
      {/* Brand Logo & Header */}
      <div className="p-6 border-b border-slate-900 flex items-center space-x-3">
        <div className="bg-violet-600/10 p-2.5 rounded-xl border border-violet-500/20 glow-purple">
          <Shield className="h-6 w-6 text-violet-500 animate-pulse" />
        </div>
        <div>
          <h1 className="text-xl font-bold tracking-tight font-display text-white">CYBER SHIELD</h1>
          <span className="text-[10px] uppercase font-bold text-violet-400 tracking-widest">B.Tech Project MVP</span>
        </div>
      </div>

      {/* Connection Status Header */}
      <div className="px-6 py-3 border-b border-slate-900 bg-slate-950/50 flex items-center justify-between text-xs">
        <span className="text-slate-400 font-medium">Server Connection</span>
        <div className="flex items-center space-x-1.5">
          {backendOnline ? (
            <>
              <Radio className="h-3 w-3 text-emerald-500 animate-pulse" />
              <span className="text-emerald-500 font-bold">ONLINE</span>
            </>
          ) : (
            <>
              <ServerCrash className="h-3 w-3 text-rose-500" />
              <span className="text-rose-500 font-bold">OFFLINE</span>
            </>
          )}
        </div>
      </div>

      {/* Main Navigation Menu */}
      <nav className="flex-1 px-4 py-6 space-y-1.5 overflow-y-auto">
        {menuItems.map((item) => {
          const Icon = item.icon;
          const isActive = activeTab === item.id;
          return (
            <button
              key={item.id}
              onClick={() => setActiveTab(item.id)}
              className={`w-full flex items-center space-x-3.5 px-4 py-3.5 rounded-xl text-sm font-medium transition-all duration-200 ${
                isActive
                  ? 'bg-gradient-to-r from-violet-600/20 to-pink-600/10 text-white border border-violet-500/25 shadow-lg shadow-violet-950/20'
                  : 'text-slate-400 hover:bg-slate-900/60 hover:text-slate-200 border border-transparent'
              }`}
            >
              <Icon className={`h-5 w-5 ${isActive ? 'text-violet-400' : 'text-slate-400'}`} />
              <span className="font-display">{item.label}</span>
            </button>
          );
        })}
      </nav>

      {/* Bottom Settings Panel: AI Model Toggle */}
      <div className="p-4 border-t border-slate-900 bg-slate-950">
        <div className="glass-panel p-4 rounded-2xl border border-slate-800/80">
          <div className="flex items-center justify-between mb-3">
            <span className="text-xs font-bold uppercase tracking-wider text-slate-400">AI Model Mode</span>
            <span className={`px-2 py-0.5 rounded text-[10px] font-bold ${
              systemStatus.use_real_model 
                ? 'bg-rose-500/10 text-rose-400 border border-rose-500/25 glow-red' 
                : 'bg-violet-500/10 text-violet-400 border border-violet-500/25 glow-purple'
            }`}>
              {systemStatus.use_real_model ? 'BERT Mode' : 'Demo Mode'}
            </span>
          </div>

          <p className="text-xs text-slate-400 leading-relaxed mb-4">
            {systemStatus.use_real_model 
              ? 'Using BERT Transformer (unitary/toxic-bert) for deep contextual classification.' 
              : 'Using fast keyword matching classifier for ultra-low latency representation.'}
          </p>

          <button
            onClick={toggleAiMode}
            disabled={!backendOnline || (!systemStatus.ml_packages_installed && !systemStatus.use_real_model)}
            className={`w-full flex items-center justify-between px-3.5 py-2.5 rounded-xl border text-sm font-semibold transition-all duration-300 ${
              !backendOnline 
                ? 'opacity-40 cursor-not-allowed border-slate-800 text-slate-600 bg-slate-950' 
                : systemStatus.use_real_model
                  ? 'border-rose-500/20 bg-rose-950/10 hover:bg-rose-950/20 text-rose-400 hover:border-rose-500/40'
                  : 'border-violet-500/20 bg-violet-950/10 hover:bg-violet-950/20 text-violet-400 hover:border-violet-500/40'
            }`}
          >
            <span className="font-display">
              {systemStatus.use_real_model ? 'Deactivate BERT' : 'Activate BERT'}
            </span>
            {systemStatus.use_real_model ? (
              <ToggleRight className="h-6 w-6 text-rose-400" />
            ) : (
              <ToggleLeft className="h-6 w-6 text-slate-500" />
            )}
          </button>

          {!systemStatus.ml_packages_installed && (
            <div className="mt-2.5 text-[10px] text-amber-500 leading-tight border-t border-slate-800/40 pt-2.5">
              ⚠️ AI dependencies (torch/transformers) not detected. BERT disabled. Running in Fast Demo Mode.
            </div>
          )}
        </div>
      </div>
    </aside>
  );
}
