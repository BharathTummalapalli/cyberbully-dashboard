import React from 'react';
import { 
  Shield, LayoutDashboard, BarChart3, Share2, ToggleLeft, ToggleRight, 
  Radio, ServerCrash, AlertOctagon, Scale, UserCheck, LogOut, KeyRound 
} from 'lucide-react';
import { useAuth } from '../context/AuthContext';

export default function Sidebar({ 
  activeTab, 
  setActiveTab, 
  systemStatus, 
  toggleAiMode, 
  backendOnline,
  pendingQueueCount = 0
}) {
  const { user, role, setIsLoginModalOpen, logout } = useAuth();

  // Role-aware navigation definitions
  const allMenuItems = [
    { id: 'dashboard', label: 'Live Stream Feed', icon: LayoutDashboard, roles: ['admin', 'moderator', 'guest'] },
    { 
      id: 'queue', 
      label: 'Review Queue', 
      icon: AlertOctagon, 
      roles: ['admin', 'moderator'],
      badge: pendingQueueCount > 0 ? pendingQueueCount : null,
      badgeColor: 'bg-amber-500 text-slate-950'
    },
    { id: 'network', label: 'Bullying Directed Graph', icon: Share2, roles: ['admin', 'moderator', 'guest'] },
    { id: 'analytics', label: 'Analytics & Benchmarks', icon: BarChart3, roles: ['admin'] },
    { id: 'fairness', label: 'Fairness & Bias Audit', icon: Scale, roles: ['admin'] },
  ];

  const filteredMenuItems = allMenuItems.filter(item => item.roles.includes(role));

  return (
    <aside className="w-80 border-r border-slate-800 bg-slate-950 flex flex-col h-screen sticky top-0">
      {/* Brand Logo & Header */}
      <div className="p-6 border-b border-slate-900 flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <div className="bg-violet-600/10 p-2.5 rounded-xl border border-violet-500/20 glow-purple">
            <Shield className="h-6 w-6 text-violet-500 animate-pulse" />
          </div>
          <div>
            <h1 className="text-xl font-bold tracking-tight font-display text-white">CYBER SHIELD</h1>
            <span className="text-[10px] uppercase font-bold text-violet-400 tracking-widest">
              Autonomous Moderation
            </span>
          </div>
        </div>
      </div>

      {/* Authenticated User Profile Pill */}
      <div className="px-6 py-3 border-b border-slate-900 bg-slate-950/70 flex items-center justify-between">
        <div className="flex items-center space-x-2.5">
          <div className="h-7 w-7 rounded-lg bg-slate-900 border border-slate-800 flex items-center justify-center font-bold text-violet-400 text-xs font-display">
            {user ? user.email.charAt(0).toUpperCase() : 'G'}
          </div>
          <div>
            <div className="text-xs font-bold text-slate-200 font-mono truncate max-w-[130px]">
              {user ? user.email : 'Guest Mode'}
            </div>
            <span className={`text-[9px] font-extrabold uppercase px-1.5 py-0.2 rounded border ${
              role === 'admin' 
                ? 'bg-rose-500/10 text-rose-400 border-rose-500/25' 
                : role === 'moderator'
                  ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/25'
                  : 'bg-slate-800 text-slate-400 border-slate-700'
            }`}>
              {role}
            </span>
          </div>
        </div>

        <button
          onClick={() => setIsLoginModalOpen(true)}
          className="p-1.5 rounded-lg border border-slate-800 hover:bg-slate-900 text-slate-400 hover:text-white transition-all text-xs font-bold"
          title="Switch role or login"
        >
          <KeyRound className="h-3.5 w-3.5" />
        </button>
      </div>

      {/* Connection Status Sub-Header */}
      <div className="px-6 py-2 border-b border-slate-900 bg-slate-950/40 flex items-center justify-between text-[11px]">
        <span className="text-slate-500 font-medium">FastAPI Engine</span>
        <div className="flex items-center space-x-1.5">
          {backendOnline ? (
            <>
              <Radio className="h-2.5 w-2.5 text-emerald-500 animate-pulse" />
              <span className="text-emerald-500 font-bold font-mono text-[10px]">CONNECTED</span>
            </>
          ) : (
            <>
              <ServerCrash className="h-2.5 w-2.5 text-rose-500" />
              <span className="text-rose-500 font-bold font-mono text-[10px]">OFFLINE</span>
            </>
          )}
        </div>
      </div>

      {/* Main Navigation Menu */}
      <nav className="flex-1 px-4 py-5 space-y-1.5 overflow-y-auto">
        {filteredMenuItems.map((item) => {
          const Icon = item.icon;
          const isActive = activeTab === item.id;
          return (
            <button
              key={item.id}
              onClick={() => setActiveTab(item.id)}
              className={`w-full flex items-center justify-between px-4 py-3 rounded-xl text-sm font-medium transition-all duration-200 cursor-pointer ${
                isActive
                  ? 'bg-gradient-to-r from-violet-600/20 to-pink-600/10 text-white border border-violet-500/25 shadow-lg shadow-violet-950/20'
                  : 'text-slate-400 hover:bg-slate-900/60 hover:text-slate-200 border border-transparent'
              }`}
            >
              <div className="flex items-center space-x-3.5">
                <Icon className={`h-4.5 w-4.5 ${isActive ? 'text-violet-400' : 'text-slate-400'}`} />
                <span className="font-display text-xs font-bold">{item.label}</span>
              </div>
              {item.badge !== undefined && item.badge !== null && (
                <span className={`px-2 py-0.5 rounded-full text-[10px] font-black ${item.badgeColor}`}>
                  {item.badge}
                </span>
              )}
            </button>
          );
        })}
      </nav>

      {/* AI Model Mode Toggle Box */}
      <div className="p-4 border-t border-slate-900 bg-slate-950">
        <div className="glass-panel p-4 rounded-2xl border border-slate-800/80">
          <div className="flex items-center justify-between mb-2">
            <span className="text-[10px] font-extrabold uppercase tracking-wider text-slate-400">Model Pipeline</span>
            <span className={`px-2 py-0.5 rounded text-[9px] font-black ${
              systemStatus.use_real_model 
                ? 'bg-rose-500/10 text-rose-400 border border-rose-500/25 glow-red' 
                : 'bg-violet-500/10 text-violet-400 border border-violet-500/25 glow-purple'
            }`}>
              {systemStatus.use_real_model ? 'Toxic-BERT' : 'Lexicon'}
            </span>
          </div>

          <p className="text-[11px] text-slate-400 leading-relaxed mb-3">
            {systemStatus.use_real_model 
              ? 'BERT Transformer + Rule-Based Context Adjustment Layer.' 
              : 'Fast keyword lexicon fallback mode.'}
          </p>

          <button
            onClick={toggleAiMode}
            disabled={!backendOnline || (!systemStatus.ml_packages_installed && !systemStatus.use_real_model)}
            className={`w-full flex items-center justify-between px-3 py-2 rounded-xl border text-xs font-bold transition-all duration-300 cursor-pointer ${
              !backendOnline 
                ? 'opacity-40 cursor-not-allowed border-slate-800 text-slate-600 bg-slate-950' 
                : systemStatus.use_real_model
                  ? 'border-rose-500/20 bg-rose-950/10 hover:bg-rose-950/20 text-rose-400'
                  : 'border-violet-500/20 bg-violet-950/10 hover:bg-violet-950/20 text-violet-400'
            }`}
          >
            <span className="font-display">
              {systemStatus.use_real_model ? 'Switch to Lexicon' : 'Activate Toxic-BERT'}
            </span>
            {systemStatus.use_real_model ? (
              <ToggleRight className="h-5 w-5 text-rose-400" />
            ) : (
              <ToggleLeft className="h-5 w-5 text-slate-500" />
            )}
          </button>
        </div>
      </div>
    </aside>
  );
}
