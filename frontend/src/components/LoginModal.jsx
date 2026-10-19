import React, { useState } from 'react';
import { Shield, Lock, Mail, AlertTriangle, CheckCircle2, UserCheck, X } from 'lucide-react';
import { useAuth } from '../context/AuthContext';

export default function LoginModal() {
  const { isLoginModalOpen, setIsLoginModalOpen, login, quickSwitch, user } = useAuth();
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  if (!isLoginModalOpen) return null;

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setLoading(true);
    const res = await login(email, password);
    setLoading(false);
    if (!res.success) {
      setError(res.error || 'Invalid credentials');
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/80 backdrop-blur-md p-4">
      <div className="glass-panel w-full max-w-md rounded-3xl border border-slate-800 p-8 shadow-2xl relative">
        {/* Close Button */}
        <button
          onClick={() => setIsLoginModalOpen(false)}
          className="absolute top-5 right-5 text-slate-400 hover:text-white p-1 rounded-lg hover:bg-slate-800 transition-all"
        >
          <X className="h-5 w-5" />
        </button>

        {/* Brand Header */}
        <div className="flex items-center space-x-3 mb-6">
          <div className="bg-violet-600/10 p-3 rounded-2xl border border-violet-500/20 glow-purple">
            <Shield className="h-7 w-7 text-violet-500" />
          </div>
          <div>
            <h2 className="text-xl font-bold font-display text-white">CYBER SHIELD ACCESS</h2>
            <span className="text-[10px] uppercase font-extrabold text-violet-400 tracking-widest">
              Role-Based Authentication
            </span>
          </div>
        </div>

        {/* Demo Credentials Warning Banner */}
        <div className="mb-6 p-3 rounded-xl bg-amber-500/10 border border-amber-500/25 flex items-start space-x-2.5">
          <AlertTriangle className="h-4 w-4 text-amber-400 shrink-0 mt-0.5" />
          <div className="text-[11px] text-amber-300 leading-snug">
            <strong>DEMO CREDENTIALS ONLY</strong>
            <p className="text-[10px] text-amber-400/80 mt-0.5">
              Select a quick preset below or log in with demo credentials:
            </p>
          </div>
        </div>

        {/* 1-Click Demo Presets */}
        <div className="grid grid-cols-2 gap-3 mb-6">
          <button
            onClick={() => quickSwitch('admin')}
            type="button"
            className="p-3 rounded-xl border border-violet-500/30 bg-violet-950/20 hover:bg-violet-900/30 text-left transition-all group"
          >
            <div className="flex items-center space-x-1.5 text-xs font-bold text-violet-300 group-hover:text-white">
              <Shield className="h-3.5 w-3.5 text-violet-400" />
              <span>Admin Role</span>
            </div>
            <span className="text-[10px] text-slate-400 block mt-1">Full system + Fairness</span>
          </button>

          <button
            onClick={() => quickSwitch('moderator')}
            type="button"
            className="p-3 rounded-xl border border-emerald-500/30 bg-emerald-950/20 hover:bg-emerald-900/30 text-left transition-all group"
          >
            <div className="flex items-center space-x-1.5 text-xs font-bold text-emerald-300 group-hover:text-white">
              <UserCheck className="h-3.5 w-3.5 text-emerald-400" />
              <span>Moderator</span>
            </div>
            <span className="text-[10px] text-slate-400 block mt-1">Review Queue & Feed</span>
          </button>
        </div>

        <div className="relative flex items-center justify-center mb-6">
          <div className="border-t border-slate-800 w-full" />
          <span className="bg-slate-900 px-3 text-[10px] uppercase font-bold text-slate-500 tracking-wider absolute">
            Or Sign In Manually
          </span>
        </div>

        {/* Error message */}
        {error && (
          <div className="mb-4 p-2.5 rounded-xl bg-rose-500/10 border border-rose-500/25 text-xs text-rose-400 font-medium">
            {error}
          </div>
        )}

        {/* Standard Form */}
        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label className="text-xs font-semibold text-slate-300 block mb-1.5">Email Address</label>
            <div className="relative">
              <Mail className="absolute left-3.5 top-3.5 h-4 w-4 text-slate-500" />
              <input
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="admin@cybershield.org"
                className="w-full bg-slate-950/90 border border-slate-800 rounded-xl pl-10 pr-4 py-2.5 text-sm text-white focus:outline-none focus:border-violet-500 transition-all font-mono text-xs"
              />
            </div>
          </div>

          <div>
            <label className="text-xs font-semibold text-slate-300 block mb-1.5">Password</label>
            <div className="relative">
              <Lock className="absolute left-3.5 top-3.5 h-4 w-4 text-slate-500" />
              <input
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="••••••••••••"
                className="w-full bg-slate-950/90 border border-slate-800 rounded-xl pl-10 pr-4 py-2.5 text-sm text-white focus:outline-none focus:border-violet-500 transition-all font-mono text-xs"
              />
            </div>
          </div>

          <button
            type="submit"
            disabled={loading}
            className="w-full mt-2 py-3 rounded-xl bg-gradient-to-r from-violet-600 to-pink-600 hover:from-violet-500 hover:to-pink-500 text-white font-bold text-xs uppercase tracking-wider transition-all shadow-lg shadow-violet-950/30 disabled:opacity-50 cursor-pointer"
          >
            {loading ? 'Authenticating...' : 'Sign In with JWT'}
          </button>
        </form>

        <div className="mt-4 text-center text-[10px] text-slate-500">
          Current Active Session: <span className="font-mono text-slate-400 font-bold">{user ? `${user.email} (${user.role})` : 'Guest'}</span>
        </div>
      </div>
    </div>
  );
}
