import React from 'react';
import { 
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, 
  PieChart, Pie, Cell, 
  BarChart, Bar, Legend
} from 'recharts';
import { ShieldCheck, EyeOff, CheckCircle2, Award, Users, AlertTriangle } from 'lucide-react';

export default function Analytics({ analytics, backendOnline }) {
  // Safe defaults if database/backend is unpopulated or offline
  const total = analytics.total_posts || 0;
  const approved = analytics.approved_count || 0;
  const hidden = analytics.hidden_count || 0;
  const accuracy = analytics.moderator_accuracy !== undefined ? analytics.moderator_accuracy : 100.0;
  
  // 1. Prepare Agreement Pie Chart Data
  // Accuracy is the percentage of posts where moderator agreed with AI prediction.
  const processedDecisionsCount = approved + hidden;
  const agreedPercent = accuracy;
  const disagreedPercent = 100 - accuracy;
  
  const agreementData = processedDecisionsCount > 0 ? [
    { name: 'Agreement (Mod == AI)', value: parseFloat(agreedPercent) },
    { name: 'Disagreement (Mod != AI)', value: parseFloat(disagreedPercent) }
  ] : [
    { name: 'No decisions yet', value: 100 }
  ];

  const PIE_COLORS = processedDecisionsCount > 0 
    ? ['#10b981', '#ef4444'] // Green for agreement, red for disagreement
    : ['#475569'];          // Slate gray if empty

  // 2. Prepare Daily Trend Data
  const trendData = analytics.daily_trends && analytics.daily_trends.length > 0
    ? analytics.daily_trends.map(t => ({
        ...t,
        // Shorten dates YYYY-MM-DD to MM/DD for axis readability
        dateStr: t.date ? t.date.slice(5) : 'Unknown'
      }))
    : [
        { dateStr: 'Day 1', bullying_count: 0, safe_count: 0 },
        { dateStr: 'Day 2', bullying_count: 0, safe_count: 0 },
        { dateStr: 'Day 3', bullying_count: 0, safe_count: 0 }
      ];

  // 3. Prepare Top Abusers Bar Chart Data (NetworkX integration)
  const topAbusers = analytics.top_abusers || [];
  const abuserChartData = topAbusers.map(u => ({
    name: u.id,
    'Attacks Sent': u.attacks_sent,
    'Out Centrality': Math.round(u.out_centrality * 100) // represent centrality out of 100
  }));

  return (
    <div className="flex-1 overflow-y-auto bg-slate-900 bg-grid p-8">
      {/* Page Title */}
      <div className="flex items-center space-x-3 mb-8">
        <div className="bg-violet-600/10 p-2.5 rounded-xl border border-violet-500/20 glow-purple">
          <Award className="h-6 w-6 text-violet-500" />
        </div>
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-white font-display">Analytics & Performance Reports</h1>
          <p className="text-xs text-slate-400">Evaluate model metrics, human-in-the-loop accuracy, and social network threats.</p>
        </div>
      </div>

      {/* Aggregate Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
        <div className="glass-panel p-5 rounded-2xl border border-slate-800 flex items-center space-x-4">
          <div className="bg-blue-500/10 p-3 rounded-xl border border-blue-500/20 text-blue-400">
            <Users className="h-6 w-6" />
          </div>
          <div>
            <span className="text-xs font-semibold text-slate-400">Database Logs</span>
            <div className="text-2xl font-extrabold text-white mt-0.5 font-display">{total}</div>
            <span className="text-[10px] text-slate-500">Total processed posts</span>
          </div>
        </div>

        <div className="glass-panel p-5 rounded-2xl border border-slate-800 flex items-center space-x-4">
          <div className="bg-emerald-500/10 p-3 rounded-xl border border-emerald-500/20 text-emerald-400">
            <ShieldCheck className="h-6 w-6" />
          </div>
          <div>
            <span className="text-xs font-semibold text-slate-400">Approved Posts</span>
            <div className="text-2xl font-extrabold text-white mt-0.5 font-display">{approved}</div>
            <span className="text-[10px] text-emerald-500">Safe/whitelist active</span>
          </div>
        </div>

        <div className="glass-panel p-5 rounded-2xl border border-slate-800 flex items-center space-x-4">
          <div className="bg-rose-500/10 p-3 rounded-xl border border-rose-500/20 text-rose-400">
            <EyeOff className="h-6 w-6" />
          </div>
          <div>
            <span className="text-xs font-semibold text-slate-400">Hidden Posts</span>
            <div className="text-2xl font-extrabold text-white mt-0.5 font-display">{hidden}</div>
            <span className="text-[10px] text-rose-500">Abuse/toxic content hidden</span>
          </div>
        </div>

        <div className="glass-panel p-5 rounded-2xl border border-slate-800 flex items-center space-x-4">
          <div className="bg-violet-500/10 p-3 rounded-xl border border-violet-500/20 text-violet-400">
            <CheckCircle2 className="h-6 w-6" />
          </div>
          <div>
            <span className="text-xs font-semibold text-slate-400">Moderator Agreement</span>
            <div className="text-2xl font-extrabold text-white mt-0.5 font-display">
              {processedDecisionsCount > 0 ? `${accuracy}%` : 'N/A'}
            </div>
            <span className="text-[10px] text-slate-500">AI prediction match rate</span>
          </div>
        </div>
      </div>

      {/* Grid of charts */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 mb-8">
        
        {/* Trend Area Chart (Cyberbullying vs Safe) */}
        <div className="glass-panel p-6 rounded-2xl border border-slate-800/80 lg:col-span-2">
          <h3 className="text-sm font-bold text-white uppercase tracking-wider mb-5 font-display">Daily Classification Trends</h3>
          <div className="h-72 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={trendData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                <defs>
                  <linearGradient id="colorBullying" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#ef4444" stopOpacity={0.2}/>
                    <stop offset="95%" stopColor="#ef4444" stopOpacity={0}/>
                  </linearGradient>
                  <linearGradient id="colorSafe" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#10b981" stopOpacity={0.2}/>
                    <stop offset="95%" stopColor="#10b981" stopOpacity={0}/>
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis dataKey="dateStr" stroke="#64748b" style={{ fontSize: '11px', fontWeight: '500' }} />
                <YAxis stroke="#64748b" style={{ fontSize: '11px', fontWeight: '500' }} />
                <Tooltip 
                  contentStyle={{ backgroundColor: '#0b0f19', border: '1px solid #1e293b', borderRadius: '12px' }}
                  labelStyle={{ color: '#94a3b8', fontWeight: 'bold' }}
                />
                <Legend wrapperStyle={{ fontSize: '12px', paddingTop: '10px' }} />
                <Area type="monotone" dataKey="bullying_count" name="Cyberbullying" stroke="#ef4444" strokeWidth={2} fillOpacity={1} fill="url(#colorBullying)" />
                <Area type="monotone" dataKey="safe_count" name="Safe Posts" stroke="#10b981" strokeWidth={2} fillOpacity={1} fill="url(#colorSafe)" />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Agreement Rate Pie Chart */}
        <div className="glass-panel p-6 rounded-2xl border border-slate-800/80 lg:col-span-1 flex flex-col justify-between">
          <div>
            <h3 className="text-sm font-bold text-white uppercase tracking-wider mb-2 font-display">AI / Human Alignment</h3>
            <p className="text-xs text-slate-400 mb-4">Measures the agreement rate between model tags and human moderator actions.</p>
          </div>
          
          <div className="h-48 w-full relative flex items-center justify-center">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={agreementData}
                  cx="50%"
                  cy="50%"
                  innerRadius={55}
                  outerRadius={75}
                  paddingAngle={5}
                  dataKey="value"
                >
                  {agreementData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={PIE_COLORS[index % PIE_COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip 
                  contentStyle={{ backgroundColor: '#0b0f19', border: '1px solid #1e293b', borderRadius: '12px', fontSize: '12px' }}
                />
              </PieChart>
            </ResponsiveContainer>
            {/* Center stat labels */}
            <div className="absolute text-center">
              <span className="text-[10px] uppercase font-bold text-slate-500 tracking-wider">Agreement</span>
              <div className="text-2xl font-black text-white font-display">
                {processedDecisionsCount > 0 ? `${accuracy}%` : '0%'}
              </div>
            </div>
          </div>

          <div className="text-xs text-slate-400 border-t border-slate-900 pt-4 mt-2">
            {processedDecisionsCount > 0 ? (
              <div className="space-y-1.5">
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-1.5">
                    <span className="h-2.5 w-2.5 rounded-full bg-emerald-500" />
                    <span>AI agreed with Human:</span>
                  </div>
                  <span className="font-bold text-white">{accuracy}%</span>
                </div>
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-1.5">
                    <span className="h-2.5 w-2.5 rounded-full bg-rose-505 bg-rose-500" />
                    <span>AI disagreed with Human:</span>
                  </div>
                  <span className="font-bold text-white">{100 - accuracy}%</span>
                </div>
              </div>
            ) : (
              <div className="italic text-center text-slate-500">Log moderator actions to populate alignment metrics.</div>
            )}
          </div>
        </div>
      </div>

      {/* NetworkX abuser centrality bars */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        
        {/* Centrality Bar Chart */}
        <div className="glass-panel p-6 rounded-2xl border border-slate-800/80 lg:col-span-2">
          <h3 className="text-sm font-bold text-white uppercase tracking-wider mb-5 font-display flex items-center space-x-2">
            <AlertTriangle className="h-4 w-4 text-rose-500" />
            <span>NetworkX Abuser Threat Centrality</span>
          </h3>
          {abuserChartData.length === 0 ? (
            <div className="h-72 w-full flex items-center justify-center text-xs text-slate-500 italic">
              No bullying interactions logged. Graph analysis unpopulated.
            </div>
          ) : (
            <div className="h-72 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={abuserChartData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                  <XAxis dataKey="name" stroke="#64748b" style={{ fontSize: '11px', fontWeight: '500' }} />
                  <YAxis stroke="#64748b" style={{ fontSize: '11px', fontWeight: '500' }} />
                  <Tooltip 
                    contentStyle={{ backgroundColor: '#0b0f19', border: '1px solid #1e293b', borderRadius: '12px' }}
                    labelStyle={{ color: '#94a3b8', fontWeight: 'bold' }}
                  />
                  <Legend wrapperStyle={{ fontSize: '12px', paddingTop: '10px' }} />
                  <Bar dataKey="Attacks Sent" name="Total Incidents Flagged" fill="#f97316" radius={[4, 4, 0, 0]} />
                  <Bar dataKey="Out Centrality" name="Centrality Activity Score (%)" fill="#8b5cf6" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}
        </div>

        {/* Detailed Abuser Profiles table */}
        <div className="glass-panel p-6 rounded-2xl border border-slate-800/80 lg:col-span-1 flex flex-col justify-between">
          <div>
            <h3 className="text-sm font-bold text-white uppercase tracking-wider mb-1 font-display">Target Threat List</h3>
            <p className="text-xs text-slate-400 mb-4">Top abusive nodes flagged by NetworkX directed centrality index.</p>
          </div>

          <div className="flex-1 overflow-y-auto space-y-3.5 max-h-[260px] pr-1">
            {topAbusers.length === 0 ? (
              <div className="text-center py-10 text-xs text-slate-500 italic">No abusers detected.</div>
            ) : (
              topAbusers.map((abuser, idx) => (
                <div key={abuser.id} className="bg-slate-950/80 border border-slate-800/50 p-3.5 rounded-xl flex items-center justify-between">
                  <div className="flex items-center space-x-3">
                    <span className="text-xs font-bold text-slate-500 font-mono">#{idx + 1}</span>
                    <div>
                      <div className="text-xs font-bold text-white">{abuser.id}</div>
                      <div className="text-[10px] text-rose-400 font-medium">Activity Ratio: {Math.round(abuser.out_centrality * 100)}%</div>
                    </div>
                  </div>
                  <div className="text-right">
                    <span className="px-2 py-0.5 rounded bg-rose-500/10 text-rose-400 border border-rose-500/20 text-[9px] font-extrabold uppercase">
                      {abuser.attacks_sent} attacks
                    </span>
                  </div>
                </div>
              ))
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
