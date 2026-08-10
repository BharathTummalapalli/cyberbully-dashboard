import React, { useState, useEffect, useRef } from 'react';
import { Share2, AlertOctagon, HelpCircle, Activity, Info } from 'lucide-react';

export default function NetworkGraph({ backendOnline }) {
  const [graphData, setGraphData] = useState({ nodes: [], edges: [], top_abusers: [], clusters: [] });
  const [loading, setLoading] = useState(true);
  const [selectedNode, setSelectedNode] = useState(null);
  const [hoveredNode, setHoveredNode] = useState(null);
  
  // Custom force layout state
  const [nodePositions, setNodePositions] = useState({});
  const width = 720;
  const height = 450;

  // Fetch NetworkX calculations from FastAPI backend
  const fetchGraph = async () => {
    try {
      setLoading(true);
      const res = await fetch('/api/graph');
      if (res.ok) {
        const data = await res.json();
        setGraphData(data);
        runForceSimulation(data.nodes, data.edges);
      }
    } catch (err) {
      console.error("Failed to fetch graph data:", err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchGraph();
  }, [backendOnline]);

  // Fast custom spring-force layout simulation
  const runForceSimulation = (nodes, edges) => {
    if (!nodes || nodes.length === 0) return;

    // Initialize positions randomly within central radius
    let pos = {};
    nodes.forEach(n => {
      pos[n.id] = {
        x: width / 2 + (Math.random() - 0.5) * 150,
        y: height / 2 + (Math.random() - 0.5) * 150
      };
    });

    const iterations = 100;
    const k = Math.sqrt((width * height) / nodes.length) * 0.85; // ideal node spacing

    for (let iter = 0; iter < iterations; iter++) {
      // 1. Repulsion forces between all nodes
      for (let i = 0; i < nodes.length; i++) {
        let n1 = nodes[i].id;
        for (let j = 0; j < nodes.length; j++) {
          if (i === j) continue;
          let n2 = nodes[j].id;
          
          let dx = pos[n1].x - pos[n2].x;
          let dy = pos[n1].y - pos[n2].y;
          let dist = Math.sqrt(dx * dx + dy * dy) || 1.0;
          
          if (dist < 250) {
            // Force vector
            let fr = (k * k) / dist;
            pos[n1].x += (dx / dist) * fr * 0.15;
            pos[n1].y += (dy / dist) * fr * 0.15;
          }
        }
      }

      // 2. Attraction forces along edges
      edges.forEach(edge => {
        let u = edge.source;
        let v = edge.target;
        if (!pos[u] || !pos[v]) return;
        
        let dx = pos[v].x - pos[u].x;
        let dy = pos[v].y - pos[u].y;
        let dist = Math.sqrt(dx * dx + dy * dy) || 1.0;
        
        // Attraction force vector (spring)
        let fa = (dist * dist) / k;
        let forceRatio = fa * 0.15;
        
        pos[u].x += (dx / dist) * forceRatio;
        pos[u].y += (dy / dist) * forceRatio;
        pos[v].x -= (dx / dist) * forceRatio;
        pos[v].y -= (dy / dist) * forceRatio;
      });

      // 3. Center gravity constraint
      nodes.forEach(n => {
        let dx = width / 2 - pos[n.id].x;
        let dy = height / 2 - pos[n.id].y;
        pos[n.id].x += dx * 0.04;
        pos[n.id].y += dy * 0.04;
      });
    }

    // 4. Bound positions to canvas viewport boundary
    let boundedPos = {};
    nodes.forEach(n => {
      boundedPos[n.id] = {
        x: Math.max(30, Math.min(width - 30, pos[n.id].x)),
        y: Math.max(30, Math.min(height - 30, pos[n.id].y))
      };
    });

    setNodePositions(boundedPos);
  };

  // Helper to determine node styling based on threat role
  const getNodeColor = (role) => {
    switch (role) {
      case 'Abuser': return '#ef4444'; // Red
      case 'Victim': return '#3b82f6'; // Blue
      case 'Aggressive Participator': return '#8b5cf6'; // Purple
      default: return '#64748b'; // Slate gray
    }
  };

  const getConnectedNodes = (nodeId) => {
    if (!nodeId) return [];
    const connected = new Set();
    graphData.edges.forEach(e => {
      if (e.source === nodeId) connected.add(e.target);
      if (e.target === nodeId) connected.add(e.source);
    });
    return Array.from(connected);
  };

  const isRelated = (nId) => {
    if (!hoveredNode) return true;
    if (hoveredNode === nId) return true;
    return getConnectedNodes(hoveredNode).includes(nId);
  };

  const isEdgeRelated = (edge) => {
    if (!hoveredNode) return true;
    return edge.source === hoveredNode || edge.target === hoveredNode;
  };

  const activeDetails = selectedNode 
    ? graphData.nodes.find(n => n.id === selectedNode) 
    : null;

  return (
    <div className="flex-1 overflow-y-auto bg-slate-900 bg-grid p-8">
      {/* Page Header */}
      <div className="flex items-center justify-between mb-8">
        <div className="flex items-center space-x-3">
          <div className="bg-violet-600/10 p-2.5 rounded-xl border border-violet-500/20 glow-purple">
            <Share2 className="h-6 w-6 text-violet-500 animate-pulse" />
          </div>
          <div>
            <h1 className="text-2xl font-bold tracking-tight text-white font-display">Cyberbullying Directed Network</h1>
            <p className="text-xs text-slate-400">Map attacker-victim dynamics, repeated attack counts (edge weight), and isolate group cliques.</p>
          </div>
        </div>

        <button
          onClick={fetchGraph}
          className="px-4 py-2 rounded-xl text-xs font-bold bg-slate-800 text-slate-200 border border-slate-700 hover:bg-slate-750 transition-all cursor-pointer"
        >
          REFRESH NETWORK
        </button>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* SVG Interactive Canvas */}
        <div className="glass-panel p-4 rounded-2xl border border-slate-800/80 lg:col-span-2 flex flex-col justify-between">
          <div className="flex items-center justify-between mb-4 px-2">
            <span className="text-xs font-bold uppercase tracking-wider text-slate-400">Social Network Canvas</span>
            {/* Color key legend */}
            <div className="flex space-x-4 text-[10px] font-semibold">
              <div className="flex items-center space-x-1.5"><span className="h-2 w-2 rounded-full bg-rose-500 shadow shadow-rose-500" /> <span className="text-slate-400">Abuser</span></div>
              <div className="flex items-center space-x-1.5"><span className="h-2 w-2 rounded-full bg-blue-500 shadow shadow-blue-500" /> <span className="text-slate-400">Victim</span></div>
              <div className="flex items-center space-x-1.5"><span className="h-2 w-2 rounded-full bg-purple-500 shadow shadow-purple-500" /> <span className="text-slate-400">Participator</span></div>
            </div>
          </div>

          <div className="bg-slate-950/60 rounded-xl border border-slate-900 overflow-hidden relative min-h-[450px]">
            {loading ? (
              <div className="absolute inset-0 flex items-center justify-center text-xs text-slate-500">
                Executing NetworkX Calculations...
              </div>
            ) : graphData.nodes.length === 0 ? (
              <div className="absolute inset-0 flex items-center justify-center text-xs text-slate-500 italic">
                No cyberbullying nodes detected. Run feed simulation first to register interactions.
              </div>
            ) : (
              <svg width="100%" height="450" viewBox={`0 0 ${width} ${height}`} className="select-none">
                <defs>
                  {/* Arrowhead definitions for directed edges */}
                  <marker
                    id="arrowhead"
                    viewBox="0 0 10 10"
                    refX="23"
                    refY="5"
                    markerWidth="6"
                    markerHeight="6"
                    orient="auto-start-reverse"
                  >
                    <path d="M 0 0 L 10 5 L 0 10 z" fill="#475569" />
                  </marker>
                  <marker
                    id="arrowhead-highlight"
                    viewBox="0 0 10 10"
                    refX="23"
                    refY="5"
                    markerWidth="6"
                    markerHeight="6"
                    orient="auto-start-reverse"
                  >
                    <path d="M 0 0 L 10 5 L 0 10 z" fill="#8b5cf6" />
                  </marker>
                </defs>

                {/* Draw directed link lines */}
                {graphData.edges.map((edge, idx) => {
                  const from = nodePositions[edge.source];
                  const to = nodePositions[edge.target];
                  if (!from || !to) return null;
                  
                  const isHighlighted = isEdgeRelated(edge);
                  
                  return (
                    <line
                      key={idx}
                      x1={from.x}
                      y1={from.y}
                      x2={to.x}
                      y2={to.y}
                      stroke={isHighlighted && hoveredNode ? '#c084fc' : '#334155'}
                      strokeWidth={isHighlighted && hoveredNode ? edge.weight * 1.5 + 1.5 : edge.weight * 1.0}
                      strokeOpacity={isHighlighted ? 0.8 : 0.15}
                      markerEnd={isHighlighted && hoveredNode ? "url(#arrowhead-highlight)" : "url(#arrowhead)"}
                    />
                  );
                })}

                {/* Draw interaction weights counts labels on links (only on hover) */}
                {hoveredNode && graphData.edges.map((edge, idx) => {
                  if (edge.source !== hoveredNode && edge.target !== hoveredNode) return null;
                  const from = nodePositions[edge.source];
                  const to = nodePositions[edge.target];
                  if (!from || !to) return null;
                  
                  const midX = (from.x + to.x) / 2;
                  const midY = (from.y + to.y) / 2;
                  
                  return (
                    <g key={`lbl-${idx}`}>
                      <rect 
                        x={midX - 10} 
                        y={midY - 8} 
                        width="20" 
                        height="14" 
                        rx="3" 
                        fill="#0b0f19" 
                        stroke="#8b5cf6" 
                        strokeWidth="0.5" 
                      />
                      <text
                        x={midX}
                        y={midY + 2}
                        fill="#c084fc"
                        fontSize="9"
                        fontWeight="bold"
                        textAnchor="middle"
                        fontFamily="monospace"
                      >
                        {edge.weight}
                      </text>
                    </g>
                  );
                })}

                {/* Draw nodes circles */}
                {graphData.nodes.map((node) => {
                  const pos = nodePositions[node.id];
                  if (!pos) return null;
                  
                  const isNodeRelated = isRelated(node.id);
                  const isHovered = hoveredNode === node.id;
                  const isSelected = selectedNode === node.id;
                  
                  // Scale node radius dynamically based on network activity centrality (attacks sent)
                  // Minimum radius: 10, Max radius: 24
                  const radius = 12 + (node.attacks_sent * 1.8);
                  
                  return (
                    <g
                      key={node.id}
                      transform={`translate(${pos.x}, ${pos.y})`}
                      onMouseEnter={() => setHoveredNode(node.id)}
                      onMouseLeave={() => setHoveredNode(null)}
                      onClick={() => setSelectedNode(node.id)}
                      className="cursor-pointer"
                    >
                      {/* Outer pulse/highlight ring */}
                      {(isSelected || isHovered) && (
                        <circle
                          r={radius + 6}
                          fill="transparent"
                          stroke={getNodeColor(node.role)}
                          strokeWidth="2"
                          strokeOpacity={isHovered ? 0.7 : 0.4}
                          className={isHovered ? 'animate-ping' : ''}
                        />
                      )}

                      {/* Main node bubble */}
                      <circle
                        r={radius}
                        fill={getNodeColor(node.role)}
                        fillOpacity={isNodeRelated ? 0.9 : 0.15}
                        stroke="#070a13"
                        strokeWidth="1.5"
                      />

                      {/* Node initial label inside */}
                      <text
                        fill="#ffffff"
                        fillOpacity={isNodeRelated ? 1.0 : 0.25}
                        fontSize="9"
                        fontWeight="extrabold"
                        textAnchor="middle"
                        y="3.5"
                        fontFamily="sans-serif"
                      >
                        {node.id.substring(1, 3).toUpperCase()}
                      </text>

                      {/* Floating text tag overlay */}
                      <text
                        y={-radius - 5}
                        fill={isHovered || isSelected ? '#ffffff' : '#94a3b8'}
                        fillOpacity={isNodeRelated ? 1.0 : 0.15}
                        fontSize="10"
                        fontWeight={isHovered || isSelected ? 'bold' : 'normal'}
                        textAnchor="middle"
                      >
                        {node.id}
                      </text>
                    </g>
                  );
                })}
              </svg>
            )}
          </div>
          <div className="text-[10px] text-slate-500 mt-2 px-1 flex items-center space-x-1.5">
            <Info className="h-3 w-3" />
            <span>Hover on bubbles to filter connections and inspect repeated weights; click nodes to extract full threat logs.</span>
          </div>
        </div>

        {/* Info detail inspector & Community Clusters */}
        <div className="space-y-6 lg:col-span-1">
          {/* Node details */}
          <div className="glass-panel p-5 rounded-2xl border border-slate-800/80 min-h-[220px] flex flex-col justify-between">
            <h3 className="text-sm font-bold text-white uppercase tracking-wider mb-3.5 font-display flex items-center space-x-2">
              <Activity className="h-4.5 w-4.5 text-violet-400" />
              <span>Threat Inspector</span>
            </h3>

            {activeDetails ? (
              <div className="space-y-4">
                <div>
                  <div className="flex items-center justify-between">
                    <span className="text-sm font-extrabold text-white font-display">{activeDetails.id}</span>
                    <span className={`px-2 py-0.5 rounded text-[9px] font-extrabold border ${
                      activeDetails.role === 'Abuser' 
                        ? 'bg-rose-500/10 text-rose-400 border-rose-500/25 glow-red' 
                        : activeDetails.role === 'Victim' 
                          ? 'bg-blue-500/10 text-blue-400 border-blue-500/25 glow-blue' 
                          : 'bg-purple-500/10 text-purple-400 border-purple-500/25'
                    }`}>
                      {activeDetails.role}
                    </span>
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-3 text-xs bg-slate-950/80 border border-slate-900 p-3 rounded-xl">
                  <div>
                    <span className="text-[10px] text-slate-500 block">Out Centrality</span>
                    <span className="font-mono font-bold text-slate-200">{Math.round(activeDetails.out_centrality * 100)}%</span>
                  </div>
                  <div>
                    <span className="text-[10px] text-slate-500 block">In Centrality</span>
                    <span className="font-mono font-bold text-slate-200">{Math.round(activeDetails.in_centrality * 100)}%</span>
                  </div>
                  <div className="mt-2">
                    <span className="text-[10px] text-slate-500 block">Attacks Logged</span>
                    <span className="font-bold text-rose-400">{activeDetails.attacks_sent}</span>
                  </div>
                  <div className="mt-2">
                    <span className="text-[10px] text-slate-500 block">Victim Incidents</span>
                    <span className="font-bold text-blue-400">{activeDetails.attacks_received}</span>
                  </div>
                </div>

                <div className="text-xs">
                  <div className="flex justify-between border-b border-slate-900 pb-1.5 mb-1.5">
                    <span className="text-slate-500">Clustering Coefficient:</span>
                    <span className="font-bold font-mono text-slate-200">{activeDetails.clustering}</span>
                  </div>
                </div>
              </div>
            ) : (
              <div className="flex-1 flex items-center justify-center text-xs text-slate-500 italic text-center py-6">
                Click a node inside the canvas to load its profile.
              </div>
            )}
          </div>

          {/* Group Clusters (Connected Components) */}
          <div className="glass-panel p-5 rounded-2xl border border-slate-800/80">
            <h3 className="text-sm font-bold text-white uppercase tracking-wider mb-4 font-display flex items-center space-x-2">
              <AlertOctagon className="h-4.5 w-4.5 text-rose-500" />
              <span>Isolated Bullying Cliques</span>
            </h3>
            
            <div className="space-y-3 max-h-[170px] overflow-y-auto pr-1">
              {graphData.clusters.length === 0 ? (
                <div className="text-center py-6 text-xs text-slate-500 italic">No clusters resolved.</div>
              ) : (
                graphData.clusters.map((cluster, idx) => (
                  <div key={idx} className="bg-slate-950/80 border border-slate-900 p-3 rounded-xl">
                    <div className="text-[10px] uppercase font-extrabold text-slate-500 tracking-wider mb-2">Clique Group #{idx + 1} ({cluster.length} nodes)</div>
                    <div className="flex flex-wrap gap-1.5">
                      {cluster.map(node => (
                        <span key={node} className="px-2 py-0.5 rounded text-[10px] bg-slate-900 border border-slate-800 text-slate-300 font-semibold font-mono">
                          {node}
                        </span>
                      ))}
                    </div>
                  </div>
                ))
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
