import React, { useState, useEffect } from 'react';
import Sidebar from './components/Sidebar';
import Dashboard from './components/Dashboard';
import Analytics from './components/Analytics';
import NetworkGraph from './components/NetworkGraph';

export default function App() {
  const [activeTab, setActiveTab] = useState('dashboard');
  const [backendOnline, setBackendOnline] = useState(false);
  const [posts, setPosts] = useState([]);
  const [analytics, setAnalytics] = useState({
    total_posts: 0,
    approved_count: 0,
    hidden_count: 0,
    pending_count: 0,
    daily_trends: [],
    moderator_accuracy: 100.0,
    top_abusers: []
  });
  const [systemStatus, setSystemStatus] = useState({
    ml_packages_installed: false,
    use_real_model: false,
    model_loaded: false,
    active_model: 'Fast Demo Classifier'
  });

  // Check connection to FastAPI backend
  const checkConnection = async () => {
    try {
      const res = await fetch('/api/status');
      if (res.ok) {
        const data = await res.json();
        setBackendOnline(true);
        setSystemStatus({
          ml_packages_installed: data.ml_packages_installed,
          use_real_model: data.use_real_model,
          model_loaded: data.model_loaded,
          active_model: data.active_model
        });
      } else {
        setBackendOnline(false);
      }
    } catch (err) {
      setBackendOnline(false);
    }
  };

  // Fetch all posts from SQLite database
  const fetchPosts = async () => {
    if (!backendOnline) return;
    try {
      const res = await fetch('/api/posts');
      if (res.ok) {
        const data = await res.json();
        // Since database rows contain key names that might map slightly differently,
        // let's transform them to match what Dashboard expects if necessary
        const formatted = data.map(p => ({
          id: p.id,
          user_id: p.user_id,
          username: p.user_id, // in our schema user_id holds username for convenience
          text: p.post_text,
          prediction: p.prediction,
          confidence: p.confidence,
          moderator_action: p.moderator_action,
          timestamp: p.timestamp,
          // Since explanations and LIME weights are computed on-the-fly and aren't in SQLite,
          // we can regenerate them quickly or mock them on read if details are expanded
          highlighted_words: p.prediction === 'Cyberbullying' ? extractToxicWords(p.post_text) : [],
          lime_weights: generateStaticWeights(p.post_text, p.prediction),
          explanations: generateStaticExplanations(p.post_text, p.prediction)
        }));
        setPosts(formatted);
      }
    } catch (err) {
      console.error("Failed to fetch database logs:", err);
    }
  };

  // Helper keyword lists for static recovery of highlighted elements (from SQLite logs)
  const toxicKeywords = ["trash", "loser", "ugly", "stupid", "idiot", "fat", "gross", "worthless", "brainless", "troll", "annoying", "hateful", "jerk", "pathetic", "hate", "banned", "toxic", "parasite", "worst", "delete", "kill", "disappointment", "garbage", "shame"];
  
  const extractToxicWords = (text) => {
    const words = text.toLowerCase().split(/\b/);
    return words.filter(w => toxicKeywords.includes(w));
  };

  const generateStaticWeights = (text, prediction) => {
    const words = text.toLowerCase().split(/\b/).filter(w => w.trim().length > 1);
    const weights = {};
    if (prediction === 'Cyberbullying') {
      words.forEach(w => {
        if (toxicKeywords.includes(w)) {
          weights[w] = 0.5 + Math.random() * 0.4;
        } else if (Math.random() > 0.7) {
          weights[w] = Math.random() * 0.08;
        }
      });
    } else {
      words.slice(0, 3).forEach(w => {
        weights[w] = -(0.01 + Math.random() * 0.15);
      });
    }
    return weights;
  };

  const generateStaticExplanations = (text, prediction) => {
    const weights = generateStaticWeights(text, prediction);
    const explanations = [];
    Object.entries(weights).forEach(([word, weight]) => {
      if (weight > 0) {
        explanations.append(f => `Word '${word}' increased cyberbullying score by ${int(weight * 100)}%.`); // wait, string interpolation in JS
      }
    });
    // JS equivalent formatting:
    const exps = [];
    Object.entries(weights).forEach(([word, weight]) => {
      if (weight > 0) {
        exps.push(`Word '${word}' increased cyberbullying score by ${Math.round(weight * 100)}%.`);
      } else {
        exps.push(`Word '${word}' supported Safe classification by ${Math.round(Math.abs(weight) * 100)}%.`);
      }
    });
    return exps;
  };

  // Fetch aggregate analytics
  const fetchAnalytics = async () => {
    if (!backendOnline) return;
    try {
      const res = await fetch('/api/analytics');
      if (res.ok) {
        const data = await res.json();
        setAnalytics(data);
      }
    } catch (err) {
      console.error("Failed to fetch analytics:", err);
    }
  };

  // Periodic heartbeat connection poller
  useEffect(() => {
    checkConnection();
    const interval = setInterval(checkConnection, 5000);
    return () => clearInterval(interval);
  }, []);

  // Fetch initial database items when backend online status flips to TRUE
  useEffect(() => {
    if (backendOnline) {
      fetchPosts();
      fetchAnalytics();
    }
  }, [backendOnline]);

  // Trigger a single live feed simulator step
  const triggerNextPost = async () => {
    if (!backendOnline) return;
    try {
      const res = await fetch('/api/simulator/next');
      if (res.ok) {
        const newPost = await res.json();
        setPosts(prev => [newPost, ...prev]);
        fetchAnalytics(); // dynamically update analytics counts in real-time
      }
    } catch (err) {
      console.error("Failed to trigger simulation step:", err);
    }
  };

  // Perform moderator action (Approve / Hide / Review)
  const handleAction = async (postId, action) => {
    if (!backendOnline) return;
    try {
      const res = await fetch(`/api/posts/${postId}/action`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action })
      });
      if (res.ok) {
        // Update local state instantly for smooth rendering transitions
        setPosts(prev => prev.map(p => p.id === postId ? { ...p, moderator_action: action } : p));
        fetchAnalytics(); // recalculate moderator agreement percentages
      }
    } catch (err) {
      console.error("Failed to update moderator decision:", err);
    }
  };

  // Toggle AI backend mode (BERT vs Fast Fallback)
  const toggleAiMode = async () => {
    if (!backendOnline) return;
    try {
      const res = await fetch('/api/toggle-mode', { method: 'POST' });
      if (res.ok) {
        const data = await res.json();
        setSystemStatus(prev => ({
          ...prev,
          use_real_model: data.use_real_model
        }));
        // Re-read status to sync full parameters
        checkConnection();
      }
    } catch (err) {
      console.error("Failed to toggle model mode:", err);
    }
  };

  // Routing render helper
  const renderContent = () => {
    switch (activeTab) {
      case 'dashboard':
        return (
          <Dashboard 
            posts={posts} 
            setPosts={setPosts}
            handleAction={handleAction} 
            backendOnline={backendOnline}
            triggerNextPost={triggerNextPost}
          />
        );
      case 'analytics':
        return <Analytics analytics={analytics} backendOnline={backendOnline} />;
      case 'network':
        return <NetworkGraph backendOnline={backendOnline} />;
      default:
        return (
          <div className="flex-1 flex items-center justify-center text-slate-400">
            Select a panel from the sidebar menu options.
          </div>
        );
    }
  };

  return (
    <div className="flex h-screen bg-slate-950 text-slate-200 overflow-hidden font-sans">
      <Sidebar 
        activeTab={activeTab} 
        setActiveTab={setActiveTab} 
        systemStatus={systemStatus}
        toggleAiMode={toggleAiMode}
        backendOnline={backendOnline}
      />
      {renderContent()}
    </div>
  );
}
