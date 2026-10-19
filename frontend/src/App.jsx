import React, { useState, useEffect } from 'react';
import { AuthProvider, useAuth } from './context/AuthContext';
import Sidebar from './components/Sidebar';
import Dashboard from './components/Dashboard';
import ReviewQueue from './components/ReviewQueue';
import Analytics from './components/Analytics';
import NetworkGraph from './components/NetworkGraph';
import FairnessPanel from './components/FairnessPanel';
import LoginModal from './components/LoginModal';

function MainApp() {
  const { role, user } = useAuth();
  const [activeTab, setActiveTab] = useState('dashboard');
  const [backendOnline, setBackendOnline] = useState(false);
  const [posts, setPosts] = useState([]);
  const [pendingQueueCount, setPendingQueueCount] = useState(0);
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
    active_model: 'Keyword Fallback Classifier'
  });

  const [evaluation, setEvaluation] = useState(null);
  const [loadingEvaluation, setLoadingEvaluation] = useState(false);

  // Role routing safety check: if moderator somehow lands on admin-only tabs, redirect to dashboard
  useEffect(() => {
    if (role === 'moderator' && (activeTab === 'analytics' || activeTab === 'fairness')) {
      setActiveTab('dashboard');
    }
  }, [role, activeTab]);

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

  // Fetch pending review queue count
  const fetchQueueCount = async () => {
    if (!backendOnline) return;
    try {
      const res = await fetch('/api/queue?status=pending');
      if (res.ok) {
        const data = await res.json();
        setPendingQueueCount(data.length);
      }
    } catch (err) {
      // quiet fail
    }
  };

  // Fetch posts from database
  const fetchPosts = async () => {
    if (!backendOnline) return;
    try {
      const res = await fetch('/api/posts');
      if (res.ok) {
        const data = await res.json();
        setPosts(data);
      }
    } catch (err) {
      console.error("Failed to fetch database logs:", err);
    }
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

  const fetchEvaluation = async () => {
    if (!backendOnline) return;
    setLoadingEvaluation(true);
    try {
      const res = await fetch('/api/evaluation');
      if (res.ok) {
        const data = await res.json();
        setEvaluation(data);
      }
    } catch (err) {
      console.error("Failed to fetch evaluation metrics:", err);
    } finally {
      setLoadingEvaluation(false);
    }
  };

  // Heartbeat polling
  useEffect(() => {
    checkConnection();
    const interval = setInterval(() => {
      checkConnection();
      fetchQueueCount();
    }, 5000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    if (backendOnline) {
      fetchPosts();
      fetchAnalytics();
      fetchQueueCount();
    }
  }, [backendOnline]);

  useEffect(() => {
    if (activeTab === 'analytics') {
      fetchEvaluation();
    }
  }, [activeTab, backendOnline]);

  // Single post step trigger (for manual fallback button)
  const triggerNextPost = async () => {
    if (!backendOnline) return;
    try {
      const res = await fetch('/api/simulator/next');
      if (res.ok) {
        const newPost = await res.json();
        setPosts(prev => [newPost, ...prev]);
        fetchAnalytics();
        fetchQueueCount();
      }
    } catch (err) {
      console.error("Failed to trigger simulation step:", err);
    }
  };

  // Handle quick action (Approve / Hide)
  const handleAction = async (postId, action) => {
    if (!backendOnline) return;
    try {
      const res = await fetch(`/api/posts/${postId}/action`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action })
      });
      if (res.ok) {
        setPosts(prev => prev.map(p => p.id === postId ? { ...p, moderator_action: action } : p));
        fetchAnalytics();
        fetchQueueCount();
      }
    } catch (err) {
      console.error("Failed to update moderator action:", err);
    }
  };

  // Toggle AI model mode
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
        checkConnection();
      }
    } catch (err) {
      console.error("Failed to toggle model mode:", err);
    }
  };

  // View routing helper
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
            setActiveTab={setActiveTab}
            refreshAnalytics={fetchAnalytics}
          />
        );
      case 'queue':
        return <ReviewQueue backendOnline={backendOnline} />;
      case 'network':
        return <NetworkGraph backendOnline={backendOnline} />;
      case 'analytics':
        return (
          <Analytics 
            analytics={analytics} 
            backendOnline={backendOnline} 
            evaluation={evaluation}
            loadingEvaluation={loadingEvaluation}
            refetchEvaluation={fetchEvaluation}
            setActiveTab={setActiveTab}
          />
        );
      case 'fairness':
        return <FairnessPanel backendOnline={backendOnline} />;
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
        pendingQueueCount={pendingQueueCount}
      />
      {renderContent()}
      <LoginModal />
    </div>
  );
}

export default function App() {
  return (
    <AuthProvider>
      <MainApp />
    </AuthProvider>
  );
}
