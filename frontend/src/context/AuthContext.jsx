import React, { createContext, useContext, useState, useEffect } from 'react';

const AuthContext = createContext(null);

export function AuthProvider({ children }) {
  // Pre-seed with default demo admin for smooth first load, allowing immediate testing
  const [user, setUser] = useState({
    id: "admin-default-id",
    email: "admin@cybershield.org",
    role: "admin"
  });
  const [token, setToken] = useState(localStorage.getItem('cyber_auth_token') || 'demo-admin-token');
  const [isLoginModalOpen, setIsLoginModalOpen] = useState(false);

  useEffect(() => {
    const savedToken = localStorage.getItem('cyber_auth_token');
    if (savedToken) {
      // Validate token with backend /api/auth/me
      fetch('/api/auth/me', {
        headers: { 'Authorization': `Bearer ${savedToken}` }
      })
      .then(res => {
        if (res.ok) return res.json();
        throw new Error('Token expired');
      })
      .then(userData => {
        setUser(userData);
        setToken(savedToken);
      })
      .catch(() => {
        // If expired, clear token and fallback to demo admin
        localStorage.removeItem('cyber_auth_token');
        setToken('demo-admin-token');
        setUser({ id: "admin-default-id", email: "admin@cybershield.org", role: "admin" });
      });
    }
  }, []);

  const login = async (email, password) => {
    try {
      const res = await fetch('/api/auth/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, password })
      });
      if (!res.ok) {
        const errorData = await res.json().catch(() => ({}));
        throw new Error(errorData.detail || 'Login failed');
      }
      const data = await res.json();
      setToken(data.access_token);
      setUser(data.user);
      localStorage.setItem('cyber_auth_token', data.access_token);
      setIsLoginModalOpen(false);
      return { success: true };
    } catch (err) {
      return { success: false, error: err.message };
    }
  };

  const quickSwitch = (role) => {
    if (role === 'admin') {
      setUser({ id: "admin-seed-id", email: "admin@cybershield.org", role: "admin" });
      setToken('demo-admin-token');
      localStorage.setItem('cyber_auth_token', 'demo-admin-token');
    } else if (role === 'moderator') {
      setUser({ id: "mod-seed-id", email: "mod@cybershield.org", role: "moderator" });
      setToken('demo-mod-token');
      localStorage.setItem('cyber_auth_token', 'demo-mod-token');
    } else {
      setUser(null);
      setToken(null);
      localStorage.removeItem('cyber_auth_token');
    }
    setIsLoginModalOpen(false);
  };

  const logout = () => {
    setUser(null);
    setToken(null);
    localStorage.removeItem('cyber_auth_token');
    setIsLoginModalOpen(true);
  };

  return (
    <AuthContext.Provider value={{
      user,
      token,
      role: user ? user.role : 'guest',
      isAuthenticated: !!user,
      login,
      logout,
      quickSwitch,
      isLoginModalOpen,
      setIsLoginModalOpen
    }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
}
