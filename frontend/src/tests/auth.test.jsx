import { describe, it, expect, beforeEach, vi } from 'vitest';
import React from 'react';
import { renderHook, act } from '@testing-library/react';
import { AuthProvider, useAuth } from '../context/AuthContext';

describe('AuthContext Unit Tests', () => {
  beforeEach(() => {
    localStorage.clear();
    vi.restoreAllMocks();
  });

  it('initializes with demo admin user by default', () => {
    const wrapper = ({ children }) => <AuthProvider>{children}</AuthProvider>;
    const { result } = renderHook(() => useAuth(), { wrapper });

    expect(result.current.isAuthenticated).toBe(true);
    expect(result.current.role).toBe('admin');
    expect(result.current.user.email).toBe('admin@cybershield.org');
  });

  it('switches role to moderator properly using quickSwitch', () => {
    const wrapper = ({ children }) => <AuthProvider>{children}</AuthProvider>;
    const { result } = renderHook(() => useAuth(), { wrapper });

    act(() => {
      result.current.quickSwitch('moderator');
    });

    expect(result.current.role).toBe('moderator');
    expect(result.current.user.email).toBe('mod@cybershield.org');
    expect(localStorage.getItem('cyber_auth_token')).toBe('demo-mod-token');
  });

  it('clears state on logout and opens login modal', () => {
    const wrapper = ({ children }) => <AuthProvider>{children}</AuthProvider>;
    const { result } = renderHook(() => useAuth(), { wrapper });

    act(() => {
      result.current.logout();
    });

    expect(result.current.isAuthenticated).toBe(false);
    expect(result.current.role).toBe('guest');
    expect(result.current.user).toBeNull();
    expect(localStorage.getItem('cyber_auth_token')).toBeNull();
    expect(result.current.isLoginModalOpen).toBe(true);
  });
});
