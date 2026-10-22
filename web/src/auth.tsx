import { createContext, useCallback, useContext, useEffect, useState, type ReactNode } from "react";
import { getJson, login as apiLogin, onUnauthorised, readSession, signup as apiSignup, writeSession } from "./api";
import type { User } from "./types";

interface AuthState {
  user: User | null;
  /** True until a stored session has been checked against the backend. */
  checking: boolean;
  signIn: (email: string, password: string) => Promise<void>;
  signUp: (email: string, password: string) => Promise<void>;
  signOut: () => void;
}

const AuthContext = createContext<AuthState | null>(null);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [checking, setChecking] = useState<boolean>(() => readSession() !== null);

  // The identity always comes from /auth/me, never from the token's claims: the
  // backend reads the role from the user store, and so does this screen.
  useEffect(() => {
    if (!readSession()) return;
    getJson<User>("/auth/me")
      .then(setUser)
      .catch(() => { writeSession(null); setUser(null); })
      .finally(() => setChecking(false));
  }, []);

  useEffect(() => onUnauthorised(() => setUser(null)), []);

  const signIn = useCallback(async (username: string, password: string) => {
    const token = await apiLogin(username, password);
    writeSession({ token });
    setUser(await getJson<User>("/auth/me"));
  }, []);

  const signUp = useCallback(async (email: string, password: string) => {
    const token = await apiSignup(email, password);
    writeSession({ token });
    setUser(await getJson<User>("/auth/me"));
  }, []);

  const signOut = useCallback(() => {
    writeSession(null);
    setUser(null);
  }, []);

  return (
    <AuthContext.Provider value={{ user, checking, signIn, signUp, signOut }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth(): AuthState {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth outside AuthProvider");
  return ctx;
}
