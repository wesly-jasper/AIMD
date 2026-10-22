import { useEffect, useState, type ReactNode } from "react";
import { Link, NavLink, useLocation } from "react-router-dom";
import { useAuth } from "../auth";

export function Mark({ size = 20 }: { size?: number }) {
  // A media frame with one marked area: the product's own object.
  return (
    <svg width={size} height={size} viewBox="0 0 20 20" aria-hidden="true" className="mark">
      <rect x="1.5" y="1.5" width="17" height="17" rx="4" fill="none" stroke="currentColor" strokeWidth="1.5" />
      <rect x="9" y="5" width="6.5" height="6" rx="1" fill="var(--hue)" />
    </svg>
  );
}

const LINKS = [
  { to: "/", label: "Check media" },
  { to: "/history", label: "History" },
  { to: "/folders", label: "Folders" },
];

export function Shell({ children }: { children: ReactNode }) {
  const { user, signOut } = useAuth();
  const [open, setOpen] = useState(false);
  const location = useLocation();
  useEffect(() => setOpen(false), [location.pathname]);

  return (
    <div className="app">
      <header className="topbar">
        <div className="bar-inner">
          <Link to="/" className="brand"><Mark /> <span>AIMD Sentinel</span></Link>
          <nav className={"nav-links" + (open ? " open" : "")} aria-label="Primary" id="primary-nav">
            {LINKS.map((l) => (
              <NavLink key={l.to} to={l.to} end={l.to === "/"} className="link">{l.label}</NavLink>
            ))}
            {user && (
              <span className="account in-menu">
                <span className="who">{user.username}</span>
                <button className="text-btn" onClick={signOut}>Sign out</button>
              </span>
            )}
          </nav>
          {user && (
            <span className="account in-bar">
              <span className="avatar" aria-hidden="true">{user.username.charAt(0).toUpperCase()}</span>
              <span className="who">{user.username}</span>
              <button className="text-btn" onClick={signOut}>Sign out</button>
            </span>
          )}
          <button className="menu-btn" aria-expanded={open} aria-controls="primary-nav" onClick={() => setOpen((v) => !v)}>
            {open ? "Close" : "Menu"}
          </button>
        </div>
      </header>
      <main className="column">{children}</main>
    </div>
  );
}
