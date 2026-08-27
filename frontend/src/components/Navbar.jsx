import React from 'react';
import { ShieldCheck, ShieldAlert, Cpu, Activity, Database, FileText } from 'lucide-react';

export default function Navbar({ systemStatus, currentAnalysis, onViewReport, onNewCase }) {
  const isOnline = systemStatus?.status === 'ok';

  return (
    <header className="glass-panel" style={{ borderRadius: 0, borderTop: 0, borderLeft: 0, borderRight: 0, padding: '1rem 2rem' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', maxWidth: '1600px', margin: '0 auto' }}>
        {/* Brand */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
          <div style={{
            width: '40px',
            height: '40px',
            borderRadius: '10px',
            background: 'linear-gradient(135deg, #06B6D4 0%, #3B82F6 100%)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            boxShadow: '0 0 15px rgba(6, 182, 212, 0.4)'
          }}>
            <Cpu size={24} color="#FFFFFF" />
          </div>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <span className="font-display" style={{ fontSize: '1.25rem', fontWeight: 800, letterSpacing: '-0.02em', color: '#F8FAFC' }}>
                AIMD
              </span>
              <span style={{ fontSize: '0.7rem', background: 'rgba(6, 182, 212, 0.2)', color: '#38BDF8', padding: '0.15rem 0.5rem', borderRadius: '4px', border: '1px solid rgba(6, 182, 212, 0.4)', fontWeight: 600 }}>
                FORENSICS V2.0
              </span>
            </div>
            <p style={{ fontSize: '0.75rem', color: '#64748B' }}>AI Media Detection & Provenance Platform</p>
          </div>
        </div>

        {/* System Status & Actions */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '1.5rem' }}>
          {/* Active Case Tag */}
          {currentAnalysis && (
            <div style={{
              display: 'flex',
              alignItems: 'center',
              gap: '0.5rem',
              background: 'rgba(15, 23, 42, 0.6)',
              border: '1px solid rgba(255, 255, 255, 0.1)',
              padding: '0.35rem 0.75rem',
              borderRadius: '8px',
              fontSize: '0.8rem',
              fontFamily: 'var(--font-mono)'
            }}>
              <span style={{ color: '#94A3B8' }}>CASE:</span>
              <span style={{ color: '#38BDF8', fontWeight: 600 }}>
                {currentAnalysis.id ? currentAnalysis.id.slice(0, 8) : 'ACTIVE'}
              </span>
              <span className={`badge badge-${(currentAnalysis.assessment || 'INCONCLUSIVE').toLowerCase()}`} style={{ fontSize: '0.65rem', padding: '0.1rem 0.4rem' }}>
                {currentAnalysis.assessment || 'INCONCLUSIVE'}
              </span>
            </div>
          )}

          {/* Connection Status */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.8rem' }}>
            <span style={{
              width: '8px',
              height: '8px',
              borderRadius: '50%',
              backgroundColor: isOnline ? '#10B981' : '#EF4444',
              boxShadow: isOnline ? '0 0 8px #10B981' : '0 0 8px #EF4444'
            }}></span>
            <span style={{ color: isOnline ? '#10B981' : '#EF4444', fontWeight: 500 }}>
              {isOnline ? 'BACKEND ONLINE' : 'BACKEND OFFLINE'}
            </span>
          </div>

          {/* Action Buttons */}
          {currentAnalysis && (
            <button
              onClick={onViewReport}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '0.5rem',
                background: 'rgba(6, 182, 212, 0.15)',
                border: '1px solid rgba(6, 182, 212, 0.4)',
                color: '#38BDF8',
                padding: '0.5rem 1rem',
                borderRadius: '8px',
                fontSize: '0.85rem',
                fontWeight: 600,
              }}
            >
              <FileText size={16} />
              View Report
            </button>
          )}

          <button
            onClick={onNewCase}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '0.5rem',
              background: 'linear-gradient(135deg, #06B6D4 0%, #3B82F6 100%)',
              border: 'none',
              color: '#FFFFFF',
              padding: '0.5rem 1.25rem',
              borderRadius: '8px',
              fontSize: '0.85rem',
              fontWeight: 600,
              boxShadow: '0 0 15px rgba(6, 182, 212, 0.3)'
            }}
          >
            + New Investigation
          </button>
        </div>
      </div>
    </header>
  );
}
