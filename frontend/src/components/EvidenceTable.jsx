import React, { useState } from 'react';
import { Database, Filter, Search, Info } from 'lucide-react';

export default function EvidenceTable({ evidenceItems = [], limitations = [] }) {
  const [activeTab, setActiveTab] = useState('ALL');
  const [searchTerm, setSearchTerm] = useState('');

  // Normalize items if structured as classified dict or list
  let itemsList = [];
  if (Array.isArray(evidenceItems)) {
    itemsList = evidenceItems;
  } else if (typeof evidenceItems === 'object' && evidenceItems !== null) {
    for (const [cat, list] of Object.entries(evidenceItems)) {
      if (Array.isArray(list)) {
        itemsList.push(...list.map(item => ({ ...item, category: cat })));
      }
    }
  }

  // Filter items
  const filtered = itemsList.filter((item) => {
    const matchesTab = activeTab === 'ALL' || item.category?.toUpperCase() === activeTab;
    const matchesSearch = !searchTerm || (item.description || '').toLowerCase().includes(searchTerm.toLowerCase());
    return matchesTab && matchesSearch;
  });

  const getCategoryBadge = (cat) => {
    switch (cat?.toUpperCase()) {
      case 'FACT':
        return <span className="badge badge-fact">[FACT]</span>;
      case 'OBSERVATION':
        return <span className="badge badge-observation">[OBSERVATION]</span>;
      case 'INFERENCE':
        return <span className="badge badge-inference">[INFERENCE]</span>;
      case 'UNCERTAINTY':
        return <span className="badge badge-uncertainty">[UNCERTAINTY]</span>;
      default:
        return <span className="badge badge-inconclusive">[{cat}]</span>;
    }
  };

  return (
    <div className="glass-panel" style={{ padding: '1.75rem', marginBottom: '1.5rem' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1.25rem' }}>
        <div>
          <h3 style={{ fontSize: '1.15rem', fontWeight: 700, color: '#F8FAFC', marginBottom: '0.25rem' }}>
            Forensic Evidentiary Classification Log
          </h3>
          <p style={{ fontSize: '0.8rem', color: '#94A3B8' }}>
            Strict epistemic separation of empirical facts, algorithmic observations, inferences, and uncertainties
          </p>
        </div>

        {/* Tab Filters */}
        <div style={{ display: 'flex', gap: '0.5rem', background: 'rgba(15, 23, 42, 0.6)', padding: '0.25rem', borderRadius: '8px', border: '1px solid rgba(255, 255, 255, 0.08)' }}>
          {['ALL', 'FACT', 'OBSERVATION', 'INFERENCE', 'UNCERTAINTY'].map((tab) => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              style={{
                background: activeTab === tab ? 'rgba(6, 182, 212, 0.2)' : 'transparent',
                color: activeTab === tab ? '#38BDF8' : '#94A3B8',
                border: activeTab === tab ? '1px solid rgba(6, 182, 212, 0.4)' : '1px solid transparent',
                padding: '0.35rem 0.75rem',
                borderRadius: '6px',
                fontSize: '0.75rem',
                fontWeight: 600,
                fontFamily: 'var(--font-mono)',
              }}
            >
              {tab}
            </button>
          ))}
        </div>
      </div>

      {/* Search Input */}
      <div style={{ marginBottom: '1rem', position: 'relative' }}>
        <input
          type="text"
          placeholder="Filter evidence descriptions..."
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
          style={{
            width: '100%',
            background: 'rgba(15, 23, 42, 0.4)',
            border: '1px solid rgba(255, 255, 255, 0.1)',
            padding: '0.6rem 1rem 0.6rem 2.25rem',
            borderRadius: '8px',
            color: '#F8FAFC',
            fontSize: '0.85rem',
            outline: 'none',
          }}
        />
        <Search size={16} color="#64748B" style={{ position: 'absolute', left: '0.75rem', top: '50%', transform: 'translateY(-50%)' }} />
      </div>

      {/* Table */}
      <div style={{ overflowX: 'auto', border: '1px solid rgba(255, 255, 255, 0.08)', borderRadius: '8px' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.85rem', textAlign: 'left' }}>
          <thead>
            <tr style={{ background: 'rgba(15, 23, 42, 0.8)', borderBottom: '1px solid rgba(255, 255, 255, 0.08)' }}>
              <th style={{ padding: '0.75rem 1rem', color: '#94A3B8', fontWeight: 600, width: '150px' }}>Category</th>
              <th style={{ padding: '0.75rem 1rem', color: '#94A3B8', fontWeight: 600 }}>Description</th>
              <th style={{ padding: '0.75rem 1rem', color: '#94A3B8', fontWeight: 600, width: '140px' }}>Source Stage</th>
              <th style={{ padding: '0.75rem 1rem', color: '#94A3B8', fontWeight: 600, width: '100px' }}>Confidence</th>
            </tr>
          </thead>
          <tbody>
            {filtered.length > 0 ? (
              filtered.map((item, idx) => (
                <tr
                  key={idx}
                  style={{
                    borderBottom: '1px solid rgba(255, 255, 255, 0.04)',
                    background: idx % 2 === 0 ? 'transparent' : 'rgba(255, 255, 255, 0.01)',
                  }}
                >
                  <td style={{ padding: '0.75rem 1rem' }}>
                    {getCategoryBadge(item.category)}
                  </td>
                  <td style={{ padding: '0.75rem 1rem', color: '#F8FAFC', lineHeight: 1.4 }}>
                    {item.description}
                  </td>
                  <td style={{ padding: '0.75rem 1rem', color: '#94A3B8', fontFamily: 'var(--font-mono)', fontSize: '0.75rem' }}>
                    {item.source_stage || 'pipeline'}
                  </td>
                  <td style={{ padding: '0.75rem 1rem', color: '#38BDF8', fontFamily: 'var(--font-mono)', fontSize: '0.75rem' }}>
                    {item.confidence !== undefined && item.confidence !== null ? `${(item.confidence * 100).toFixed(0)}%` : '—'}
                  </td>
                </tr>
              ))
            ) : (
              <tr>
                <td colSpan={4} style={{ padding: '2rem', textAlign: 'center', color: '#64748B' }}>
                  No evidence items found for this filter.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>

      {/* Limitations Footnote */}
      {limitations.length > 0 && (
        <div style={{
          marginTop: '1.25rem',
          background: 'rgba(100, 116, 139, 0.08)',
          border: '1px solid rgba(100, 116, 139, 0.2)',
          borderRadius: '8px',
          padding: '1rem',
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.5rem', color: '#94A3B8', fontSize: '0.8rem', fontWeight: 600 }}>
            <Info size={16} />
            <span>Epistemic Limitations & Technical Caveats [UNCERTAINTY]:</span>
          </div>
          <ul style={{ paddingLeft: '1.5rem', fontSize: '0.75rem', color: '#94A3B8', display: 'flex', flexDirection: 'column', gap: '0.25rem' }}>
            {limitations.map((lim, i) => (
              <li key={i}>{lim}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
