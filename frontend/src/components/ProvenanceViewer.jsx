import React from 'react';
import { GitFork, ExternalLink, Calendar, ShieldCheck, ArrowRight } from 'lucide-react';

export default function ProvenanceViewer({ provenance = {}, sourceTrace = {} }) {
  const nodes = provenance.nodes || [];
  const edges = provenance.edges || [];
  const earliest = sourceTrace.earliest_known_occurrence;
  const candidates = sourceTrace.candidates || [];

  return (
    <div className="glass-panel" style={{ padding: '1.75rem', marginBottom: '1.5rem' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1.25rem' }}>
        <div>
          <h3 style={{ fontSize: '1.15rem', fontWeight: 700, color: '#F8FAFC', marginBottom: '0.25rem' }}>
            Provenance Graph & Source Lineage [INFERENCE]
          </h3>
          <p style={{ fontSize: '0.8rem', color: '#94A3B8' }}>
            Topological reconstruction of discovery occurrences and derivative relationships
          </p>
        </div>

        <span className="badge badge-inference">
          {nodes.length} Nodes · {edges.length} Edges
        </span>
      </div>

      {/* Earliest Known Occurrence Highlight */}
      {earliest ? (
        <div style={{
          background: 'rgba(245, 158, 11, 0.08)',
          border: '1px solid rgba(245, 158, 11, 0.3)',
          borderRadius: '10px',
          padding: '1.25rem',
          marginBottom: '1.5rem',
        }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.5rem' }}>
            <span style={{ fontSize: '0.75rem', fontWeight: 700, color: '#FBBF24', textTransform: 'uppercase', fontFamily: 'var(--font-mono)' }}>
              EARLIEST-KNOWN OCCURRENCE DISCOVERED
            </span>
            <span style={{ fontSize: '0.75rem', color: '#94A3B8' }}>
              Similarity: {(earliest.similarity * 100).toFixed(1)}%
            </span>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: '#F8FAFC', fontWeight: 600, fontSize: '0.95rem', marginBottom: '0.25rem' }}>
            <ExternalLink size={16} color="#FBBF24" />
            <a href={earliest.url} target="_blank" rel="noopener noreferrer" style={{ color: '#FBBF24', textDecoration: 'none' }}>
              {earliest.url || earliest.title || 'Discovered Source URL'}
            </a>
          </div>

          <div style={{ fontSize: '0.75rem', color: '#94A3B8', display: 'flex', gap: '1.5rem', marginTop: '0.5rem' }}>
            <span>Provider: <strong>{earliest.provider || 'Search Engine'}</strong></span>
            {earliest.source_timestamp && (
              <span>Timestamp: <strong>{new Date(earliest.source_timestamp).toLocaleDateString()}</strong></span>
            )}
          </div>
        </div>
      ) : (
        <div style={{
          background: 'rgba(15, 23, 42, 0.5)',
          border: '1px solid rgba(255, 255, 255, 0.06)',
          borderRadius: '8px',
          padding: '1rem',
          fontSize: '0.8rem',
          color: '#94A3B8',
          marginBottom: '1.5rem',
        }}>
          <strong>External Search:</strong> {sourceTrace.provider_status || 'UNAVAILABLE'} — No historical web occurrences discovered or indexed.
        </div>
      )}

      {/* Visual Provenance Flow */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
        <h4 style={{ fontSize: '0.9rem', fontWeight: 600, color: '#F8FAFC' }}>
          Graph Relationships
        </h4>

        {nodes.length > 0 ? (
          <div style={{
            display: 'flex',
            flexWrap: 'wrap',
            alignItems: 'center',
            gap: '1rem',
            padding: '1.5rem',
            background: 'rgba(15, 23, 42, 0.5)',
            border: '1px solid rgba(255, 255, 255, 0.06)',
            borderRadius: '10px',
          }}>
            {nodes.map((node, i) => (
              <React.Fragment key={node.id || i}>
                <div style={{
                  background: node.type === 'uploaded' ? 'rgba(6, 182, 212, 0.15)' : 'rgba(245, 158, 11, 0.15)',
                  border: node.type === 'uploaded' ? '1px solid rgba(6, 182, 212, 0.4)' : '1px solid rgba(245, 158, 11, 0.4)',
                  padding: '0.75rem 1.25rem',
                  borderRadius: '8px',
                  display: 'flex',
                  flexDirection: 'column',
                  gap: '0.25rem',
                }}>
                  <span style={{ fontSize: '0.65rem', textTransform: 'uppercase', color: node.type === 'uploaded' ? '#38BDF8' : '#FBBF24', fontFamily: 'var(--font-mono)', fontWeight: 700 }}>
                    {node.type || 'NODE'}
                  </span>
                  <span style={{ fontSize: '0.85rem', fontWeight: 600, color: '#F8FAFC' }}>
                    {node.label || node.id}
                  </span>
                </div>

                {i < nodes.length - 1 && (
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.25rem', color: '#64748B' }}>
                    <ArrowRight size={18} />
                  </div>
                )}
              </React.Fragment>
            ))}
          </div>
        ) : (
          <div style={{ padding: '1rem', textAlign: 'center', color: '#64748B', fontSize: '0.8rem' }}>
            No graph nodes recorded for this investigation.
          </div>
        )}
      </div>
    </div>
  );
}
