import React from 'react';
import { CheckCircle2, Clock, Loader2, AlertCircle } from 'lucide-react';

const STAGES = [
  { id: 1, name: 'Ingestion', desc: 'SHA-256 & Type Guard', key: 'ingestion' },
  { id: 2, name: 'Metadata', desc: 'EXIF & Codec Specs', key: 'metadata' },
  { id: 3, name: 'Preprocessing', desc: 'Frames & Keyframes', key: 'preprocessing' },
  { id: 4, name: 'Detection', desc: 'Multimodal Ensembles', key: 'detection' },
  { id: 5, name: 'Localization', desc: 'Heatmaps & Bounding Boxes', key: 'localization' },
  { id: 6, name: 'Fingerprints', desc: 'pHash, dHash, Vector', key: 'fingerprinting' },
  { id: 7, name: 'Similarity', desc: 'Local & External Matches', key: 'similarity' },
  { id: 8, name: 'Provenance', desc: 'Lineage Graph & Origin', key: 'provenance' },
  { id: 9, name: 'Evidence & Report', desc: 'Standardized Case Report', key: 'report' },
];

export default function PipelineTracker({ status, activeStage = 9, error = null }) {
  const isCompleted = status === 'COMPLETED' || status === 'PARTIAL';
  const isProcessing = status === 'PROCESSING';

  return (
    <div className="glass-panel" style={{ padding: '1.5rem', marginBottom: '1.5rem' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1.25rem' }}>
        <div>
          <h3 style={{ fontSize: '1rem', fontWeight: 700, color: '#F8FAFC', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <span>9-Stage Forensic Analysis Pipeline</span>
            {isProcessing && <Loader2 size={16} color="#06B6D4" className="animate-spin" />}
          </h3>
          <p style={{ fontSize: '0.75rem', color: '#94A3B8' }}>
            Standard operating procedure tracking from ingest to verified evidence reporting
          </p>
        </div>

        <div>
          <span className={`badge badge-${(status || 'PENDING').toLowerCase()}`}>
            {status || 'PENDING'}
          </span>
        </div>
      </div>

      {/* Steps Track */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(130px, 1fr))',
        gap: '0.75rem',
      }}>
        {STAGES.map((stage, idx) => {
          let stageState = 'pending';
          if (isCompleted) {
            stageState = 'done';
          } else if (isProcessing) {
            if (idx < activeStage) stageState = 'done';
            else if (idx === activeStage) stageState = 'active';
            else stageState = 'pending';
          }

          let borderColor = 'rgba(255, 255, 255, 0.08)';
          let bgColor = 'rgba(15, 23, 42, 0.3)';
          let textColor = '#64748B';

          if (stageState === 'done') {
            borderColor = 'rgba(16, 185, 129, 0.3)';
            bgColor = 'rgba(16, 185, 129, 0.08)';
            textColor = '#10B981';
          } else if (stageState === 'active') {
            borderColor = 'rgba(6, 182, 212, 0.5)';
            bgColor = 'rgba(6, 182, 212, 0.12)';
            textColor = '#38BDF8';
          }

          return (
            <div
              key={stage.id}
              style={{
                border: `1px solid ${borderColor}`,
                background: bgColor,
                borderRadius: '8px',
                padding: '0.75rem',
                display: 'flex',
                flexDirection: 'column',
                gap: '0.35rem',
                position: 'relative',
                overflow: 'hidden',
              }}
            >
              {stageState === 'active' && <div className="laser-bar" />}

              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <span style={{ fontSize: '0.7rem', fontWeight: 700, color: textColor, fontFamily: 'var(--font-mono)' }}>
                  STAGE 0{stage.id}
                </span>
                {stageState === 'done' ? (
                  <CheckCircle2 size={14} color="#10B981" />
                ) : stageState === 'active' ? (
                  <Loader2 size={14} color="#06B6D4" className="animate-spin" />
                ) : (
                  <Clock size={14} color="#475569" />
                )}
              </div>

              <div style={{ fontSize: '0.8rem', fontWeight: 600, color: stageState !== 'pending' ? '#F8FAFC' : '#64748B' }}>
                {stage.name}
              </div>

              <div style={{ fontSize: '0.65rem', color: '#64748B', lineHeight: 1.2 }}>
                {stage.desc}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
