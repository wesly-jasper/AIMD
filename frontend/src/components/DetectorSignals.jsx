import React from 'react';
import { ShieldCheck, ShieldAlert, AlertTriangle, HelpCircle, Eye, Activity, Waves, Video } from 'lucide-react';

export default function DetectorSignals({ detections = [], overallConfidence = 0.0, assessment = 'INCONCLUSIVE' }) {
  const getAssessmentBadge = (ass) => {
    switch (ass?.toUpperCase()) {
      case 'CLEAN':
        return <span className="badge badge-clean">CLEAN</span>;
      case 'SUSPICIOUS':
        return <span className="badge badge-suspicious">SUSPICIOUS</span>;
      case 'UNAVAILABLE':
        return <span className="badge badge-unavailable">UNAVAILABLE</span>;
      default:
        return <span className="badge badge-inconclusive">INCONCLUSIVE</span>;
    }
  };

  const getDetectorIcon = (mType) => {
    switch (mType?.toLowerCase()) {
      case 'face':
        return <Eye size={18} color="#38BDF8" />;
      case 'audio':
        return <Waves size={18} color="#F59E0B" />;
      case 'video':
      case 'temporal':
        return <Video size={18} color="#8B5CF6" />;
      default:
        return <Activity size={18} color="#06B6D4" />;
    }
  };

  return (
    <div className="glass-panel" style={{ padding: '1.75rem' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1.5rem' }}>
        <div>
          <h3 style={{ fontSize: '1.15rem', fontWeight: 700, color: '#F8FAFC', marginBottom: '0.25rem' }}>
            Multimodal Forensic Detector Signals
          </h3>
          <p style={{ fontSize: '0.8rem', color: '#94A3B8' }}>
            Independent neural and signal-level anomaly extractors
          </p>
        </div>

        {/* Overall Verdict Meter */}
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: '1rem',
          background: 'rgba(15, 23, 42, 0.8)',
          border: '1px solid rgba(255, 255, 255, 0.1)',
          padding: '0.5rem 1rem',
          borderRadius: '8px',
        }}>
          <div>
            <span style={{ fontSize: '0.65rem', color: '#64748B', display: 'block', textTransform: 'uppercase', fontFamily: 'var(--font-mono)' }}>
              COMPOSITE VERDICT
            </span>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              {getAssessmentBadge(assessment)}
              <span style={{ fontSize: '0.9rem', fontWeight: 700, color: '#F8FAFC', fontFamily: 'var(--font-mono)' }}>
                {(overallConfidence * 100).toFixed(1)}%
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Detector Cards Grid */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '1rem' }}>
        {detections.length > 0 ? (
          detections.map((det, idx) => {
            const conf = det.confidence || 0.0;
            const meta = det.metadata || {};
            const regions = det.regions || [];

            return (
              <div
                key={idx}
                style={{
                  background: 'rgba(15, 23, 42, 0.5)',
                  border: '1px solid rgba(255, 255, 255, 0.08)',
                  borderRadius: '10px',
                  padding: '1.25rem',
                  display: 'flex',
                  flexDirection: 'column',
                  gap: '0.75rem',
                }}
              >
                {/* Card Header */}
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                    <div style={{
                      width: '32px',
                      height: '32px',
                      borderRadius: '8px',
                      background: 'rgba(255, 255, 255, 0.05)',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                    }}>
                      {getDetectorIcon(det.manipulation_type)}
                    </div>
                    <div>
                      <div style={{ fontSize: '0.85rem', fontWeight: 600, color: '#F8FAFC' }}>
                        {det.detector}
                      </div>
                      <div style={{ fontSize: '0.7rem', color: '#64748B', textTransform: 'capitalize' }}>
                        Target: {det.manipulation_type}
                      </div>
                    </div>
                  </div>
                  {getAssessmentBadge(det.assessment)}
                </div>

                {/* Confidence Bar */}
                <div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', marginBottom: '0.25rem' }}>
                    <span style={{ color: '#94A3B8' }}>Signal Confidence</span>
                    <span style={{ color: '#F8FAFC', fontFamily: 'var(--font-mono)', fontWeight: 600 }}>
                      {(conf * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div style={{
                    width: '100%',
                    height: '6px',
                    backgroundColor: 'rgba(255, 255, 255, 0.08)',
                    borderRadius: '3px',
                    overflow: 'hidden',
                  }}>
                    <div
                      style={{
                        width: `${Math.min(100, Math.max(0, conf * 100))}%`,
                        height: '100%',
                        backgroundColor:
                          det.assessment === 'SUSPICIOUS' ? '#EF4444' :
                          det.assessment === 'CLEAN' ? '#10B981' : '#06B6D4',
                        borderRadius: '3px',
                        transition: 'width 0.5s ease',
                      }}
                    />
                  </div>
                </div>

                {/* Regions / Notes */}
                {meta.note && (
                  <div style={{ fontSize: '0.75rem', color: '#94A3B8', background: 'rgba(0,0,0,0.2)', padding: '0.5rem', borderRadius: '6px' }}>
                    {meta.note}
                  </div>
                )}

                {regions.length > 0 && (
                  <div style={{ fontSize: '0.75rem', color: '#38BDF8' }}>
                    📍 {regions.length} Localized Region(s) Flagged
                  </div>
                )}
              </div>
            );
          })
        ) : (
          <div style={{ gridColumn: 'span 3', padding: '2rem', textAlign: 'center', color: '#64748B', fontSize: '0.85rem' }}>
            No detector signals available for this media item yet.
          </div>
        )}
      </div>
    </div>
  );
}
