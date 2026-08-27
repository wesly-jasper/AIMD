import React, { useState, useRef } from 'react';
import { Eye, Video, Waves, Play, Pause, Layers, AlertTriangle } from 'lucide-react';

export default function ForensicViewer({ media, detections = [], suspiciousSegments = [], frameScores = [] }) {
  const [showOverlays, setShowOverlays] = useState(true);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);

  const videoRef = useRef(null);

  if (!media) return null;

  const isVideo = media.media_type === 'video';
  const isAudio = media.media_type === 'audio';
  const isImage = media.media_type === 'image';

  // Extract all flagged regions from detections
  const allRegions = detections.flatMap(d => d.regions || []);

  const togglePlay = () => {
    if (!videoRef.current) return;
    if (isPlaying) {
      videoRef.current.pause();
      setIsPlaying(false);
    } else {
      videoRef.current.play();
      setIsPlaying(true);
    }
  };

  const handleTimeUpdate = () => {
    if (videoRef.current) {
      setCurrentTime(videoRef.current.currentTime);
    }
  };

  const handleSeek = (time) => {
    if (videoRef.current) {
      videoRef.current.currentTime = time;
      setCurrentTime(time);
    }
  };

  return (
    <div className="glass-panel" style={{ padding: '1.75rem', marginBottom: '1.5rem' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1.25rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
          {isVideo && <Video size={20} color="#8B5CF6" />}
          {isAudio && <Waves size={20} color="#F59E0B" />}
          {isImage && <Eye size={20} color="#06B6D4" />}
          <div>
            <h3 style={{ fontSize: '1.15rem', fontWeight: 700, color: '#F8FAFC' }}>
              Interactive Forensic Media Inspector
            </h3>
            <p style={{ fontSize: '0.75rem', color: '#94A3B8' }}>
              Spatial & temporal anomaly visualization
            </p>
          </div>
        </div>

        {allRegions.length > 0 && (
          <button
            onClick={() => setShowOverlays(!showOverlays)}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '0.5rem',
              background: showOverlays ? 'rgba(6, 182, 212, 0.2)' : 'rgba(255, 255, 255, 0.05)',
              border: showOverlays ? '1px solid rgba(6, 182, 212, 0.4)' : '1px solid rgba(255, 255, 255, 0.1)',
              color: showOverlays ? '#38BDF8' : '#94A3B8',
              padding: '0.35rem 0.75rem',
              borderRadius: '6px',
              fontSize: '0.75rem',
              fontWeight: 600,
            }}
          >
            <Layers size={14} />
            {showOverlays ? 'Hide Anomaly Overlays' : 'Show Anomaly Overlays'}
          </button>
        )}
      </div>

      {/* Main Media Preview Area */}
      <div style={{
        background: '#030712',
        borderRadius: '12px',
        overflow: 'hidden',
        border: '1px solid rgba(255, 255, 255, 0.08)',
        position: 'relative',
        minHeight: '260px',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
      }}>
        {/* IMAGE PREVIEW */}
        {isImage && (
          <div style={{ position: 'relative', maxWidth: '100%', maxHeight: '500px' }}>
            <img
              src={`http://localhost:8000/api/v1/media/${media.id}/download`}
              alt="Investigated Asset"
              onError={(e) => {
                // Fallback placeholder display if direct download endpoint not served
                e.target.style.display = 'none';
              }}
              style={{ maxHeight: '480px', width: 'auto', display: 'block', margin: '0 auto', objectFit: 'contain' }}
            />

            {/* Bounding Box Overlays */}
            {showOverlays && allRegions.map((reg, i) => {
              if (!reg.bbox || reg.bbox.length !== 4) return null;
              const [x, y, w, h] = reg.bbox;
              return (
                <div
                  key={i}
                  style={{
                    position: 'absolute',
                    left: `${x}px`,
                    top: `${y}px`,
                    width: `${w}px`,
                    height: `${h}px`,
                    border: '2px solid #EF4444',
                    backgroundColor: 'rgba(239, 68, 68, 0.15)',
                    boxShadow: '0 0 10px rgba(239, 68, 68, 0.5)',
                    pointerEvents: 'none',
                  }}
                >
                  <span style={{
                    position: 'absolute',
                    top: '-20px',
                    left: '0',
                    background: '#EF4444',
                    color: '#FFFFFF',
                    fontSize: '0.65rem',
                    padding: '0.1rem 0.35rem',
                    fontWeight: 700,
                    fontFamily: 'var(--font-mono)',
                    borderRadius: '2px',
                  }}>
                    {reg.type || 'ANOMALY'} ({(reg.confidence * 100).toFixed(0)}%)
                  </span>
                </div>
              );
            })}
          </div>
        )}

        {/* VIDEO PREVIEW & TIMELINE */}
        {isVideo && (
          <div style={{ width: '100%', padding: '1rem' }}>
            <video
              ref={videoRef}
              src={`http://localhost:8000/api/v1/media/${media.id}/download`}
              onTimeUpdate={handleTimeUpdate}
              style={{ width: '100%', maxHeight: '400px', borderRadius: '8px', background: '#000000' }}
              controls
            />

            {/* Suspicious Segments Timeline Marker */}
            {suspiciousSegments.length > 0 && (
              <div style={{ marginTop: '1rem', background: 'rgba(15, 23, 42, 0.8)', padding: '1rem', borderRadius: '8px', border: '1px solid rgba(239, 68, 68, 0.3)' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: '#F87171', fontSize: '0.8rem', fontWeight: 600, marginBottom: '0.5rem' }}>
                  <AlertTriangle size={16} />
                  <span>Flagged Suspicious Temporal Segments ({suspiciousSegments.length})</span>
                </div>

                <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem' }}>
                  {suspiciousSegments.map((seg, idx) => (
                    <button
                      key={idx}
                      onClick={() => handleSeek(seg.start_timestamp)}
                      style={{
                        background: 'rgba(239, 68, 68, 0.15)',
                        border: '1px solid rgba(239, 68, 68, 0.4)',
                        color: '#FCA5A5',
                        padding: '0.35rem 0.75rem',
                        borderRadius: '6px',
                        fontSize: '0.75rem',
                        fontFamily: 'var(--font-mono)',
                      }}
                    >
                      {seg.start_timestamp.toFixed(2)}s – {seg.end_timestamp.toFixed(2)}s (Peak: {(seg.peak_score * 100).toFixed(0)}%)
                    </button>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        {/* AUDIO SPECTRAL VIEW */}
        {isAudio && (
          <div style={{ width: '100%', padding: '2rem', textAlign: 'center' }}>
            <Waves size={48} color="#F59E0B" style={{ margin: '0 auto 1rem auto' }} />
            <p style={{ fontSize: '0.9rem', color: '#F8FAFC', fontWeight: 600, marginBottom: '0.5rem' }}>
              Acoustic Spectral Analysis Track
            </p>
            <p style={{ fontSize: '0.75rem', color: '#94A3B8', maxWidth: '500px', margin: '0 auto 1.5rem auto' }}>
              Spectral flux, centroid distribution, and zero-crossing rates analyzed via librosa FFT.
            </p>

            <audio
              src={`http://localhost:8000/api/v1/media/${media.id}/download`}
              controls
              style={{ width: '100%', maxWidth: '500px' }}
            />
          </div>
        )}
      </div>
    </div>
  );
}
