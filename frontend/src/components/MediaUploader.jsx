import React, { useState, useRef } from 'react';
import { Upload, FileImage, FileVideo, Music, AlertCircle, CheckCircle2, ShieldAlert, ArrowRight, Loader2 } from 'lucide-react';
import { api } from '../services/api';

export default function MediaUploader({ onMediaUploaded, onStartAnalysis, isAnalyzing }) {
  const [dragActive, setDragActive] = useState(false);
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [uploadResult, setUploadResult] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState(null);

  const fileInputRef = useRef(null);

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const processFile = async (file) => {
    setError(null);
    setSelectedFile(file);

    // Create local preview
    if (file.type.startsWith('image/') || file.type.startsWith('video/')) {
      const url = URL.createObjectURL(file);
      setPreviewUrl(url);
    } else {
      setPreviewUrl(null);
    }

    // Auto upload to backend
    setUploading(true);
    try {
      const result = await api.uploadMedia(file);
      setUploadResult(result);
      if (onMediaUploaded) onMediaUploaded(result);
    } catch (err) {
      setError(err.message || 'Failed to upload media file');
    } finally {
      setUploading(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      processFile(e.dataTransfer.files[0]);
    }
  };

  const handleChange = (e) => {
    e.preventDefault();
    if (e.target.files && e.target.files[0]) {
      processFile(e.target.files[0]);
    }
  };

  const getMediaIcon = (type) => {
    if (!type) return <Upload size={32} color="#06B6D4" />;
    if (type.startsWith('image/')) return <FileImage size={32} color="#06B6D4" />;
    if (type.startsWith('video/')) return <FileVideo size={32} color="#8B5CF6" />;
    if (type.startsWith('audio/')) return <Music size={32} color="#F59E0B" />;
    return <Upload size={32} color="#06B6D4" />;
  };

  return (
    <div className="glass-panel" style={{ padding: '2rem' }}>
      <div style={{ marginBottom: '1.5rem' }}>
        <h2 style={{ fontSize: '1.25rem', fontWeight: 700, color: '#F8FAFC', marginBottom: '0.25rem' }}>
          Media Evidence Ingestion
        </h2>
        <p style={{ fontSize: '0.85rem', color: '#94A3B8' }}>
          Upload digital media (image, video, or audio) to extract cryptographic hashes, technical metadata, and execute deep multimodal forensics.
        </p>
      </div>

      {/* Drag & Drop Area */}
      <div
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
        onClick={() => fileInputRef.current?.click()}
        style={{
          border: `2px dashed ${dragActive ? '#06B6D4' : 'rgba(255, 255, 255, 0.15)'}`,
          background: dragActive ? 'rgba(6, 182, 212, 0.05)' : 'rgba(15, 23, 42, 0.4)',
          borderRadius: '12px',
          padding: '2.5rem 1.5rem',
          textAlign: 'center',
          cursor: 'pointer',
          transition: 'all 0.2s ease',
          position: 'relative',
        }}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept="image/*,video/*,audio/*"
          onChange={handleChange}
          style={{ display: 'none' }}
        />

        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '1rem' }}>
          <div style={{
            width: '64px',
            height: '64px',
            borderRadius: '50%',
            background: 'rgba(6, 182, 212, 0.1)',
            border: '1px solid rgba(6, 182, 212, 0.25)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
          }}>
            {uploading ? (
              <Loader2 size={32} color="#06B6D4" className="animate-spin" />
            ) : (
              getMediaIcon(selectedFile?.type)
            )}
          </div>

          <div>
            <p style={{ fontSize: '1rem', fontWeight: 600, color: '#F8FAFC', marginBottom: '0.25rem' }}>
              {selectedFile ? selectedFile.name : 'Drop media file here, or click to browse'}
            </p>
            <p style={{ fontSize: '0.75rem', color: '#64748B' }}>
              Supports JPEG, PNG, WEBP, MP4, AVI, MOV, WAV, MP3, FLAC (Max 500 MB)
            </p>
          </div>
        </div>
      </div>

      {/* Error Banner */}
      {error && (
        <div style={{
          marginTop: '1rem',
          padding: '0.75rem 1rem',
          background: 'rgba(239, 68, 68, 0.15)',
          border: '1px solid rgba(239, 68, 68, 0.3)',
          borderRadius: '8px',
          display: 'flex',
          alignItems: 'center',
          gap: '0.75rem',
          color: '#F87171',
          fontSize: '0.85rem',
        }}>
          <AlertCircle size={18} />
          <span>{error}</span>
        </div>
      )}

      {/* Ingested File Details Card */}
      {uploadResult && (
        <div style={{
          marginTop: '1.5rem',
          background: 'rgba(15, 23, 42, 0.6)',
          border: '1px solid rgba(6, 182, 212, 0.25)',
          borderRadius: '10px',
          padding: '1.25rem',
        }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1rem' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <CheckCircle2 size={18} color="#10B981" />
              <span style={{ fontSize: '0.85rem', fontWeight: 600, color: '#10B981' }}>
                INGESTION COMPLETE [FACT]
              </span>
            </div>
            <span className="badge badge-fact" style={{ fontSize: '0.7rem' }}>
              ID: {uploadResult.media_id.slice(0, 8)}...
            </span>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '1rem', fontSize: '0.8rem' }}>
            <div>
              <span style={{ color: '#64748B', display: 'block' }}>Media Type:</span>
              <span style={{ color: '#F8FAFC', fontWeight: 600, textTransform: 'capitalize' }}>
                {uploadResult.media_type}
              </span>
            </div>
            <div>
              <span style={{ color: '#64748B', display: 'block' }}>File Size:</span>
              <span style={{ color: '#F8FAFC', fontWeight: 600 }}>
                {(uploadResult.size_bytes / 1024).toFixed(1)} KB
              </span>
            </div>
            <div style={{ gridColumn: 'span 2' }}>
              <span style={{ color: '#64748B', display: 'block' }}>Cryptographic Hash (SHA-256):</span>
              <code style={{ color: '#38BDF8', fontSize: '0.75rem', wordBreak: 'break-all' }}>
                {uploadResult.sha256}
              </code>
            </div>
          </div>

          {/* Action to trigger full pipeline */}
          <div style={{ marginTop: '1.25rem', display: 'flex', justifyContent: 'flex-end' }}>
            <button
              onClick={() => onStartAnalysis && onStartAnalysis(uploadResult.media_id)}
              disabled={isAnalyzing}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '0.5rem',
                background: 'linear-gradient(135deg, #06B6D4 0%, #3B82F6 100%)',
                color: '#FFFFFF',
                border: 'none',
                padding: '0.75rem 1.5rem',
                borderRadius: '8px',
                fontSize: '0.9rem',
                fontWeight: 600,
                boxShadow: '0 0 20px rgba(6, 182, 212, 0.4)',
              }}
            >
              {isAnalyzing ? (
                <>
                  <Loader2 size={18} className="animate-spin" />
                  Running 9-Stage Forensic Pipeline...
                </>
              ) : (
                <>
                  <span>Execute Full Forensic Pipeline</span>
                  <ArrowRight size={18} />
                </>
              )}
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
