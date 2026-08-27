import React, { useState } from 'react';
import { X, Copy, Check, Printer, FileText, Code2, Download } from 'lucide-react';

export default function ReportModal({ report, isOpen, onClose }) {
  const [activeView, setActiveView] = useState('MARKDOWN');
  const [copied, setCopied] = useState(false);

  if (!isOpen || !report) return null;

  const markdownContent = report.markdown || '';
  const jsonContent = JSON.stringify(report.json || {}, null, 2);

  const handleCopy = () => {
    const textToCopy = activeView === 'MARKDOWN' ? markdownContent : jsonContent;
    navigator.clipboard.writeText(textToCopy);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleDownload = () => {
    const blob = new Blob([activeView === 'MARKDOWN' ? markdownContent : jsonContent], {
      type: activeView === 'MARKDOWN' ? 'text/markdown' : 'application/json',
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `AIMD_Report_${report.analysis_id || 'case'}.${activeView === 'MARKDOWN' ? 'md' : 'json'}`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  return (
    <div style={{
      position: 'fixed',
      top: 0,
      left: 0,
      right: 0,
      bottom: 0,
      backgroundColor: 'rgba(0, 0, 0, 0.8)',
      backdropFilter: 'blur(8px)',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      zIndex: 1000,
      padding: '2rem',
    }}>
      <div className="glass-panel" style={{
        width: '100%',
        maxWidth: '900px',
        maxHeight: '90vh',
        display: 'flex',
        flexDirection: 'column',
        borderRadius: '16px',
        overflow: 'hidden',
        boxShadow: '0 25px 50px -12px rgba(0, 0, 0, 0.7)',
      }}>
        {/* Modal Header */}
        <div style={{
          padding: '1.25rem 1.75rem',
          borderBottom: '1px solid rgba(255, 255, 255, 0.08)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          background: 'rgba(15, 23, 42, 0.9)',
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
            <FileText size={20} color="#06B6D4" />
            <h3 style={{ fontSize: '1.1rem', fontWeight: 700, color: '#F8FAFC' }}>
              Forensic Investigation Report
            </h3>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
            {/* View Switcher */}
            <div style={{ display: 'flex', background: 'rgba(0,0,0,0.3)', padding: '0.2rem', borderRadius: '6px' }}>
              <button
                onClick={() => setActiveView('MARKDOWN')}
                style={{
                  background: activeView === 'MARKDOWN' ? 'rgba(6, 182, 212, 0.2)' : 'transparent',
                  color: activeView === 'MARKDOWN' ? '#38BDF8' : '#94A3B8',
                  border: 'none',
                  padding: '0.3rem 0.6rem',
                  borderRadius: '4px',
                  fontSize: '0.75rem',
                  fontWeight: 600,
                }}
              >
                Markdown
              </button>
              <button
                onClick={() => setActiveView('JSON')}
                style={{
                  background: activeView === 'JSON' ? 'rgba(6, 182, 212, 0.2)' : 'transparent',
                  color: activeView === 'JSON' ? '#38BDF8' : '#94A3B8',
                  border: 'none',
                  padding: '0.3rem 0.6rem',
                  borderRadius: '4px',
                  fontSize: '0.75rem',
                  fontWeight: 600,
                }}
              >
                Raw JSON
              </button>
            </div>

            {/* Actions */}
            <button
              onClick={handleCopy}
              style={{
                background: 'rgba(255, 255, 255, 0.05)',
                border: '1px solid rgba(255, 255, 255, 0.1)',
                color: '#F8FAFC',
                padding: '0.35rem 0.75rem',
                borderRadius: '6px',
                fontSize: '0.75rem',
                display: 'flex',
                alignItems: 'center',
                gap: '0.35rem',
              }}
            >
              {copied ? <Check size={14} color="#10B981" /> : <Copy size={14} />}
              {copied ? 'Copied' : 'Copy'}
            </button>

            <button
              onClick={handleDownload}
              style={{
                background: 'rgba(255, 255, 255, 0.05)',
                border: '1px solid rgba(255, 255, 255, 0.1)',
                color: '#F8FAFC',
                padding: '0.35rem 0.75rem',
                borderRadius: '6px',
                fontSize: '0.75rem',
                display: 'flex',
                alignItems: 'center',
                gap: '0.35rem',
              }}
            >
              <Download size={14} />
              Export
            </button>

            <button
              onClick={onClose}
              style={{
                background: 'transparent',
                border: 'none',
                color: '#94A3B8',
                padding: '0.35rem',
                borderRadius: '6px',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
              }}
            >
              <X size={20} />
            </button>
          </div>
        </div>

        {/* Modal Body */}
        <div style={{
          padding: '2rem',
          overflowY: 'auto',
          flex: 1,
          fontFamily: activeView === 'JSON' ? 'var(--font-mono)' : 'var(--font-sans)',
          fontSize: activeView === 'JSON' ? '0.8rem' : '0.9rem',
          lineHeight: 1.6,
          color: '#E2E8F0',
        }}>
          {activeView === 'MARKDOWN' ? (
            <pre style={{
              whiteSpace: 'pre-wrap',
              wordBreak: 'break-word',
              fontFamily: 'inherit',
              lineHeight: 1.6,
            }}>
              {markdownContent}
            </pre>
          ) : (
            <pre style={{
              whiteSpace: 'pre-wrap',
              wordBreak: 'break-word',
              color: '#38BDF8',
            }}>
              {jsonContent}
            </pre>
          )}
        </div>
      </div>
    </div>
  );
}
