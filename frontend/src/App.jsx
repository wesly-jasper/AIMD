import React, { useState, useEffect } from 'react';
import Navbar from './components/Navbar';
import MediaUploader from './components/MediaUploader';
import PipelineTracker from './components/PipelineTracker';
import DetectorSignals from './components/DetectorSignals';
import EvidenceTable from './components/EvidenceTable';
import ProvenanceViewer from './components/ProvenanceViewer';
import ReportModal from './components/ReportModal';
import { api } from './services/api';
import { ShieldCheck, ShieldAlert, FileText, History, RefreshCw, Layers } from 'lucide-react';

export default function App() {
  const [systemStatus, setSystemStatus] = useState(null);
  const [currentAnalysis, setCurrentAnalysis] = useState(null);
  const [detections, setDetections] = useState([]);
  const [evidencePackage, setEvidencePackage] = useState(null);
  const [provenance, setProvenance] = useState(null);
  const [sourceTrace, setSourceTrace] = useState({});
  const [report, setReport] = useState(null);
  const [caseHistory, setCaseHistory] = useState([]);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [isReportOpen, setIsReportOpen] = useState(false);
  const [activeTab, setActiveTab] = useState('DETECTIONS'); // DETECTIONS | EVIDENCE | PROVENANCE

  // Fetch initial system status & history
  useEffect(() => {
    checkHealth();
    loadHistory();
    const interval = setInterval(checkHealth, 15000);
    return () => clearInterval(interval);
  }, []);

  const checkHealth = async () => {
    try {
      const status = await api.getHealth();
      setSystemStatus(status);
    } catch {
      setSystemStatus({ status: 'offline' });
    }
  };

  const loadHistory = async () => {
    try {
      const list = await api.listAnalyses();
      setCaseHistory(list);
    } catch {
      // ignore
    }
  };

  const loadAnalysisData = async (analysisId) => {
    try {
      const [analysisRes, detRes, evRes, provRes, srcRes, repRes] = await Promise.allSettled([
        api.getAnalysis(analysisId),
        api.getDetections(analysisId),
        api.getEvidence(analysisId),
        api.getProvenance(analysisId),
        api.getSourceTrace(analysisId),
        api.getReport(analysisId),
      ]);

      if (analysisRes.status === 'fulfilled') setCurrentAnalysis(analysisRes.value);
      if (detRes.status === 'fulfilled') setDetections(detRes.value?.detections || []);
      if (evRes.status === 'fulfilled') setEvidencePackage(evRes.value);
      if (provRes.status === 'fulfilled') setProvenance(provRes.value);
      if (srcRes.status === 'fulfilled') setSourceTrace(srcRes.value);
      if (repRes.status === 'fulfilled') setReport(repRes.value);
    } catch (err) {
      console.error('Failed to load investigation details:', err);
    }
  };

  const handleStartAnalysis = async (mediaId) => {
    setIsAnalyzing(true);
    try {
      const analysisResult = await api.startAnalysis(mediaId);
      const analysisId = analysisResult.analysis_id;
      await loadAnalysisData(analysisId);
      loadHistory();
    } catch (err) {
      alert(`Analysis failed: ${err.message}`);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleSelectHistoricalCase = (analysisId) => {
    loadAnalysisData(analysisId);
  };

  const handleNewCase = () => {
    setCurrentAnalysis(null);
    setDetections([]);
    setEvidencePackage(null);
    setProvenance(null);
    setSourceTrace({});
    setReport(null);
  };

  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>
      <Navbar
        systemStatus={systemStatus}
        currentAnalysis={currentAnalysis}
        onViewReport={() => setIsReportOpen(true)}
        onNewCase={handleNewCase}
      />

      <main className="container" style={{ flex: 1, paddingTop: '2rem', paddingBottom: '3rem' }}>
        {/* Pipeline Ingestion Zone */}
        {!currentAnalysis && (
          <div style={{ maxWidth: '800px', margin: '0 auto 2rem auto' }}>
            <MediaUploader
              onStartAnalysis={handleStartAnalysis}
              isAnalyzing={isAnalyzing}
            />
          </div>
        )}

        {/* Active Analysis Dashboard */}
        {currentAnalysis && (
          <div>
            {/* 9-Stage Progress Banner */}
            <PipelineTracker
              status={currentAnalysis.status}
              activeStage={9}
            />

            {/* Navigation Tabs */}
            <div style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              marginBottom: '1.5rem',
              borderBottom: '1px solid rgba(255, 255, 255, 0.08)',
              paddingBottom: '0.75rem',
            }}>
              <div style={{ display: 'flex', gap: '1rem' }}>
                <button
                  onClick={() => setActiveTab('DETECTIONS')}
                  style={{
                    background: activeTab === 'DETECTIONS' ? 'rgba(6, 182, 212, 0.15)' : 'transparent',
                    color: activeTab === 'DETECTIONS' ? '#38BDF8' : '#94A3B8',
                    border: activeTab === 'DETECTIONS' ? '1px solid rgba(6, 182, 212, 0.4)' : '1px solid transparent',
                    padding: '0.5rem 1.25rem',
                    borderRadius: '8px',
                    fontSize: '0.85rem',
                    fontWeight: 600,
                  }}
                >
                  Detector Signals
                </button>

                <button
                  onClick={() => setActiveTab('EVIDENCE')}
                  style={{
                    background: activeTab === 'EVIDENCE' ? 'rgba(6, 182, 212, 0.15)' : 'transparent',
                    color: activeTab === 'EVIDENCE' ? '#38BDF8' : '#94A3B8',
                    border: activeTab === 'EVIDENCE' ? '1px solid rgba(6, 182, 212, 0.4)' : '1px solid transparent',
                    padding: '0.5rem 1.25rem',
                    borderRadius: '8px',
                    fontSize: '0.85rem',
                    fontWeight: 600,
                  }}
                >
                  Evidence Classification Log
                </button>

                <button
                  onClick={() => setActiveTab('PROVENANCE')}
                  style={{
                    background: activeTab === 'PROVENANCE' ? 'rgba(6, 182, 212, 0.15)' : 'transparent',
                    color: activeTab === 'PROVENANCE' ? '#38BDF8' : '#94A3B8',
                    border: activeTab === 'PROVENANCE' ? '1px solid rgba(6, 182, 212, 0.4)' : '1px solid transparent',
                    padding: '0.5rem 1.25rem',
                    borderRadius: '8px',
                    fontSize: '0.85rem',
                    fontWeight: 600,
                  }}
                >
                  Provenance Lineage Graph
                </button>
              </div>

              <button
                onClick={() => setIsReportOpen(true)}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.5rem',
                  background: 'rgba(255, 255, 255, 0.05)',
                  border: '1px solid rgba(255, 255, 255, 0.1)',
                  color: '#F8FAFC',
                  padding: '0.45rem 1rem',
                  borderRadius: '8px',
                  fontSize: '0.8rem',
                  fontWeight: 600,
                }}
              >
                <FileText size={16} color="#38BDF8" />
                <span>Forensic Report</span>
              </button>
            </div>

            {/* Tab Views */}
            {activeTab === 'DETECTIONS' && (
              <DetectorSignals
                detections={detections}
                overallConfidence={currentAnalysis.overall_confidence}
                assessment={currentAnalysis.assessment}
              />
            )}

            {activeTab === 'EVIDENCE' && (
              <EvidenceTable
                evidenceItems={evidencePackage?.classified_evidence || []}
                limitations={evidencePackage?.limitations || []}
              />
            )}

            {activeTab === 'PROVENANCE' && (
              <ProvenanceViewer
                provenance={provenance || {}}
                sourceTrace={sourceTrace || {}}
              />
            )}
          </div>
        )}

        {/* Historical Investigations Log */}
        {caseHistory.length > 0 && (
          <div className="glass-panel" style={{ marginTop: '2.5rem', padding: '1.75rem' }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1.25rem' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <History size={18} color="#06B6D4" />
                <h3 style={{ fontSize: '1.1rem', fontWeight: 700, color: '#F8FAFC' }}>
                  Database Investigation Records ({caseHistory.length})
                </h3>
              </div>
              <button
                onClick={loadHistory}
                style={{
                  background: 'transparent',
                  border: 'none',
                  color: '#94A3B8',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.35rem',
                  fontSize: '0.75rem',
                }}
              >
                <RefreshCw size={14} />
                Refresh
              </button>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))', gap: '1rem' }}>
              {caseHistory.map((item) => (
                <div
                  key={item.id}
                  onClick={() => handleSelectHistoricalCase(item.id)}
                  className="glass-panel-interactive"
                  style={{
                    background: currentAnalysis?.id === item.id ? 'rgba(6, 182, 212, 0.15)' : 'rgba(15, 23, 42, 0.5)',
                    border: currentAnalysis?.id === item.id ? '1px solid #06B6D4' : '1px solid rgba(255, 255, 255, 0.08)',
                    borderRadius: '8px',
                    padding: '1rem',
                    cursor: 'pointer',
                  }}
                >
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.5rem' }}>
                    <span style={{ fontSize: '0.75rem', color: '#38BDF8', fontFamily: 'var(--font-mono)', fontWeight: 600 }}>
                      CASE: {item.id.slice(0, 8)}...
                    </span>
                    <span className={`badge badge-${(item.assessment || 'INCONCLUSIVE').toLowerCase()}`} style={{ fontSize: '0.65rem' }}>
                      {item.assessment || 'INCONCLUSIVE'}
                    </span>
                  </div>

                  <div style={{ fontSize: '0.75rem', color: '#94A3B8', display: 'flex', justifyContent: 'space-between' }}>
                    <span>Confidence: <strong>{item.overall_confidence ? `${(item.overall_confidence * 100).toFixed(0)}%` : '—'}</strong></span>
                    <span>{new Date(item.created_at).toLocaleDateString()}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </main>

      {/* Forensic Report Modal */}
      <ReportModal
        report={report}
        isOpen={isReportOpen}
        onClose={() => setIsReportOpen(false)}
      />
    </div>
  );
}
