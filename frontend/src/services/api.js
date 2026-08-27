/**
 * AIMD Backend API Client
 * Connects to the FastAPI forensic backend service.
 */

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000/api/v1';

async function handleResponse(response) {
  if (!response.ok) {
    let errorDetail = `HTTP ${response.status}: ${response.statusText}`;
    try {
      const data = await response.json();
      if (data.detail) {
        errorDetail = typeof data.detail === 'string' ? data.detail : JSON.stringify(data.detail);
      }
    } catch {
      // ignore json parse error
    }
    throw new Error(errorDetail);
  }
  return response.json();
}

export const api = {
  // System Health
  async getHealth() {
    return handleResponse(await fetch(`${API_BASE}/health`));
  },

  // Media Operations
  async uploadMedia(file, onProgress) {
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch(`${API_BASE}/media/upload`, {
      method: 'POST',
      body: formData,
    });
    return handleResponse(response);
  },

  async listMedia() {
    return handleResponse(await fetch(`${API_BASE}/media/`));
  },

  async getMedia(mediaId) {
    return handleResponse(await fetch(`${API_BASE}/media/${mediaId}`));
  },

  // Analysis Pipeline Operations
  async startAnalysis(mediaId) {
    const response = await fetch(`${API_BASE}/analysis/`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ media_id: mediaId }),
    });
    return handleResponse(response);
  },

  async getAnalysis(analysisId) {
    return handleResponse(await fetch(`${API_BASE}/analysis/${analysisId}`));
  },

  async listAnalyses() {
    return handleResponse(await fetch(`${API_BASE}/analysis/`));
  },

  // Forensic Investigation Artifacts
  async getDetections(analysisId) {
    return handleResponse(await fetch(`${API_BASE}/detection/${analysisId}`));
  },

  async getSourceTrace(analysisId) {
    return handleResponse(await fetch(`${API_BASE}/source/${analysisId}`));
  },

  async getProvenance(analysisId) {
    return handleResponse(await fetch(`${API_BASE}/provenance/${analysisId}`));
  },

  async getEvidence(analysisId) {
    return handleResponse(await fetch(`${API_BASE}/evidence/${analysisId}`));
  },

  async getReport(analysisId) {
    return handleResponse(await fetch(`${API_BASE}/report/${analysisId}`));
  },
};
