import { useMutation, useQuery } from '@tanstack/react-query';
import { useState } from 'react';
import axios from 'axios';

import type {
  RegisterRequest,
  LoginRequest,
  AuthResponse,
  MediaUploadResponse,
  MediaMetadata,
  MediaListResponse,
  AnalysisRequest,
  AnalysisResponse,
  AsyncAnalysisResponse,
  JobStatus,
  SimilaritySearchResponse,
  EvidenceListResponse,
  ReportRequest,
  ReportResponse,
  HeatmapResponse,
  RegionsResponse,
  AnalysisTimeline,
} from '../types/api';
// Base API URL from env
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000/api/v1';

// Axios instance with token interceptor
const apiClient = axios.create({
  baseURL: API_BASE_URL,
});

// Request interceptor to add auth token
apiClient.interceptors.request.use((config) => {
  const token = localStorage.getItem('access_token');
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

// Response interceptor to handle 401 (optional)
apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      // Optionally redirect to login or logout
      localStorage.removeItem('access_token');
      // window.location.href = '/login';
    }
    return Promise.reject(error);
  }
);

/* ========== AUTH ========== */
export const useRegister = () => {
  return useMutation<AuthResponse, Error, RegisterRequest>({
    mutationFn: async (data) => {
      const response = await apiClient.post('/auth/register', data);
      // Store token
      localStorage.setItem('access_token', response.data.access_token);
      return response.data;
    },
  });
};

export const useLogin = () => {
  return useMutation<AuthResponse, Error, LoginRequest>({
    mutationFn: async (data) => {
      const response = await apiClient.post('/auth/login', data);
      localStorage.setItem('access_token', response.data.access_token);
      return response.data;
    },
  });
};

export const useLogout = () => {
  return useMutation<void, Error, void>({
    mutationFn: async () => {
      await apiClient.post('/auth/logout');
      localStorage.removeItem('access_token');
    },
  });
};

/* ========== MEDIA ========== */
export const useUploadMedia = () => {
  return useMutation<MediaUploadResponse, Error, File>({
    mutationFn: async (file) => {
      const formData = new FormData();
      formData.append('file', file);
      const response = await apiClient.post('/media/', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      return response.data;
    },
  });
};

export const useMediaList = (page = 1, pageSize = 10) => {
  return useQuery<MediaListResponse, Error>({
    queryKey: ['media', 'list', page, pageSize],
    queryFn: async () => {
      const response = await apiClient.get('/media/', {
        params: { page, page_size: pageSize },
      });
      return response.data;
    },
  });
};

export const useMediaDetail = (mediaId: string) => {
  return useQuery<MediaMetadata, Error>({
    queryKey: ['media', 'detail', mediaId],
    queryFn: async () => {
      const response = await apiClient.get(`/media/${mediaId}`);
      return response.data;
    },
    enabled: !!mediaId,
  });
};

/* ========== ANALYSIS ========== */
export const useAnalysis = () => {
  return useMutation<AnalysisResponse, Error, AnalysisRequest>({
    mutationFn: async (data) => {
      const response = await apiClient.post('/analysis/', data);
      return response.data;
    },
  });
};

export const useAsyncAnalysis = () => {
  return useMutation<AsyncAnalysisResponse, Error, AnalysisRequest>({
    mutationFn: async (data) => {
      const response = await apiClient.post('/analysis/async', data);
      return response.data;
    },
  });
};

/* ========== JOB STATUS ========== */
export const useJobStatus = (jobId: string) => {
  return useQuery<JobStatus, Error>({
    queryKey: ['job', 'status', jobId],
    queryFn: async () => {
      const response = await apiClient.get(`/jobs/${jobId}`);
      return response.data;
    },
    // Poll every 2 seconds when job is processing
    refetchInterval: (query) => {
      const data = query.state.data;
      if (data && data.status == "processing") {
        return 2000;
      }
      return false;
    },
    enabled: !!jobId,
  });
};

/* ========== ANALYSIS EVENTS (SSE) ========== */
// Note: React Query is not ideal for SSE; we'll implement a custom hook using useState/useEffect
export const useAnalysisEvents = (analysisId: string) => {
  const [events] = useState<Array<any>>([]);
  const [status] = useState<'connecting' | 'open' | 'closed' | 'error'>(
    'connecting'
  );
  const [error] = useState<Error | null>(null);

  // In a real implementation, we'd use EventSource
  // For simplicity, we'll simulate with polling (as in the mock)
  // But we'll keep the hook signature

  // For now, we'll return a placeholder; actual implementation would connect to SSE
  return {
    events,
    status,
    error,
    connect: () => {
      // Implementation would open EventSource
      console.log('Connecting to SSE for', analysisId);
    },
    disconnect: () => {
      // Implementation would close EventSource
      console.log('Disconnecting from SSE');
    },
  };
};

/* ========== VISUALIZATION ========== */
export const useHeatmap = (analysisId: string) => {
  return useQuery<HeatmapResponse, Error>({
    queryKey: ['heatmap', analysisId],
    queryFn: async () => {
      const response = await apiClient.get(`/analysis/${analysisId}/heatmap`);
      return response.data;
    },
    enabled: !!analysisId,
  });
};

export const useRegions = (analysisId: string) => {
  return useQuery<RegionsResponse, Error>({
    queryKey: ['regions', analysisId],
    queryFn: async () => {
      const response = await apiClient.get(`/analysis/${analysisId}/regions`);
      return response.data;
    },
    enabled: !!analysisId,
  });
};

export const useTimeline = (analysisId: string) => {
  return useQuery<AnalysisTimeline, Error>({
    queryKey: ['timeline', analysisId],
    queryFn: async () => {
      const response = await apiClient.get(`/analysis/${analysisId}/timeline`);
      return response.data;
    },
    enabled: !!analysisId,
  });
};

export const useProvenanceGraph = (analysisId: string) => {
  return useQuery<ProvenanceGraph, Error>({
    queryKey: ['graph', analysisId],
    queryFn: async () => {
      const response = await apiClient.get(`/analysis/${analysisId}/graph`);
      return response.data;
    },
    enabled: !!analysisId,
  });
};

/* ========== ADDITIONAL SERVICES ========== */
export const useSimilaritySearch = () => {
  return useMutation<SimilaritySearchResponse, Error, string>({
    mutationFn: async (mediaId) => {
      const response = await apiClient.get(`/similarity/search/${mediaId}`);
      return response.data;
    },
  });
};

export const useEvidenceList = () => {
  return useMutation<EvidenceListResponse, Error, string>({
    mutationFn: async (analysisId) => {
      const response = await apiClient.get(`/evidence/${analysisId}`);
      return response.data;
    },
  });
};

export const useGenerateReport = () => {
  return useMutation<ReportResponse, Error, ReportRequest>({
    mutationFn: async (data) => {
      const response = await apiClient.post('/reports/generate', data);
      return response.data;
    },
  });
};

/* ========== WORKFLOW ========== */
// Combined hook for upload + async analysis + job status + events
export const useAnalysisWorkflow = () => {
  const uploadMedia = useUploadMedia();
  const startAsyncAnalysis = useAsyncAnalysis();
  const [jobId, setJobId] = useState<string | null>(null);
  const jobStatus = useJobStatus(jobId || '');
  const { events, connect, disconnect } = useAnalysisEvents(
    jobStatus.data && jobStatus.data.status === "completed" ? jobStatus.data.analysis_id : ""
  );

  const handleFileUpload = async (file: File) => {
    const uploadResult = await uploadMedia.mutateAsync(file);
    const analysisResult = await startAsyncAnalysis.mutateAsync({
      media_id: uploadResult.media_id,
    });
    setJobId(analysisResult.job_id);
    connect(); // Start listening for real-time events
  };

  // Job completion handling
  if (jobStatus.data?.status === 'completed') {
    // Analysis finished, disconnect from events
    // disconnect(); // keep connected for final events?
  }

  return {
    uploadMedia,
    startAsyncAnalysis,
    jobId,
    jobStatus,
    events,
    handleFileUpload,
    connect,
    disconnect,
  };
};