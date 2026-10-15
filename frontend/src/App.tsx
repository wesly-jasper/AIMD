import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import UploadModal from './components/UploadModal';
import AnalysisViewer from './components/AnalysisViewer';
import './App.css';

const queryClient = new QueryClient();

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <div className="app">
          <header className="app-header">
            <h1>AIMD — AI Media Detection Platform</h1>
            <p>Forensic Analysis &amp; Deepfake Detection</p>
          </header>
          <Routes>
            <Route
              path="/"
              element={
                <UploadModal />
              }
            >
              {/* After successful upload, we could redirect to analysis page */}
              {/* For simplicity, we stay on upload page */}
            </Route>
            <Route
              path="/analysis/:analysisId"
              element={
                <AnalysisViewer />
              }
            />
            {/* Redirect root to upload if needed */}
            <Route
              path="*"
              element={
                <Navigate to="/" replace />
              }
            />
          </Routes>
          <footer className="app-footer">
            <p>
              &copy; {new Date().getFullYear()} AIMD Platform. All rights reserved.
            </p>
          </footer>
        </div>
      </BrowserRouter>
    </QueryClientProvider>
  );
}

export default App;