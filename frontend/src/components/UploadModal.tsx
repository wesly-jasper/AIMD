import React, { useState } from 'react';
import { useAnalysisWorkflow } from '../services';

const UploadModal: React.FC = () => {
  const { handleFileUpload, jobStatus, events } = useAnalysisWorkflow();
  const [file, setFile] = useState<File | null>(null);
  const [isUploading, setIsUploading] = useState(false);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!file) return;
    setIsUploading(true);
    try {
      await handleFileUpload(file);
      // reset file input
      setFile(null);
    } catch (err) {
      console.error('Upload failed', err);
    } finally {
      setIsUploading(false);
    }
  };

  const jobStatusData = jobStatus.data;

  return (
    <div className="upload-modal">
      <h2>Upload Media for Analysis</h2>
      <form onSubmit={handleSubmit} className="upload-form">
        <div>
          <label htmlFor="file-input">Choose file:</label>
          <input
            id="file-input"
            type="file"
            accept="image/*,video/*,audio/*"
            onChange={handleChange}
            disabled={isUploading}
          />
          {file && (
            <p className="file-info">
              Selected: {file.name} ({Math.round(file.size / 1024)} KB)
            </p>
          )}
        </div>
        <button type="submit" disabled={isUploading || !file}>
          {isUploading ? 'Uploading...' : 'Upload & Analyze'}
        </button>
      </form>

      {jobStatusData && (
        <div className="job-status">
          <h3>Analysis Job Status</h3>
          <p>Status: {jobStatusData.status}</p>
          {jobStatusData.status === 'processing' && (
            <p>
              Stage: {jobStatusData.stage}/9 &nbsp;|&nbsp; Progress: {
                ' '}
                {Math.round(jobStatusData.progress * 100)}%
              </p>
          )}
          {jobStatusData.status === 'completed' && (
            <p>Analysis completed! View results below.</p>
          )}
          {jobStatusData.status === 'failed' && (
            <p style={{ color: 'red' }}>
              Failed: {jobStatusData.error}
            </p>
          )}
        </div>
      )}

      {events.length > 0 && (
        <div className="events-log">
          <h3>Real-time Events</h3>
          <ul>
            {events.map((e, idx) => (
              <li key={idx}>{JSON.stringify(e)}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
};

export default UploadModal;
