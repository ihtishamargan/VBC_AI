/**
 * DocumentUpload Component - Handles VBC contract PDF uploads with analysis
 */
import React, { useState } from 'react';
import { Upload, FileText, CheckCircle, XCircle, AlertCircle } from 'lucide-react';
import { VBCApiService } from '../services/VBCApiService';
import './DocumentUpload.css';

interface UploadedDocument {
  id: string;
  filename: string;
  status: 'DONE' | 'ERROR' | 'PROCESSING';
  analysis?: any;
  vbc_contract_data?: any;
  error_message?: string;
  uploaded_at: string;
}

export const DocumentUpload: React.FC = () => {
  const [isUploading, setIsUploading] = useState(false);
  const [uploadedDocuments, setUploadedDocuments] = useState<UploadedDocument[]>([]);
  const [dragActive, setDragActive] = useState(false);

  // Function to generate automatic document analysis
  const generateDocumentAnalysisChat = async (uploadResponse: any) => {
    try {
      console.log('Creating structured analysis for:', uploadResponse.filename);
      
      // Create structured analysis message using the new endpoint
      const analysisResponse = await VBCApiService.createDocumentAnalysis(
        uploadResponse.document_id,
        uploadResponse.filename,
        uploadResponse.vbc_contract_data
      );
      
      console.log('Structured analysis created:', analysisResponse.answer.substring(0, 100) + '...');
      
      // Show the analysis preview
      const preview = analysisResponse.answer.substring(0, 300) + (analysisResponse.answer.length > 300 ? '...' : '');
      
      // Show notification with structured analysis
      setTimeout(() => {
        const shouldOpenChat = window.confirm(
          `📄 Document Analysis Ready!\n\n${preview}\n\nWould you like to open the chat interface to see the full structured analysis and ask questions?`
        );
        
        if (shouldOpenChat) {
          // Emit a custom event to switch to chat with the analysis already loaded
          window.dispatchEvent(new CustomEvent('switchToChat', { 
            detail: { 
              analysis: analysisResponse,
              autoLoad: true
            }
          }));
        }
      }, 1000);
      
    } catch (error) {
      console.error('Failed to generate structured document analysis:', error);
      alert('Failed to generate automatic analysis. You can still ask questions about the uploaded document in the chat interface.');
    }
  };

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  };

  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      handleFile(e.target.files[0]);
    }
  };

  const handleFile = async (file: File) => {
    // Validate file type
    if (!file.name.toLowerCase().endsWith('.pdf')) {
      alert('Please upload a PDF file.');
      return;
    }

    // Validate file size (10MB limit)
    if (file.size > 10 * 1024 * 1024) {
      alert('File size must be less than 10MB.');
      return;
    }

    setIsUploading(true);

    try {
      const response = await VBCApiService.uploadDocument(file);
      
      const newDocument: UploadedDocument = {
        id: response.document_id,
        filename: response.filename,
        status: response.status as 'DONE' | 'ERROR' | 'PROCESSING',
        analysis: {
          processing_metrics: {
            total_pages: response.pages_processed,
            total_chunks: response.chunks_created,
            vectors_stored: response.vectors_stored,
            processing_time: response.processing_time_seconds,
          }
        },
        vbc_contract_data: response.vbc_contract_data,
        error_message: response.error_message,
        uploaded_at: new Date().toISOString(),
      };

      setUploadedDocuments(prev => [newDocument, ...prev]);
      
      if (response.status === 'DONE') {
        // Show immediate success with processing stats
        const processingStats = `✅ Document "${file.name}" processed successfully!\n\n📊 Processing Summary:\n• ${response.pages_processed} pages processed\n• ${response.chunks_created} text chunks created\n• ${response.vectors_stored} vectors stored\n• Processing time: ${response.processing_time_seconds.toFixed(1)}s`;
        
        alert(processingStats);
        
        // Trigger automatic analysis chat message
        generateDocumentAnalysisChat(response);
      } else if (response.status === 'ERROR') {
        alert(`❌ Error processing "${file.name}": ${response.error_message}`);
      }
    } catch (error: any) {
      alert(`❌ Upload failed: ${error.message}`);
    } finally {
      setIsUploading(false);
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'DONE':
        return <CheckCircle className="status-icon success" />;
      case 'ERROR':
        return <XCircle className="status-icon error" />;
      case 'PROCESSING':
        return <AlertCircle className="status-icon processing" />;
      default:
        return <FileText className="status-icon" />;
    }
  };

  const formatFileSize = (bytes: number) => {
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    if (bytes === 0) return '0 Byte';
    const i = parseInt(Math.floor(Math.log(bytes) / Math.log(1024)).toString());
    return Math.round(bytes / Math.pow(1024, i) * 100) / 100 + ' ' + sizes[i];
  };

  return (
    <div className="document-upload">
      <div className="upload-header">
        <h2>📄 Upload VBC Contracts</h2>
        <p>Upload PDF contracts for AI-powered analysis and chat</p>
      </div>

      {/* Upload Area */}
      <div 
        className={`upload-area ${dragActive ? 'drag-active' : ''} ${isUploading ? 'uploading' : ''}`}
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
      >
        <input
          type="file"
          id="file-upload"
          accept=".pdf"
          onChange={handleFileInput}
          disabled={isUploading}
          style={{ display: 'none' }}
        />
        
        <label htmlFor="file-upload" className="upload-content">
          {isUploading ? (
            <div className="uploading-state">
              <div className="spinner"></div>
              <p>Processing your VBC contract...</p>
              <small>This may take a few moments</small>
            </div>
          ) : (
            <div className="upload-prompt">
              <Upload className="upload-icon" />
              <p>
                <strong>Click to upload</strong> or drag and drop
              </p>
              <small>PDF files only, max 10MB</small>
            </div>
          )}
        </label>
      </div>

      {/* Uploaded Documents List */}
      {uploadedDocuments.length > 0 && (
        <div className="uploaded-documents">
          <h3>📋 Uploaded Documents ({uploadedDocuments.length})</h3>
          
          <div className="documents-list">
            {uploadedDocuments.map((doc) => (
              <div key={doc.id} className="document-item">
                <div className="document-header">
                  {getStatusIcon(doc.status)}
                  <div className="document-info">
                    <h4>{doc.filename}</h4>
                    <small>
                      Uploaded: {new Date(doc.uploaded_at).toLocaleString()} • 
                      Status: <span className={`status ${doc.status.toLowerCase()}`}>{doc.status}</span>
                    </small>
                  </div>
                </div>

                {/* VBC Analysis Results */}
                {doc.status === 'DONE' && doc.vbc_contract_data && (
                  <div className="vbc-analysis">
                    <h5>🔍 VBC Contract Analysis</h5>
                    <div className="analysis-grid">
                      <div className="analysis-item">
                        <strong>Agreement:</strong> {doc.vbc_contract_data.agreement_title}
                      </div>
                      <div className="analysis-item">
                        <strong>Country:</strong> {doc.vbc_contract_data.country}
                      </div>
                      <div className="analysis-item">
                        <strong>Disease Area:</strong> {doc.vbc_contract_data.disease_area}
                      </div>
                      <div className="analysis-item">
                        <strong>Payment Model:</strong> {doc.vbc_contract_data.payment_model}
                      </div>
                      <div className="analysis-item">
                        <strong>Patient Population:</strong> {doc.vbc_contract_data.patient_population_size?.toLocaleString() || 'N/A'}
                      </div>
                      <div className="analysis-item">
                        <strong>Confidence:</strong> 
                        <span className="confidence-score">
                          {(doc.vbc_contract_data.extraction_confidence * 100).toFixed(0)}%
                        </span>
                      </div>
                    </div>

                    {/* Parties */}
                    {doc.vbc_contract_data.parties && doc.vbc_contract_data.parties.length > 0 && (
                      <div className="parties-section">
                        <strong>🏢 Parties:</strong>
                        <ul>
                          {doc.vbc_contract_data.parties.map((party: any, idx: number) => (
                            <li key={idx}>
                              {party.name} ({party.role})
                            </li>
                          ))}
                        </ul>
                      </div>
                    )}

                    {/* Outcome Metrics */}
                    {doc.vbc_contract_data.outcome_metrics && doc.vbc_contract_data.outcome_metrics.length > 0 && (
                      <div className="metrics-section">
                        <strong>📊 Outcome Metrics:</strong>
                        <ul>
                          {doc.vbc_contract_data.outcome_metrics.map((metric: any, idx: number) => (
                            <li key={idx}>
                              {metric.name} ({metric.type})
                            </li>
                          ))}
                        </ul>
                      </div>
                    )}
                  </div>
                )}

                {/* Processing Metrics */}
                {doc.status === 'DONE' && doc.analysis?.processing_metrics && (
                  <div className="processing-metrics">
                    <small>
                      📈 {doc.analysis.processing_metrics.total_pages} pages • 
                      {doc.analysis.processing_metrics.total_chunks} chunks • 
                      {doc.analysis.processing_metrics.vectors_stored} vectors • 
                      {doc.analysis.processing_metrics.processing_time.toFixed(1)}s processing time
                    </small>
                  </div>
                )}

                {/* Error Message */}
                {doc.status === 'ERROR' && doc.error_message && (
                  <div className="error-message">
                    <small>❌ Error: {doc.error_message}</small>
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};
