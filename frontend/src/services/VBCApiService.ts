/**
 * VBC API Service - Handles communication with the FastAPI backend
 */
import axios, { AxiosResponse } from 'axios';

// API Base URL - adjust based on your backend configuration
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

// API Response interfaces
interface ChatRequest {
  message: string;
  filters?: Record<string, any>;
}

interface ChatResponse {
  answer: string;
  sources: Array<{
    document_id: string;
    chunk_id: string;
    content: string;
    score: number;
  }>;
}

interface DocumentUploadResponse {
  document_id: string;
  status: 'DONE' | 'ERROR' | 'PROCESSING';
  uploaded_at: string;
  filename: string;
  analysis?: {
    document_type: string;
    summary: string;
    key_topics: string[];
    entities: Array<{
      text: string;
      type: string;
      confidence: number;
    }>;
    confidence_score: number;
    processing_metrics: {
      total_pages: number;
      total_chunks: number;
      vectors_stored: number;
      processing_time: number;
      file_size: number;
    };
    chunks_preview: Array<{
      content: string;
      metadata: Record<string, any>;
    }>;
  };
  vbc_contract_data?: {
    agreement_title: string;
    parties: Array<{
      name: string;
      type: string;
      role: string;
    }>;
    country: string;
    disease_area: string;
    patient_population_size: number;
    payment_model: string;
    currency: string;
    outcome_metrics: Array<{
      name: string;
      type: string;
      target_value: string;
    }>;
    extraction_confidence: number;
  };
  error_message?: string;
}

interface DocumentStatusResponse {
  document_id: string;
  status: 'PENDING' | 'PROCESSING' | 'DONE' | 'ERROR';
  created_at: string;
  updated_at: string;
  error_message?: string;
}

export class VBCApiService {
  private static axiosInstance = axios.create({
    baseURL: API_BASE_URL,
    timeout: 30000,
    headers: {
      'Content-Type': 'application/json',
    },
  });

  /**
   * Send a chat message to the AI assistant
   */
  static async sendChatMessage(message: string, filters?: Record<string, any>): Promise<ChatResponse> {
    try {
      const response: AxiosResponse<ChatResponse> = await this.axiosInstance.post('/chat', {
        message,
        filters: filters || null,
      } as ChatRequest);

      return response.data;
    } catch (error) {
      console.error('Error sending chat message:', error);
      throw new Error('Failed to send chat message. Please try again.');
    }
  }

  /**
   * Upload a document for processing
   */
  static async uploadDocument(file: File): Promise<DocumentUploadResponse> {
    try {
      const formData = new FormData();
      formData.append('file', file);

      const response: AxiosResponse<DocumentUploadResponse> = await this.axiosInstance.post(
        '/upload',
        formData,
        {
          headers: {
            'Content-Type': 'multipart/form-data',
          },
          timeout: 120000, // 2 minutes for document processing
        }
      );

      return response.data;
    } catch (error: any) {
      console.error('Error uploading document:', error);
      if (error.response?.data?.detail) {
        throw new Error(error.response.data.detail);
      }
      throw new Error('Failed to upload document. Please try again.');
    }
  }

  /**
   * Get document processing status
   */
  static async getDocumentStatus(documentId: string): Promise<DocumentStatusResponse> {
    try {
      const response: AxiosResponse<DocumentStatusResponse> = await this.axiosInstance.get(
        `/documents/${documentId}/status`
      );

      return response.data;
    } catch (error) {
      console.error('Error getting document status:', error);
      throw new Error('Failed to get document status.');
    }
  }

  /**
   * Search through documents
   */
  static async searchDocuments(query: string, filters?: Record<string, any>): Promise<any> {
    try {
      const params = new URLSearchParams();
      params.append('q', query);
      if (filters) {
        params.append('filters', JSON.stringify(filters));
      }

      const response = await this.axiosInstance.get(`/search?${params}`);
      return response.data;
    } catch (error) {
      console.error('Error searching documents:', error);
      throw new Error('Failed to search documents.');
    }
  }

  /**
   * Get application health status
   */
  static async getHealthStatus(): Promise<{ status: string; timestamp: string }> {
    try {
      const response = await this.axiosInstance.get('/healthz');
      return response.data;
    } catch (error) {
      console.error('Error getting health status:', error);
      throw new Error('Failed to get health status.');
    }
  }

  /**
   * Get application metrics
   */
  static async getMetrics(): Promise<{
    total_documents: number;
    processing_documents: number;
    total_queries: number;
    uptime_seconds: number;
  }> {
    try {
      const response = await this.axiosInstance.get('/metrics');
      return response.data;
    } catch (error) {
      console.error('Error getting metrics:', error);
      throw new Error('Failed to get metrics.');
    }
  }
}
