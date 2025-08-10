-- VBC AI Database Schema
-- Initialize database for storing document analysis and extracted data

-- Create extension for UUID generation
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Documents table - stores document metadata and analysis results
CREATE TABLE documents (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id VARCHAR(255) UNIQUE NOT NULL,
    filename VARCHAR(255) NOT NULL,
    file_size INTEGER,
    upload_timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    processing_status VARCHAR(50) DEFAULT 'pending',
    
    -- Document analysis results
    document_type VARCHAR(100),
    summary TEXT,
    confidence_score FLOAT,
    
    -- Processing metrics
    pages_processed INTEGER,
    chunks_created INTEGER,
    vectors_stored INTEGER,
    processing_time_seconds FLOAT,
    
    -- Additional metadata
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Topics table - stores extracted key topics
CREATE TABLE document_topics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id UUID REFERENCES documents(id) ON DELETE CASCADE,
    topic VARCHAR(255) NOT NULL,
    relevance_score FLOAT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Entities table - stores extracted entities
CREATE TABLE document_entities (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id UUID REFERENCES documents(id) ON DELETE CASCADE,
    entity_text VARCHAR(500) NOT NULL,
    entity_type VARCHAR(100) NOT NULL,
    confidence FLOAT,
    start_pos INTEGER,
    end_pos INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Chunks table - stores document chunks for vector search reference
CREATE TABLE document_chunks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id UUID REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    vector_id VARCHAR(255), -- Reference to Qdrant vector ID
    page_number INTEGER,
    chunk_metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Search queries table - for analytics and improvement
CREATE TABLE search_queries (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    query_text TEXT NOT NULL,
    results_count INTEGER,
    response_time_ms INTEGER,
    user_session VARCHAR(255),
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Chat conversations table - for storing chat history
CREATE TABLE chat_conversations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id VARCHAR(255) NOT NULL,
    user_message TEXT NOT NULL,
    ai_response TEXT NOT NULL,
    sources_used JSONB, -- Array of document sources used
    response_time_ms INTEGER,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for better performance
CREATE INDEX idx_documents_document_id ON documents(document_id);
CREATE INDEX idx_documents_upload_timestamp ON documents(upload_timestamp);
CREATE INDEX idx_documents_processing_status ON documents(processing_status);
CREATE INDEX idx_documents_document_type ON documents(document_type);

CREATE INDEX idx_document_topics_document_id ON document_topics(document_id);
CREATE INDEX idx_document_topics_topic ON document_topics(topic);

CREATE INDEX idx_document_entities_document_id ON document_entities(document_id);
CREATE INDEX idx_document_entities_type ON document_entities(entity_type);
CREATE INDEX idx_document_entities_text ON document_entities(entity_text);

CREATE INDEX idx_document_chunks_document_id ON document_chunks(document_id);
CREATE INDEX idx_document_chunks_vector_id ON document_chunks(vector_id);

CREATE INDEX idx_search_queries_timestamp ON search_queries(timestamp);
CREATE INDEX idx_search_queries_session ON search_queries(user_session);

CREATE INDEX idx_chat_conversations_session_id ON chat_conversations(session_id);
CREATE INDEX idx_chat_conversations_timestamp ON chat_conversations(timestamp);

-- Function to update the updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Trigger to automatically update updated_at for documents
CREATE TRIGGER update_documents_updated_at 
    BEFORE UPDATE ON documents 
    FOR EACH ROW 
    EXECUTE FUNCTION update_updated_at_column();

-- VBC Contract Tables
-- Main VBC contracts table - stores core contract information
CREATE TABLE vbc_contracts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id UUID REFERENCES documents(id) ON DELETE CASCADE,
    
    -- Core identification
    agreement_id VARCHAR(255),
    agreement_title TEXT NOT NULL,
    
    -- Location and clinical focus
    country VARCHAR(100) NOT NULL,
    disease_area VARCHAR(100) NOT NULL,
    disease_area_details TEXT,
    
    -- Patient population
    patient_population_size INTEGER,
    patient_population_description TEXT,
    
    -- Contract overview
    agreement_overview TEXT NOT NULL,
    contract_background TEXT,
    pilot_program_results TEXT,
    
    -- Duration and timeline
    duration_months INTEGER,
    duration_description TEXT,
    start_date DATE,
    end_date DATE,
    
    -- Quality and compliance
    data_collection_frequency VARCHAR(255),
    regulatory_framework VARCHAR(255),
    governance_structure TEXT,
    contract_complexity VARCHAR(100),
    
    -- Processing metadata
    extraction_confidence FLOAT DEFAULT 0.0,
    processing_notes TEXT[],
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Contract parties table
CREATE TABLE vbc_contract_parties (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    contract_id UUID REFERENCES vbc_contracts(id) ON DELETE CASCADE,
    party_name VARCHAR(255) NOT NULL,
    party_type VARCHAR(100) NOT NULL,
    party_country VARCHAR(100),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Financial structure table
CREATE TABLE vbc_contract_financials (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    contract_id UUID REFERENCES vbc_contracts(id) ON DELETE CASCADE,
    
    -- Payment details
    initial_payment DECIMAL(15,2),
    currency VARCHAR(10) DEFAULT 'USD',
    payment_model VARCHAR(50) NOT NULL,
    base_reimbursement DECIMAL(15,2),
    shared_savings_percentage FLOAT,
    
    -- Risk corridors
    risk_corridor_upper DECIMAL(15,2),
    risk_corridor_lower DECIMAL(15,2),
    maximum_payout DECIMAL(15,2),
    minimum_guarantee DECIMAL(15,2),
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Risk protection mechanisms table
CREATE TABLE vbc_contract_risk_protection (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    contract_id UUID REFERENCES vbc_contracts(id) ON DELETE CASCADE,
    
    -- Risk protection flags
    has_stop_loss BOOLEAN DEFAULT FALSE,
    has_risk_cap BOOLEAN DEFAULT FALSE,
    has_non_responder_fund BOOLEAN DEFAULT FALSE,
    
    -- Thresholds
    stop_loss_threshold DECIMAL(15,2),
    risk_cap_percentage FLOAT,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Outcome metrics table
CREATE TABLE vbc_contract_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    contract_id UUID REFERENCES vbc_contracts(id) ON DELETE CASCADE,
    
    metric_name VARCHAR(255) NOT NULL,
    metric_type VARCHAR(100) NOT NULL,
    target_value VARCHAR(255),
    measurement_period VARCHAR(255),
    data_source VARCHAR(255),
    weight FLOAT,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Contract endpoints and measures
CREATE TABLE vbc_contract_endpoints (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    contract_id UUID REFERENCES vbc_contracts(id) ON DELETE CASCADE,
    endpoint_type VARCHAR(50) NOT NULL, -- 'primary', 'secondary', 'quality_measure', 'performance_benchmark'
    endpoint_description TEXT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Contract compliance and reporting
CREATE TABLE vbc_contract_reporting (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    contract_id UUID REFERENCES vbc_contracts(id) ON DELETE CASCADE,
    requirement_type VARCHAR(50) NOT NULL, -- 'reporting', 'innovation_element'
    requirement_description TEXT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for VBC tables
CREATE INDEX idx_vbc_contracts_document_id ON vbc_contracts(document_id);
CREATE INDEX idx_vbc_contracts_disease_area ON vbc_contracts(disease_area);
CREATE INDEX idx_vbc_contracts_country ON vbc_contracts(country);
CREATE INDEX idx_vbc_contracts_agreement_title ON vbc_contracts(agreement_title);

CREATE INDEX idx_vbc_contract_parties_contract_id ON vbc_contract_parties(contract_id);
CREATE INDEX idx_vbc_contract_parties_party_type ON vbc_contract_parties(party_type);

CREATE INDEX idx_vbc_contract_financials_contract_id ON vbc_contract_financials(contract_id);
CREATE INDEX idx_vbc_contract_financials_payment_model ON vbc_contract_financials(payment_model);

CREATE INDEX idx_vbc_contract_metrics_contract_id ON vbc_contract_metrics(contract_id);
CREATE INDEX idx_vbc_contract_metrics_type ON vbc_contract_metrics(metric_type);

-- Triggers for VBC contracts
CREATE TRIGGER update_vbc_contracts_updated_at 
    BEFORE UPDATE ON vbc_contracts 
    FOR EACH ROW 
    EXECUTE FUNCTION update_updated_at_column();

-- Insert initial data for testing (optional)
INSERT INTO documents (document_id, filename, processing_status, document_type) 
VALUES ('test-doc-1', 'sample_contract.pdf', 'completed', 'vbc_contract')
ON CONFLICT (document_id) DO NOTHING;

-- Create a view for document summary with counts
CREATE VIEW document_summary AS
SELECT 
    d.id,
    d.document_id,
    d.filename,
    d.document_type,
    d.upload_timestamp,
    d.processing_status,
    d.confidence_score,
    d.pages_processed,
    d.chunks_created,
    d.vectors_stored,
    COUNT(DISTINCT dt.id) as topics_count,
    COUNT(DISTINCT de.id) as entities_count,
    COUNT(DISTINCT dc.id) as chunks_count
FROM documents d
LEFT JOIN document_topics dt ON d.id = dt.document_id
LEFT JOIN document_entities de ON d.id = de.document_id  
LEFT JOIN document_chunks dc ON d.id = dc.document_id
GROUP BY d.id, d.document_id, d.filename, d.document_type, d.upload_timestamp, 
         d.processing_status, d.confidence_score, d.pages_processed, 
         d.chunks_created, d.vectors_stored;

COMMENT ON TABLE documents IS 'Main table for storing document metadata and analysis results';
COMMENT ON TABLE document_topics IS 'Extracted key topics from document analysis';
COMMENT ON TABLE document_entities IS 'Named entities extracted from documents';
COMMENT ON TABLE document_chunks IS 'Document chunks with references to vector store';
COMMENT ON TABLE search_queries IS 'Search query analytics and performance tracking';
COMMENT ON TABLE chat_conversations IS 'Chat history and AI responses with sources';
