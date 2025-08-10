"""Document ingestion service for LLM analysis, chunking, and vector storage."""
import logging
import uuid
from typing import List, Optional, Dict, Any
from datetime import datetime

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

from backend.app.services.llm_analyzer import LLMDocumentAnalyzer, DocumentAnalysis
from backend.app.services.vbc_analyzer import VBCContractAnalyzer
from backend.app.services.qdrant_store import QdrantStoreDense as QdrantStore
from backend.app.services.database import DatabaseService
from backend.app.services.deduplication import deduplication_service
from backend.app.config import settings

logger = logging.getLogger(__name__)


class DocumentChunk:
    """Represents a document chunk with metadata."""
    
    def __init__(self, content: str, metadata: Dict[str, Any]):
        self.chunk_id = str(uuid.uuid4())
        self.content = content
        self.metadata = metadata
        self.created_at = datetime.now()
    
    def to_dict(self) -> dict:
        """Convert chunk to dictionary."""
        return {
            "chunk_id": self.chunk_id,
            "content": self.content,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat()
        }


class DocumentIngestionService:
    """Service for analyzing documents, creating chunks, and storing in vector DB."""
    
    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap: int = None,
        llm_model: str = None,
        collection_name: str = None,
        use_vbc_analyzer: bool = True
    ):
        """Initialize the ingestion service.
        
        Args:
            chunk_size: Size of text chunks (defaults to config)
            chunk_overlap: Overlap between chunks (defaults to config)
            llm_model: OpenAI model to use (defaults to config)
            collection_name: Qdrant collection name (defaults to config)
            use_vbc_analyzer: Whether to use VBC-specific analyzer (default: True)
        """
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
        self.use_vbc_analyzer = use_vbc_analyzer
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        # Initialize VBC analyzer (preferred for contract documents)
        if use_vbc_analyzer:
            try:
                self.vbc_analyzer = VBCContractAnalyzer(
                    model=llm_model or settings.openai_model,
                    api_key=settings.openai_api_key
                )
                logger.info("VBC contract analyzer initialized successfully")
            except Exception as e:
                logger.warning(f"VBC analyzer initialization failed: {e}")
                self.vbc_analyzer = None
                use_vbc_analyzer = False
        
        # Initialize generic LLM analyzer (fallback or alternative)
        if not use_vbc_analyzer or not hasattr(self, 'vbc_analyzer') or self.vbc_analyzer is None:
            try:
                self.llm_analyzer = LLMDocumentAnalyzer(
                    api_key=settings.openai_api_key,
                    model=llm_model or settings.openai_model
                )
                logger.info("Generic LLM analyzer initialized successfully")
            except Exception as e:
                logger.warning(f"LLM analyzer initialization failed: {e}")
                self.llm_analyzer = None
        
        # Initialize vector store
        try:
            self.vector_store = QdrantStore(
                collection_name=collection_name or settings.qdrant_collection_name
            )
            logger.info("Vector store initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {e}")
            self.vector_store = None
        
        # Initialize database service
        try:
            self.database_service = DatabaseService()
            logger.info("Database service initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize database service: {e}")
            self.database_service = None
    
    async def analyze_document(
        self, 
        pages: List[Document], 
        document_id: str
    ) -> Optional[DocumentAnalysis]:
        """Analyze document content using VBC analyzer or generic LLM.
        
        Args:
            pages: List of document pages
            document_id: Unique document identifier
            
        Returns:
            LLM analysis results or None if disabled
        """
        # Check if any analyzer is available
        has_vbc_analyzer = hasattr(self, 'vbc_analyzer') and self.vbc_analyzer is not None
        has_llm_analyzer = hasattr(self, 'llm_analyzer') and self.llm_analyzer is not None
        
        if not has_vbc_analyzer and not has_llm_analyzer:
            logger.info("No analyzer available")
            return None
        
        try:
            # Combine all page content for analysis
            full_content = " ".join([page.page_content for page in pages])
            
            # Use VBC analyzer if available (preferred for contract documents)
            if has_vbc_analyzer:
                logger.info(f"Starting VBC contract analysis for document {document_id}")
                
                vbc_response = await self.vbc_analyzer.analyze_contract(full_content, document_id)
                
                if vbc_response.success and vbc_response.contract_data:
                    # Convert VBC data to generic DocumentAnalysis format for compatibility
                    contract_data = vbc_response.contract_data
                    
                    # Extract entities from VBC contract data
                    entities = []
                    for party in contract_data.parties:
                        entities.append({
                            "text": party.name,
                            "type": "ORGANIZATION", 
                            "confidence": 0.95,
                            "context": f"{party.type} based in {party.country or contract_data.country}"
                        })
                    
                    # Extract key topics from VBC contract data
                    key_topics = [
                        contract_data.disease_area.value.replace('_', ' ').title(),
                        contract_data.financial_structure.payment_model.value.replace('_', ' ').title(),
                        contract_data.country
                    ]
                    if contract_data.outcome_metrics:
                        key_topics.extend([metric.name for metric in contract_data.outcome_metrics[:3]])
                    
                    # Create compatible DocumentAnalysis object
                    analysis = DocumentAnalysis(
                        document_type="vbc_contract",
                        summary=contract_data.agreement_overview[:500],
                        key_topics=key_topics[:10],
                        entities=entities[:20],
                        confidence_score=contract_data.extraction_confidence,
                        language="english",
                        sentiment="neutral",
                        key_insights=[
                            f"Agreement: {contract_data.agreement_title}",
                            f"Parties: {len(contract_data.parties)} organizations",
                            f"Disease Area: {contract_data.disease_area.value.replace('_', ' ').title()}",
                            f"Payment Model: {contract_data.financial_structure.payment_model.value.replace('_', ' ').title()}",
                            f"Patient Population: {contract_data.patient_population_size or 'Not specified'}"
                        ][:5]
                    )
                    
                    # Note: Store VBC contract data separately since DocumentAnalysis has strict validation
                    # The full VBC data is available in the vbc_response for database persistence
                    
                    # Store the VBC contract data for database persistence
                    self._vbc_contract_data = contract_data
                    
                    logger.info(f"VBC contract analysis completed for document {document_id}")
                    return analysis
                else:
                    logger.warning(f"VBC analysis failed, falling back to generic analyzer for {document_id}")
            
            # Fall back to generic LLM analyzer
            if has_llm_analyzer:
                # Use first 10k characters for generic analysis
                sample_content = full_content[:10000]
                
                # Prepare document info
                document_info = {
                    "document_id": document_id,
                    "total_pages": len(pages),
                    "content_length": len(full_content),
                    "filename": pages[0].metadata.get("filename", "unknown.pdf") if pages else "unknown.pdf"
                }
                
                logger.info(f"Starting generic LLM analysis for document {document_id}")
                analysis = await self.llm_analyzer.analyze_document(sample_content, document_info)
                
                logger.info(f"Generic LLM analysis completed for document {document_id}")
                return analysis
            
            return None
            
        except Exception as e:
            logger.error(f"Document analysis failed for document {document_id}: {str(e)}")
            return None
    
    def create_chunks(
        self, 
        pages: List[Document], 
        document_id: str,
        analysis: Optional[DocumentAnalysis] = None
    ) -> List[DocumentChunk]:
        """Create text chunks from document pages.
        
        Args:
            pages: List of document pages
            document_id: Unique document identifier
            analysis: Optional LLM analysis results
            
        Returns:
            List of document chunks with metadata
        """
        try:
            logger.info(f"Creating chunks for document {document_id}")
            
            all_chunks = []
            
            for page in pages:
                # Split page content into chunks
                page_chunks = self.text_splitter.split_text(page.page_content)
                
                for i, chunk_content in enumerate(page_chunks):
                    # Build chunk metadata
                    chunk_metadata = {
                        "document_id": document_id,
                        "page_number": page.metadata.get("page_number", 0),
                        "chunk_index": i,
                        "filename": page.metadata.get("filename"),
                        "file_path": page.metadata.get("file_path"),
                        "total_pages": page.metadata.get("total_pages"),
                        "chunk_length": len(chunk_content)
                    }
                    
                    # Add analysis metadata if available
                    if analysis:
                        chunk_metadata.update({
                            "document_type": analysis.document_type,
                            "language": analysis.language,
                            "confidence": analysis.confidence_score,
                            "sentiment": analysis.sentiment
                        })
                    
                    # Create chunk
                    chunk = DocumentChunk(
                        content=chunk_content,
                        metadata=chunk_metadata
                    )
                    all_chunks.append(chunk)
            
            logger.info(f"Created {len(all_chunks)} chunks for document {document_id}")
            return all_chunks
            
        except Exception as e:
            logger.error(f"Chunk creation failed for document {document_id}: {str(e)}")
            raise Exception(f"Failed to create chunks: {str(e)}")
    
    async def store_chunks_in_vector_db(
        self, 
        chunks: List[DocumentChunk]
    ) -> List[str]:
        """Store document chunks in vector database.
        
        Args:
            chunks: List of document chunks to store
            
        Returns:
            List of vector IDs created in the database
        """
        if not self.vector_store:
            logger.info("Vector storage disabled")
            return []
        
        try:
            logger.info(f"Storing {len(chunks)} chunks in vector database")
            
            # Convert chunks to LangChain documents for vector storage
            documents = []
            for chunk in chunks:
                doc = Document(
                    page_content=chunk.content,
                    metadata=chunk.metadata
                )
                documents.append(doc)
            
            # Add documents to vector store
            vector_ids = self.vector_store.add_documents(documents)
            
            logger.info(f"Successfully stored {len(vector_ids)} vectors")
            return vector_ids
            
        except Exception as e:
            logger.error(f"Vector storage failed: {str(e)}")
            return []
    
    async def ingest_document(
        self,
        pages: List[Document],
        document_id: str,
        file_content: Optional[bytes] = None,
        filename: str = "unknown.pdf"
    ) -> dict:
        """Complete document ingestion pipeline.
        
        Args:
            pages: List of document pages
            document_id: Unique document identifier
            
        Returns:
            Dictionary with ingestion results
        """
        start_time = datetime.now()
        
        try:
            logger.info(f"Starting document ingestion for {document_id}")
            
            # Step 0: Deduplication check (if file content is provided)
            dedup_result = None
            text_content = " ".join([page.page_content for page in pages])
            
            if file_content:
                logger.info(f"Performing deduplication check for {document_id}")
                
                # Extract agreement ID from VBC analysis for deduplication
                agreement_id = None
                if hasattr(self, 'vbc_analyzer') and self.vbc_analyzer:
                    try:
                        # Quick VBC analysis just for agreement ID extraction
                        vbc_response = await self.vbc_analyzer.analyze_contract(text_content, document_id)
                        if vbc_response.success and vbc_response.contract_data:
                            agreement_id = vbc_response.contract_data.agreement_id
                            logger.info(f"Extracted agreement ID for deduplication: {agreement_id}")
                    except Exception as e:
                        logger.warning(f"Failed to extract agreement ID for deduplication: {e}")
                
                # Perform comprehensive deduplication check
                dedup_result = deduplication_service.perform_comprehensive_check(
                    file_content=file_content,
                    text_content=text_content,
                    filename=filename,
                    agreement_id=agreement_id
                )
                
                if dedup_result['is_duplicate']:
                    duplicate_type = dedup_result['duplicate_type']
                    existing_doc = dedup_result['existing_document']
                    
                    logger.info(f"Duplicate document detected ({duplicate_type}): {document_id}")
                    logger.info(f"Existing document: {existing_doc['document_id']}")
                    
                    if dedup_result['should_update']:
                        # Update existing document (e.g., same agreement, new version)
                        logger.info(f"Updating existing document {existing_doc['document_id']}")
                        
                        # Update deduplication registry
                        updated_doc = deduplication_service.update_document(
                            existing_doc, filename, {"updated_version": True}
                        )
                        
                        # Continue with ingestion but log as update
                        logger.info(f"Proceeding with ingestion as document update")
                    else:
                        # Skip ingestion for exact duplicates
                        logger.info(f"Skipping ingestion for exact duplicate")
                        processing_time = (datetime.now() - start_time).total_seconds()
                        
                        return {
                            "success": True,
                            "document_id": document_id,
                            "duplicate_detected": True,
                            "duplicate_type": duplicate_type,
                            "existing_document_id": existing_doc['document_id'],
                            "skipped": True,
                            "message": f"Document already exists ({duplicate_type})",
                            "processing_time": processing_time
                        }
            
            # Step 1: Analyze document with LLM
            analysis = await self.analyze_document(pages, document_id)
            
            # Step 2: Create chunks
            chunks = self.create_chunks(pages, document_id, analysis)
            
            # Step 3: Store in vector database
            vector_ids = await self.store_chunks_in_vector_db(chunks)
            
            # Step 4: Persist VBC contract data to database (if available)
            vbc_contract_id = None
            if hasattr(self, '_vbc_contract_data') and self._vbc_contract_data and self.database_service:
                try:
                    # First save basic document metadata (create a document UUID for foreign key)
                    doc_uuid = str(uuid.uuid4())
                    await self.database_service.save_document_metadata(
                        doc_uuid=doc_uuid,
                        document_id=document_id,
                        filename=pages[0].metadata.get("filename", "unknown.pdf") if pages else "unknown.pdf",
                        file_size=sum(len(page.page_content) for page in pages),
                        document_type="vbc_contract",
                        summary=analysis.summary if analysis else "VBC Contract",
                        confidence_score=analysis.confidence_score if analysis else 0.0,
                        pages_processed=len(pages),
                        chunks_created=len(chunks),
                        vectors_stored=len(vector_ids),
                        processing_time_seconds=0.0  # Will update after calculation
                    )
                    
                    # Save VBC contract data
                    vbc_contract_id = await self.database_service.save_vbc_contract(
                        doc_uuid, self._vbc_contract_data
                    )
                    logger.info(f"Saved VBC contract to database with ID: {vbc_contract_id}")
                    
                    # Clean up stored VBC data
                    delattr(self, '_vbc_contract_data')
                    
                except Exception as e:
                    logger.warning(f"Failed to persist VBC contract data to database: {e}")
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Step 5: Register document in deduplication system (if not a duplicate)
            if file_content and (not dedup_result or not dedup_result['is_duplicate']):
                try:
                    # Extract agreement ID from VBC data if available
                    final_agreement_id = None
                    if hasattr(self, '_vbc_contract_data') and self._vbc_contract_data:
                        final_agreement_id = self._vbc_contract_data.agreement_id
                    elif dedup_result:
                        final_agreement_id = dedup_result.get('agreement_id')
                    
                    # Register the document
                    deduplication_service.register_document(
                        document_id=document_id,
                        filename=filename,
                        file_hash=dedup_result['file_hash'] if dedup_result else deduplication_service.generate_file_hash(file_content),
                        content_hash=dedup_result['content_hash'] if dedup_result else deduplication_service.generate_content_hash(text_content),
                        agreement_id=final_agreement_id,
                        metadata={
                            "pages": len(pages),
                            "chunks": len(chunks), 
                            "vectors": len(vector_ids),
                            "analysis_confidence": analysis.confidence_score if analysis else 0.0,
                            "document_type": analysis.document_type if analysis else "unknown",
                            "vbc_contract_id": vbc_contract_id
                        }
                    )
                    logger.info(f"Registered document in deduplication system: {document_id}")
                    
                except Exception as e:
                    logger.warning(f"Failed to register document in deduplication system: {e}")
            
            result = {
                "success": True,
                "document_id": document_id,
                "pages_processed": len(pages),
                "chunks_created": len(chunks),
                "vectors_stored": len(vector_ids),
                "processing_time": processing_time,
                "analysis": analysis.model_dump() if analysis else None,
                "chunks": [chunk.to_dict() for chunk in chunks],
                "vector_ids": vector_ids,
                "vbc_contract_id": vbc_contract_id,
                "deduplication_info": {
                    "checked": file_content is not None,
                    "is_duplicate": dedup_result['is_duplicate'] if dedup_result else False,
                    "duplicate_type": dedup_result.get('duplicate_type') if dedup_result else None,
                    "was_update": dedup_result.get('should_update') if dedup_result else False
                } if dedup_result else {"checked": False}
            }
            
            logger.info(
                f"Document ingestion completed for {document_id}: "
                f"{len(chunks)} chunks, {len(vector_ids)} vectors, {processing_time:.2f}s"
            )
            
            return result
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Document ingestion failed for {document_id}: {str(e)}")
            
            return {
                "success": False,
                "document_id": document_id,
                "error": str(e),
                "processing_time": processing_time
            }
