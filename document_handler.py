"""
Modular Document Processing Handler
Orchestrates the complete document processing pipeline with clear separation of concerns:
1. Parse PDF to extract text
2. Analyze text with LLM for insights and entities
3. Create chunks with metadata
4. Push chunks to vector store
"""

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from pydantic import BaseModel

from llm_analyzer import DocumentAnalysis, LLMDocumentAnalyzer
from qdrant_store import QdrantStoreDense, chunks_to_documents

logger = logging.getLogger(__name__)


class DocumentMetadata(BaseModel):
    """Metadata for processed documents."""
    document_id: str
    filename: str
    file_path: str
    file_size: int
    total_pages: int
    total_chunks: int
    processing_time: float
    processed_at: datetime


class ProcessedChunk(BaseModel):
    """Individual processed chunk with metadata."""
    chunk_id: str
    document_id: str
    page_number: int
    content: str
    start_index: int
    metadata: Dict[str, Any]


class DocumentProcessingResult(BaseModel):
    """Complete result of document processing."""
    success: bool
    document_id: str
    metadata: DocumentMetadata
    analysis: Optional[DocumentAnalysis] = None
    chunks: List[ProcessedChunk] = []
    vector_ids: List[str] = []
    error: Optional[str] = None


class DocumentHandler:
    """
    Modular document processing handler with clear pipeline stages.
    """
    
    def __init__(
        self,
        chunk_size: int = 1200,
        chunk_overlap: int = 200,
        llm_model: str = "o4-mini",
        collection_name: str = "pdf_documents",
        enable_vector_store: bool = True,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.enable_vector_store = enable_vector_store
        
        # Initialize text splitter for chunking
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
            add_start_index=True,
        )
        
        # Initialize LLM analyzer
        self.llm_analyzer = LLMDocumentAnalyzer(model=llm_model)
        
        # Initialize vector store if enabled
        self.vector_store = None
        if self.enable_vector_store:
            try:
                self.vector_store = QdrantStoreDense(collection_name=collection_name)
                logger.info(f"Vector store initialized: collection={collection_name}")
            except Exception as e:
                logger.warning(f"Vector store initialization failed: {e}. Continuing without vector storage.")
                self.enable_vector_store = False
        
        logger.info(
            f"DocumentHandler initialized: llm_model={llm_model}, chunk_size={chunk_size}, "
            f"overlap={chunk_overlap}, vector_store={self.enable_vector_store}"
        )

    async def process_document(
        self, 
        file_path: str, 
        document_id: str
    ) -> DocumentProcessingResult:
        """
        Process a document through the complete pipeline.
        
        Args:
            file_path: Path to the document file
            document_id: Unique identifier for the document
            
        Returns:
            DocumentProcessingResult with all processing details
        """
        start_time = datetime.now()
        
        try:
            # Step 1: Parse PDF to extract text
            pages, file_metadata = await self._parse_pdf_to_text(file_path)
            
            # Step 2: Analyze text with LLM for insights and entities
            analysis = await self._analyze_text_with_llm(pages, file_metadata, document_id)
            
            # Step 3: Create chunks with metadata
            chunks = await self._create_chunks_with_metadata(pages, document_id)
            
            # Step 4: Push chunks to vector store
            vector_ids = await self._push_chunks_to_vector_store(chunks, document_id)
            
            # Create final metadata
            processing_time = (datetime.now() - start_time).total_seconds()
            metadata = DocumentMetadata(
                document_id=document_id,
                filename=file_metadata["filename"],
                file_path=file_path,
                file_size=file_metadata["file_size"],
                total_pages=len(pages),
                total_chunks=len(chunks),
                processing_time=processing_time,
                processed_at=datetime.now(),
            )
            
            logger.info(
                f"Document processing completed for {document_id}: "
                f"{len(pages)} pages, {len(chunks)} chunks, {len(vector_ids)} vectors stored"
            )
            
            return DocumentProcessingResult(
                success=True,
                document_id=document_id,
                metadata=metadata,
                analysis=analysis,
                chunks=chunks,
                vector_ids=vector_ids,
            )
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Document processing failed for {document_id}: {str(e)}")
            
            return DocumentProcessingResult(
                success=False,
                document_id=document_id,
                metadata=DocumentMetadata(
                    document_id=document_id,
                    filename=Path(file_path).name,
                    file_path=file_path,
                    file_size=0,
                    total_pages=0,
                    total_chunks=0,
                    processing_time=processing_time,
                    processed_at=datetime.now(),
                ),
                error=str(e),
            )

    async def _parse_pdf_to_text(self, file_path: str) -> Tuple[List[Document], Dict[str, Any]]:
        """
        Step 1: Parse PDF and extract text content.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Tuple of (pages, file_metadata)
        """
        logger.info(f"Parsing PDF: {file_path}")
        
        # Get file metadata
        file_stat = os.stat(file_path)
        filename = Path(file_path).name
        
        # Load PDF using LangChain
        loader = PyPDFLoader(file_path)
        pages = []
        
        async for page in loader.alazy_load():
            pages.append(page)
        
        if not pages:
            raise ValueError(f"No pages found in PDF: {file_path}")
        
        file_metadata = {
            "filename": filename,
            "file_size": file_stat.st_size,
            "total_pages": len(pages),
        }
        
        logger.info(f"PDF parsed successfully: {len(pages)} pages extracted")
        return pages, file_metadata

    async def _analyze_text_with_llm(
        self, 
        pages: List[Document], 
        file_metadata: Dict[str, Any], 
        document_id: str
    ) -> DocumentAnalysis:
        """
        Step 2: Analyze text with LLM for insights and entity extraction.
        
        Args:
            pages: List of page documents
            file_metadata: File metadata
            document_id: Document identifier
            
        Returns:
            DocumentAnalysis with insights and entities
        """
        logger.info(f"Analyzing text with LLM for document: {document_id}")
        
        # Combine text from all pages for analysis
        full_content = " ".join([page.page_content for page in pages])
        
        # Prepare document info for LLM
        document_info = {
            "document_id": document_id,
            "total_pages": len(pages),
            "content_length": len(full_content),
            **file_metadata,
        }
        
        # Perform LLM analysis
        analysis = await self.llm_analyzer.analyze_document(full_content, document_info)
        
        logger.info(
            f"LLM analysis completed: type={analysis.document_type}, "
            f"confidence={analysis.confidence_score:.3f}, "
            f"entities={len(analysis.entities)}, topics={len(analysis.key_topics)}"
        )
        
        return analysis

    async def _create_chunks_with_metadata(
        self, 
        pages: List[Document], 
        document_id: str
    ) -> List[ProcessedChunk]:
        """
        Step 3: Create chunks with metadata.
        
        Args:
            pages: List of page documents
            document_id: Document identifier
            
        Returns:
            List of ProcessedChunk objects
        """
        logger.info(f"Creating chunks for document: {document_id}")
        
        # Split documents into chunks
        chunks = self.text_splitter.split_documents(pages)
        
        # Create ProcessedChunk objects with metadata
        processed_chunks = []
        for i, chunk in enumerate(chunks):
            page_no = int(chunk.metadata.get("page", 0))
            start_idx = int(chunk.metadata.get("start_index", 0))
            
            processed_chunk = ProcessedChunk(
                chunk_id=f"{document_id}_chunk_{i}",
                document_id=document_id,
                page_number=page_no,
                content=chunk.page_content,
                start_index=start_idx,
                metadata={
                    **chunk.metadata,
                    "chunk_index": i,
                    "start_index": start_idx,
                    "document_id": document_id,
                },
            )
            processed_chunks.append(processed_chunk)
        
        logger.info(f"Created {len(processed_chunks)} chunks for document: {document_id}")
        return processed_chunks

    async def _push_chunks_to_vector_store(
        self, 
        chunks: List[ProcessedChunk], 
        document_id: str
    ) -> List[str]:
        """
        Step 4: Push chunks to vector store.
        
        Args:
            chunks: List of processed chunks
            document_id: Document identifier
            
        Returns:
            List of vector IDs from the vector store
        """
        vector_ids = []
        
        if not (self.enable_vector_store and self.vector_store):
            logger.info(f"Vector store disabled, skipping storage for {document_id}")
            return vector_ids
        
        try:
            logger.info(f"Pushing {len(chunks)} chunks to vector store for document: {document_id}")
            
            # Convert ProcessedChunk objects to dictionaries
            chunk_dicts = [chunk.model_dump() for chunk in chunks]
            
            # Convert to LangChain Documents for vector storage
            documents = chunks_to_documents(chunk_dicts)
            
            # Add documents to vector store
            vector_ids = self.vector_store.add_documents(documents)
            
            logger.info(f"Successfully stored {len(vector_ids)} chunks in vector database for {document_id}")
            
        except Exception as e:
            logger.warning(f"Failed to store chunks in vector database for {document_id}: {e}")
        
        return vector_ids

    def get_retriever(self, k: int = 8, mmr: bool = True, score_threshold: Optional[float] = None):
        """
        Get a retriever for semantic search over stored documents.
        
        Args:
            k: Number of documents to retrieve
            mmr: Whether to use maximum marginal relevance
            score_threshold: Minimum score threshold for retrieval
            
        Returns:
            Retriever object for semantic search
        """
        if not (self.enable_vector_store and self.vector_store):
            raise ValueError("Vector store is not enabled or not initialized")
        
        return self.vector_store.retriever(
            k=k, 
            mmr=mmr, 
            score_threshold=score_threshold
        )

    @staticmethod
    def save_uploaded_file(file_content: bytes, filename: str, upload_dir: str = "uploads") -> str:
        """
        Save uploaded file to disk and return file path.
        
        Args:
            file_content: File content as bytes
            filename: Original filename
            upload_dir: Directory to save files
            
        Returns:
            Path to saved file
        """
        upload_path = Path(upload_dir)
        upload_path.mkdir(exist_ok=True)
        
        file_path = upload_path / filename
        
        # Ensure unique filename
        counter = 1
        while file_path.exists():
            name, ext = filename.rsplit(".", 1) if "." in filename else (filename, "")
            new_filename = f"{name}_{counter}.{ext}" if ext else f"{name}_{counter}"
            file_path = upload_path / new_filename
            counter += 1
        
        with open(file_path, "wb") as f:
            f.write(file_content)
        
        logger.info(f"Saved uploaded file: {file_path}")
        return str(file_path)
