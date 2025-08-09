"""
PDF Processing Pipeline using LangChain
- Async load with PyPDFLoader.alazy_load
- Simple, reliable chunking with start indices
- Delegates analysis to LLMDocumentAnalyzer (o4-mini)
- Pushes chunks to Qdrant vector store for retrieval
"""

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from pydantic import BaseModel

from llm_analyzer import DocumentAnalysis, LLMDocumentAnalyzer
from qdrant_store import QdrantStoreDense, chunks_to_documents

logger = logging.getLogger(__name__)


class PDFMetadata(BaseModel):
    document_id: str
    filename: str
    file_path: str
    total_pages: int
    total_chunks: int
    processing_time: float
    processed_at: datetime
    file_size: int


class ExtractedInsights(BaseModel):
    document_id: str
    summary: str
    key_topics: List[str]
    entities: List[Dict[str, Any]]
    document_type: str
    confidence_score: float
    metadata: Dict[str, Any]


class PDFChunk(BaseModel):
    chunk_id: str
    document_id: str
    page_number: int
    content: str
    metadata: Dict[str, Any]


class PDFProcessor:
    def __init__(
        self,
        chunk_size: int = 1200,
        chunk_overlap: int = 200,
        upload_dir: str = "uploads",
        llm_model: str = "o4-mini",
        collection_name: str = "pdf_documents",
        enable_vector_store: bool = True,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.upload_dir = Path(upload_dir)
        self.upload_dir.mkdir(exist_ok=True)
        self.enable_vector_store = enable_vector_store

        # Character-based splitter with start indices in metadata.
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
            add_start_index=True,  # exposes 'start_index' in chunk.metadata
        )

        self.llm_analyzer = LLMDocumentAnalyzer(model=llm_model)
        
        # Initialize Qdrant vector store if enabled
        self.vector_store = None
        if self.enable_vector_store:
            try:
                self.vector_store = QdrantStoreDense(collection_name=collection_name)
                logger.info(f"Vector store initialized: collection={collection_name}")
            except Exception as e:
                logger.warning(f"Vector store initialization failed: {e}. Continuing without vector storage.")
                self.enable_vector_store = False
        
        logger.info(
            f"PDFProcessor initialized: model={llm_model}, chunk_size={chunk_size}, overlap={chunk_overlap}, vector_store={self.enable_vector_store}"
        )

    async def process_pdf(self, file_path: str, document_id: str) -> Dict[str, Any]:
        start_time = datetime.now()
        try:
            filename = Path(file_path).name
            file_stat = os.stat(file_path)

            logger.info(f"Processing PDF {document_id}: {file_path}")

            # Async page loading (modern pattern).
            loader = PyPDFLoader(file_path)
            pages: List[Document] = []
            async for page in loader.alazy_load():
                pages.append(page)

            if not pages:
                raise ValueError(f"No pages found in PDF: {file_path}")

            # Split into chunks; preserve page numbers from source metadata if present.
            chunks: List[Document] = self.text_splitter.split_documents(pages)

            pdf_chunks: List[PDFChunk] = []
            for i, chunk in enumerate(chunks):
                page_no = int(chunk.metadata.get("page", 0))
                start_idx = int(chunk.metadata.get("start_index", 0))

                pdf_chunks.append(
                    PDFChunk(
                        chunk_id=f"{document_id}_chunk_{i}",
                        document_id=document_id,
                        page_number=page_no,
                        content=chunk.page_content,
                        metadata={
                            **chunk.metadata,
                            "chunk_index": i,
                            "start_index": start_idx,
                        },
                    )
                )

            # LLM analysis on a compact sample of the full text
            full_content = " ".join([p.page_content for p in pages])
            document_info = {
                "document_id": document_id,
                "total_pages": len(pages),
                "total_chunks": len(chunks),
                "content_length": len(full_content),
                "filename": filename,
                "file_size": file_stat.st_size,
            }
            analysis: DocumentAnalysis = await self.llm_analyzer.analyze_document(
                full_content, document_info
            )

            insights = ExtractedInsights(
                document_id=document_id,
                summary=analysis.summary,
                key_topics=analysis.key_topics,
                entities=[
                    {
                        "text": e.text,
                        "type": e.type,
                        "confidence": e.confidence,
                        "context": e.context,
                        "start_pos": e.start_pos,
                        "end_pos": e.end_pos,
                    }
                    for e in analysis.entities
                ],
                document_type=analysis.document_type,
                confidence_score=analysis.confidence_score,
                metadata={
                    "total_pages": len(pages),
                    "total_chunks": len(chunks),
                    "extraction_method": "structured_llm",
                    "model": self.llm_analyzer.model,
                    "language": analysis.language,
                    "sentiment": analysis.sentiment,
                    "key_insights": analysis.key_insights,
                },
            )

            # Store chunks in vector database if enabled
            vector_ids = []
            if self.enable_vector_store and self.vector_store:
                try:
                    # Convert chunks to LangChain Documents for vector storage
                    chunk_dicts = [c.model_dump() for c in pdf_chunks]
                    documents = chunks_to_documents(chunk_dicts)
                    
                    # Add documents to vector store
                    vector_ids = self.vector_store.add_documents(documents)
                    logger.info(f"Stored {len(vector_ids)} chunks in vector database for {document_id}")
                except Exception as e:
                    logger.warning(f"Failed to store chunks in vector database: {e}")

            processing_time = (datetime.now() - start_time).total_seconds()
            metadata = PDFMetadata(
                document_id=document_id,
                filename=filename,
                file_path=file_path,
                total_pages=len(pages),
                total_chunks=len(chunks),
                processing_time=processing_time,
                processed_at=datetime.now(),
                file_size=file_stat.st_size,
            )

            return {
                "success": True,
                "metadata": metadata.model_dump(),
                "chunks": [c.model_dump() for c in pdf_chunks],
                "insights": insights.model_dump(),
                # Keep previews tiny to avoid payload bloat
                "raw_pages": [
                    {
                        "page": i,
                        "content": (
                            page.page_content[:500] + "..."
                            if len(page.page_content) > 500
                            else page.page_content
                        ),
                    }
                    for i, page in enumerate(pages[:3])
                ],
            }

        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.exception(f"Error processing PDF {document_id}")
            return {
                "success": False,
                "error": str(e),
                "processing_time": processing_time,
                "document_id": document_id,
            }

    def save_uploaded_file(self, file_content: bytes, filename: str) -> str:
        file_path = self.upload_dir / filename
        counter = 1
        while file_path.exists():
            name, ext = filename.rsplit(".", 1)
            file_path = self.upload_dir / f"{name}_{counter}.{ext}"
            counter += 1
        with open(file_path, "wb") as f:
            f.write(file_content)
        logger.info(f"Saved uploaded file: {file_path}")
        return str(file_path)
