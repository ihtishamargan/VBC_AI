"""Document ingestion service - refactored for modularity and maintainability."""

from langchain.schema import Document

from backend.app.models.ingestion import (
    AnalysisStrategy,
    IngestionConfig,
    IngestionResult,
)
from backend.app.services.ingestion.pipeline import DocumentIngestionPipeline
from backend.app.utils.logger import get_module_logger

logger = get_module_logger(__name__)


class DocumentIngestionService:
    """Simplified document ingestion service using modular pipeline architecture."""

    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap: int = None,
        llm_model: str = None,
        collection_name: str = None,
        use_vbc_analyzer: bool = True,
    ):
        """Initialize the ingestion service with modular pipeline.

        Args:
            chunk_size: Size of text chunks (defaults to config)
            chunk_overlap: Overlap between chunks (defaults to config)
            llm_model: OpenAI model to use (defaults to config)
            collection_name: Qdrant collection name (defaults to config)
            use_vbc_analyzer: Whether to use VBC-specific analyzer (default: True)
        """
        # Create configuration
        self.config = IngestionConfig(
            chunk_size=chunk_size or 1000,
            chunk_overlap=chunk_overlap or 200,
            analysis_strategy=AnalysisStrategy.VBC_CONTRACT
            if use_vbc_analyzer
            else AnalysisStrategy.GENERIC_LLM,
            llm_model=llm_model or "gpt-4",
            collection_name=collection_name or "vbc_documents",
        )

        # Initialize pipeline
        self.pipeline = DocumentIngestionPipeline(self.config)

        logger.info("DocumentIngestionService initialized with modular pipeline")

    async def analyze_document(
        self, pages: list[Document], document_id: str
    ) -> dict | None:
        """Analyze document content using the pipeline.

        Args:
            pages: List of document pages
            document_id: Unique document identifier

        Returns:
            Analysis results or None if disabled
        """
        logger.info(f"Delegating document analysis to pipeline for {document_id}")
        return await self.pipeline.analysis_service.analyze_document(pages, document_id)

    def create_chunks(
        self, pages: list[Document], document_id: str, analysis: dict | None = None
    ) -> list[dict]:
        """Create text chunks from document pages using the pipeline.

        Args:
            pages: List of document pages
            document_id: Unique document identifier
            analysis: Optional analysis results

        Returns:
            List of document chunks with metadata
        """
        logger.info(f"Delegating chunk creation to pipeline for {document_id}")
        chunks = self.pipeline.chunking_service.create_chunks(
            pages, document_id, analysis
        )
        return [chunk.to_dict() for chunk in chunks]

    async def store_chunks_in_vector_db(self, chunks: list[dict]) -> list[str]:
        """Store document chunks in vector database using the pipeline.

        Args:
            chunks: List of document chunks to store

        Returns:
            List of vector IDs created in the database
        """
        logger.info(f"Delegating vector storage to pipeline for {len(chunks)} chunks")
        # Convert dict chunks back to DocumentChunk objects for the pipeline
        from backend.app.models.ingestion import DocumentChunk

        chunk_objects = [
            DocumentChunk(content=c["content"], metadata=c["metadata"]) for c in chunks
        ]
        return await self.pipeline.vector_service.store_chunks(chunk_objects)

    async def ingest_document(
        self,
        pages: list[Document],
        document_id: str,
        file_content: bytes | None = None,
        filename: str = "unknown.pdf",
    ) -> IngestionResult:
        """Complete document ingestion pipeline using modular architecture.

        Args:
            pages: List of document pages
            document_id: Unique document identifier
            file_content: Optional raw file content for deduplication
            filename: Original filename

        Returns:
            IngestionResult with processing details
        """
        logger.info(
            f"Delegating complete document ingestion to pipeline for {document_id}"
        )
        return await self.pipeline.ingest_document(
            pages, document_id, file_content, filename
        )
