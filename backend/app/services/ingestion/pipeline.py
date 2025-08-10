"""Document ingestion pipeline orchestrator."""

from datetime import datetime
from typing import Any

from langchain.schema import Document

from backend.app.models.ingestion import (
    DeduplicationInfo,
    IngestionConfig,
    IngestionResult,
    IngestionStatus,
)
from backend.app.services.database import database_service
from backend.app.utils.deduplication import deduplication_service
from backend.app.services.ingestion.analysis_service import DocumentAnalysisService
from backend.app.services.ingestion.chunking_service import DocumentChunkingService
from backend.app.services.ingestion.vector_service import VectorStorageService
from backend.app.utils.logger import get_module_logger

logger = get_module_logger(__name__)


class DocumentIngestionPipeline:
    """Orchestrates the complete document ingestion process."""

    def __init__(self, config: IngestionConfig = None):
        """Initialize the ingestion pipeline."""
        self.config = config or IngestionConfig()

        # Initialize services
        self.analysis_service = DocumentAnalysisService(self.config.analysis_strategy)
        self.chunking_service = DocumentChunkingService(self.config)
        self.vector_service = VectorStorageService(self.config)

        logger.info("Document ingestion pipeline initialized successfully")

    async def ingest_document(
        self,
        pages: list[Document],
        document_id: str,
        file_content: bytes | None = None,
        filename: str = "unknown.pdf",
    ) -> IngestionResult:
        """Execute the complete document ingestion pipeline.

        Args:
            pages: List of document pages
            document_id: Unique document identifier
            file_content: Optional raw file content for deduplication
            filename: Original filename

        Returns:
            IngestionResult with processing details
        """
        start_time = datetime.now()

        try:
            logger.info(f"Starting document ingestion pipeline for {document_id}")

            # Step 1: Check for duplicates (if enabled)
            dedup_info = None
            if self.config.enable_deduplication and file_content:
                dedup_info = await self._check_deduplication(
                    file_content, pages, document_id, filename
                )

                if dedup_info and dedup_info.is_duplicate and not dedup_info.was_update:
                    logger.info(
                        f"Document {document_id} is a duplicate, skipping processing"
                    )
                    return self._create_duplicate_result(
                        document_id, start_time, dedup_info
                    )

            # Step 2: Analyze document
            analysis = None
            if self.analysis_service.is_available:
                logger.info(f"Analyzing document {document_id}")
                analysis = await self.analysis_service.analyze_document(
                    pages, document_id
                )

            # Step 3: Create chunks
            logger.info(f"Creating chunks for document {document_id}")
            chunks = self.chunking_service.create_chunks(pages, document_id, analysis)

            if not chunks:
                raise ValueError("No chunks were created from document")

            # Validate chunks
            if not self.chunking_service.validate_chunks(chunks):
                raise ValueError("Chunk validation failed")

            # Step 4: Store in vector database
            vector_ids = []
            if self.vector_service.is_available:
                logger.info(f"Storing {len(chunks)} chunks in vector database")
                vector_ids = await self.vector_service.store_chunks(chunks)

                if not vector_ids:
                    logger.warning("No vectors were stored in database")

            # Step 5: Store in database (if enabled)
            vbc_contract_id = None
            if self.config.enable_database_storage:
                vbc_contract_id = await self._store_in_database(
                    document_id, pages, chunks, analysis, vector_ids
                )

            # Step 6: Register in deduplication system
            if dedup_info and not dedup_info.is_duplicate:
                await self._register_deduplication(
                    document_id,
                    filename,
                    file_content,
                    pages,
                    chunks,
                    vector_ids,
                    analysis,
                    vbc_contract_id,
                    dedup_info,
                )

            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()

            # Create success result
            result = IngestionResult.success_result(
                document_id=document_id,
                processing_time=processing_time,
                pages_processed=len(pages),
                chunks_created=len(chunks),
                vectors_stored=len(vector_ids),
                analysis=analysis.model_dump() if analysis else None,
                vbc_analysis=getattr(analysis, "vbc_data", {}) if analysis else None,
                vbc_contract_id=vbc_contract_id,
                chunks=[chunk.to_dict() for chunk in chunks],
                vector_ids=vector_ids,
                deduplication_info=dedup_info,
            )

            logger.info(
                f"Document ingestion completed successfully for {document_id}: "
                f"{len(chunks)} chunks, {len(vector_ids)} vectors, {processing_time:.2f}s"
            )

            return result

        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Document ingestion failed for {document_id}: {e}")

            return IngestionResult.failure_result(
                document_id=document_id,
                processing_time=processing_time,
                error=str(e),
                error_details={"exception_type": type(e).__name__},
            )

    async def _check_deduplication(
        self,
        file_content: bytes,
        pages: list[Document],
        document_id: str,
        filename: str,
    ) -> DeduplicationInfo:
        """Check for document duplicates."""
        try:
            logger.info(f"Checking for duplicates: {document_id}")

            # Combine page content for deduplication
            text_content = "\n\n".join([page.page_content for page in pages])

            # Check for duplicates
            dedup_result = deduplication_service.check_duplicate(
                file_content=file_content, text_content=text_content, filename=filename
            )

            return DeduplicationInfo(
                checked=True,
                is_duplicate=dedup_result.get("is_duplicate", False),
                duplicate_type=dedup_result.get("duplicate_type"),
                was_update=dedup_result.get("should_update", False),
                file_hash=dedup_result.get("file_hash"),
                content_hash=dedup_result.get("content_hash"),
            )

        except Exception as e:
            logger.error(f"Deduplication check failed: {e}")
            return DeduplicationInfo(checked=False)

    async def _store_in_database(
        self,
        document_id: str,
        pages: list[Document],
        chunks: list,
        analysis,
        vector_ids: list[str],
    ) -> str | None:
        """Store document data in database."""
        try:
            logger.info(f"Storing document data in database: {document_id}")

            # Store document metadata
            await database_service.save_document_metadata(
                doc_uuid=document_id,
                document_id=document_id,
                filename=getattr(pages[0], "metadata", {}).get(
                    "filename", "unknown.pdf"
                ),
                file_size=0,  # TODO: Get actual file size
                document_type=analysis.document_type if analysis else "unknown",
                summary=analysis.summary if analysis else "",
                confidence_score=analysis.confidence_score if analysis else 0.0,
                pages_processed=len(pages),
                chunks_created=len(chunks),
                vectors_stored=len(vector_ids),
                processing_time_seconds=0,  # Will be updated later
            )

            # Store VBC contract data if available
            vbc_contract_id = None
            if analysis and hasattr(analysis, "vbc_data") and analysis.vbc_data:
                vbc_contract_id = await database_service.save_vbc_contract(
                    doc_uuid=document_id, vbc_data=analysis.vbc_data
                )

            return vbc_contract_id

        except Exception as e:
            logger.error(f"Database storage failed: {e}")
            return None

    async def _register_deduplication(
        self,
        document_id: str,
        filename: str,
        file_content: bytes | None,
        pages: list[Document],
        chunks: list,
        vector_ids: list[str],
        analysis,
        vbc_contract_id: str | None,
        dedup_info: DeduplicationInfo,
    ) -> None:
        """Register document in deduplication system."""
        try:
            # Extract agreement ID if available
            agreement_id = None
            if analysis and hasattr(analysis, "vbc_data") and analysis.vbc_data:
                agreement_id = getattr(analysis.vbc_data, "agreement_id", None)

            # Register document
            deduplication_service.register_document(
                document_id=document_id,
                filename=filename,
                file_hash=dedup_info.file_hash
                or deduplication_service.generate_file_hash(file_content),
                content_hash=dedup_info.content_hash
                or deduplication_service.generate_content_hash(
                    "\n\n".join([page.page_content for page in pages])
                ),
                agreement_id=agreement_id,
                metadata={
                    "pages": len(pages),
                    "chunks": len(chunks),
                    "vectors": len(vector_ids),
                    "analysis_confidence": analysis.confidence_score
                    if analysis
                    else 0.0,
                    "document_type": analysis.document_type if analysis else "unknown",
                    "vbc_contract_id": vbc_contract_id,
                },
            )

            logger.info(f"Registered document in deduplication system: {document_id}")

        except Exception as e:
            logger.warning(f"Failed to register document in deduplication system: {e}")

    def _create_duplicate_result(
        self, document_id: str, start_time: datetime, dedup_info: DeduplicationInfo
    ) -> IngestionResult:
        """Create result for duplicate documents."""
        processing_time = (datetime.now() - start_time).total_seconds()

        return IngestionResult(
            success=True,
            document_id=document_id,
            status=IngestionStatus.COMPLETED,
            processing_time=processing_time,
            pages_processed=0,
            chunks_created=0,
            vectors_stored=0,
            deduplication_info=dedup_info,
        )

    def update_config(self, new_config: IngestionConfig) -> None:
        """Update pipeline configuration."""
        logger.info("Updating ingestion pipeline configuration")

        # Update strategy if changed
        if new_config.analysis_strategy != self.config.analysis_strategy:
            self.analysis_service.change_strategy(new_config.analysis_strategy)

        # Update services
        self.chunking_service.update_config(new_config)
        self.vector_service.update_config(new_config)

        self.config = new_config
        logger.info("Pipeline configuration updated successfully")

    async def get_pipeline_status(self) -> dict[str, Any]:
        """Get status of all pipeline components."""
        return {
            "analysis_service": {
                "available": self.analysis_service.is_available,
                "strategy": self.config.analysis_strategy.value,
            },
            "chunking_service": {
                "available": True,
                "chunk_size": self.config.chunk_size,
                "chunk_overlap": self.config.chunk_overlap,
            },
            "vector_service": {
                "available": self.vector_service.is_available,
                "collection": self.config.collection_name,
            },
            "deduplication": {"enabled": self.config.enable_deduplication},
            "database_storage": {"enabled": self.config.enable_database_storage},
        }
