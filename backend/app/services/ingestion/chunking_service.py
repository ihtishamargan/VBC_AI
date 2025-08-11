"""Document chunking service for text processing."""

from typing import Any

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from backend.app.models import DocumentChunk, IngestionConfig
from backend.app.services.llm_analyzer import DocumentAnalysis
from backend.app.utils.logger import get_module_logger

logger = get_module_logger(__name__)


class DocumentChunkingService:
    """Service for creating document chunks from pages."""

    def __init__(self, config: IngestionConfig):
        """Initialize chunking service with configuration."""
        self.config = config
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        logger.info(
            f"Document chunking service initialized with chunk_size={config.chunk_size}, overlap={config.chunk_overlap}"
        )

    def create_chunks(
        self,
        pages: list[Document],
        document_id: str,
        analysis: DocumentAnalysis | None = None,
    ) -> list[DocumentChunk]:
        """Create text chunks from document pages.

        Args:
            pages: List of document pages
            document_id: Unique document identifier
            analysis: Optional analysis results for enhanced metadata

        Returns:
            List of document chunks with metadata
        """
        try:
            logger.info(
                f"Creating chunks for document {document_id} from {len(pages)} pages"
            )

            chunks = []

            for page_idx, page in enumerate(pages):
                # Split page content into chunks
                page_chunks = self.text_splitter.split_text(page.page_content)

                for chunk_idx, chunk_text in enumerate(page_chunks):
                    # Create base metadata
                    metadata = {
                        "document_id": document_id,
                        "page_number": page_idx + 1,
                        "chunk_index": chunk_idx,
                        "total_pages": len(pages),
                        "chunk_size": len(chunk_text),
                        "source": "document_ingestion",
                    }

                    # Add page metadata if available
                    if hasattr(page, "metadata") and page.metadata:
                        metadata.update(
                            {
                                f"page_{key}": value
                                for key, value in page.metadata.items()
                            }
                        )

                    # Add analysis metadata if available
                    if analysis:
                        # Handle different analysis types
                        if hasattr(analysis, 'document_type'):
                            # DocumentAnalysis object
                            metadata.update(
                                {
                                    "document_type": analysis.document_type,
                                    "confidence_score": analysis.confidence_score,
                                    "analysis_summary": analysis.summary[:200]
                                    if analysis.summary
                                    else None,
                                }
                            )
                        else:
                            # VBCContractAnalysisResponse or other analysis type
                            metadata.update(
                                {
                                    "document_type": "vbc_contract",
                                    "confidence_score": 1.0,  # VBC analysis is deterministic
                                    "analysis_summary": f"VBC Contract Analysis - Success: {getattr(analysis, 'success', False)}"
                                }
                            )

                    # Create chunk
                    chunk = DocumentChunk(content=chunk_text.strip(), metadata=metadata)
                    chunks.append(chunk)

            logger.info(f"Created {len(chunks)} chunks for document {document_id}")
            return chunks

        except Exception as e:
            logger.error(f"Failed to create chunks for document {document_id}: {e}")
            raise

    def validate_chunks(self, chunks: list[DocumentChunk]) -> bool:
        """Validate that chunks meet quality requirements.

        Args:
            chunks: List of chunks to validate

        Returns:
            True if chunks are valid, False otherwise
        """
        if not chunks:
            logger.warning("No chunks to validate")
            return False

        try:
            for i, chunk in enumerate(chunks):
                # Check minimum content length
                if len(chunk.content.strip()) < 10:
                    logger.warning(
                        f"Chunk {i} has insufficient content (length: {len(chunk.content)})"
                    )
                    return False

                # Check maximum content length
                if len(chunk.content) > self.config.chunk_size * 2:
                    logger.warning(
                        f"Chunk {i} exceeds maximum size (length: {len(chunk.content)})"
                    )
                    return False

                # Check required metadata
                required_fields = ["document_id", "page_number", "chunk_index"]
                for field in required_fields:
                    if field not in chunk.metadata:
                        logger.warning(
                            f"Chunk {i} missing required metadata field: {field}"
                        )
                        return False

            logger.info(f"Validated {len(chunks)} chunks successfully")
            return True

        except Exception as e:
            logger.error(f"Chunk validation failed: {e}")
            return False

    def get_chunk_statistics(self, chunks: list[DocumentChunk]) -> dict[str, Any]:
        """Get statistics about the created chunks.

        Args:
            chunks: List of chunks to analyze

        Returns:
            Dictionary containing chunk statistics
        """
        if not chunks:
            return {"total_chunks": 0}

        try:
            chunk_sizes = [len(chunk.content) for chunk in chunks]
            pages_covered = {
                chunk.metadata.get("page_number", 0) for chunk in chunks
            }

            stats = {
                "total_chunks": len(chunks),
                "total_characters": sum(chunk_sizes),
                "avg_chunk_size": sum(chunk_sizes) / len(chunks),
                "min_chunk_size": min(chunk_sizes),
                "max_chunk_size": max(chunk_sizes),
                "pages_covered": len(pages_covered),
                "chunks_per_page": len(chunks) / len(pages_covered)
                if pages_covered
                else 0,
            }

            logger.info(f"Chunk statistics: {stats}")
            return stats

        except Exception as e:
            logger.error(f"Failed to calculate chunk statistics: {e}")
            return {"total_chunks": len(chunks), "error": str(e)}

    def update_config(self, new_config: IngestionConfig) -> None:
        """Update chunking configuration and reinitialize splitter.

        Args:
            new_config: New configuration to apply
        """
        if (
            new_config.chunk_size != self.config.chunk_size
            or new_config.chunk_overlap != self.config.chunk_overlap
        ):
            logger.info(
                f"Updating chunking config: size {self.config.chunk_size}->{new_config.chunk_size}, "
                f"overlap {self.config.chunk_overlap}->{new_config.chunk_overlap}"
            )

            self.config = new_config
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=new_config.chunk_size,
                chunk_overlap=new_config.chunk_overlap,
                separators=["\n\n", "\n", ". ", " ", ""],
            )
