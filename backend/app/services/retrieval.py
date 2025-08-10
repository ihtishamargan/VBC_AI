"""Document retrieval service for search and chat functionality."""

from typing import Any

from backend.app.config import settings
from backend.app.models import SearchChunk
from backend.app.services.qdrant_store import QdrantStoreDense as QdrantStore
from backend.app.utils.logger import get_module_logger

logger = get_module_logger(__name__)


class DocumentRetrievalService:
    """Service for retrieving and searching document chunks."""

    def __init__(self, collection_name: str = None):
        """Initialize the retrieval service.

        Args:
            collection_name: Qdrant collection name (defaults to config)
        """
        # Initialize vector store for retrieval
        try:
            self.vector_store = QdrantStore(
                collection_name=collection_name or settings.qdrant_collection_name
            )
            logger.info("Vector store initialized for retrieval")
        except Exception as e:
            logger.warning(f"Vector store initialization failed: {e}")
            self.vector_store = None

    async def search_documents(
        self, query: str, filters: dict[str, Any] | None = None, limit: int = 10
    ) -> list[SearchChunk]:
        """Search for relevant document chunks.

        Args:
            query: Search query string
            filters: Optional metadata filters
            limit: Maximum number of results to return

        Returns:
            List of relevant document chunks with scores
        """
        if not self.vector_store:
            logger.warning("Vector store not available, returning mock results")
            return self._get_mock_search_results(query, limit)

        try:
            logger.info(f"Searching for: '{query}' with filters: {filters}")

            # Perform hybrid search (dense + sparse) for better results
            if hasattr(self.vector_store, "hybrid_search"):
                logger.info("Using hybrid search (dense + sparse vectors)")
                results = self.vector_store.hybrid_search(
                    query=query,
                    k=limit,
                    dense_weight=0.6,  # 60% semantic understanding
                    sparse_weight=0.4,  # 40% keyword matching
                    filter=filters,
                    fusion_algorithm="rrf",  # Reciprocal Rank Fusion
                )
            else:
                logger.info("Using dense-only search (fallback)")
                results = self.vector_store.similarity_search_with_score(
                    query=query, k=limit, filter=filters
                )

            # Convert to SearchChunk objects
            search_chunks = []
            for doc, score in results:
                chunk = SearchChunk(
                    chunk_id=doc.metadata.get(
                        "chunk_id", f"chunk-{len(search_chunks)}"
                    ),
                    document_id=doc.metadata.get("document_id", "unknown"),
                    content=doc.page_content,
                    score=float(score),
                    metadata=doc.metadata,
                )
                search_chunks.append(chunk)

            logger.info(f"Found {len(search_chunks)} relevant chunks")
            return search_chunks

        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            return self._get_mock_search_results(query, limit)

    def _generate_mock_response(self, message: str, source_count: int) -> str:
        """Generate a mock response when LLM is not available.

        Args:
            message: User's message
            source_count: Number of relevant sources found

        Returns:
            Mock response string
        """
        if source_count > 0:
            return (
                f"Based on your question '{message}', I found {source_count} relevant sections "
                f"in the uploaded documents. The documents contain information that may help "
                f"answer your query about contract terms, payment conditions, and service agreements."
            )
        return (
            f"I understand you're asking about '{message}', but I couldn't find directly "
            f"relevant information in the currently uploaded documents. Could you try "
            f"rephrasing your question or uploading more specific documents?"
        )

    def _get_mock_search_results(self, query: str, limit: int) -> list[SearchChunk]:
        """Generate mock search results when vector store is not available.

        Args:
            query: Search query
            limit: Maximum results to return

        Returns:
            List of mock SearchChunk objects
        """
        mock_results = []

        for i in range(min(2, limit)):  # Return up to 2 mock results
            chunk = SearchChunk(
                chunk_id=f"mock-chunk-{i + 1}",
                document_id=f"mock-doc-{i + 1}",
                content=f"Mock content related to '{query}' from document section {i + 1}. This would contain relevant information about your search query.",
                score=0.85 - (i * 0.1),  # Decreasing relevance scores
                metadata={
                    "page": i + 1,
                    "document_type": "contract",
                    "section": f"Section {i + 1}",
                    "mock": True,
                },
            )
            mock_results.append(chunk)

        return mock_results

    def get_service_status(self) -> dict:
        """Get the status of retrieval service components.

        Returns:
            Dictionary with service status information
        """
        return {
            "vector_store_available": self.vector_store is not None,
            "llm_available": self.llm_analyzer is not None,
            "vector_store_url": settings.qdrant_url,
            "llm_model": settings.openai_model,
            "collection_name": settings.qdrant_collection_name,
        }
