"""Vector storage service for document chunks."""

from typing import Any

from backend.app.models.ingestion import DocumentChunk, IngestionConfig
from backend.app.services.qdrant_store import QdrantStoreDense as QdrantStore
from backend.app.utils.logger import get_module_logger

logger = get_module_logger(__name__)


class VectorStorageService:
    """Service for storing document chunks in vector database."""

    def __init__(self, config: IngestionConfig):
        """Initialize vector storage service."""
        self.config = config
        self._vector_store: QdrantStore | None = None
        self._initialize_vector_store()

    def _initialize_vector_store(self) -> None:
        """Initialize the vector store connection."""
        try:
            self._vector_store = QdrantStore(
                collection_name=self.config.collection_name
            )
            logger.info(
                f"Vector store initialized successfully with collection: {self.config.collection_name}"
            )
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {e}")
            self._vector_store = None

    async def store_chunks(self, chunks: list[DocumentChunk]) -> list[str]:
        """Store document chunks in vector database.

        Args:
            chunks: List of document chunks to store

        Returns:
            List of vector IDs created in the database
        """
        if not self._vector_store:
            logger.error("Vector store not available for chunk storage")
            return []

        if not chunks:
            logger.warning("No chunks provided for vector storage")
            return []

        try:
            logger.info(f"Storing {len(chunks)} chunks in vector database")

            # Prepare texts and metadata for vector storage
            texts = [chunk.content for chunk in chunks]
            metadatas = [chunk.metadata for chunk in chunks]

            # Store in vector database
            vector_ids = await self._vector_store.add_texts(
                texts=texts, metadatas=metadatas
            )

            if vector_ids:
                logger.info(
                    f"Successfully stored {len(vector_ids)} vectors in database"
                )
                return vector_ids
            logger.warning("Vector storage returned no IDs")
            return []

        except Exception as e:
            logger.error(f"Failed to store chunks in vector database: {e}")
            return []

    async def verify_storage(self, vector_ids: list[str]) -> bool:
        """Verify that vectors were stored successfully.

        Args:
            vector_ids: List of vector IDs to verify

        Returns:
            True if all vectors are accessible, False otherwise
        """
        if not self._vector_store or not vector_ids:
            return False

        try:
            # Try to retrieve a sample of the stored vectors
            sample_size = min(3, len(vector_ids))
            sample_ids = vector_ids[:sample_size]

            for vector_id in sample_ids:
                # Attempt to retrieve vector by ID
                result = await self._vector_store.get_by_id(vector_id)
                if not result:
                    logger.warning(f"Vector {vector_id} not found during verification")
                    return False

            logger.info(f"Verified {sample_size} vectors successfully")
            return True

        except Exception as e:
            logger.error(f"Vector verification failed: {e}")
            return False

    async def get_storage_stats(self) -> dict[str, Any]:
        """Get statistics about vector storage.

        Returns:
            Dictionary containing storage statistics
        """
        if not self._vector_store:
            return {"available": False, "error": "Vector store not initialized"}

        try:
            # Get collection info
            collection_info = await self._vector_store.get_collection_info()

            stats = {
                "available": True,
                "collection_name": self.config.collection_name,
                "total_vectors": collection_info.get("vectors_count", 0),
                "collection_status": collection_info.get("status", "unknown"),
            }

            logger.info(f"Vector storage stats: {stats}")
            return stats

        except Exception as e:
            logger.error(f"Failed to get storage stats: {e}")
            return {"available": False, "error": str(e)}

    async def delete_vectors(self, vector_ids: list[str]) -> bool:
        """Delete vectors from the database.

        Args:
            vector_ids: List of vector IDs to delete

        Returns:
            True if deletion was successful, False otherwise
        """
        if not self._vector_store or not vector_ids:
            return False

        try:
            logger.info(f"Deleting {len(vector_ids)} vectors from database")

            success = await self._vector_store.delete_vectors(vector_ids)

            if success:
                logger.info(f"Successfully deleted {len(vector_ids)} vectors")
            else:
                logger.warning("Vector deletion failed")

            return success

        except Exception as e:
            logger.error(f"Failed to delete vectors: {e}")
            return False

    async def search_similar(
        self,
        query_text: str,
        limit: int = 10,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Search for similar vectors in the database.

        Args:
            query_text: Text to search for
            limit: Maximum number of results
            filter_metadata: Optional metadata filters

        Returns:
            List of similar documents with scores
        """
        if not self._vector_store:
            logger.error("Vector store not available for search")
            return []

        try:
            logger.info(f"Searching for similar vectors: '{query_text[:50]}...'")

            results = await self._vector_store.similarity_search_with_score(
                query=query_text, k=limit, filter=filter_metadata
            )

            logger.info(f"Found {len(results)} similar vectors")
            return results

        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []

    @property
    def is_available(self) -> bool:
        """Check if vector store is available."""
        return self._vector_store is not None

    def update_config(self, new_config: IngestionConfig) -> None:
        """Update vector storage configuration.

        Args:
            new_config: New configuration to apply
        """
        if new_config.collection_name != self.config.collection_name:
            logger.info(
                f"Updating vector store collection: {self.config.collection_name} -> {new_config.collection_name}"
            )
            self.config = new_config
            self._initialize_vector_store()
