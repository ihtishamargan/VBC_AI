"""
Simple LangChain Qdrant Vector Store Integration
Following: https://python.langchain.com/docs/integrations/vectorstores/qdrant/
Clean, simple implementation using LangChain's built-in Qdrant features
"""

import logging
import os
from typing import Any

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

logger = logging.getLogger(__name__)


class QdrantStoreDense:
    """
    Simple Qdrant vector store using LangChain's built-in features.
    Clean wrapper around LangChain's QdrantVectorStore for document search.
    """

    def __init__(
        self,
        collection_name: str = "contracts-index",
        qdrant_url: str | None = None,
        qdrant_api_key: str | None = None,
    ):
        self.collection_name = collection_name

        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small", api_key=os.getenv("OPENAI_API_KEY")
        )

        # Initialize Qdrant client
        if qdrant_url and qdrant_api_key:
            self.client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        else:
            # Use environment variables or local instance
            qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
            qdrant_api_key = os.getenv("QDRANT_API_KEY")
            if qdrant_api_key:
                self.client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
            else:
                self.client = QdrantClient(host="localhost", port=6333)

        # Initialize LangChain vector store (creates collection automatically)
        self.vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embedding=self.embeddings,
        )

        logger.info(f"Vector store initialized: {self.collection_name}")

    def add_documents(self, documents: list[Document]):
        """Add documents to the vector store."""
        try:
            self.vector_store.add_documents(documents)
            logger.info(f"Added {len(documents)} documents to vector store")
        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            raise

    def similarity_search_with_score(
        self, query: str, k: int = 5, filter: dict[str, Any] | None = None
    ) -> list[tuple[Document, float]]:
        """Search for similar documents with scores."""
        try:
            results = self.vector_store.similarity_search_with_score(
                query=query, k=k, filter=filter
            )
            logger.info(f"Search found {len(results)} results for query: '{query}'")
            return results
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    def similarity_search(
        self, query: str, k: int = 5, filter: dict[str, Any] | None = None
    ) -> list[Document]:
        """Search for similar documents."""
        try:
            results = self.vector_store.similarity_search(
                query=query, k=k, filter=filter
            )
            logger.info(f"Search found {len(results)} documents for query: '{query}'")
            return results
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    def as_retriever(self, **kwargs):
        """Get a LangChain retriever interface.

        Returns:
            LangChain retriever
        """
        return self.vector_store.as_retriever(**kwargs)

    def get_collection_info(self) -> dict:
        """Get information about the collection.

        Returns:
            Dictionary with collection information
        """
        try:
            collection_info = self.client.get_collection(self.collection_name)
            return {
                "collection_name": self.collection_name,
                "vectors_count": collection_info.vectors_count,
                "points_count": collection_info.points_count,
                "embedding_model": self.embeddings.model,
            }
        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            return {
                "collection_name": self.collection_name,
                "embedding_model": self.embeddings.model,
                "error": str(e),
            }


# For backward compatibility
QdrantStore = QdrantStoreDense
QdrantHybridStore = QdrantStoreDense  # Remove complex hybrid logic
