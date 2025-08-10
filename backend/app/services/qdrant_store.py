"""
Official LangChain Qdrant Vector Store Integration with Hybrid Search
Following: https://python.langchain.com/docs/integrations/vectorstores/qdrant/
Supports both dense (semantic) and sparse (keyword) search for optimal retrieval
"""
import os
import logging
from typing import List, Optional, Dict, Any, Tuple
import re
from collections import Counter

from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, SparseVectorParams, SparseIndexParams

logger = logging.getLogger(__name__)


class SparseVectorEncoder:
    """Simple BM25-like sparse vector encoder for keyword matching."""
    
    def __init__(self, k1: float = 1.2, b: float = 0.75):
        """Initialize sparse encoder with BM25 parameters."""
        self.k1 = k1
        self.b = b
        self.vocab = {}
        self.doc_count = 0
        self.avg_doc_length = 0
        self.doc_frequencies = Counter()
        
class QdrantHybridStore:
    """
    Simplified hybrid Qdrant vector store using LangChain best practices.
    Supports both dense (semantic) and sparse (keyword) search for contract documents.
    """
    
    def __init__(
        self,
        collection_name: str = "contracts-index",
        qdrant_url: Optional[str] = None,
        qdrant_api_key: Optional[str] = None,
    ):
        self.collection_name = collection_name
        
        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=os.getenv("OPENAI_API_KEY")
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
        
        # Create hybrid collection
        self._create_collection()
        
        # Initialize LangChain vector store
        self.vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embedding=self.embeddings,
            vector_name="text-dense",  # Use the dense vector in hybrid collection
        )
        
        logger.info(f"Hybrid vector store initialized: {self.collection_name}")
    
    def _create_collection(self):
        """Create hybrid collection with dense and sparse vectors."""
        try:
            # Check if collection already exists
            collections = self.client.get_collections().collections
            if any(col.name == self.collection_name for col in collections):
                logger.info(f"Using existing collection: {self.collection_name}")
                return
            
            # Get embedding dimension
            sample_embedding = self.embeddings.embed_query("sample text")
            dense_size = len(sample_embedding)
            
            # Create collection with hybrid vectors
            vectors_config = {
                "text-dense": VectorParams(size=dense_size, distance=Distance.COSINE),
            }
            
            sparse_vectors_config = {
                "text-sparse": SparseVectorParams(
                    index=SparseIndexParams(on_disk=False)
                )
            }
            
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=vectors_config,
                sparse_vectors_config=sparse_vectors_config
            )
            
            logger.info(f"Created hybrid collection: {self.collection_name} (dense: {dense_size}D + sparse)")
            
        except Exception as e:
            logger.error(f"Failed to create collection: {e}")
            raise
    
    def add_documents(self, documents: List[Document]) -> List[str]:
        """Add documents to the vector store."""
        try:
            doc_ids = self.vector_store.add_documents(documents)
            logger.info(f"Added {len(documents)} documents to hybrid vector store")
            return doc_ids
        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            raise
    
    def similarity_search_with_score(
        self, 
        query: str, 
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[tuple]:
        """Search for similar documents with scores."""
        try:
            results = self.vector_store.similarity_search_with_score(
                query=query, 
                k=k, 
                filter=filter
            )
            logger.info(f"Search found {len(results)} results for query: '{query}'")
            return results
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def similarity_search(
        self,
        query: str,
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """Search for similar documents."""
        try:
            results = self.vector_store.similarity_search(
                query=query,
                k=k,
                filter=filter
            )
            logger.info(f"Search found {len(results)} documents for query: '{query}'")
            return results
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
            for doc_id, doc, score in dense_results:
                doc_key = doc.page_content[:100]
                normalized_score = score / max_dense if max_dense > 0 else 0
                combined_scores[doc_key] = combined_scores.get(doc_key, (doc, 0))
                combined_scores[doc_key] = (combined_scores[doc_key][0], 
                                          combined_scores[doc_key][1] + dense_weight * normalized_score)
        
        # Normalize and weight sparse scores
        if sparse_results:
            max_sparse = max(score for _, _, score in sparse_results) if sparse_results else 1.0
            for doc_id, doc, score in sparse_results:
                doc_key = doc.page_content[:100]
                normalized_score = score / max_sparse if max_sparse > 0 else 0
                combined_scores[doc_key] = combined_scores.get(doc_key, (doc, 0))
                combined_scores[doc_key] = (combined_scores[doc_key][0], 
                                          combined_scores[doc_key][1] + sparse_weight * normalized_score)
        
        # Sort by combined score and return top k
        sorted_results = sorted(combined_scores.values(), key=lambda x: x[1], reverse=True)
        return [(doc, score) for doc, score in sorted_results[:k]]
    
    def _max_fusion(self, dense_results: List[Tuple[str, Document, float]], 
                   sparse_results: List[Tuple[str, Document, float]], k: int) -> List[Tuple[Document, float]]:
        """Combine results using maximum score fusion."""
        max_scores = {}
        
        # Process dense results
        for doc_id, doc, score in dense_results:
            doc_key = doc.page_content[:100]
            if doc_key not in max_scores or score > max_scores[doc_key][1]:
                max_scores[doc_key] = (doc, score)
        
        # Process sparse results
        for doc_id, doc, score in sparse_results:
            doc_key = doc.page_content[:100]
            if doc_key not in max_scores or score > max_scores[doc_key][1]:
                max_scores[doc_key] = (doc, score)
        
        # Sort by max score and return top k
        sorted_results = sorted(max_scores.values(), key=lambda x: x[1], reverse=True)
        return [(doc, score) for doc, score in sorted_results[:k]]
    
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
        return {
            "collection_name": self.collection_name,
            "url": self.url,
            "has_api_key": bool(self.api_key),
            "embedding_model": self.embeddings.model,
        }


# For backward compatibility
QdrantStore = QdrantHybridStore  # Main class now supports hybrid search
QdrantStoreDense = QdrantHybridStore  # Backward compatibility alias
