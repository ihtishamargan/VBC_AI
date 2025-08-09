# qdrant_store_dense.py
import os, logging
from typing import Iterable, List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from langchain_core.documents import Document

log = logging.getLogger(__name__)

class QdrantStoreDense:
    def __init__(self, collection_name: str, model: str = "text-embedding-3-small", prefer_grpc: bool = True):
        url = os.getenv("QDRANT_URL"); key = os.getenv("QDRANT_API_KEY")
        if not (url and key): raise ValueError("QDRANT_URL and QDRANT_API_KEY must be set")

        self.emb = OpenAIEmbeddings(model=model)
        self.client = QdrantClient(url=url, api_key=key, prefer_grpc=prefer_grpc)

        # Create collection if missing (dense only)
        try:
            self.client.get_collection(collection_name)
        except Exception:
            dim = len(self.emb.embed_query("probe"))
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config={"text": qmodels.VectorParams(size=dim, distance=qmodels.Distance.COSINE)},
                optimizers_config=qmodels.OptimizersConfigDiff(indexing_threshold=20000),
                hnsw_config=qmodels.HnswConfigDiff(m=16, ef_construct=128),
            )

        self.store = QdrantVectorStore(
            client=self.client,
            collection_name=collection_name,
            embedding=self.emb,
            retrieval_mode=RetrievalMode.DENSE,  # dense only
        )

    def add_documents(self, docs: Iterable[Document]) -> List[str]:
        return self.store.add_documents(list(docs))

    def retriever(self, k: int = 8, mmr: bool = True, score_threshold: Optional[float] = None, filter_=None):
        return self.store.as_retriever(
            search_type="mmr" if mmr else "similarity",
            search_kwargs={"k": k, "score_threshold": score_threshold, "filter": filter_},
        )

def chunks_to_documents(chunks: List[Dict[str, Any]]) -> List[Document]:
    docs = []
    for c in chunks:
        meta = c.get("metadata", {}) or {}
        meta.setdefault("document_id", c.get("document_id"))
        meta.setdefault("page_number", c.get("page_number"))
        meta.setdefault("chunk_id", c.get("chunk_id"))
        docs.append(Document(page_content=c.get("content",""), metadata=meta))
    return docs
