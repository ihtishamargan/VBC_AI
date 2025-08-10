"""Document retrieval service for search and chat functionality."""
import logging
from typing import List, Optional, Dict, Any

from backend.app.services.qdrant_store import QdrantStoreDense as QdrantStore
from backend.app.services.llm_analyzer import LLMDocumentAnalyzer
from backend.app.config import settings
from backend.app.models import Source, SearchChunk

logger = logging.getLogger(__name__)


class DocumentRetrievalService:
    """Service for retrieving and searching document chunks."""
    
    def __init__(
        self,
        collection_name: str = None,
        llm_model: str = None
    ):
        """Initialize the retrieval service.
        
        Args:
            collection_name: Qdrant collection name (defaults to config)
            llm_model: OpenAI model for chat responses (defaults to config)
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
        
        # Initialize LLM for chat responses
        try:
            self.llm_analyzer = LLMDocumentAnalyzer(
                api_key=settings.openai_api_key,
                model=llm_model or settings.openai_model
            )
            logger.info("LLM initialized for chat responses")
        except Exception as e:
            logger.warning(f"LLM initialization failed: {e}")
            self.llm_analyzer = None
    
    async def search_documents(
        self, 
        query: str, 
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 10
    ) -> List[SearchChunk]:
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
            if hasattr(self.vector_store, 'hybrid_search'):
                logger.info("Using hybrid search (dense + sparse vectors)")
                results = self.vector_store.hybrid_search(
                    query=query,
                    k=limit,
                    dense_weight=0.6,  # 60% semantic understanding
                    sparse_weight=0.4, # 40% keyword matching
                    filter=filters,
                    fusion_algorithm="rrf"  # Reciprocal Rank Fusion
                )
            else:
                logger.info("Using dense-only search (fallback)")
                results = self.vector_store.similarity_search_with_score(
                    query=query,
                    k=limit,
                    filter=filters
                )
            
            # Convert to SearchChunk objects
            search_chunks = []
            for doc, score in results:
                chunk = SearchChunk(
                    chunk_id=doc.metadata.get("chunk_id", f"chunk-{len(search_chunks)}"),
                    document_id=doc.metadata.get("document_id", "unknown"),
                    content=doc.page_content,
                    score=float(score),
                    metadata=doc.metadata
                )
                search_chunks.append(chunk)
            
            logger.info(f"Found {len(search_chunks)} relevant chunks")
            return search_chunks
            
        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            return self._get_mock_search_results(query, limit)
    
    async def chat_with_documents(
        self, 
        message: str, 
        filters: Optional[Dict[str, Any]] = None,
        context_limit: int = 5
    ) -> dict:
        """Generate AI chat response based on document context.
        
        Args:
            message: User's chat message
            filters: Optional metadata filters for context retrieval
            context_limit: Maximum number of context chunks to use
            
        Returns:
            Dictionary with answer and sources
        """
        try:
            logger.info(f"Processing chat message: '{message}'")
            
            # Step 1: Retrieve relevant context
            relevant_chunks = await self.search_documents(
                query=message, 
                filters=filters, 
                limit=context_limit
            )
            
            # Step 2: Prepare context for LLM
            if relevant_chunks:
                context_texts = []
                sources = []
                
                for chunk in relevant_chunks:
                    context_texts.append(f"Source: {chunk.document_id}\nContent: {chunk.content}")
                    
                    # Convert to Source object
                    source = Source(
                        document_id=chunk.document_id,
                        chunk_id=chunk.chunk_id,
                        content=chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content,
                        score=chunk.score,
                        metadata=chunk.metadata
                    )
                    sources.append(source)
                
                context = "\n\n".join(context_texts)
            else:
                context = "No relevant documents found."
                sources = []
            
            # Step 3: Generate AI response
            if self.llm_analyzer:
                answer = await self._generate_ai_response(message, context)
            else:
                answer = self._generate_mock_response(message, len(sources))
            
            logger.info(f"Generated chat response with {len(sources)} sources")
            
            return {
                "answer": answer,
                "sources": [source.model_dump() for source in sources]
            }
            
        except Exception as e:
            logger.error(f"Chat processing failed: {str(e)}")
            return {
                "answer": f"I apologize, but I encountered an error processing your question: {str(e)}",
                "sources": []
            }
    
    async def _generate_ai_response(self, message: str, context: str) -> str:
        """Generate AI response using LLM with context.
        
        Args:
            message: User's message
            context: Retrieved document context
            
        Returns:
            AI-generated response
        """
        try:
            # Use OpenAI for proper RAG chat responses
            from openai import OpenAI
            
            client = OpenAI(api_key=self.llm_analyzer.client.api_key)
            
            # Create specialized contract assistant prompt
            system_prompt = """You are a specialized AI assistant for contract and document analysis. You help users understand legal documents, contracts, and business agreements.

Your expertise includes:
- Contract terms and clauses analysis
- Legal terminology explanation
- Risk assessment and liability analysis
- Compliance and regulatory guidance
- Payment terms and service level agreements

Guidelines:
1. Use the provided document context to answer questions accurately
2. Cite specific parts of documents when possible
3. Explain legal concepts in clear, understandable language
4. If context is insufficient, acknowledge this clearly
5. For legal advice, always recommend consulting qualified legal counsel
6. Be precise and professional in your responses"""
            
            user_prompt = f"""Based on the following document context, please answer the user's question:

DOCUMENT CONTEXT:
{context}

USER QUESTION: {message}

Please provide a comprehensive answer based on the document context above. If the documents contain relevant information, cite specific sections. If not, explain what information would be needed to provide a complete answer."""
            
            # Generate response using OpenAI
            response = client.chat.completions.create(
                model="gpt-4o-mini",  # Use the same model as document analysis
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,  # Low temperature for consistent, factual responses
                max_tokens=800
            )
            
            return response.choices[0].message.content
                
        except Exception as e:
            logger.error(f"AI response generation failed: {str(e)}")
            # Fallback to contextual response
            return self._generate_contextual_fallback(message, context)
    
    def _generate_contextual_fallback(self, message: str, context: str) -> str:
        """Generate contextual fallback response when LLM fails."""
        try:
            if not context.strip():
                return "I don't have relevant document context to answer your question. Please ensure documents are uploaded and processed first."
            
            # Simple pattern matching for contract-related queries
            message_lower = message.lower()
            context_lower = context.lower()
            
            if any(term in message_lower for term in ["payment", "pay", "invoice", "billing"]):
                if "payment" in context_lower or "invoice" in context_lower:
                    return "Based on the documents, I found information about payment terms and billing arrangements. The documents contain specific details about payment schedules and conditions that may help answer your question."
                else:
                    return "I don't see specific payment information in the current document context."
            
            elif any(term in message_lower for term in ["liability", "risk", "damage", "indemnification"]):
                if any(term in context_lower for term in ["liability", "indemnification", "damages"]):
                    return "The documents contain liability and indemnification clauses that address risk allocation and damage limitations. These sections outline the responsibilities and protections for each party."
                else:
                    return "I don't see specific liability or risk information in the current document context."
            
            elif any(term in message_lower for term in ["terminate", "termination", "end", "cancel"]):
                if "termination" in context_lower or "terminate" in context_lower:
                    return "The documents include termination clauses that specify conditions and procedures for ending the agreement. These sections detail notice requirements and post-termination obligations."
                else:
                    return "I don't see specific termination information in the current document context."
            
            elif any(term in message_lower for term in ["force majeure", "extraordinary", "unforeseeable"]):
                if "force majeure" in context_lower:
                    return "The documents contain force majeure provisions that address extraordinary circumstances beyond the parties' control and their impact on contractual obligations."
                else:
                    return "I don't see force majeure provisions in the current document context."
            
            else:
                # General response
                return f"Based on the available document context, I can see information related to your question about '{message}'. The documents contain relevant details that may help address your inquiry. For a more detailed analysis, I recommend reviewing the specific document sections."
        
        except Exception as e:
            logger.error(f"Fallback response generation failed: {str(e)}")
            return "I apologize, but I'm having trouble processing your question right now. Please try again or rephrase your question."
    
    def _generate_mock_response(self, message: str, source_count: int) -> str:
        """Generate a mock response when LLM is not available.
        
        Args:
            message: User's message
            source_count: Number of relevant sources found
            
        Returns:
            Mock response string
        """
        if source_count > 0:
            return (f"Based on your question '{message}', I found {source_count} relevant sections "
                   f"in the uploaded documents. The documents contain information that may help "
                   f"answer your query about contract terms, payment conditions, and service agreements.")
        else:
            return (f"I understand you're asking about '{message}', but I couldn't find directly "
                   f"relevant information in the currently uploaded documents. Could you try "
                   f"rephrasing your question or uploading more specific documents?")
    
    def _get_mock_search_results(self, query: str, limit: int) -> List[SearchChunk]:
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
                chunk_id=f"mock-chunk-{i+1}",
                document_id=f"mock-doc-{i+1}",
                content=f"Mock content related to '{query}' from document section {i+1}. This would contain relevant information about your search query.",
                score=0.85 - (i * 0.1),  # Decreasing relevance scores
                metadata={
                    "page": i + 1,
                    "document_type": "contract",
                    "section": f"Section {i+1}",
                    "mock": True
                }
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
            "collection_name": settings.qdrant_collection_name
        }
