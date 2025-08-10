"""Chat service for handling LLM interactions and conversation management."""

import json
from typing import Any

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from qdrant_client import QdrantClient

from backend.app.config import settings
from backend.app.models import Source
from backend.app.prompts import RETRIEVAL_SYSTEM_PROMPT, RETRIEVAL_USER_PROMPT_TEMPLATE
from backend.app.utils.logger import get_module_logger

# Configure logging
logger = get_module_logger(__name__)

from backend.app.prompts import QUERY_REWRITE_PROMPT, RESPONSE_GENERATION_PROMPT


class ChatService:
    """Service class for handling chat functionality and component initialization."""

    def __init__(self):
        """Initialize the chat service with all required components."""
        self.chat_queries_count = 0
        self.conversation_memory: dict[str, ConversationBufferMemory] = {}

        # Initialize LangChain components
        self._initialize_llm()
        self._initialize_embeddings()
        self._initialize_vector_store()

        # Initialize retrieval service
        self.document_retrieval = DocumentRetrievalService()

        logger.info("ChatService initialized successfully")

    def _initialize_llm(self) -> None:
        """Initialize the LangChain LLM."""
        self.llm = ChatOpenAI(
            model="gpt-4o-mini", openai_api_key=settings.openai_api_key, temperature=0.1
        )
        logger.info("LLM initialized")

    def _initialize_embeddings(self) -> None:
        """Initialize the OpenAI embeddings."""
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small", openai_api_key=settings.openai_api_key
        )
        logger.info("Embeddings initialized")

    def _initialize_vector_store(self) -> None:
        """Initialize Qdrant client and vector store."""
        try:
            self.qdrant_client = QdrantClient(
                url=settings.qdrant_url,
                api_key=settings.qdrant_api_key,
            )

            self.vector_store = QdrantVectorStore(
                client=self.qdrant_client,
                collection_name=settings.qdrant_collection_name,
                embedding=self.embeddings,
                vector_name="text-dense",  # Specify the dense vector name
            )
            logger.info("Qdrant vector store initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Qdrant: {e}")
            self.vector_store = None

    def get_or_create_memory(
        self, session_id: str = "default"
    ) -> ConversationBufferMemory:
        """Get or create conversation memory for a session."""
        if session_id not in self.conversation_memory:
            self.conversation_memory[session_id] = ConversationBufferMemory(
                return_messages=True,
                memory_key="chat_history",
                max_token_limit=40000,  # Limit memory to prevent token overflow
            )
        return self.conversation_memory[session_id]

    async def rewrite_query_with_llm(self, message: str) -> dict[str, Any]:
        """Use LangChain LLM to rewrite query and extract filters."""
        try:
            # Modern LangChain pattern: prompt | llm
            chain = QUERY_REWRITE_PROMPT | self.llm
            response = await chain.ainvoke({"message": message})

            # Try to parse JSON response
            try:
                result = json.loads(response.content.strip())
                return result
            except json.JSONDecodeError:
                # Fallback if JSON parsing fails
                logger.warning(f"Failed to parse LLM JSON response: {response.content}")
                return {"rewritten_query": message, "filters": None}

        except Exception as e:
            logger.error(f"Query rewrite failed: {e}")
            return {"rewritten_query": message, "filters": None}

    async def generate_llm_response(
        self, query: str, context_chunks: list
    ) -> tuple[str, list[int]]:
        """Generate LLM response using LangChain with context chunks.

        Returns:
            tuple[str, list[int]]: (response_text, list_of_used_source_indices)
        """
        if not context_chunks:
            return (
                "I don't have any relevant document context to answer your question. Please ensure documents are uploaded and processed first.",
                [],
            )

        context_text = "\n\n".join(
            [
                f"Document {i + 1}: {chunk.page_content}"
                for i, chunk in enumerate(context_chunks)
            ]
        )

        try:
            chain = RESPONSE_GENERATION_PROMPT | self.llm
            response = await chain.ainvoke({"context": context_text, "query": query})
            response_text = response.content.strip()

            # Extract which sources were used
            used_sources = []
            if "Sources used:" in response_text:
                # Split response and source citation
                parts = response_text.rsplit("Sources used:", 1)
                if len(parts) == 2:
                    main_response = parts[0].strip()
                    sources_part = parts[1].strip().lower()

                    # Parse source numbers (1-indexed in response, convert to 0-indexed)
                    if sources_part != "none":
                        import re

                        numbers = re.findall(r"\d+", sources_part)
                        used_sources = [
                            int(n) - 1
                            for n in numbers
                            if 1 <= int(n) <= len(context_chunks)
                        ]

                    # Return main response without source citation
                    return main_response, used_sources

            # Fallback: assume all sources were used if parsing fails
            logger.warning("Could not parse source usage from LLM response")
            return response_text, list(range(len(context_chunks)))

        except Exception as e:
            logger.error(f"LLM response generation failed: {e}")
            return (
                "I apologize, but I'm unable to generate a response at this time. Please try again.",
                [],
            )

    def increment_query_count(self) -> None:
        """Increment the chat queries counter."""
        self.chat_queries_count += 1

    def get_chat_metrics(self) -> dict[str, Any]:
        """Get chat-related metrics."""
        return {
            "total_queries": self.chat_queries_count,
            "active_sessions": len(self.conversation_memory),
            "vector_store_available": self.vector_store is not None,
        }

    async def chat_with_documents(
        self,
        message: str,
        filters: dict[str, Any] | None = None,
        context_limit: int = 5,
    ) -> dict:
        """Generate AI chat response based on document context using retrieval.

        Args:
            message: User's chat message
            filters: Optional metadata filters for context retrieval
            context_limit: Maximum number of context chunks to use

        Returns:
            Dictionary with answer and sources
        """
        try:
            logger.info(f"Processing chat message with document context: '{message}'")

            # Step 1: Retrieve relevant context using document retrieval service
            relevant_chunks = await self.document_retrieval.search_documents(
                query=message, filters=filters, limit=context_limit
            )

            # Step 2: Prepare context for LLM
            if relevant_chunks:
                context_texts = []
                sources = []

                for chunk in relevant_chunks:
                    context_texts.append(
                        f"Source: {chunk.document_id}\nContent: {chunk.content}"
                    )

                    # Convert to Source object
                    source = Source(
                        document_id=chunk.document_id,
                        chunk_id=chunk.chunk_id,
                        content=chunk.content[:200] + "..."
                        if len(chunk.content) > 200
                        else chunk.content,
                        score=chunk.score,
                        metadata=chunk.metadata,
                    )
                    sources.append(source)

                context = "\n\n".join(context_texts)
            else:
                context = "No relevant documents found."
                sources = []

            # Step 3: Generate AI response using enhanced retrieval prompts
            if context.strip() and context != "No relevant documents found.":
                answer = await self._generate_document_based_response(message, context)
            else:
                answer = "I don't have relevant document context to answer your question. Please ensure documents are uploaded and processed first."

            # Increment query count
            self.increment_query_count()

            logger.info(
                f"Generated document-based chat response with {len(sources)} sources"
            )

            return {
                "answer": answer,
                "sources": [source.model_dump() for source in sources],
            }

        except Exception as e:
            logger.error(f"Document-based chat processing failed: {str(e)}")
            return {
                "answer": f"I apologize, but I encountered an error processing your question: {str(e)}",
                "sources": [],
            }

    async def _generate_document_based_response(
        self, message: str, context: str
    ) -> str:
        """Generate AI response using document context and specialized prompts."""
        try:
            # Use OpenAI for document-based chat responses with specialized prompts
            from openai import OpenAI

            client = OpenAI(api_key=settings.openai_api_key)

            # Use centralized prompts for document retrieval
            system_prompt = RETRIEVAL_SYSTEM_PROMPT
            user_prompt = RETRIEVAL_USER_PROMPT_TEMPLATE.format(
                context=context, message=message
            )

            # Generate response using OpenAI
            response = client.chat.completions.create(
                model="gpt-4o-mini",  # Use the same model as document analysis
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,  # Low temperature for consistent, factual responses
                max_tokens=800,
            )

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"Document-based AI response generation failed: {str(e)}")
            # Simple fallback
            return (
                f"I'm having trouble generating a detailed response about '{message}' right now. "
                f"However, I found relevant document context that may contain the information you need. "
                f"Please try rephrasing your question or check the document content directly."
            )


# Global chat service instance (lazy initialization)
_chat_service_instance: ChatService | None = None


def get_chat_service() -> ChatService:
    """Get the global chat service instance (singleton pattern with lazy initialization)."""
    global _chat_service_instance
    if _chat_service_instance is None:
        _chat_service_instance = ChatService()
    return _chat_service_instance


def reset_chat_service() -> None:
    """Reset the chat service instance (useful for testing)."""
    global _chat_service_instance
    _chat_service_instance = None


# For backward compatibility (will be removed in future)
chat_service = get_chat_service()
