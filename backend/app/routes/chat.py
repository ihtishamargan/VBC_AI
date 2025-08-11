"""Chat and search endpoints."""

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from backend.app.auth import AuthUser, require_auth
from backend.app.models import ChatRequest, ChatResponse, Source
from backend.app.prompts import (
    get_fallback_analysis_template,
    get_vbc_analysis_template,
)
from backend.app.services import chat_service
from backend.app.utils.logger import get_module_logger

# Configure logging
logger = get_module_logger(__name__)

# Create router
router = APIRouter(tags=["chat"])


class DocumentAnalysisRequest(BaseModel):
    document_id: str
    filename: str
    vbc_data: dict[str, Any] | None = None


@router.post("/chat/document-analysis", response_model=ChatResponse)
async def create_document_analysis_message(
    request: DocumentAnalysisRequest, _current_user: AuthUser = Depends(require_auth)
):
    """Create a structured analysis message for newly uploaded document."""
    try:
        # Format structured analysis based on VBC contract data
        if request.vbc_data:
            analysis = get_vbc_analysis_template(request.filename, request.vbc_data)
        else:
            # Fallback for documents without VBC analysis
            analysis = get_fallback_analysis_template(request.filename)

        return ChatResponse(
            answer=analysis,
            sources=[
                Source(
                    document_id=request.document_id,
                    chunk_id="analysis_summary",
                    content=f"Document analysis for {request.filename}",
                    score=1.0,
                    metadata={
                        "type": "document_analysis",
                        "filename": request.filename,
                    },
                )
            ],
        )

    except Exception as e:
        logger.error(f"Failed to create document analysis: {e}")
        raise HTTPException(
            status_code=500, detail="Failed to create document analysis"
        )


@router.post("/chat", response_model=ChatResponse)
async def chat_with_documents(
    request: ChatRequest, _current_user: AuthUser = Depends(require_auth)
):
    """
    Simple 4-step chat flow:
    1. Receive string message
    2. LLM call to rewrite query (+ detect filters)
    3. Qdrant hybrid search
    4. Top 3 chunks as context + LLM response generation
    """
    chat_service.increment_query_count()

    logger.info(f"📝 Chat query received: {request.message}")

    if not hasattr(chat_service, "vector_store") or not chat_service.vector_store:
        logger.error("Vector store not available")
        raise HTTPException(status_code=503, detail="Vector store service unavailable")

    try:
        # Step 1: Receive message ✅
        original_message = request.message.strip()

        # Step 2: LLM call to rewrite query (+ detect filters)
        logger.info("🔄 Step 2: Rewriting query with LLM...")
        query_info = await chat_service.rewrite_query_with_llm(original_message)
        rewritten_query = query_info.get("rewritten_query", original_message)
        search_filters = query_info.get("filters")

        logger.info(f"   Original: {original_message}")
        logger.info(f"   Rewritten: {rewritten_query}")
        logger.info(f"   Filters: {search_filters}")

        # Step 3: Qdrant hybrid search (top 3 chunks)
        logger.info("🔍 Step 3: Performing Qdrant search...")
        search_results = chat_service.vector_store.similarity_search(
            query=rewritten_query,
            k=3,  # Exactly 3 chunks as requested
            filter=search_filters,
        )

        logger.info(f"   Found {len(search_results)} chunks")
        for i, doc in enumerate(search_results):
            logger.info(
                f"   Doc {i}: metadata={doc.metadata}, content_length={len(doc.page_content)}"
            )

        # Step 4: Top 3 chunks as context + LLM response generation
        logger.info("🧠 Step 4: Generating LLM response with context...")
        if search_results:
            answer, used_source_indices = await chat_service.generate_llm_response(
                original_message, search_results
            )

            # Only include sources that were actually used in the response
            sources = []
            for i in used_source_indices:
                if 0 <= i < len(search_results):
                    doc = search_results[i]
                    doc_metadata = doc.metadata or {}

                    source = Source(
                        document_id=doc_metadata.get("document_id", f"doc_{i}"),
                        chunk_id=doc_metadata.get("chunk_id", f"chunk_{i}"),
                        content=doc.page_content[:200] + "..."
                        if len(doc.page_content) > 200
                        else doc.page_content,
                        score=doc_metadata.get(
                            "score", 0.8
                        ),  # Use actual score if available
                        metadata=doc_metadata if doc_metadata else {},
                    )
                    sources.append(source)

            logger.info(
                f"   Generated response: {len(answer)} chars, {len(sources)}/{len(search_results)} sources used"
            )
        else:
            answer = "I don't have relevant VBC contract information to answer your question. Please ensure documents are uploaded and processed."
            sources = []
            logger.info("   No relevant documents found")

        # Store conversation in memory
        session_id = "default"  # Could be extracted from request in future
        memory = chat_service.get_or_create_memory(session_id)
        memory.chat_memory.add_user_message(original_message)
        memory.chat_memory.add_ai_message(answer)

        return ChatResponse(answer=answer, sources=sources)

    except Exception as e:
        logger.error(f"❌ Chat processing failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Chat processing failed: {str(e)}")
