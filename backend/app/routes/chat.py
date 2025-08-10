"""Chat and search endpoints."""
import logging
from typing import Optional, Dict, Any
import json

from fastapi import APIRouter, Query, HTTPException
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage
from qdrant_client import QdrantClient

from backend.app.models import ChatRequest, ChatResponse, SearchResponse, Source
from backend.app.config import settings
from backend.app.services.retrieval import DocumentRetrievalService

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(tags=["chat"])

# Initialize LangChain components
llm = ChatOpenAI(
    model="gpt-4o-mini",
    openai_api_key=settings.openai_api_key,
    temperature=0.1
)

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=settings.openai_api_key
)

# Initialize Qdrant client and vector store
try:
    qdrant_client = QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
    )
    
    vector_store = QdrantVectorStore(
        client=qdrant_client,
        collection_name=settings.qdrant_collection_name,
        embedding=embeddings,
        vector_name="text-dense"  # Specify the dense vector name
    )
    logger.info("Qdrant vector store initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Qdrant: {e}")
    vector_store = None

# Global counter for tracking queries
chat_queries_count = 0

# Retrieval service instance (used by /search endpoint)
document_retrieval = DocumentRetrievalService()

# In-memory conversation history (in production, use Redis or database)
conversation_memory: Dict[str, ConversationBufferMemory] = {}

def get_or_create_memory(session_id: str = "default") -> ConversationBufferMemory:
    """Get or create conversation memory for a session."""
    if session_id not in conversation_memory:
        conversation_memory[session_id] = ConversationBufferMemory(
            return_messages=True,
            memory_key="chat_history",
            max_token_limit=4000  # Limit memory to prevent token overflow
        )
    return conversation_memory[session_id]


async def rewrite_query_with_llm(message: str) -> Dict[str, Any]:
    """Use LangChain LLM to rewrite query and extract filters."""
    
    query_rewrite_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a VBC (Value-Based Care) contract analysis assistant. 
Your job is to:
1. Rewrite the user query to be more specific and searchable for VBC contracts
2. Identify if any filters should be applied to the search

Respond in JSON format:
{{"rewritten_query": "improved search query", "filters": {{"field": "value"}} or null}}"""),
        ("user", "{message}")
    ])
    
    try:
        # Modern LangChain pattern: prompt | llm
        chain = query_rewrite_prompt | llm
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


async def generate_llm_response(query: str, context_chunks: list) -> str:
    """Generate LLM response using LangChain with context chunks."""
    
    if not context_chunks:
        return "I don't have any relevant document context to answer your question. Please ensure documents are uploaded and processed first."
    
    context_text = "\n\n".join([
        f"Document {i+1}: {chunk.page_content}" 
        for i, chunk in enumerate(context_chunks)
    ])
    
    response_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a VBC (Value-Based Care) contract analysis expert. 
Answer questions about VBC contracts using the provided context.

Guidelines:
- Give precise, professional answers based on the context
- Reference specific contract terms when relevant
- If context doesn't contain the answer, say so clearly
- Focus on VBC-specific aspects like outcome metrics, payment models, risk sharing"""),
        ("user", """Context from VBC contracts:
{context}

Question: {query}

Please provide a comprehensive answer based on the context provided.""")
    ])
    
    try:
        # Modern LangChain pattern: prompt | llm
        chain = response_prompt | llm
        response = await chain.ainvoke({"context": context_text, "query": query})
        return response.content.strip()
        
    except Exception as e:
        logger.error(f"LLM response generation failed: {e}")
        return "I apologize, but I'm unable to generate a response at this time. Please try again."


@router.post("/chat/document-analysis", response_model=ChatResponse)
async def create_document_analysis_message(
    document_id: str = Query(..., description="Document ID"),
    filename: str = Query(..., description="Document filename"),
    vbc_data: Optional[Dict[str, Any]] = None
):
    """Create a structured analysis message for newly uploaded document."""
    try:
        # Format structured analysis based on VBC contract data
        if vbc_data:
            analysis = f"""📄 **Document Analysis: {filename}**

**🔍 VBC Contract Summary:**
• Agreement: {vbc_data.get('agreement_title', 'N/A')}
• Country: {vbc_data.get('country', 'N/A')}
• Disease Area: {vbc_data.get('disease_area', 'N/A')}  
• Payment Model: {vbc_data.get('payment_model', 'N/A')}
• Patient Population: {vbc_data.get('patient_population_size', 'N/A')}

**🏢 Parties:**
"""
            if vbc_data.get('parties'):
                for party in vbc_data['parties']:
                    analysis += f"• {party.get('name', 'Unknown')} ({party.get('role', 'Unknown role')})\n"
            else:
                analysis += "• No parties identified\n"

            analysis += f"""
**📊 Outcome Metrics:**
"""
            if vbc_data.get('outcome_metrics'):
                for metric in vbc_data['outcome_metrics']:
                    analysis += f"• {metric.get('name', 'Unknown')} ({metric.get('type', 'Unknown type')})\n"
            else:
                analysis += "• No outcome metrics identified\n"

            analysis += f"""
**🎯 Extraction Confidence:** {(vbc_data.get('extraction_confidence', 0) * 100):.0f}%

You can now ask me specific questions about this contract!"""
        
        else:
            # Fallback for documents without VBC analysis
            analysis = f"""📄 **Document Analysis: {filename}**

✅ Document successfully processed and indexed for search.

The document has been parsed and is ready for analysis. You can ask me questions like:
• What are the key terms in this contract?
• Who are the parties involved?
• What are the payment terms?
• What outcome metrics are defined?

I'll search through the document content to provide detailed answers."""

        return ChatResponse(
            answer=analysis,
            sources=[Source(
                document_id=document_id,
                chunk_id="analysis_summary", 
                content=f"Document analysis for {filename}",
                score=1.0,
                metadata={"type": "document_analysis", "filename": filename}
            )]
        )

    except Exception as e:
        logger.error(f"Failed to create document analysis: {e}")
        raise HTTPException(status_code=500, detail="Failed to create document analysis")


@router.post("/chat", response_model=ChatResponse)
async def chat_with_documents(request: ChatRequest):
    """
    Simple 4-step chat flow:
    1. Receive string message  
    2. LLM call to rewrite query (+ detect filters)
    3. Qdrant hybrid search
    4. Top 3 chunks as context + LLM response generation
    """
    global chat_queries_count
    chat_queries_count += 1
    
    logger.info(f"📝 Chat query received: {request.message}")
    
    if not vector_store:
        logger.error("Vector store not available")
        raise HTTPException(status_code=503, detail="Vector store service unavailable")
    
    try:
        # Step 1: Receive string message ✅
        original_message = request.message.strip()
        
        # Step 2: LLM call to rewrite query (+ detect filters)
        logger.info("🔄 Step 2: Rewriting query with LLM...")
        query_info = await rewrite_query_with_llm(original_message)
        rewritten_query = query_info.get("rewritten_query", original_message)
        search_filters = query_info.get("filters")
        
        logger.info(f"   Original: {original_message}")
        logger.info(f"   Rewritten: {rewritten_query}")
        logger.info(f"   Filters: {search_filters}")
        
        # Step 3: Qdrant hybrid search (top 3 chunks)
        logger.info("🔍 Step 3: Performing Qdrant search...")
        search_results = vector_store.similarity_search(
            query=rewritten_query,
            k=3,  # Exactly 3 chunks as requested
            filter=search_filters
        )
        
        logger.info(f"   Found {len(search_results)} chunks")
        for i, doc in enumerate(search_results):
            logger.info(f"   Doc {i}: metadata={doc.metadata}, content_length={len(doc.page_content)}")
        
        # Step 4: Top 3 chunks as context + LLM response generation  
        logger.info("🧠 Step 4: Generating LLM response with context...")
        if search_results:
            answer = await generate_llm_response(original_message, search_results)
            
            # Prepare sources from search results
            sources = []
            for i, doc in enumerate(search_results):
                doc_metadata = doc.metadata or {}
                
                source = Source(
                    document_id=doc_metadata.get("document_id", f"doc_{i}"),
                    chunk_id=doc_metadata.get("chunk_id", f"chunk_{i}"),
                    content=doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                    score=doc_metadata.get("score", 0.8),  # Use actual score if available
                    metadata=doc_metadata if doc_metadata else {}
                )
                sources.append(source)
                
            logger.info(f"   Generated response: {len(answer)} chars, {len(sources)} sources")
        else:
            answer = "I don't have relevant VBC contract information to answer your question. Please ensure documents are uploaded and processed."
            sources = []
            logger.info("   No relevant documents found")
        
        return ChatResponse(
            answer=answer,
            sources=sources
        )
        
    except Exception as e:
        logger.error(f"❌ Chat processing failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Chat processing failed: {str(e)}")


@router.get("/search", response_model=SearchResponse)
async def search_documents(
    q: str = Query(..., description="Search query"), 
    filters: Optional[str] = Query(None, description="JSON filters")
):
    """Search through document chunks with optional filters."""
    logger.info(f"Search query: {q}")
    
    try:
        # Parse filters if provided
        parsed_filters = None
        if filters:
            import json
            try:
                parsed_filters = json.loads(filters)
            except json.JSONDecodeError:
                logger.warning(f"Invalid JSON filters: {filters}")
        
        # Use the retrieval service for search
        search_chunks = await document_retrieval.search_documents(
            query=q,
            filters=parsed_filters,
            limit=20
        )
        
        return SearchResponse(
            query=q,
            chunks=search_chunks,
            total_results=len(search_chunks)
        )
        
    except Exception as e:
        logger.error(f"Search failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


def get_chat_metrics():
    """Get chat-related metrics."""
    return {
        "total_queries": chat_queries_count
    }
