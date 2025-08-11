"""
LLM-based Document Analysis using OpenAI o4-mini
- Uses LangChain structured outputs instead of brittle manual JSON parsing.
"""

import contextlib
import logging
import os
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from backend.app.config import settings
from backend.app.prompts import LLM_DOCUMENT_ANALYSIS_PROMPT

logger = logging.getLogger(__name__)


class ExtractedEntity(BaseModel):
    text: str = Field(description="The extracted entity text")
    type: str = Field(
        description="Entity type (PERSON, ORGANIZATION, DATE, MONEY, etc.)"
    )
    confidence: float = Field(description="Confidence score between 0-1")
    context: str = Field(description="Surrounding context where entity was found")
    start_pos: int | None = Field(None, description="Start position in text")
    end_pos: int | None = Field(None, description="End position in text")


class DocumentAnalysis(BaseModel):
    document_type: str = Field(
        description="Type of document (contract, research_paper, report, etc.)"
    )
    summary: str = Field(description="Comprehensive summary of the document")
    key_topics: list[str] = Field(description="Main topics and themes in the document")
    entities: list[ExtractedEntity] = Field(
        description="Extracted entities with confidence scores"
    )
    confidence_score: float = Field(
        description="Overall confidence in the analysis (0-1)"
    )
    language: str = Field(description="Primary language of the document")
    sentiment: str = Field(
        description="Overall sentiment (positive, negative, neutral)"
    )
    key_insights: list[str] = Field(description="Important insights or takeaways")

    class Config:
        extra = "forbid"  # This ensures additionalProperties: false in JSON schema


class LLMDocumentAnalyzer:
    """
    Lightweight analyzer using OpenAI o4-mini with structured outputs.
    """

    def __init__(
        self,
        model: str = settings.openai_model,
        api_key: str | None = None,
    ):
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required")

        # Store API key for retrieval service access
        self.api_key = api_key

        # Use o4-mini for structured outputs
        self.llm = ChatOpenAI(
            model=model,
            api_key=api_key,
        )

        # Create OpenAI client for direct API access (used by retrieval service)
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key)

        # Bind the schema natively (function/JSON-schema under the hood).
        self.structured = self.llm.with_structured_output(
            DocumentAnalysis, method="json_schema", strict=True
        )

        self.model = model
        logger.info(f"LLMDocumentAnalyzer initialized with model: {model}")

    @staticmethod
    def _system_prompt() -> str:
        return LLM_DOCUMENT_ANALYSIS_PROMPT

    def _make_human_message(self, content: str, document_info: dict[str, Any]) -> str:
        # Keep it compact; schema does the heavy lifting.
        fname = document_info.get("filename", "unknown")
        pages = document_info.get("total_pages", 0)
        size = document_info.get("file_size", 0)
        return (
            f"Document metadata:\n"
            f"- filename: {fname}\n- total_pages: {pages}\n- file_size: {size}\n\n"
            f"Analyze the following content excerpt and return the structured object:\n\n"
            f"{content}"
        )

    async def analyze_document(
        self, content: str, document_info: dict[str, Any]
    ) -> DocumentAnalysis:
        """
        Perform analysis using structured outputs. Returns a DocumentAnalysis instance.
        """
        # Guardrail: keep prompt small; you can tune this threshold.
        snippet = content[:10_000]

        messages = [
            SystemMessage(content=self._system_prompt()),
            HumanMessage(content=self._make_human_message(snippet, document_info)),
        ]

        try:
            result = await self.structured.ainvoke(messages)
            # Optional: log usage for observability/cost control
            with contextlib.suppress(Exception):
                logger.debug(f"Token usage: {getattr(result, 'usage_metadata', None)}")
            return result
        except Exception as e:
            logger.exception("LLM structured analysis failed")
            # Minimal, typed fallback
            return DocumentAnalysis(
                document_type="unknown",
                summary=f"LLM analysis failed. Content preview: {snippet[:200]}...",
                key_topics=[],
                entities=[],
                confidence_score=0.0,
                language="unknown",
                sentiment="neutral",
                key_insights=[f"Analysis error: {str(e)}"],
            )

    def get_supported_entity_types(self) -> list[str]:
        return [
            "PERSON",
            "ORGANIZATION",
            "DATE",
            "MONEY",
            "LOCATION",
            "CONTRACT_TERMS",
            "LEGAL_REFERENCES",
            "PRODUCT",
            "SERVICE",
            "EMAIL",
            "PHONE",
            "PERCENTAGE",
            "DURATION",
            "ADDRESS",
            "LAW",
            "REGULATION",
            "CLAUSE",
            "DEADLINE",
            "PARTY",
        ]
