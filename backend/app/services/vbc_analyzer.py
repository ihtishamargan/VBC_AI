"""Specialized analyzer for Value-Based Care contracts."""

import logging
from datetime import datetime

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from openai import OpenAI

from backend.app.config import settings
from backend.app.models import VBCContractAnalysisResponse, VBCContractData
from backend.app.prompts import VBC_EXTRACTION_PROMPT

logger = logging.getLogger(__name__)


class VBCContractAnalyzer:
    """Specialized analyzer for extracting structured data from VBC contracts."""

    def __init__(self, model: str = settings.openai_model, api_key: str | None = None):
        """Initialize the VBC contract analyzer."""
        api_key = api_key or settings.openai_api_key
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required")

        # Store API key for direct access
        self.api_key = api_key

        # Use LangChain ChatOpenAI for structured outputs
        self.llm = ChatOpenAI(
            model=model,
            api_key=api_key,
        )

        # Create OpenAI client for direct API access if needed
        self.client = OpenAI(api_key=api_key)

        # Bind the VBC schema for structured output
        self.structured = self.llm.with_structured_output(
            VBCContractData, method="json_schema", strict=True
        )

        self.model = model
        logger.info(f"VBCContractAnalyzer initialized with model: {model}")

    def get_extraction_prompt(self) -> str:
        """Get the specialized prompt for VBC contract extraction."""
        return VBC_EXTRACTION_PROMPT

    async def analyze_contract(
        self, contract_text: str, document_id: str
    ) -> VBCContractAnalysisResponse:
        """Analyze a VBC contract and extract structured data."""
        start_time = datetime.now()

        try:
            logger.info(f"Starting VBC contract analysis for document: {document_id}")

            # Guardrail: keep prompt manageable for large contracts
            snippet = contract_text[
                :15000
            ]  # VBC contracts can be longer than regular docs

            # Prepare messages using LangChain format
            messages = [
                SystemMessage(content=self.get_extraction_prompt()),
                HumanMessage(
                    content=f"Please analyze this VBC contract and extract all structured data:\n\n{snippet}"
                ),
            ]

            # Call LangChain structured output
            contract_data = await self.structured.ainvoke(messages)

            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()

            # Estimate pages analyzed (rough estimate based on text length)
            estimated_pages = max(1, len(contract_text) // 2000)

            logger.info(
                f"VBC contract analysis completed for {document_id}: {processing_time:.2f}s"
            )
            logger.info(f"Extracted data for {contract_data.agreement_title}")
            logger.info(
                f"Parties: {len(contract_data.parties)}, Disease: {contract_data.disease_area}"
            )
            logger.info(f"Patient population: {contract_data.patient_population_size}")

            return VBCContractAnalysisResponse(
                success=True,
                contract_data=contract_data,
                error_message=None,
                processing_time_seconds=processing_time,
                pages_analyzed=estimated_pages,
                extraction_method="llm_structured_vbc",
            )

        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            error_msg = f"VBC contract analysis failed: {str(e)}"
            logger.error(error_msg)

            return VBCContractAnalysisResponse(
                success=False,
                contract_data=None,
                error_message=error_msg,
                processing_time_seconds=processing_time,
                pages_analyzed=0,
                extraction_method="llm_structured_vbc",
            )

    def validate_extraction(self, contract_data: VBCContractData) -> list[str]:
        """Validate the extracted contract data and return any issues."""
        issues = []

        # Check required fields
        if not contract_data.agreement_title:
            issues.append("Missing agreement title")

        if not contract_data.parties:
            issues.append("No parties identified")

        if not contract_data.country:
            issues.append("Country not specified")

        if not contract_data.agreement_overview:
            issues.append("Missing agreement overview")

        # Validate financial structure
        financial = contract_data.financial_structure
        if not financial.currency:
            issues.append("Currency not specified")

        # Check for reasonable confidence score
        if contract_data.extraction_confidence < 0.5:
            issues.append(
                f"Low extraction confidence: {contract_data.extraction_confidence}"
            )

        return issues

    def format_summary(self, contract_data: VBCContractData) -> str:
        """Format a human-readable summary of the extracted contract data."""
        summary_parts = []

        # Basic info
        summary_parts.append(f"**Agreement:** {contract_data.agreement_title}")
        summary_parts.append(
            f"**Disease Area:** {contract_data.disease_area.value.replace('_', ' ').title()}"
        )
        summary_parts.append(f"**Country:** {contract_data.country}")

        # Parties
        if contract_data.parties:
            party_names = [p.name for p in contract_data.parties]
            summary_parts.append(f"**Parties:** {', '.join(party_names)}")

        # Population
        if contract_data.patient_population_size:
            summary_parts.append(
                f"**Population:** {contract_data.patient_population_size:,} patients"
            )

        # Financial
        financial = contract_data.financial_structure
        if financial.initial_payment:
            summary_parts.append(
                f"**Initial Payment:** {financial.currency} {financial.initial_payment:,.2f}"
            )

        summary_parts.append(
            f"**Payment Model:** {financial.payment_model.value.replace('_', ' ').title()}"
        )

        # Risk protection
        risk_features = []
        if contract_data.risk_protection.has_stop_loss:
            risk_features.append("Stop-Loss")
        if contract_data.risk_protection.has_risk_cap:
            risk_features.append("Risk Cap")
        if contract_data.risk_protection.has_non_responder_fund:
            risk_features.append("Non-Responder Fund")

        if risk_features:
            summary_parts.append(f"**Risk Protection:** {', '.join(risk_features)}")

        # Metrics
        if contract_data.outcome_metrics:
            summary_parts.append(
                f"**Outcome Metrics:** {len(contract_data.outcome_metrics)} defined"
            )

        return "\n".join(summary_parts)
