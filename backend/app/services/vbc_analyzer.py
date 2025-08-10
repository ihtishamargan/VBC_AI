"""Specialized analyzer for Value-Based Care contracts."""
import logging
from typing import List, Optional
from datetime import datetime

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from openai import OpenAI

from backend.app.models import VBCContractData, VBCContractAnalysisResponse
from backend.app.config import settings

logger = logging.getLogger(__name__)

class VBCContractAnalyzer:
    """Specialized analyzer for extracting structured data from VBC contracts."""
    
    def __init__(self, model: str = settings.openai_model, api_key: Optional[str] = None):
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
        return """
You are a specialized AI analyst for Value-Based Care (VBC) healthcare contracts. Your task is to extract specific structured data from pharmaceutical and healthcare payment agreements.

CRITICAL INSTRUCTIONS:
1. Extract ALL available information from the contract text
2. Use exact values when found (don't estimate or guess)
3. For financial amounts, extract the exact number and identify the currency
4. Identify ALL parties involved and their types (pharma company, payer, provider, etc.)
5. Extract specific disease areas and patient population details
6. Identify payment models, risk protection mechanisms, and outcome metrics
7. Return "null" or appropriate defaults for fields not found in the text

REQUIRED EXTRACTION FIELDS:

**Core Information:**
- agreement_id: Look for contract ID, agreement number, or unique identifier
- agreement_title: Full title of the contract/agreement
- parties: All organizations involved (with their types and countries)
- country: Primary operating country
- disease_area: Specific therapeutic area (diabetes, RA, heart failure, etc.)

**Financial Structure:**
- initial_payment: Any upfront or initial payment amounts
- currency: Currency used (USD, EUR, etc.)
- payment_model: Type of VBC model (shared savings, risk sharing, bundled, etc.)
- base_reimbursement: Base payment amounts
- shared_savings_percentage: Percentage of shared savings
- risk protections: stop-loss, risk caps, non-responder funds

**Clinical Details:**
- patient_population_size: Number of patients covered
- patient_population_description: Demographics and characteristics
- outcome_metrics: Clinical endpoints and quality measures
- performance_benchmarks: Target outcomes and thresholds

**Contract Terms:**
- duration: Contract length in months or years
- start_date and end_date: If specified
- reporting_requirements: Data collection and reporting needs

**Risk Management:**
- has_stop_loss: Boolean for stop-loss protection
- has_risk_cap: Boolean for risk cap mechanisms  
- has_non_responder_fund: Boolean for non-responder protection
- Specific thresholds and percentages for risk mechanisms

Extract information precisely as written in the contract. If information is not present, use appropriate null values or defaults.
"""

    async def analyze_contract(
        self, 
        contract_text: str, 
        document_id: str
    ) -> VBCContractAnalysisResponse:
        """Analyze a VBC contract and extract structured data."""
        start_time = datetime.now()
        
        try:
            logger.info(f"Starting VBC contract analysis for document: {document_id}")
            
            # Guardrail: keep prompt manageable for large contracts
            snippet = contract_text[:15000]  # VBC contracts can be longer than regular docs
            
            # Prepare messages using LangChain format
            messages = [
                SystemMessage(content=self.get_extraction_prompt()),
                HumanMessage(content=f"Please analyze this VBC contract and extract all structured data:\n\n{snippet}")
            ]
            
            # Call LangChain structured output
            contract_data = await self.structured.ainvoke(messages)
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Estimate pages analyzed (rough estimate based on text length)
            estimated_pages = max(1, len(contract_text) // 2000)
            
            logger.info(f"VBC contract analysis completed for {document_id}: {processing_time:.2f}s")
            logger.info(f"Extracted data for {contract_data.agreement_title}")
            logger.info(f"Parties: {len(contract_data.parties)}, Disease: {contract_data.disease_area}")
            logger.info(f"Patient population: {contract_data.patient_population_size}")
            
            return VBCContractAnalysisResponse(
                success=True,
                contract_data=contract_data,
                error_message=None,
                processing_time_seconds=processing_time,
                pages_analyzed=estimated_pages,
                extraction_method="llm_structured_vbc"
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
                extraction_method="llm_structured_vbc"
            )
    
    def validate_extraction(self, contract_data: VBCContractData) -> List[str]:
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
            issues.append(f"Low extraction confidence: {contract_data.extraction_confidence}")
        
        return issues
    
    def format_summary(self, contract_data: VBCContractData) -> str:
        """Format a human-readable summary of the extracted contract data."""
        summary_parts = []
        
        # Basic info
        summary_parts.append(f"**Agreement:** {contract_data.agreement_title}")
        summary_parts.append(f"**Disease Area:** {contract_data.disease_area.value.replace('_', ' ').title()}")
        summary_parts.append(f"**Country:** {contract_data.country}")
        
        # Parties
        if contract_data.parties:
            party_names = [p.name for p in contract_data.parties]
            summary_parts.append(f"**Parties:** {', '.join(party_names)}")
        
        # Population
        if contract_data.patient_population_size:
            summary_parts.append(f"**Population:** {contract_data.patient_population_size:,} patients")
        
        # Financial
        financial = contract_data.financial_structure
        if financial.initial_payment:
            summary_parts.append(f"**Initial Payment:** {financial.currency} {financial.initial_payment:,.2f}")
        
        summary_parts.append(f"**Payment Model:** {financial.payment_model.value.replace('_', ' ').title()}")
        
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
            summary_parts.append(f"**Outcome Metrics:** {len(contract_data.outcome_metrics)} defined")
        
        return "\n".join(summary_parts)
