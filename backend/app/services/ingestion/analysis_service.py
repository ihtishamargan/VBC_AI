"""Document analysis service with strategy pattern."""

from abc import ABC, abstractmethod

from langchain.schema import Document

from backend.app.config import settings
from backend.app.models import AnalysisStrategy
from backend.app.services.llm_analyzer import DocumentAnalysis, LLMDocumentAnalyzer
from backend.app.services.vbc_analyzer import VBCContractAnalyzer as VBCAnalyzer
from backend.app.utils.logger import get_module_logger

logger = get_module_logger(__name__)


class DocumentAnalyzer(ABC):
    """Abstract base class for document analyzers."""

    @abstractmethod
    async def analyze(
        self, pages: list[Document], document_id: str
    ) -> DocumentAnalysis | None:
        """Analyze document pages and return analysis results."""
        pass


class VBCContractAnalyzer(DocumentAnalyzer):
    """VBC contract-specific document analyzer."""

    def __init__(self, model: str = None, api_key: str = None):
        """Initialize VBC contract analyzer."""
        try:
            self.analyzer = VBCAnalyzer(
                model=model or settings.openai_model,
                api_key=api_key or settings.openai_api_key,
            )
            logger.info("VBC contract analyzer initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize VBC contract analyzer: {e}")
            raise

    async def analyze(
        self, pages: list[Document], document_id: str
    ) -> DocumentAnalysis | None:
        """Analyze document using VBC contract analyzer."""
        try:
            logger.info(f"Starting VBC contract analysis for document {document_id}")

            # Combine all page content
            full_text = "\n\n".join([page.page_content for page in pages])

            # Perform VBC analysis
            result = await self.analyzer.analyze_contract(full_text, document_id)

            if result:
                logger.info(f"VBC analysis completed for document {document_id}")
                return result
            logger.warning(
                f"VBC analysis returned no results for document {document_id}"
            )
            return None

        except Exception as e:
            logger.error(f"VBC analysis failed for document {document_id}: {e}")
            return None


class GenericLLMAnalyzer(DocumentAnalyzer):
    """Generic LLM document analyzer."""

    def __init__(self, model: str = None, api_key: str = None):
        """Initialize generic LLM analyzer."""
        try:
            self.analyzer = LLMDocumentAnalyzer(
                api_key=api_key or settings.openai_api_key,
                model=model or settings.openai_model,
            )
            logger.info("Generic LLM analyzer initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize generic LLM analyzer: {e}")
            raise

    async def analyze(
        self, pages: list[Document], document_id: str
    ) -> DocumentAnalysis | None:
        """Analyze document using generic LLM analyzer."""
        try:
            logger.info(f"Starting generic LLM analysis for document {document_id}")

            # Combine all page content
            full_text = "\n\n".join([page.page_content for page in pages])

            # Perform generic analysis
            result = await self.analyzer.analyze_document(full_text, document_id)

            if result:
                logger.info(
                    f"Generic LLM analysis completed for document {document_id}"
                )
                return result
            logger.warning(
                f"Generic LLM analysis returned no results for document {document_id}"
            )
            return None

        except Exception as e:
            logger.error(f"Generic LLM analysis failed for document {document_id}: {e}")
            return None


class NoAnalyzer(DocumentAnalyzer):
    """No-op analyzer that skips analysis."""

    async def analyze(
        self, _pages: list[Document], document_id: str
    ) -> DocumentAnalysis | None:
        """Skip analysis and return None."""
        logger.info(f"Skipping analysis for document {document_id}")
        return None


class DocumentAnalysisService:
    """Service for document analysis with strategy pattern."""

    def __init__(self, strategy: AnalysisStrategy = AnalysisStrategy.VBC_CONTRACT):
        """Initialize analysis service with strategy."""
        self.strategy = strategy
        self._analyzer: DocumentAnalyzer | None = None
        self._initialize_analyzer()

    def _initialize_analyzer(self) -> None:
        """Initialize the appropriate analyzer based on strategy."""
        try:
            if self.strategy == AnalysisStrategy.VBC_CONTRACT:
                self._analyzer = VBCContractAnalyzer()
            elif self.strategy == AnalysisStrategy.GENERIC_LLM:
                self._analyzer = GenericLLMAnalyzer()
            elif self.strategy == AnalysisStrategy.NONE:
                self._analyzer = NoAnalyzer()
            else:
                logger.warning(
                    f"Unknown analysis strategy: {self.strategy}, using VBC contract analyzer"
                )
                self._analyzer = VBCContractAnalyzer()

        except Exception as e:
            logger.error(
                f"Failed to initialize analyzer for strategy {self.strategy}: {e}"
            )
            # Fallback to no analysis
            self._analyzer = NoAnalyzer()

    async def analyze_document(
        self, pages: list[Document], document_id: str
    ) -> DocumentAnalysis | None:
        """Analyze document using the configured strategy."""
        if not self._analyzer:
            logger.error("No analyzer available for document analysis")
            return None

        try:
            return await self._analyzer.analyze(pages, document_id)
        except Exception as e:
            logger.error(f"Document analysis failed: {e}")
            return None

    def change_strategy(self, new_strategy: AnalysisStrategy) -> None:
        """Change the analysis strategy and reinitialize analyzer."""
        if new_strategy != self.strategy:
            logger.info(
                f"Changing analysis strategy from {self.strategy} to {new_strategy}"
            )
            self.strategy = new_strategy
            self._initialize_analyzer()

    @property
    def is_available(self) -> bool:
        """Check if analyzer is available and ready."""
        return self._analyzer is not None and not isinstance(self._analyzer, NoAnalyzer)
