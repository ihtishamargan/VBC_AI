"""Database service for PostgreSQL operations."""

from contextlib import asynccontextmanager
from typing import Any

from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import sessionmaker

from backend.app.config import settings
from backend.app.database.queries import (
    DocumentQueries,
    SearchQueries,
    SystemQueries,
    VBCContractQueries,
)
from backend.app.utils.logger import get_module_logger

logger = get_module_logger(__name__)


class DatabaseService:
    """Service for managing PostgreSQL database operations."""

    def __init__(self):
        """Initialize the database service."""
        self.database_url = settings.database_url

        # Create synchronous engine for initial setup
        self.sync_engine = create_engine(
            self.database_url.replace("postgresql://", "postgresql://"),
            pool_pre_ping=True,
            pool_recycle=300,
            echo=False,
        )

        # Create async engine for application operations
        async_url = self.database_url.replace("postgresql://", "postgresql+asyncpg://")
        self.async_engine = create_async_engine(
            async_url, pool_pre_ping=True, pool_recycle=300, echo=False
        )

        # Create session factories
        self.SessionLocal = sessionmaker(
            autocommit=False, autoflush=False, bind=self.sync_engine
        )

        self.AsyncSessionLocal = async_sessionmaker(
            self.async_engine, class_=AsyncSession, expire_on_commit=False
        )

        logger.info(
            f"Database service initialized: {settings.db_host}:{settings.db_port}/{settings.db_name}"
        )

    def test_connection(self) -> bool:
        """Test database connection."""
        try:
            with self.sync_engine.connect() as conn:
                result = conn.execute(text(SystemQueries.TEST_CONNECTION))
                logger.info("Database connection successful")
                return True
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            return False

    @asynccontextmanager
    async def get_session(self):
        """Async context manager for database sessions."""
        async with self.AsyncSessionLocal() as session:
            try:
                yield session
                await session.commit()
            except Exception as e:
                await session.rollback()
                logger.error(f"Database session error: {e}")
                raise
            finally:
                await session.close()

    # Document operations
    async def save_document_metadata(
        self,
        doc_uuid: str,
        document_id: str,
        filename: str,
        file_size: int,
        document_type: str,
        summary: str,
        confidence_score: float,
        pages_processed: int,
        chunks_created: int,
        vectors_stored: int,
        processing_time_seconds: float,
    ) -> str:
        """Save document metadata to the documents table."""
        try:
            async with self.get_session() as session:
                document_query = text(DocumentQueries.INSERT_DOCUMENT)
                await session.execute(
                    document_query,
                    {
                        "id": doc_uuid,
                        "document_id": document_id,
                        "filename": filename,
                        "file_size": file_size,
                        "document_type": document_type,
                        "summary": summary,
                        "confidence_score": confidence_score,
                        "pages_processed": pages_processed,
                        "chunks_created": chunks_created,
                        "vectors_stored": vectors_stored,
                        "processing_time_seconds": processing_time_seconds,
                        "processing_status": "completed",
                    },
                )

                await session.commit()
                logger.info(
                    f"Saved document metadata for {document_id} with UUID {doc_uuid}"
                )
                return doc_uuid

        except SQLAlchemyError as e:
            logger.error(f"Failed to save document metadata: {e}")
            raise

    async def save_document(self, document_data: dict[str, Any]) -> str:
        """Save document metadata and analysis results."""
        try:
            async with self.get_session() as session:
                # Insert document
                doc_query = text(DocumentQueries.INSERT_DOCUMENT)

                result = await session.execute(
                    doc_query,
                    {
                        "document_id": document_data["document_id"],
                        "filename": document_data["filename"],
                        "file_size": document_data.get("file_size"),
                        "document_type": document_data.get("document_type"),
                        "summary": document_data.get("summary"),
                        "confidence_score": document_data.get("confidence_score"),
                        "pages_processed": document_data.get("pages_processed"),
                        "chunks_created": document_data.get("chunks_created"),
                        "vectors_stored": document_data.get("vectors_stored"),
                        "processing_time_seconds": document_data.get(
                            "processing_time_seconds"
                        ),
                        "processing_status": document_data.get(
                            "processing_status", "completed"
                        ),
                    },
                )

                doc_uuid = result.fetchone()[0]
                logger.info(f"Saved document: {document_data['document_id']}")
                return str(doc_uuid)

        except SQLAlchemyError as e:
            logger.error(f"Failed to save document: {e}")
            raise

    async def save_document_topics(self, doc_uuid: str, topics: list[str]) -> None:
        """Save document topics."""
        try:
            async with self.get_session() as session:
                for i, topic in enumerate(topics):
                    topic_query = text(DocumentQueries.INSERT_DOCUMENT_TOPIC)
                    await session.execute(
                        topic_query,
                        {
                            "document_id": doc_uuid,
                            "topic": topic,
                            "relevance_score": 1.0 - (i * 0.1),  # Decreasing relevance
                        },
                    )

                logger.info(f"Saved {len(topics)} topics for document {doc_uuid}")

        except SQLAlchemyError as e:
            logger.error(f"Failed to save topics: {e}")
            raise

    async def save_document_entities(
        self, doc_uuid: str, entities: list[dict[str, Any]]
    ) -> None:
        """Save document entities."""
        try:
            async with self.get_session() as session:
                for entity in entities:
                    entity_query = text(DocumentQueries.INSERT_DOCUMENT_ENTITY)

                    await session.execute(
                        entity_query,
                        {
                            "document_id": doc_uuid,
                            "entity_text": entity.get("text", entity.get("entity", "")),
                            "entity_type": entity.get(
                                "type", entity.get("label", "unknown")
                            ),
                            "confidence": entity.get("confidence", 0.5),
                        },
                    )

                logger.info(f"Saved {len(entities)} entities for document {doc_uuid}")

        except SQLAlchemyError as e:
            logger.error(f"Failed to save entities: {e}")
            raise

    async def save_document_chunks(
        self, doc_uuid: str, chunks_data: list[dict[str, Any]]
    ) -> None:
        """Save document chunks with vector references."""
        try:
            async with self.get_session() as session:
                for i, chunk_data in enumerate(chunks_data):
                    chunk_query = text(DocumentQueries.INSERT_DOCUMENT_CHUNK)

                    await session.execute(
                        chunk_query,
                        {
                            "document_id": doc_uuid,
                            "chunk_index": i,
                            "content": chunk_data["content"],
                            "vector_id": chunk_data.get("vector_id"),
                            "page_number": chunk_data.get("page_number"),
                            "chunk_metadata": chunk_data.get("metadata", {}),
                        },
                    )

                logger.info(f"Saved {len(chunks_data)} chunks for document {doc_uuid}")

        except SQLAlchemyError as e:
            logger.error(f"Failed to save chunks: {e}")
            raise

    # Query operations
    async def get_documents(self, limit: int = 50) -> list[dict[str, Any]]:
        """Get list of documents with summary info."""
        try:
            async with self.get_session() as session:
                query = text(DocumentQueries.GET_DOCUMENTS)

                result = await session.execute(query, {"limit": limit})
                documents = [dict(row._mapping) for row in result.fetchall()]

                logger.info(f"Retrieved {len(documents)} documents")
                return documents

        except SQLAlchemyError as e:
            logger.error(f"Failed to get documents: {e}")
            return []

    async def get_document_by_id(self, document_id: str) -> dict[str, Any] | None:
        """Get document details by document_id."""
        try:
            async with self.get_session() as session:
                query = text(DocumentQueries.GET_DOCUMENT_BY_ID)

                result = await session.execute(query, {"document_id": document_id})
                row = result.fetchone()

                if row:
                    return dict(row._mapping)
                return None

        except SQLAlchemyError as e:
            logger.error(f"Failed to get document {document_id}: {e}")
            return None

    # Search and analytics
    async def log_search_query(
        self,
        query_text: str,
        results_count: int,
        response_time_ms: int,
        user_session: str = None,
    ) -> None:
        """Log search query for analytics."""
        try:
            async with self.get_session() as session:
                search_query = text(SearchQueries.LOG_SEARCH_QUERY)

                await session.execute(
                    search_query,
                    {
                        "query_text": query_text,
                        "results_count": results_count,
                        "response_time_ms": response_time_ms,
                        "user_session": user_session or "anonymous",
                    },
                )

        except SQLAlchemyError as e:
            logger.error(f"Failed to log search query: {e}")

    async def log_chat_conversation(
        self,
        session_id: str,
        user_message: str,
        ai_response: str,
        sources_used: list[str],
        response_time_ms: int,
    ) -> None:
        """Log chat conversation."""
        try:
            async with self.get_session() as session:
                chat_query = text(SearchQueries.LOG_CHAT_CONVERSATION)

                await session.execute(
                    chat_query,
                    {
                        "session_id": session_id,
                        "user_message": user_message,
                        "ai_response": ai_response,
                        "sources_used": sources_used,
                        "response_time_ms": response_time_ms,
                    },
                )

        except SQLAlchemyError as e:
            logger.error(f"Failed to log chat conversation: {e}")

    # VBC Contract persistence
    async def save_vbc_contract(
        self, doc_uuid: str, vbc_data: "VBCContractData"
    ) -> str:
        """Save VBC contract data to database tables."""
        try:
            async with self.get_session() as session:
                # Insert main VBC contract record
                contract_query = text(VBCContractQueries.INSERT_VBC_CONTRACT)

                contract_result = await session.execute(
                    contract_query,
                    {
                        "document_id": doc_uuid,
                        "agreement_id": vbc_data.agreement_id,
                        "agreement_title": vbc_data.agreement_title,
                        "country": vbc_data.country,
                        "disease_area": vbc_data.disease_area,
                        "disease_area_details": vbc_data.disease_area_details,
                        "patient_population_size": vbc_data.patient_population_size,
                        "patient_population_description": vbc_data.patient_population_description,
                        "agreement_overview": vbc_data.agreement_overview,
                        "contract_background": vbc_data.contract_background,
                        "pilot_program_results": vbc_data.pilot_program_results,
                        "duration_months": vbc_data.duration_months,
                        "duration_description": vbc_data.duration_description,
                        "data_collection_frequency": vbc_data.data_collection_frequency,
                        "regulatory_framework": vbc_data.regulatory_framework,
                        "governance_structure": vbc_data.governance_structure,
                        "contract_complexity": vbc_data.contract_complexity.value
                        if vbc_data.contract_complexity
                        else None,
                        "extraction_confidence": vbc_data.extraction_confidence,
                    },
                )

                contract_id = contract_result.fetchone().id
                logger.info(f"Saved VBC contract with ID: {contract_id}")

                # Save related data
                await self._save_vbc_parties(session, contract_id, vbc_data)
                await self._save_vbc_financials(session, contract_id, vbc_data)
                await self._save_vbc_risk_protection(session, contract_id, vbc_data)
                await self._save_vbc_metrics(session, contract_id, vbc_data)
                await self._save_vbc_endpoints(session, contract_id, vbc_data)

                await session.commit()
                logger.info(
                    f"Successfully saved complete VBC contract data for ID: {contract_id}"
                )
                return str(contract_id)

        except SQLAlchemyError as e:
            logger.error(f"Failed to save VBC contract: {e}")
            raise

    async def _save_vbc_parties(
        self, session, contract_id: str, vbc_data: "VBCContractData"
    ):
        """Save contract parties."""
        if vbc_data.parties:
            for party in vbc_data.parties:
                party_query = text(VBCContractQueries.INSERT_VBC_PARTY)

                await session.execute(
                    party_query,
                    {
                        "contract_id": contract_id,
                        "party_name": party.name,
                        "party_type": party.type,
                        "party_country": party.country,
                    },
                )

            logger.info(f"Saved {len(vbc_data.parties)} contract parties")

    async def _save_vbc_financials(
        self, session, contract_id: str, vbc_data: "VBCContractData"
    ):
        """Save financial structure."""
        if vbc_data.financial_structure:
            financial_query = text(VBCContractQueries.INSERT_VBC_FINANCIAL)

            await session.execute(
                financial_query,
                {
                    "contract_id": contract_id,
                    "initial_payment": vbc_data.financial_structure.initial_payment,
                    "currency": vbc_data.financial_structure.currency,
                    "payment_model": vbc_data.financial_structure.payment_model,
                    "base_reimbursement": vbc_data.financial_structure.base_reimbursement,
                    "shared_savings_percentage": vbc_data.financial_structure.shared_savings_percentage,
                    "risk_corridor_upper": vbc_data.financial_structure.risk_corridor_upper,
                    "risk_corridor_lower": vbc_data.financial_structure.risk_corridor_lower,
                    "maximum_payout": vbc_data.financial_structure.maximum_payout,
                    "minimum_guarantee": vbc_data.financial_structure.minimum_guarantee,
                },
            )

            logger.info("Saved VBC contract financial structure")

    async def _save_vbc_risk_protection(
        self, session, contract_id: str, vbc_data: "VBCContractData"
    ):
        """Save risk protection mechanisms."""
        if vbc_data.risk_protection:
            risk_query = text(VBCContractQueries.INSERT_VBC_RISK_PROTECTION)

            await session.execute(
                risk_query,
                {
                    "contract_id": contract_id,
                    "has_stop_loss": vbc_data.risk_protection.has_stop_loss,
                    "has_risk_cap": vbc_data.risk_protection.has_risk_cap,
                    "has_non_responder_fund": vbc_data.risk_protection.has_non_responder_fund,
                    "stop_loss_threshold": vbc_data.risk_protection.stop_loss_threshold,
                    "risk_cap_percentage": vbc_data.risk_protection.risk_cap_percentage,
                },
            )

            logger.info("Saved VBC contract risk protection")

    async def _save_vbc_metrics(
        self, session, contract_id: str, vbc_data: "VBCContractData"
    ):
        """Save outcome metrics."""
        if vbc_data.outcome_metrics:
            for metric in vbc_data.outcome_metrics:
                metric_query = text(VBCContractQueries.INSERT_VBC_METRIC)

                await session.execute(
                    metric_query,
                    {
                        "contract_id": contract_id,
                        "metric_name": metric.name,
                        "metric_type": metric.type,
                        "target_value": metric.target_value,
                        "measurement_period": metric.measurement_period,
                        "data_source": metric.data_source,
                        "weight": metric.weight,
                    },
                )

            logger.info(f"Saved {len(vbc_data.outcome_metrics)} outcome metrics")

    async def _save_vbc_endpoints(
        self, session, contract_id: str, vbc_data: "VBCContractData"
    ):
        """Save contract endpoints."""
        if vbc_data.primary_endpoints:
            for endpoint in vbc_data.primary_endpoints:
                endpoint_query = text(VBCContractQueries.INSERT_VBC_ENDPOINT)

                await session.execute(
                    endpoint_query,
                    {
                        "contract_id": contract_id,
                        "endpoint_type": "primary",
                        "endpoint_description": endpoint,
                    },
                )

        if vbc_data.secondary_endpoints:
            for endpoint in vbc_data.secondary_endpoints:
                endpoint_query = text(VBCContractQueries.INSERT_VBC_ENDPOINT)

                await session.execute(
                    endpoint_query,
                    {
                        "contract_id": contract_id,
                        "endpoint_type": "secondary",
                        "endpoint_description": endpoint,
                    },
                )

    # Health and status
    async def get_database_stats(self) -> dict[str, Any]:
        """Get database statistics."""
        try:
            async with self.get_session() as session:
                stats_query = text(SystemQueries.GET_DATABASE_STATS)

                result = await session.execute(stats_query)
                stats = dict(result.fetchone()._mapping)

                return stats

        except SQLAlchemyError as e:
            logger.error(f"Failed to get database stats: {e}")
            return {}


# Global database service instance
database_service = DatabaseService()
