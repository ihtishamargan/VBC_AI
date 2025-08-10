"""SQL queries for VBC AI database operations."""

from typing import Any


class DocumentQueries:
    """SQL queries for document operations."""

    INSERT_DOCUMENT = """
        INSERT INTO documents (
            id, document_id, filename, file_size, document_type,
            summary, confidence_score, pages_processed, chunks_created,
            vectors_stored, processing_time_seconds, processing_status
        ) VALUES (
            :id, :document_id, :filename, :file_size, :document_type,
            :summary, :confidence_score, :pages_processed, :chunks_created,
            :vectors_stored, :processing_time_seconds, :processing_status
        )
        ON CONFLICT (document_id) 
        DO UPDATE SET
            id = EXCLUDED.id,
            filename = EXCLUDED.filename,
            file_size = EXCLUDED.file_size,
            document_type = EXCLUDED.document_type,
            summary = EXCLUDED.summary,
            confidence_score = EXCLUDED.confidence_score,
            pages_processed = EXCLUDED.pages_processed,
            chunks_created = EXCLUDED.chunks_created,
            vectors_stored = EXCLUDED.vectors_stored,
            processing_time_seconds = EXCLUDED.processing_time_seconds,
            processing_status = EXCLUDED.processing_status,
            updated_at = CURRENT_TIMESTAMP
        RETURNING id
    """

    INSERT_DOCUMENT_TOPIC = """
        INSERT INTO document_topics (document_id, topic, confidence_score)
        VALUES (:document_id, :topic, :confidence_score)
        ON CONFLICT (document_id, topic) DO NOTHING
    """

    INSERT_DOCUMENT_ENTITY = """
        INSERT INTO document_entities (
            document_id, entity_text, entity_type, confidence_score,
            start_position, end_position
        ) VALUES (
            :document_id, :entity_text, :entity_type, :confidence_score,
            :start_position, :end_position
        )
    """

    INSERT_DOCUMENT_CHUNK = """
        INSERT INTO document_chunks (
            document_id, chunk_index, chunk_text, chunk_embedding_id,
            token_count, start_page, end_page
        ) VALUES (
            :document_id, :chunk_index, :chunk_text, :chunk_embedding_id,
            :token_count, :start_page, :end_page
        )
    """

    GET_DOCUMENTS = """
        SELECT 
            id, document_id, filename, file_size, document_type,
            summary, confidence_score, pages_processed, chunks_created,
            vectors_stored, processing_time_seconds, processing_status,
            created_at, updated_at
        FROM documents 
        ORDER BY created_at DESC 
        LIMIT :limit
    """

    GET_DOCUMENT_BY_ID = """
        SELECT 
            id, document_id, filename, file_size, document_type,
            summary, confidence_score, pages_processed, chunks_created,
            vectors_stored, processing_time_seconds, processing_status,
            created_at, updated_at
        FROM documents 
        WHERE document_id = :document_id
    """


class SearchQueries:
    """SQL queries for search and analytics operations."""

    LOG_SEARCH_QUERY = """
        INSERT INTO search_queries (
            query_text, results_count, response_time_ms, user_session, created_at
        ) VALUES (
            :query_text, :results_count, :response_time_ms, :user_session, :created_at
        )
    """

    LOG_CHAT_CONVERSATION = """
        INSERT INTO chat_conversations (
            session_id, user_message, ai_response, sources_used,
            response_time_ms, created_at
        ) VALUES (
            :session_id, :user_message, :ai_response, :sources_used,
            :response_time_ms, :created_at
        )
    """


class VBCContractQueries:
    """SQL queries for VBC contract operations."""

    INSERT_VBC_CONTRACT = """
        INSERT INTO vbc_contracts (
            id, document_id, contract_name, contract_type, effective_date,
            expiration_date, total_contract_value, risk_sharing_percentage,
            payment_model, provider_organization, payer_organization,
            target_population, geographic_scope, created_at
        ) VALUES (
            :id, :document_id, :contract_name, :contract_type, :effective_date,
            :expiration_date, :total_contract_value, :risk_sharing_percentage,
            :payment_model, :provider_organization, :payer_organization,
            :target_population, :geographic_scope, :created_at
        )
        ON CONFLICT (document_id) 
        DO UPDATE SET
            contract_name = EXCLUDED.contract_name,
            contract_type = EXCLUDED.contract_type,
            effective_date = EXCLUDED.effective_date,
            expiration_date = EXCLUDED.expiration_date,
            total_contract_value = EXCLUDED.total_contract_value,
            risk_sharing_percentage = EXCLUDED.risk_sharing_percentage,
            payment_model = EXCLUDED.payment_model,
            provider_organization = EXCLUDED.provider_organization,
            payer_organization = EXCLUDED.payer_organization,
            target_population = EXCLUDED.target_population,
            geographic_scope = EXCLUDED.geographic_scope,
            updated_at = CURRENT_TIMESTAMP
        RETURNING id
    """

    INSERT_VBC_PARTY = """
        INSERT INTO vbc_contract_parties (
            contract_id, party_name, party_type, party_role
        ) VALUES (:contract_id, :party_name, :party_type, :party_role)
    """

    INSERT_VBC_FINANCIAL = """
        INSERT INTO vbc_contract_financials (
            contract_id, financial_component, amount, percentage,
            calculation_method, payment_schedule
        ) VALUES (
            :contract_id, :financial_component, :amount, :percentage,
            :calculation_method, :payment_schedule
        )
    """

    INSERT_VBC_RISK_PROTECTION = """
        INSERT INTO vbc_contract_risk_protection (
            contract_id, protection_type, threshold_value, cap_amount,
            description
        ) VALUES (
            :contract_id, :protection_type, :threshold_value, :cap_amount,
            :description
        )
    """

    INSERT_VBC_METRIC = """
        INSERT INTO vbc_contract_metrics (
            contract_id, metric_name, metric_type, target_value,
            measurement_period, data_source, weight
        ) VALUES (
            :contract_id, :metric_name, :metric_type, :target_value,
            :measurement_period, :data_source, :weight
        )
    """

    INSERT_VBC_ENDPOINT = """
        INSERT INTO vbc_contract_endpoints (
            contract_id, endpoint_type, endpoint_description
        ) VALUES (:contract_id, :endpoint_type, :endpoint_description)
    """


class SystemQueries:
    """SQL queries for system operations and statistics."""

    TEST_CONNECTION = "SELECT 1"

    GET_DATABASE_STATS = """
        SELECT 
            (SELECT COUNT(*) FROM documents) as total_documents,
            (SELECT COUNT(*) FROM document_topics) as total_topics,
            (SELECT COUNT(*) FROM document_entities) as total_entities,
            (SELECT COUNT(*) FROM document_chunks) as total_chunks,
            (SELECT COUNT(*) FROM search_queries) as total_searches,
            (SELECT COUNT(*) FROM chat_conversations) as total_chats,
            (SELECT COUNT(*) FROM vbc_contracts) as total_vbc_contracts,
            (SELECT COUNT(*) FROM vbc_contract_metrics) as total_vbc_metrics
    """


class QueryBuilder:
    """Helper class for building dynamic queries."""

    @staticmethod
    def build_search_filter(filters: dict[str, Any]) -> str:
        """Build WHERE clause from filters."""
        conditions = []
        for field, value in filters.items():
            if value is not None:
                conditions.append(f"{field} = :{field}")

        return " AND ".join(conditions) if conditions else "1=1"

    @staticmethod
    def build_pagination(limit: int = 50, offset: int = 0) -> str:
        """Build LIMIT and OFFSET clause."""
        return f"LIMIT {limit} OFFSET {offset}"

    @staticmethod
    def build_order_by(field: str = "created_at", direction: str = "DESC") -> str:
        """Build ORDER BY clause."""
        return f"ORDER BY {field} {direction}"


# Query collections for easy access
ALL_QUERIES = {
    "documents": DocumentQueries,
    "search": SearchQueries,
    "vbc_contracts": VBCContractQueries,
    "system": SystemQueries,
}
