"""Configuration management for VBC AI RAG backend."""
import os
from typing import Optional
from pydantic_settings import BaseSettings

from dotenv import load_dotenv
load_dotenv()

class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # Database Settings
    database_url: str = os.getenv("DATABASE_URL", "postgresql://vbc_user:vbc_secure_password_2024@localhost:5432/vbc_ai")
    db_host: str = os.getenv("DB_HOST", "localhost")
    db_port: int = int(os.getenv("DB_PORT", "5432"))
    db_name: str = os.getenv("DB_NAME", "vbc_ai")
    db_user: str = os.getenv("DB_USER", "vbc_user")
    db_password: str = os.getenv("DB_PASSWORD", "vbc_secure_password_2024")
    
    # OpenAI Settings
    openai_api_key: str = os.getenv("OPENAI_API_KEY")
    openai_model: str = "gpt-4.1-mini-2025-04-14"
    
    # Qdrant Vector Store Settings
    qdrant_url: str = os.getenv("QDRANT_URL")
    qdrant_api_key: str = os.getenv("QDRANT_API_KEY")
    qdrant_collection_name: str = "contracts-index"
    
    # Document Processing Settings
    chunk_size: int = 4000
    chunk_overlap: int = 200
    max_file_size_mb: int = 10
    
    # Application Settings
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    upload_dir: str = "./data/uploads"
    processed_dir: str = "./data/processed"

    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"  # Ignore extra fields in .env file


# Global settings instance
settings = Settings()