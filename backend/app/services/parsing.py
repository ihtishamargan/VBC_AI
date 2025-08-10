"""Document parsing service for extracting text from PDFs."""
import logging
import os
from pathlib import Path
from typing import List, Optional

from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import Document

logger = logging.getLogger(__name__)


class DocumentParser:
    """Service for parsing PDF documents and extracting text content."""
    
    def __init__(self, max_file_size_mb: int = 10):
        """Initialize the document parser.
        
        Args:
            max_file_size_mb: Maximum file size allowed in megabytes
        """
        self.max_file_size_mb = max_file_size_mb
        self.max_file_size_bytes = max_file_size_mb * 1024 * 1024
    
    async def parse_pdf(self, file_path: str) -> List[Document]:
        """Parse a PDF file and extract text content.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            List of LangChain Documents with page content and metadata
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If file is too large or invalid
            Exception: For parsing errors
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Check file size
        file_size = os.path.getsize(file_path)
        if file_size > self.max_file_size_bytes:
            raise ValueError(
                f"File too large: {file_size / 1024 / 1024:.1f}MB. "
                f"Maximum allowed: {self.max_file_size_mb}MB"
            )
        
        try:
            logger.info(f"Parsing PDF: {file_path} ({file_size / 1024 / 1024:.1f}MB)")
            
            # Load and parse PDF
            loader = PyPDFLoader(file_path)
            pages = loader.load()
            
            if not pages:
                raise ValueError("No content extracted from PDF")
            
            # Add file-level metadata to each page
            filename = Path(file_path).name
            for i, page in enumerate(pages):
                page.metadata.update({
                    "filename": filename,
                    "file_path": file_path,
                    "file_size": file_size,
                    "page_number": i,
                    "total_pages": len(pages)
                })
            
            logger.info(f"Successfully parsed PDF: {len(pages)} pages extracted")
            return pages
            
        except Exception as e:
            logger.error(f"Error parsing PDF {file_path}: {str(e)}")
            raise Exception(f"Failed to parse PDF: {str(e)}")
    
    def validate_pdf_file(self, file_path: str) -> bool:
        """Validate if a file is a valid PDF.
        
        Args:
            file_path: Path to the file to validate
            
        Returns:
            True if valid PDF, False otherwise
        """
        try:
            if not file_path.lower().endswith('.pdf'):
                return False
            
            if not os.path.exists(file_path):
                return False
            
            # Basic PDF header check
            with open(file_path, 'rb') as f:
                header = f.read(4)
                return header == b'%PDF'
                
        except Exception:
            return False
    
    def get_file_info(self, file_path: str) -> dict:
        """Get basic file information.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Dictionary with file information
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        stat = os.stat(file_path)
        return {
            "filename": Path(file_path).name,
            "file_path": file_path,
            "size_bytes": stat.st_size,
            "size_mb": stat.st_size / 1024 / 1024,
            "created_at": stat.st_ctime,
            "modified_at": stat.st_mtime,
            "is_pdf": self.validate_pdf_file(file_path)
        }
