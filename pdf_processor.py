"""
PDF processing module.

Handles PDF upload, text extraction, chunking, and metadata enrichment
for large documents (500+ pages).
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

# Chunking configuration per project requirements
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 300


class PDFProcessor:
    """Extract and chunk text from PDF documents."""

    def __init__(
        self,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    @staticmethod
    def compute_file_hash(file_path: str | Path) -> str:
        """Compute SHA-256 hash of a PDF file for cache invalidation."""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for block in iter(lambda: f.read(65536), b""):
                sha256.update(block)
        return sha256.hexdigest()

    def load_pdf(self, file_path: str | Path) -> list[Document]:
        """
        Load and extract text from a PDF file.

        Args:
            file_path: Path to the PDF file.

        Returns:
            List of LangChain Document objects with page metadata.

        Raises:
            FileNotFoundError: If the PDF file does not exist.
            ValueError: If no text could be extracted from the PDF.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF file not found: {path}")

        logger.info("Loading PDF: %s", path.name)
        loader = PyMuPDFLoader(str(path))
        documents = loader.load()

        if not documents:
            raise ValueError(f"No content extracted from PDF: {path.name}")

        # Filter out empty pages to reduce noise
        documents = [doc for doc in documents if doc.page_content.strip()]

        if not documents:
            raise ValueError(f"PDF contains no readable text: {path.name}")

        logger.info("Extracted %d pages from %s", len(documents), path.name)
        return documents

    def chunk_documents(self, documents: list[Document]) -> list[Document]:
        """
        Split documents into overlapping chunks for embedding.

        Args:
            documents: Raw page-level documents from PDF loader.

        Returns:
            Chunked documents with preserved metadata.
        """
        chunks = self.text_splitter.split_documents(documents)

        # Enrich metadata for traceability during retrieval
        for idx, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = idx
            chunk.metadata.setdefault("source", "uploaded_pdf")

        logger.info("Created %d chunks (size=%d, overlap=%d)",
                    len(chunks), self.chunk_size, self.chunk_overlap)
        return chunks

    def process_pdf(self, file_path: str | Path) -> dict[str, Any]:
        """
        Full PDF processing pipeline: load, chunk, and return metadata.

        Args:
            file_path: Path to the PDF file.

        Returns:
            Dictionary containing chunks, page count, file hash, and filename.
        """
        path = Path(file_path)
        documents = self.load_pdf(path)
        chunks = self.chunk_documents(documents)
        file_hash = self.compute_file_hash(path)

        return {
            "chunks": chunks,
            "page_count": len(documents),
            "chunk_count": len(chunks),
            "file_hash": file_hash,
            "filename": path.name,
        }
