"""
Vector store module.

Manages HuggingFace embeddings and FAISS index creation, persistence,
and retrieval for semantic search.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)

# Model and retrieval configuration per project requirements
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
RETRIEVER_K = 10

# Local persistence paths
BASE_DIR = Path(__file__).parent
FAISS_INDEX_DIR = BASE_DIR / "faiss_index"
EMBEDDINGS_DIR = BASE_DIR / "embeddings"
METADATA_FILE = EMBEDDINGS_DIR / "index_metadata.json"


class VectorStoreManager:
    """Create, persist, and query FAISS vector stores with HuggingFace embeddings."""

    def __init__(
        self,
        embedding_model: str = EMBEDDING_MODEL,
        index_dir: Path | None = None,
        embeddings_dir: Path | None = None,
    ) -> None:
        self.embedding_model = embedding_model
        self.index_dir = index_dir or FAISS_INDEX_DIR
        self.embeddings_dir = embeddings_dir or EMBEDDINGS_DIR
        self._embeddings: HuggingFaceEmbeddings | None = None
        self._vector_store: FAISS | None = None

        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)

    @property
    def embeddings(self) -> HuggingFaceEmbeddings:
        """Lazy-load embedding model to reduce startup memory usage."""
        if self._embeddings is None:
            logger.info("Loading embedding model: %s", self.embedding_model)
            self._embeddings = HuggingFaceEmbeddings(
                model_name=self.embedding_model,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
        return self._embeddings

    def create_vector_store(self, chunks: list[Document]) -> FAISS:
        """
        Create a new FAISS index from document chunks.

        Args:
            chunks: Chunked documents to embed and index.

        Returns:
            FAISS vector store instance.
        """
        if not chunks:
            raise ValueError("Cannot create vector store from empty chunk list.")

        logger.info("Creating FAISS index from %d chunks...", len(chunks))
        self._vector_store = FAISS.from_documents(chunks, self.embeddings)
        logger.info("FAISS index created successfully.")
        return self._vector_store

    def save_index(self, file_hash: str, filename: str, page_count: int, chunk_count: int) -> None:
        """
        Persist FAISS index and metadata to disk for reuse.

        Args:
            file_hash: SHA-256 hash of the source PDF.
            filename: Original PDF filename.
            page_count: Number of pages in the PDF.
            chunk_count: Number of chunks created.
        """
        if self._vector_store is None:
            raise RuntimeError("No vector store to save. Create index first.")

        # Clear previous index before saving new one
        if self.index_dir.exists():
            shutil.rmtree(self.index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)

        self._vector_store.save_local(str(self.index_dir))
        logger.info("FAISS index saved to %s", self.index_dir)

        metadata = {
            "file_hash": file_hash,
            "filename": filename,
            "page_count": page_count,
            "chunk_count": chunk_count,
            "embedding_model": self.embedding_model,
        }
        with open(METADATA_FILE, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        logger.info("Index metadata saved.")

    def load_index(self) -> FAISS | None:
        """
        Load a previously saved FAISS index from disk.

        Returns:
            FAISS vector store if available, otherwise None.
        """
        index_file = self.index_dir / "index.faiss"
        if not index_file.exists():
            logger.info("No persisted FAISS index found.")
            return None

        try:
            self._vector_store = FAISS.load_local(
                str(self.index_dir),
                self.embeddings,
                allow_dangerous_deserialization=True,
            )
            logger.info("FAISS index loaded from %s", self.index_dir)
            return self._vector_store
        except Exception as exc:
            logger.error("Failed to load FAISS index: %s", exc)
            return None

    def get_cached_metadata(self) -> dict[str, Any] | None:
        """Return metadata for the cached index, if it exists."""
        if not METADATA_FILE.exists():
            return None
        try:
            with open(METADATA_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Could not read index metadata: %s", exc)
            return None

    def is_cache_valid(self, file_hash: str) -> bool:
        """Check if a cached index matches the given PDF file hash."""
        metadata = self.get_cached_metadata()
        if metadata is None:
            return False
        return metadata.get("file_hash") == file_hash

    def get_retriever(self, k: int = RETRIEVER_K):
        """
        Return a retriever for semantic search.

        Args:
            k: Number of top chunks to retrieve.

        Returns:
            LangChain retriever bound to the current vector store.
        """
        if self._vector_store is None:
            self.load_index()
        if self._vector_store is None:
            raise RuntimeError("Vector store not initialized. Process a PDF first.")
        return self._vector_store.as_retriever(search_kwargs={"k": k})

    def similarity_search(self, query: str, k: int = RETRIEVER_K) -> list[Document]:
        """
        Perform semantic similarity search.

        Args:
            query: User question.
            k: Number of results to return.

        Returns:
            List of most relevant document chunks.
        """
        if self._vector_store is None:
            self.load_index()
        if self._vector_store is None:
            raise RuntimeError("Vector store not initialized. Process a PDF first.")
        return self._vector_store.similarity_search(query, k=k)

    def clear_index(self) -> None:
        """Remove persisted index and metadata."""
        if self.index_dir.exists():
            shutil.rmtree(self.index_dir)
            self.index_dir.mkdir(parents=True, exist_ok=True)
        if METADATA_FILE.exists():
            METADATA_FILE.unlink()
        self._vector_store = None
        logger.info("Vector store cleared.")
