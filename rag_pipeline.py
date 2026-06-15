"""
RAG pipeline module.

Orchestrates PDF processing, vector retrieval, and LLM-based question
answering with strict anti-hallucination controls.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import ollama
from langchain_core.documents import Document

from pdf_processor import PDFProcessor
from prompts import NO_ANSWER_MESSAGE, format_qa_prompt
from vector_store import RETRIEVER_K, VectorStoreManager

logger = logging.getLogger(__name__)

# LLM configuration per project requirements
OLLAMA_MODEL = "llama3"
OLLAMA_HOST = "http://localhost:11434"

# Minimum relevance threshold for retrieved chunks (cosine similarity proxy)
MIN_RELEVANCE_SCORE = 0.25


class RAGPipeline:
    """
    End-to-end RAG pipeline for PDF question answering.

    Workflow:
        1. Process PDF into chunks
        2. Create/load FAISS vector store
        3. Retrieve semantically relevant chunks
        4. Verify context contains answer
        5. Generate answer via Ollama LLM
    """

    def __init__(
        self,
        ollama_model: str = OLLAMA_MODEL,
        ollama_host: str = OLLAMA_HOST,
        retriever_k: int = RETRIEVER_K,
    ) -> None:
        self.ollama_model = ollama_model
        self.ollama_host = ollama_host
        self.retriever_k = retriever_k

        self.pdf_processor = PDFProcessor()
        self.vector_store = VectorStoreManager()
        self._is_ready = False
        self._pdf_metadata: dict[str, Any] = {}

    @property
    def is_ready(self) -> bool:
        """Whether a PDF has been processed and the pipeline is ready for Q&A."""
        return self._is_ready

    @property
    def pdf_metadata(self) -> dict[str, Any]:
        """Metadata about the currently loaded PDF."""
        return self._pdf_metadata

    def _format_context(self, documents: list[Document]) -> str:
        """Combine retrieved chunks into a single context string."""
        context_parts = []
        for i, doc in enumerate(documents, start=1):
            page = doc.metadata.get("page", "N/A")
            content = doc.page_content.strip()
            context_parts.append(f"[Chunk {i} | Page {page}]\n{content}")
        return "\n\n".join(context_parts)

    def _retrieve_with_scores(self, question: str) -> list[tuple[Document, float]]:
        """
        Retrieve chunks with similarity scores for relevance filtering.

        Uses FAISS similarity_search_with_score for semantic ranking.
        Lower L2 distance = higher relevance for normalized embeddings.
        """
        if self.vector_store._vector_store is None:
            self.vector_store.load_index()

        store = self.vector_store._vector_store
        if store is None:
            raise RuntimeError("Vector store not available.")

        results = store.similarity_search_with_score(question, k=self.retriever_k)

        # Convert L2 distance to a 0-1 relevance score (higher = more relevant)
        scored_docs = []
        for doc, distance in results:
            relevance = 1.0 / (1.0 + distance)
            scored_docs.append((doc, relevance))

        return scored_docs

    def _filter_relevant_chunks(
        self, scored_docs: list[tuple[Document, float]]
    ) -> list[Document]:
        """
        Filter retrieved chunks by relevance threshold.

        Removes chunks that are semantically too distant from the question,
        preventing keyword-adjacent but irrelevant context from reaching the LLM.
        """
        relevant = [
            doc for doc, score in scored_docs if score >= MIN_RELEVANCE_SCORE
        ]
        logger.info(
            "Retrieved %d chunks, %d passed relevance filter (threshold=%.2f)",
            len(scored_docs),
            len(relevant),
            MIN_RELEVANCE_SCORE,
        )
        return relevant

    def _is_no_answer_response(self, response: str) -> bool:
        """Check if the LLM response indicates no answer was found."""
        normalized = response.strip().lower()
        no_answer_patterns = [
            "i don't know",
            "i do not know",
            "not available in the uploaded pdf",
            "not available in the pdf",
            "cannot be found in the pdf",
            "cannot be found in the context",
            "no information",
            "not mentioned in the pdf",
            "not mentioned in the context",
        ]
        return any(pattern in normalized for pattern in no_answer_patterns)

    def _normalize_response(self, response: str) -> str:
        """Ensure no-answer responses use the exact required message."""
        if self._is_no_answer_response(response):
            return NO_ANSWER_MESSAGE
        return response.strip()

    def _call_ollama(self, prompt: str) -> str:
        """
        Send prompt to Ollama LLM and return the generated response.

        Args:
            prompt: Fully formatted QA prompt with context and question.

        Returns:
            LLM-generated answer string.
        """
        try:
            client = ollama.Client(host=self.ollama_host)
            response = client.generate(
                model=self.ollama_model,
                prompt=prompt,
                options={
                    "temperature": 0.0,
                    "top_p": 0.9,
                    "num_predict": 1024,
                },
            )
            return response.get("response", "").strip()
        except Exception as exc:
            logger.error("Ollama generation failed: %s", exc)
            raise ConnectionError(
                f"Failed to connect to Ollama at {self.ollama_host}. "
                f"Ensure Ollama is running and model '{self.ollama_model}' is installed. "
                f"Error: {exc}"
            ) from exc

    def process_pdf(self, file_path: str | Path, use_cache: bool = True) -> dict[str, Any]:
        """
        Process a PDF: extract text, create embeddings, and persist index.

        Args:
            file_path: Path to the uploaded PDF.
            use_cache: Reuse cached index if the PDF hash matches.

        Returns:
            Processing result with status and metadata.
        """
        path = Path(file_path)
        file_hash = self.pdf_processor.compute_file_hash(path)

        # Attempt to load cached index for the same PDF
        if use_cache and self.vector_store.is_cache_valid(file_hash):
            logger.info("Using cached FAISS index for %s", path.name)
            self.vector_store.load_index()
            cached_meta = self.vector_store.get_cached_metadata() or {}
            self._pdf_metadata = cached_meta
            self._is_ready = True
            return {
                "status": "cached",
                "filename": cached_meta.get("filename", path.name),
                "page_count": cached_meta.get("page_count", 0),
                "chunk_count": cached_meta.get("chunk_count", 0),
                "file_hash": file_hash,
            }

        # Full processing pipeline for new or changed PDFs
        result = self.pdf_processor.process_pdf(path)
        chunks = result["chunks"]

        self.vector_store.create_vector_store(chunks)
        self.vector_store.save_index(
            file_hash=result["file_hash"],
            filename=result["filename"],
            page_count=result["page_count"],
            chunk_count=result["chunk_count"],
        )

        self._pdf_metadata = {
            "filename": result["filename"],
            "page_count": result["page_count"],
            "chunk_count": result["chunk_count"],
            "file_hash": result["file_hash"],
        }
        self._is_ready = True

        return {
            "status": "processed",
            "filename": result["filename"],
            "page_count": result["page_count"],
            "chunk_count": result["chunk_count"],
            "file_hash": result["file_hash"],
        }

    def try_load_cached_index(self) -> bool:
        """
        Attempt to load a previously persisted index on app startup.

        Returns:
            True if a valid cached index was loaded.
        """
        store = self.vector_store.load_index()
        if store is not None:
            metadata = self.vector_store.get_cached_metadata()
            if metadata:
                self._pdf_metadata = metadata
                self._is_ready = True
                logger.info("Loaded cached index for: %s", metadata.get("filename"))
                return True
        return False

    def answer_question(self, question: str) -> dict[str, Any]:
        """
        Answer a user question using RAG with strict PDF-only constraints.

        Steps:
            1. Understand question intent (via semantic retrieval)
            2. Retrieve top-k relevant chunks
            3. Filter by relevance score
            4. Verify context supports an answer (via LLM)
            5. Return answer or exact no-answer message

        Args:
            question: User's natural language question.

        Returns:
            Dictionary with answer, source chunks, and retrieval metadata.
        """
        if not self._is_ready:
            raise RuntimeError("No PDF loaded. Upload and process a PDF first.")

        question = question.strip()
        if not question:
            raise ValueError("Question cannot be empty.")

        # Step 1 & 2: Semantic retrieval
        scored_docs = self._retrieve_with_scores(question)

        # Step 3: Relevance filtering
        relevant_docs = self._filter_relevant_chunks(scored_docs)

        if not relevant_docs:
            logger.info("No relevant chunks found for question: %s", question[:80])
            return {
                "answer": NO_ANSWER_MESSAGE,
                "sources": [],
                "chunks_retrieved": len(scored_docs),
                "chunks_used": 0,
            }

        # Step 4: Build context and verify answer exists
        context = self._format_context(relevant_docs)
        prompt = format_qa_prompt(context=context, question=question)

        # Step 5: Generate answer
        raw_answer = self._call_ollama(prompt)
        answer = self._normalize_response(raw_answer)

        sources = [
            {
                "page": doc.metadata.get("page", "N/A"),
                "content_preview": doc.page_content[:200] + "..."
                if len(doc.page_content) > 200
                else doc.page_content,
            }
            for doc in relevant_docs
        ]

        return {
            "answer": answer,
            "sources": sources,
            "chunks_retrieved": len(scored_docs),
            "chunks_used": len(relevant_docs),
        }

    def reset(self) -> None:
        """Clear vector store and reset pipeline state."""
        self.vector_store.clear_index()
        self._is_ready = False
        self._pdf_metadata = {}
        logger.info("RAG pipeline reset.")

    def check_ollama_health(self) -> dict[str, Any]:
        """
        Verify Ollama connectivity and model availability.

        Returns:
            Health check result with status and available models.
        """
        try:
            client = ollama.Client(host=self.ollama_host)
            models = client.list()
            model_names = [m.get("model", m.get("name", "")) for m in models.get("models", [])]
            model_available = any(
                self.ollama_model in name for name in model_names
            )
            return {
                "connected": True,
                "model_available": model_available,
                "available_models": model_names,
                "required_model": self.ollama_model,
            }
        except Exception as exc:
            return {
                "connected": False,
                "model_available": False,
                "error": str(exc),
                "required_model": self.ollama_model,
            }
