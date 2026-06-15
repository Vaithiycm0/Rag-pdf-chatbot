"""
PDF Question Answering Chatbot — Streamlit Application.

A production-ready RAG chatbot that answers questions strictly from
uploaded PDF documents using semantic search and Ollama LLM.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import streamlit as st

from rag_pipeline import RAGPipeline
from prompts import NO_ANSWER_MESSAGE

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Page config & custom styling
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="PDF Q&A Chatbot",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    /* Main container */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 900px;
    }

    /* Header styling */
    .main-header {
        font-size: 2rem;
        font-weight: 700;
        color: var(--text-color);
        margin-bottom: 0.25rem;
    }
    .sub-header {
        font-size: 1rem;
        color: var(--text-color);
        opacity: 0.8;
        margin-bottom: 1.5rem;
    }

    /* Status badges */
    .status-ready {
        background-color: #d4edda;
        color: #155724;
        padding: 0.4rem 0.8rem;
        border-radius: 0.5rem;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
    }
    .status-waiting {
        background-color: #fff3cd;
        color: #856404;
        padding: 0.4rem 0.8rem;
        border-radius: 0.5rem;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
    }

    /* Chat message refinement */
    [data-testid="stChatMessage"] {
        border-radius: 0.75rem;
        margin-bottom: 0.5rem;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #f8f9fa;
    }

    /* Hide Streamlit branding for cleaner look */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------------------------


def init_session_state() -> None:
    """Initialize Streamlit session state variables."""
    if "pipeline" not in st.session_state:
        st.session_state.pipeline = RAGPipeline()
        st.session_state.pipeline.try_load_cached_index()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "pdf_processed" not in st.session_state:
        st.session_state.pdf_processed = st.session_state.pipeline.is_ready

    if "processing_status" not in st.session_state:
        st.session_state.processing_status = None


init_session_state()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### 📄 PDF Upload")
    st.markdown(
        "Upload a PDF document. The chatbot will answer questions "
        "**only** from its content."
    )

    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        help="Supports academic papers, novels, manuals, reports, and more.",
        label_visibility="collapsed",
    )

    # Process uploaded PDF
    if uploaded_file is not None:
        if st.button("🚀 Process PDF", use_container_width=True, type="primary"):
            with st.spinner("Processing PDF... This may take a few minutes for large files."):
                try:
                    # Save uploaded file to data directory
                    pdf_path = DATA_DIR / uploaded_file.name
                    with open(pdf_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    result = st.session_state.pipeline.process_pdf(pdf_path)

                    st.session_state.pdf_processed = True
                    st.session_state.processing_status = result
                    st.session_state.messages = []

                    status_label = "cached" if result["status"] == "cached" else "processed"
                    st.success(
                        f"PDF {status_label}! "
                        f"{result['page_count']} pages → {result['chunk_count']} chunks"
                    )
                except Exception as exc:
                    st.error(f"Error processing PDF: {exc}")
                    logger.exception("PDF processing failed")

    st.divider()

    # PDF status
    st.markdown("### 📊 Status")
    if st.session_state.pdf_processed and st.session_state.pipeline.is_ready:
        meta = st.session_state.pipeline.pdf_metadata
        st.markdown('<span class="status-ready">✅ Ready</span>', unsafe_allow_html=True)
        st.markdown(f"**File:** {meta.get('filename', 'Unknown')}")
        st.markdown(f"**Pages:** {meta.get('page_count', 'N/A')}")
        st.markdown(f"**Chunks:** {meta.get('chunk_count', 'N/A')}")
    else:
        st.markdown('<span class="status-waiting">⏳ Waiting for PDF</span>', unsafe_allow_html=True)

    st.divider()

    # Ollama health check
    st.markdown("### 🤖 LLM Status")
    health = st.session_state.pipeline.check_ollama_health()
    if health["connected"]:
        if health["model_available"]:
            st.success(f"Ollama connected — `{health['required_model']}` available")
        else:
            st.warning(
                f"Ollama connected but `{health['required_model']}` not found. "
                f"Run: `ollama pull {health['required_model']}`"
            )
    else:
        st.error(
            f"Ollama not reachable. Start Ollama and pull the model:\n\n"
            f"`ollama pull {health['required_model']}`"
        )

    st.divider()

    # Action buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    with col2:
        if st.button("📤 New PDF", use_container_width=True):
            st.session_state.messages = []
            st.session_state.pdf_processed = False
            st.session_state.processing_status = None
            st.session_state.pipeline.reset()
            st.rerun()

    st.divider()
    st.markdown(
        "<small>Answers are generated strictly from the uploaded PDF. "
        "No external knowledge is used.</small>",
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Main content area
# ---------------------------------------------------------------------------

st.markdown('<p class="main-header">📄 PDF Question Answering Chatbot</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">Ask questions about your uploaded PDF — '
    "answers come only from the document.</p>",
    unsafe_allow_html=True,
)

# Show welcome message if no chat history
if not st.session_state.messages:
    if not st.session_state.pdf_processed:
        st.info(
            "👈 Upload a PDF in the sidebar to get started. "
            "I can answer questions about academic papers, novels, manuals, "
            "reports, and more — using only the document content."
        )
    else:
        st.info(
            "✅ PDF loaded and ready! Ask me anything about the document. "
            "I'll answer only if the information exists in the PDF."
        )

# Render chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("sources"):
            with st.expander("📎 Source chunks used", expanded=False):
                for i, source in enumerate(message["sources"], 1):
                    st.markdown(f"**Chunk {i}** (Page {source['page']})")
                    st.text(source["content_preview"])

# Chat input
if prompt := st.chat_input(
    "Ask a question about the PDF...",
    disabled=not st.session_state.pdf_processed,
):
    # Display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate assistant response
    with st.chat_message("assistant"):
        with st.spinner("Searching document and generating answer..."):
            try:
                result = st.session_state.pipeline.answer_question(prompt)
                answer = result["answer"]
                st.markdown(answer)

                if result.get("sources") and answer != NO_ANSWER_MESSAGE:
                    with st.expander("📎 Source chunks used", expanded=False):
                        for i, source in enumerate(result["sources"], 1):
                            st.markdown(f"**Chunk {i}** (Page {source['page']})")
                            st.text(source["content_preview"])

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": result.get("sources", []),
                })

            except ConnectionError as exc:
                error_msg = str(exc)
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"⚠️ {error_msg}",
                })
            except Exception as exc:
                error_msg = f"An error occurred: {exc}"
                st.error(error_msg)
                logger.exception("Question answering failed")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"⚠️ {error_msg}",
                })
