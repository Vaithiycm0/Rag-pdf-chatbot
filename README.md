# PDF Question Answering Chatbot

A production-ready RAG (Retrieval-Augmented Generation) chatbot that allows you to upload a PDF document and ask questions about its content. Built with **Streamlit**, **LangChain**, **FAISS**, and **Ollama**, this chatbot ensures privacy by running entirely locally and enforces strict anti-hallucination controls so that answers are drawn *only* from the provided document.

## 🌟 Features

- **Strict Fact-Checking**: The bot only answers questions if the information is explicitly found in the uploaded PDF. If it's not, it responds with a strict "I don't know" fallback message.
- **100% Local and Private**: Uses local embeddings (HuggingFace `all-MiniLM-L6-v2`) and local LLM execution via Ollama (`llama3`). Your documents never leave your machine.
- **Dual Interface**:
  - **Streamlit Web UI**: An interactive, user-friendly chat interface (`app.py`).
  - **FastAPI Backend**: A robust REST API for programmatic access (`server.py`).
- **Efficient Document Processing**: Fast PDF text extraction and intelligent chunking using PyMuPDF and LangChain's RecursiveCharacterTextSplitter.
- **Smart Retrieval**: FAISS vector store with relevance score filtering to prevent keyword-adjacent but semantically irrelevant context from reaching the LLM.
- **Caching mechanism**: Computes a file hash to cache the FAISS index, eliminating the need to re-process large PDFs if they haven't changed.

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **API Server**: FastAPI, Uvicorn
- **LLM Engine**: Ollama (`llama3` model)
- **RAG Framework**: LangChain
- **Embeddings**: HuggingFace (`all-MiniLM-L6-v2`)
- **Vector Store**: FAISS (CPU)
- **PDF Processing**: PyMuPDF (`pymupdf`)

## 📋 Prerequisites

Before running the application, make sure you have the following installed:

1. **Python 3.9+**
2. **Ollama**: Download and install [Ollama](https://ollama.com/).
3. Pull the required Llama 3 model by running the following command in your terminal:
   ```bash
   ollama pull llama3
   ```

## 🚀 Installation & Setup

1. **Clone the repository** (or download the source code):
   ```bash
   git clone <repository-url>
   cd "Pdf Chatbot"
   ```

2. **(Optional) Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install the dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 🎮 Usage

You can run this application either as an interactive Streamlit web app or as a FastAPI REST server.

### Option 1: Streamlit Web UI (Recommended)
To launch the interactive chat interface, run:
```bash
python run.py
```
*Alternatively, you can run `streamlit run app.py`.*

This will open the web application in your default browser. Upload a PDF using the sidebar and start asking questions!

### Option 2: FastAPI Server
To start the REST API server, run:
```bash
python server.py
```
The API will be available at `http://localhost:8000`. 
- **Upload Endpoint**: `POST /upload` (Upload a PDF file)
- **Chat Endpoint**: `POST /chat` (Send a JSON payload with `{"question": "..."}`)
- **Status Endpoint**: `GET /status` (Check pipeline and Ollama health)

## 📁 Project Structure

- `app.py`: The Streamlit web application interface.
- `server.py`: The FastAPI server implementation.
- `rag_pipeline.py`: Orchestrator linking document processing, retrieval, and LLM question-answering.
- `pdf_processor.py`: Handles loading PDFs, extracting text, and splitting content into chunks.
- `vector_store.py`: Manages the FAISS vector database and HuggingFace embeddings.
- `prompts.py`: Defines the strict system prompt that prevents LLM hallucinations.
- `run.py`: Entry point script to easily start the Streamlit application.
- `requirements.txt`: Python package dependencies.
- `data/`: Directory where uploaded PDFs are temporarily saved.
- `faiss_index/`: Local directory for caching vector embeddings to improve startup times.

## 🤝 How it Works (The RAG Pipeline)

1. **Extraction**: `PyMuPDF` reads the uploaded PDF and extracts text.
2. **Chunking**: The text is split into chunks of 1500 characters with an overlap of 300 characters.
3. **Embedding**: `all-MiniLM-L6-v2` generates vector embeddings for each text chunk.
4. **Indexing**: Embeddings are stored locally in a FAISS index.
5. **Retrieval**: When a question is asked, the pipeline performs a similarity search in the vector database and applies a relevance threshold filter.
6. **Generation**: The retrieved context is formatted with a strict system prompt and sent to Ollama (`llama3`). If the context doesn't contain the answer, the LLM outputs a predefined "I don't know" response.

## ⚠️ Disclaimer

While the system prompt enforces strict anti-hallucination rules, Large Language Models can occasionally produce unexpected results. Always verify critical information from the provided source chunks.
