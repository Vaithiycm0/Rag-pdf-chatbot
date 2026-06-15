import os
import tempfile
import logging
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from rag_pipeline import RAGPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="PDF Q&A API")

# Allow CORS for the React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def read_root():
    return FileResponse("static/index.html")

pipeline = RAGPipeline()
pipeline.try_load_cached_index()

class QuestionRequest(BaseModel):
    question: str

@app.get("/status")
def get_status():
    return {
        "ready": pipeline.is_ready,
        "pdf_metadata": pipeline.pdf_metadata,
        "ollama_health": pipeline.check_ollama_health()
    }

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    
    # Save the uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        result = pipeline.process_pdf(tmp_path)
        return {"status": "success", "result": result}
    except Exception as e:
        logger.exception("Failed to process PDF")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.remove(tmp_path)

@app.post("/chat")
def chat(request: QuestionRequest):
    if not pipeline.is_ready:
        raise HTTPException(status_code=400, detail="No PDF loaded. Please upload one first.")
    
    try:
        result = pipeline.answer_question(request.question)
        return result
    except Exception as e:
        logger.exception("Failed to answer question")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
