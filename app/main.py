import os
import shutil
import logging
import time
import uuid
from fastapi import FastAPI, UploadFile, Depends, HTTPException, BackgroundTasks, status

# --- Database & Models Imports ---
from sqlalchemy.orm import Session
from app.database import get_db, engine, Base
from app.models import Document, DocumentStatus

# --- Services & Config Imports ---
from app.services.rag_service import ask_rag_question
from app.services.processing_service import _background_worker_logic
from app import config

# --- Utilities Imports ---
from app.utils.validators import validate_pdf_file
from app.utils.hashing import calculate_file_hash

# Initialize Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("API_Main")

# Create database tables if they don't exist
Base.metadata.create_all(bind=engine)

# Initialize FastAPI
app = FastAPI(
    title="Investor Intel Platform API",
    description="RAG-powered API for extracting insights from financial documents.",
    version="1.0.0"
)

@app.get("/")
def read_root():
    return {"status": "online", "message": "Welcome to the Investor Intel Platform API"}

@app.get("/api/ask")
def ask_question(query: str):
    answer = ask_rag_question(query)
    return {"query": query, "answer": answer}


@app.post("/api/upload", status_code=status.HTTP_202_ACCEPTED)
async def upload_document_async(
    background_tasks: BackgroundTasks,
    file: UploadFile = Depends(validate_pdf_file),
    db: Session = Depends(get_db)
):
    """
    Secure, deduplicated, asynchronous PDF upload handler.
    """
    logger.info(f"Received upload request for '{file.filename}'.")
    
    file_fingerprint = await calculate_file_hash(file)

    # 1. DEDUPLICATION CHECK (Checks the hash, not the filename)
    if not config.ALLOW_DUPLICATE_UPLOADS:
        existing_doc = db.query(Document).filter(Document.file_hash == file_fingerprint).first()
        if existing_doc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Duplicate file detected. This document was already uploaded as '{existing_doc.original_filename}'."
            )

    name, ext = os.path.splitext(file.filename)
    system_filename = f"{name}_{uuid.uuid4().hex}{ext}"

    # 3. Define Paths using the SYSTEM filename
    raw_dir = os.path.join(config.PROJECT_ROOT, config.RAW_PDF_DIR)
    markdown_dir = os.path.join(config.PROJECT_ROOT, config.MARKDOWN_DIR)
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(markdown_dir, exist_ok=True)
    
    raw_path = os.path.join(raw_dir, system_filename)

    # 4. Save the file to raw storage FIRST (Your Logic)
    try:
        with open(raw_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        # If disk fails, we just return an error. The DB is completely untouched!
        raise HTTPException(status_code=500, detail=f"Disk I/O Error: Failed to save file to server: {str(e)}")

    # 5. Create SQLite Entry mapping original to system name SECOND
    new_doc = Document(
        original_filename=file.filename,
        system_filename=system_filename,
        raw_pdf_path=raw_path,
        status=DocumentStatus.PROCESSING,
        file_hash=file_fingerprint
    )
    
    try:
        db.add(new_doc)
        db.commit()
        db.refresh(new_doc)
        doc_id = new_doc.id
    except Exception as e:
        # CRITICAL CLEANUP: If the DB insert fails, we must delete the orphaned file!
        db.rollback() # Revert any pending database state
        if os.path.exists(raw_path):
            os.remove(raw_path)
        raise HTTPException(status_code=500, detail=f"Database Error: Failed to register document: {str(e)}")

    # 6. Hand off to Background Task (passing the system name)
    background_tasks.add_task(
        _background_worker_logic, 
        document_id=doc_id, 
        filename=system_filename, 
        raw_path=raw_path, 
        collection_name=config.COLLECTION_NAME, 
        markdown_dir=markdown_dir
    )
    
    return {
        "doc_id": doc_id,
        "original_filename": file.filename,
        "status": "queued",
        "message": "Upload successful. Document is being processed."
    }