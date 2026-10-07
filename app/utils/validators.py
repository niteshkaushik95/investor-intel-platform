from fastapi import UploadFile, File, HTTPException, status
# We import the new global constants directly
from app.config import settings

async def validate_pdf_file(file: UploadFile = File(...)) -> UploadFile:
    """
    Validates PDF requirements using simple, clean global configuration.
    """
    
    # 1. Size Validation (USING BYTES)
    if file.size > settings.MAX_PDF_FILE_UPLOAD_SIZE_BYTES:
        
        # FIX: We use the MB constant DIRECTLY. No division necessary.
        max_mb = settings.MAX_PDF_FILE_UPLOAD_SIZE_MB
        
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Validation error: File too large. Maximum allowed size is {max_mb}MB."
        )

    # 2. Basic MIME check
    if file.content_type != "application/pdf":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Validation error: Expected application/pdf, got '{file.content_type}'."
        )

    # 3. Magic Byte Security check
    magic_bytes = await file.read(5)
    
    if magic_bytes != b"%PDF-":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Security error: File header does not match PDF format."
        )
        
    await file.seek(0)
    return file