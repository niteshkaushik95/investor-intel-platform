import hashlib
from fastapi import UploadFile

async def calculate_file_hash(file: UploadFile) -> str:
    """
    Reads an UploadFile in memory-safe 1MB chunks to calculate 
    its SHA-256 cryptographic hash.
    """
    sha256_hash = hashlib.sha256()
    
    # Read the file in 1MB chunks to preserve RAM
    chunk_size = 1024 * 1024 
    
    while chunk := await file.read(chunk_size):
        sha256_hash.update(chunk)
        
    # CRITICAL: We just read the whole file to hash it. 
    # We MUST reset the cursor back to 0 so the rest of the application can save it!
    await file.seek(0)
    
    return sha256_hash.hexdigest()