import chromadb
import os
from datetime import datetime, timezone  # <-- NEW IMPORT
from app.config import settings
from .chunker import split_markdown_into_chunks

DB_PATH = os.path.join(settings.PROJECT_ROOT, settings.CHROMA_DB_PATH)

def get_chroma_client():
    return chromadb.PersistentClient(path=DB_PATH)

def add_chunks_to_vector_store(collection_name: str, chunks: list[str], source_file_name: str):
    client = get_chroma_client()
    collection = client.get_or_create_collection(name=collection_name)
    
    ids = [f"{source_file_name}_chunk_{i}" for i in range(len(chunks))]
    
    # NEW: Generate a UTC timestamp string for exactly right now
    current_time = datetime.now(timezone.utc).isoformat()
    
    # NEW: Add ingested_at to our metadata dictionary
    metadatas = [
        {"source": source_file_name, "ingested_at": current_time} 
        for _ in chunks
    ]
    
    print(f"Adding {len(chunks)} chunks to ChromaDB from {source_file_name} at {current_time}...")
    collection.add(documents=chunks, metadatas=metadatas, ids=ids)
    print("Done!")

def search_similar_chunks(collection_name: str, query: str, top_k: int = 2):
    """Searches the database for chunks with similar meaning to the query."""
    client = get_chroma_client()
    collection = client.get_collection(name=collection_name)
    
    print(f"\nSearching for: '{query}'")
    results = collection.query(
        query_texts=[query],
        n_results=top_k 
    )
    return results

if __name__ == "__main__":
    # 1. Dynamically build the path to the markdown file
    md_path = os.path.join(settings.PROJECT_ROOT, settings.MARKDOWN_DIR, "sample.md")
    
    # 2. Get chunks using our chunker
    my_chunks = split_markdown_into_chunks(md_path)
    
    if my_chunks:
        # NEW: Extract just the filename (e.g., "sample.md") from the full path
        file_name = os.path.basename(md_path)
        
        # FIX: Pass all THREE required arguments now!
        add_chunks_to_vector_store(settings.COLLECTION_NAME, my_chunks, source_file_name=file_name)
        
        # 4. Search using the configured collection name
        my_query = "What is the main topic of this document?"
        search_results = search_similar_chunks(settings.COLLECTION_NAME, my_query, top_k=1)
        
        print("\n--- Top Search Result ---")
        print(search_results["documents"][0][0])