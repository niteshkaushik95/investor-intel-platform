import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.config import settings  # <-- NEW

# NEW: We set the default values to our settings!
def split_markdown_into_chunks(markdown_path: str, chunk_size: int = settings.CHUNK_SIZE, chunk_overlap: int = settings.CHUNK_OVERLAP):
    if not os.path.exists(markdown_path):
        print(f"Error: File not found at {markdown_path}")
        return []

    with open(markdown_path, "r", encoding="utf-8") as f:
        text = f.read()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,        
        chunk_overlap=chunk_overlap,  
        separators=["\n\n", "\n", " ", ""]
    )

    chunks = splitter.split_text(text)
    return chunks

if __name__ == "__main__":
    # NEW: Robust pathing so this script works no matter where you run it from!
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    sample_md_path = os.path.join(PROJECT_ROOT, settings.MARKDOWN_DIR, "sample.md")
    
    # We no longer need to pass the sizes manually, it uses the config defaults!
    chunks = split_markdown_into_chunks(sample_md_path)
    
    print(f"\n--- Total Chunks Created: {len(chunks)} ---")
    for index, chunk in enumerate(chunks[:3]):
        print(f"\n[Chunk {index + 1}]:")
        print(chunk)
        print("-" * 30)