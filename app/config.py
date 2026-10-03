import os
from dotenv import load_dotenv

# Load variables from the .env file into Python's environment
load_dotenv()

# We use os.getenv() to fetch the value. 
# The second argument is a default fallback just in case the .env file is missing.
class Settings:
    # NEW: Calculate the root folder once, right here!
    # __file__ is config.py. We go up one level to 'app', and one more level to 'iip' (root)
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
    
    CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "data/chroma_db")
    RAW_PDF_DIR = os.getenv("RAW_PDF_DIR", "data/raw")
    MARKDOWN_DIR = os.getenv("MARKDOWN_DIR", "data/markdown")
    
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 500))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50))
    
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "qwen/qwen3.8-27b")
    
    # NEW: Let's also define our collection name here so it's not hardcoded!
    COLLECTION_NAME = "investor_reports"
# We create a single instance of this class to use across our application
settings = Settings()