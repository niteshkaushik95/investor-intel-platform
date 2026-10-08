import chromadb
from app.config import settings

# 1. Use the exact path your app uses!
print(f"Connecting to ChromaDB at: {settings.CHROMA_DIR}")
client = chromadb.PersistentClient(path=str(settings.CHROMA_DIR))

# 2. See what collections actually exist
collections = [c.name for c in client.list_collections()]
print("Available collections:", collections)

# 3. If a collection exists, grab the first one dynamically
if collections:
    # Use the first collection it finds (instead of hardcoding "investor_reports")
    collection_name = collections[0]
    print(f"\nConnecting to collection: {collection_name}")
    collection = client.get_collection(name=collection_name)
    
    print("\nFetching first 2 chunks...\n")
    results = collection.get(
        limit=20,
        include=["metadatas", "documents"]
    )
    
    for i in range(len(results['ids'])):
        print(f"=== CHUNK ID: {results['ids'][i]} ===")
        print(f"METADATA: {results['metadatas'][i]}")
        print(f"TEXT:\n{results['documents'][i][:300]}...\n")
else:
    print("No collections found! Did the documents get processed?")