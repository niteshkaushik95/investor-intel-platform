from app.services.vector_store import search_similar_chunks
from app.services.llm_service import get_llm
from app.config import settings

# FIX: We now use settings.DEFAULT_TOP_K instead of a hardcoded 3!
def ask_rag_question(query: str, top_k: int = settings.DEFAULT_TOP_K):
    """
    The core RAG pipeline: Retrieve chunks, build prompt, generate answer.
    """
    print(f"\n[1] Searching database for: '{query}'")
    
    # ... the rest of your file stays exactly the same ...
    search_results = search_similar_chunks(settings.COLLECTION_NAME, query, top_k=top_k)
    
    # Extract the text chunks from the nested ChromaDB results
    documents = search_results.get("documents", [[]])[0]
    
    if not documents:
        return "I couldn't find any relevant information in the database."
        
    # Combine all found chunks into a single block of text
    context_text = "\n\n---\n\n".join(documents)
    
    # 2. AUGMENT: Build the prompt using our exact retrieved context
    prompt = f"""
    You are a professional financial AI assistant. Answer the user's question based ONLY on the context provided below. 
    If the answer is not contained in the context, do not guess. Simply state that you do not have the information.
    
    Context from documents:
    {context_text}
    
    Question:
    {query}
    """
    
    # 3. GENERATE: Send the augmented prompt to the LLM
    print(f"[2] Reading {len(documents)} chunks of context and generating answer...")
    llm = get_llm()
    response = llm.invoke(prompt)
    
    return response.content

# Test block
if __name__ == "__main__":
    # Since we uploaded a resume earlier, let's ask about it!
    # If you uploaded a financial PDF instead, change this question accordingly.
    test_query = "What are the key projects this person worked on?"
    
    final_answer = ask_rag_question(test_query)
    
    print("\n=== FINAL AI ANSWER ===")
    print(final_answer)