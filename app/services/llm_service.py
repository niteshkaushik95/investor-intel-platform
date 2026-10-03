from langchain_groq import ChatGroq
from app.config import settings

def get_llm():
    if not settings.GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY is missing! Check your .env file.")
        
    # We initialize the model using our configured model name
    llm = ChatGroq(
        api_key=settings.GROQ_API_KEY,
        model_name=settings.LLM_MODEL_NAME, 
        temperature=0 
    )
    return llm

def ask_question(question: str):
    llm = get_llm()
    print(f"Asking AI: '{question}'...")
    response = llm.invoke(question)
    return response.content

if __name__ == "__main__":
    my_question = "What is the difference between revenue and net income in one short sentence?"
    answer = ask_question(my_question)
    print("\n--- AI Response ---")
    print(answer)