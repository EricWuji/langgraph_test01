from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from config.settings import OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL

def get_model():
    model = ChatOpenAI(
        model=OPENAI_MODEL,
        temperature=0,
        openai_api_key=OPENAI_API_KEY,
        openai_api_base=OPENAI_BASE_URL
    )
    return model

def rewrite(state):
    model = get_model()
    
    query = state.get("query", "")
    documents = state.get("documents", [])
    multiply_result = state.get("multiply_result")
    retriever_result = state.get("retriever_result")
    
    context = ""
    if multiply_result is not None:
        context += f"Calculation result: {multiply_result}\n\n"
    if retriever_result is not None:
        context += f"Retrieved information: {retriever_result}\n\n"
    
    prompt = (
        f"Rewrite and improve the following query based on the provided context:\n\n"
        f"Query: {query}\n\n"
        f"Context: {context}\n\n"
        f"Provide a comprehensive, well-structured response."
    )
    
    messages = [HumanMessage(content=prompt)]
    response = model.invoke(messages)
    
    return {
        "rewrite_result": response.content,
        "final_output": response.content
    }