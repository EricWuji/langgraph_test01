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

def grade_documents(state):
    model = get_model()
    
    # Grade the documents to decide which route to take
    query = state.get("query", "")
    multiply_result = state.get("multiply_result")
    retriever_result = state.get("retriever_result")
    
    available_routes = []
    if multiply_result is not None:
        available_routes.append("multiply")
    if retriever_result is not None:
        available_routes.append("retriever")
    
    if not available_routes:
        # If no results, default to generate
        return {"route_after_grade": "generate"}
    
    prompt = (
        f"Based on the query '{query}' and the following tool results, "
        f"which route should I take next: 'generate' or 'rewrite'?\n\n"
    )
    
    if "multiply" in available_routes:
        prompt += f"Multiply result: {multiply_result}\n\n"
    if "retriever" in available_routes:
        prompt += f"Retrieved documents: {retriever_result}\n\n"
    
    prompt += "Respond with just one word: 'generate' or 'rewrite'"
    
    messages = [HumanMessage(content=prompt)]
    response = model.invoke(messages)
    
    if "generate" in response.content.lower():
        return {"route_after_grade": "generate"}
    else:
        return {"route_after_grade": "rewrite"}