from typing import Dict, List, Any
from langchain_core.messages import HumanMessage, AIMessage
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

def agent(state):
    model = get_model()
    
    # Determine tools condition and where to route next
    messages = [
        HumanMessage(
            content=f"Based on the following query, determine if tools are needed. "
                    f"Query: {state.get('query', '')}\n\n"
                    f"Respond with 'tools_needed: true' or 'tools_needed: false'"
        )
    ]
    
    response = model.invoke(messages)
    tools_needed = "true" in response.content.lower()
    
    return {
        "tools_condition": tools_needed
    }