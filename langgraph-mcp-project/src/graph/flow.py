from typing import Dict, Any, TypedDict, Annotated, Literal
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from src.graph.state import State, RouteChoices
from src.agents.agent import agent
from src.agents.grade import grade_documents
from src.agents.generate import generate
from src.agents.rewrite import rewrite
from src.tools.multiply import multiply_tool
from src.tools.retriever import retriever_tool

def build_graph():
    # Define the workflow graph
    workflow = StateGraph(State)
    
    # Add nodes
    workflow.add_node("agent", agent)
    workflow.add_node("call_tools", call_tools)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("generate", generate)
    workflow.add_node("rewrite", rewrite)
    
    # Define edges
    
    # From start to agent
    workflow.set_entry_point("agent")
    
    # From agent to call_tools or directly to end
    workflow.add_conditional_edges(
        "agent",
        lambda state: "call_tools" if state.get("tools_condition") else END,
        {
            "call_tools": "call_tools",
            END: END
        }
    )
    
    # From call_tools to grade_documents
    workflow.add_edge("call_tools", "grade_documents")
    
    # From grade_documents to either generate or rewrite
    workflow.add_conditional_edges(
        "grade_documents",
        lambda state: state.get("route_after_grade", "generate"),
        {
            "generate": "generate",
            "rewrite": "rewrite"
        }
    )
    
    # From generate and rewrite to end
    workflow.add_edge("generate", END)
    workflow.add_edge("rewrite", END)
    
    return workflow.compile()

def call_tools(state):
    """Call appropriate tools based on the query"""
    query = state.get("query", "")
    
    # Call both tools (in a more sophisticated implementation, 
    # you would determine which tools to call based on the query)
    multiply_result = multiply_tool(state)
    retriever_result = retriever_tool(state)
    
    # Merge results
    results = {**multiply_result, **retriever_result}
    
    # Determine which tool returned useful results
    has_multiply = multiply_result.get("multiply_result") is not None
    has_retriever = retriever_result.get("retriever_result") is not None
    
    route = None
    if has_multiply and has_retriever:
        route = "multiply@retriever"
    elif has_multiply:
        route = "multiply"
    elif has_retriever:
        route = "retriever"
        
    results["route_after_tools"] = route
    return results