from typing import Dict, Any, TypedDict, Annotated, Literal
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from src.tools.tool_manager import call_tools
from src.tools.retriever import RetrieverTool
from src.tools.multiply import MultiplyTool

from src.graph.state import State, RouteChoices
from src.agents.agent import agent
from src.agents.grade import grade_documents
from src.agents.generate import generate
from src.agents.rewrite import rewrite

def build_graph():
    """构建工作流图"""
    # Define the workflow graph
    workflow = StateGraph(State)
    
    # 创建工具实例
    retriever_tool = RetrieverTool()
    multiply_tool = MultiplyTool()
    
    # Add nodes
    workflow.add_node("agent", agent)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("generate", generate)
    workflow.add_node("rewrite", rewrite)
    
    # 添加工具节点 - 注意这里我们只添加一次call_tools，避免重复
    workflow.add_node("call_tools", call_tools)
    
    # 添加单独的工具节点，用于更精细的控制（如有需要）
    workflow.add_node("retriever", lambda state: retriever_tool.run(state.get("query", "")))
    workflow.add_node("multiply", lambda state: multiply_tool.run(state.get("query", "")))
    
    # From start to agent
    workflow.set_entry_point("agent")
    
    # From agent to call_tools or directly to generate
    # 如果不需要工具，直接转到generate节点而不是结束
    workflow.add_conditional_edges(
        "agent",
        lambda state: "call_tools" if state.get("tools_condition") else "generate",
        {
            "call_tools": "call_tools",
            "generate": "generate"
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
    
    # 编译并返回图
    return workflow.compile()