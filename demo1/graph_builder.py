from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict
from typing import Annotated
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.base import BaseStore
from llms import llm, embeddings
from nodes import chatbot
from graph_state import State
from langgraph.graph.message import MessagesState
def create_graph(in_memory_store: BaseStore):
    try:
        graph_builder = StateGraph(MessagesState)

        graph_builder.add_node("chatbot", chatbot)
        graph_builder.add_edge(START, "chatbot")
        graph_builder.add_edge("chatbot", END)

        memory = MemorySaver()
        return graph_builder.compile(checkpointer=memory, store=in_memory_store)
    except Exception as e:
        raise RuntimeError(f"Failed to create graph: {str(e)}")