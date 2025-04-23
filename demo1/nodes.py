import uuid
from graph_state import State
from llms import llm
from langgraph.graph.message import MessagesState
from langchain_core.runnables import RunnableConfig
from langgraph.store.base import BaseStore
from utils import filter_messages
import logging
import rich
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def chatbot(state: MessagesState, config: RunnableConfig, *, store: BaseStore) -> dict:
    namespace = ("memories", config["configurable"]["user_id"]) # 给出用户唯一标识
    # namespace 的作用是存储用户的记忆信息，以便后续对话中使用
    memories = store.search(namespace, query=str(state["messages"][-1].content))
    
    # Updated this line to handle the PostgreSQL store's return format
    # Original: info = "\n".join([d.value["data"] for d in memories])
    info = "\n".join([d.get("value", {}).get("data", "") for d in memories])
    
    system_msg = f"你是一个推荐手机流量套餐的客服代表，你的名字由用户指定。用户指定你的名称为: {info}"
    rich.print(f"system message is {system_msg}")
    last_message = state["messages"][-1]
    rich.print(f"last message is {last_message}")
    if "记住" in last_message.content.lower():
        memory = "你的名字是南哥。"
        store.put(namespace, str(uuid.uuid4()), {"data": memory})
    messages = filter_messages(state["messages"])
    logger.debug(f"User messages: {messages}")
    response = llm.invoke(
                [{"role": "system", "content": system_msg}] + messages
            )
    return {"messages": [response]}