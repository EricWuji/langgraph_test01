import os
import uuid
import time
import json
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
import uvicorn
from langchain_core.prompts import PromptTemplate
from models import ChatCompletionRequest, ChatCompletionResponse
from graph_builder import create_graph
from utils import format_response, save_graph_visualization
from llms import llm, embeddings
# Replace InMemoryStore with PostgreSQLStore
# from langgraph.store.memory import InMemoryStore
from postgresql import PostgreSQLStore
from models import ChatCompletionRequest, ChatCompletionResponse, ChatCompletionResponseChoice, Message
# Import configuration
from config import get_pg_connection_string

# 设置日志模版
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# prompt模版设置相关 根据自己的实际业务进行调整
PROMPT_TEMPLATE_TXT_SYS = "prompt_template_system.txt"
PROMPT_TEMPLATE_TXT_USER = "prompt_template_user.txt"

# openai:调用gpt模型,oneapi:调用oneapi方案支持的模型,ollama:调用本地开源大模型,qwen:调用阿里通义千问大模型
llm_type = "ollama"

# API服务设置相关
PORT = 8012

# Get PostgreSQL connection string from config
PG_CONNECTION_STRING = get_pg_connection_string()

# 申明全局变量 全局调用
graph = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global graph
    try:
        logger.info("正在初始化模型、定义Graph...")
        logger.info(f"PostgreSQL连接: {PG_CONNECTION_STRING}")
        
        # Replace InMemoryStore with PostgreSQLStore
        pg_store = PostgreSQLStore(
            connection_string=PG_CONNECTION_STRING,
            index={
                "dims": 1024,
                "embed": embeddings
            },
            table_name="langgraph_store"  # You can customize the table name
        )
        
        graph = create_graph(pg_store)
        save_graph_visualization(graph)
        logger.info("初始化完成")
    except Exception as e:
        logger.error(f"初始化过程中出错: {str(e)}")
        raise
    yield
    logger.info("正在关闭...")

app = FastAPI(lifespan=lifespan)

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    # 判断初始化是否完成
    if not graph:
        logger.error("服务未初始化")
        raise HTTPException(status_code=500, detail="服务未初始化")

    try:
        logger.info(f"收到聊天完成请求: {request}")

        query_prompt = request.messages[-1].content
        logger.info(f"用户问题是: {query_prompt}")

        config = {"configurable": {"thread_id": request.userId+"@@"+request.conversationId, "user_id": request.userId}}
        logger.info(f"用户当前会话信息: {config}")

        prompt_template_system = PromptTemplate.from_file(PROMPT_TEMPLATE_TXT_SYS)
        prompt_template_user = PromptTemplate.from_file(PROMPT_TEMPLATE_TXT_USER)
        prompt = [
            {"role": "system", "content": prompt_template_system.template},
            {"role": "user", "content": prompt_template_user.template.format(query=query_prompt)}
        ]

        # 处理流式响应
        if request.stream:
            async def generate_stream():
                chunk_id = f"chatcmpl-{uuid.uuid4().hex}"
                async for message_chunk, metadata in graph.astream({"messages": prompt}, config, stream_mode="messages"):
                    chunk = message_chunk.content
                    logger.info(f"chunk: {chunk}")
                    # 在处理过程中产生每个块
                    yield f"data: {json.dumps({'id': chunk_id,'object': 'chat.completion.chunk','created': int(time.time()),'choices': [{'index': 0,'delta': {'content': chunk},'finish_reason': None}]})}\n\n"
                # 流结束的最后一块
                yield f"data: {json.dumps({'id': chunk_id,'object': 'chat.completion.chunk','created': int(time.time()),'choices': [{'index': 0,'delta': {},'finish_reason': 'stop'}]})}\n\n"
            # 返回fastapi.responses中StreamingResponse对象
            return StreamingResponse(generate_stream(), media_type="text/event-stream")

        # 处理非流式响应处理
        else:
            try:
                events = graph.stream({"messages": prompt}, config)
                for event in events:
                    for value in event.values():
                        result = value["messages"][-1].content
            except Exception as e:
                logger.info(f"Error processing response: {str(e)}")

            formatted_response = str(format_response(result))
            logger.info(f"格式化的搜索结果: {formatted_response}")

            response = ChatCompletionResponse(
                choices=[
                    ChatCompletionResponseChoice(
                        index=0,
                        message=Message(role="assistant", content=formatted_response),
                        finish_reason="stop"
                    )
                ]
            )
            logger.info(f"发送响应内容: \n{response}")
            # 返回fastapi.responses中JSONResponse对象
            # model_dump()方法通常用于将Pydantic模型实例的内容转换为一个标准的Python字典，以便进行序列化
            return JSONResponse(content=response.model_dump())
    except Exception as e:
        logger.error(f"处理聊天完成时出错:\n\n {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    logger.info(f"在端口 {PORT} 上启动服务器")
    uvicorn.run(app, host="0.0.0.0", port=PORT)