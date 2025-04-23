import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from typing import Optional
import logging
from dotenv import load_dotenv
load_dotenv()

# Author:@南哥AGI研习社 (B站 or YouTube 搜索“南哥AGI研习社”)


# 设置日志模版
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MODEL = os.getenv("MODEL")
BASE_URL = os.getenv("BASE_URL")
API_KEY = os.getenv("API_KEY")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
llm = ChatOpenAI(base_url=BASE_URL, api_key=API_KEY, model=MODEL)
embeddings = OpenAIEmbeddings(base_url=BASE_URL, api_key=API_KEY, model=EMBEDDING_MODEL)