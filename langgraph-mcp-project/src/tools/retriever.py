import os
import psycopg2
from psycopg2.extras import RealDictCursor
from typing import Dict, List, Any, Optional, Union
from langchain_openai import OpenAIEmbeddings
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.callbacks.manager import CallbackManagerForToolRun
import numpy as np
from config.settings import (
    POSTGRES_HOST, POSTGRES_DB, POSTGRES_USER, 
    POSTGRES_PASSWORD, POSTGRES_PORT,
    OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_EMBEDDING_MODEL
)

class PostgresRetriever:
    """Vector database retriever for health records stored in PostgreSQL."""
    
    def __init__(self):
        # Connection details
        self.db_params = {
            "host": POSTGRES_HOST,
            "database": POSTGRES_DB,
            "user": POSTGRES_USER,
            "password": POSTGRES_PASSWORD,
            "port": POSTGRES_PORT,
        }
        
        # Initialize embeddings with custom configuration
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=OPENAI_API_KEY,
            openai_api_base=OPENAI_BASE_URL,
            model=OPENAI_EMBEDDING_MODEL,
        )
        
        self.table_name = "health_records"
        self.vector_dim = 1536  # Default to 1536
    
    def connect(self):
        """Establish connection to PostgreSQL"""
        try:
            conn = psycopg2.connect(**self.db_params)
            return conn
        except Exception as e:
            print(f"Error connecting to PostgreSQL: {e}")
            return None
    
    def get_vector_dimension(self):
        """Get the dimension of vectors in the database"""
        conn = self.connect()
        if not conn:
            return self.vector_dim
        
        try:
            cur = conn.cursor()
            
            # Get vector dimension
            cur.execute(f"""
                SELECT a.atttypmod
                FROM pg_attribute a
                JOIN pg_class c ON a.attrelid = c.oid
                JOIN pg_type t ON a.atttypid = t.oid
                WHERE c.relname = '{self.table_name}' 
                AND a.attname = 'embedding'
                AND t.typname = 'vector';
            """)
            
            result = cur.fetchone()
            if result:
                dimension = result[0]
                print(f"Using vector dimension from database: {dimension}")
                return dimension
            
            return self.vector_dim
        except Exception as e:
            print(f"Error getting vector dimension: {e}")
            return self.vector_dim
        finally:
            conn.close()
    
    def ensure_table_exists(self):
        """Ensure the health_records table exists"""
        conn = self.connect()
        if not conn:
            return False
        
        try:
            cur = conn.cursor()
            
            # Check if table exists
            cur.execute(f"SELECT to_regclass('public.{self.table_name}');")
            table_exists = cur.fetchone()[0]
            
            if table_exists:
                # Get the vector dimension
                self.vector_dim = self.get_vector_dimension()
                return True
            
            # If table doesn't exist, create it
            # Check if pgvector extension exists
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            
            # Create table with vector support if it doesn't exist
            cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.table_name} (
                id TEXT PRIMARY KEY,
                content TEXT,
                metadata JSONB,
                embedding VECTOR({self.vector_dim})
            );
            """)
            
            # Create index for faster similarity search
            cur.execute(f"""
            CREATE INDEX IF NOT EXISTS {self.table_name}_embedding_idx 
            ON {self.table_name} USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 100);
            """)
            
            conn.commit()
            return True
        except Exception as e:
            print(f"Error ensuring table exists: {e}")
            return False
        finally:
            conn.close()
    
    def similarity_search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Search for documents similar to the query"""
        # Ensure table exists before querying
        if not self.ensure_table_exists():
            return []
            
        query_embedding = self.embeddings.embed_query(query)
        
        # Adjust embedding dimension if needed
        if len(query_embedding) != self.vector_dim:
            print(f"Adjusting embedding dimension from {len(query_embedding)} to {self.vector_dim}")
            if len(query_embedding) < self.vector_dim:
                # Pad with zeros
                query_embedding = list(query_embedding) + [0.0] * (self.vector_dim - len(query_embedding))
            else:
                # Truncate
                query_embedding = query_embedding[:self.vector_dim]
        
        conn = self.connect()
        if not conn:
            return []
        
        try:
            cur = conn.cursor(cursor_factory=RealDictCursor)
            
            # Check if table has any rows
            cur.execute(f"SELECT COUNT(*) FROM {self.table_name}")
            count = cur.fetchone()['count']
            
            if count == 0:
                print(f"No records found in {self.table_name} table")
                return []
                
            # Format embedding as a PostgreSQL array string with square brackets
            embedding_str = '[' + ','.join([str(x) for x in query_embedding]) + ']'
            
            # Use cosine similarity in PostgreSQL
            query_sql = f"""
            SELECT id, content, metadata, 
                1 - (embedding <=> '{embedding_str}') as similarity
            FROM {self.table_name}
            ORDER BY similarity DESC
            LIMIT {top_k}
            """
            
            cur.execute(query_sql)
            results = cur.fetchall()
            
            return [dict(result) for result in results]
        except Exception as e:
            print(f"Error searching documents: {e}")
            return []
        finally:
            conn.close()


# LangGraph MCP Tool implementation
class RetrieverTool(Runnable):
    """LangGraph compatible retriever tool"""
    
    def __init__(self, top_k: int = 3):
        """Initialize the retriever tool"""
        self.retriever = PostgresRetriever()
        self.top_k = top_k
    
    def invoke(self, query: str, config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
        """Execute the retriever tool with the given query"""
        # 获取回调管理器（如果有）
        run_manager = None
        if config and config.get("callbacks"):
            run_manager = config["callbacks"]
        
        # 记录操作
        if run_manager:
            run_manager.on_text("Retrieving documents for query: " + query)
        
        # 从检索器获取结果
        results = self.retriever.similarity_search(query, self.top_k)
        
        # 格式化结果以提高可读性
        if results:
            formatted_results = []
            for i, result in enumerate(results):
                formatted_results.append(
                    f"Document {i+1} (similarity: {result['similarity']:.4f}):\n"
                    f"{result['content']}\n"
                )
            
            retriever_result = "\n".join(formatted_results)
            
            # 记录完成
            if run_manager:
                run_manager.on_text(f"Retrieved {len(results)} documents")
        else:
            retriever_result = "No relevant health records found."
            
            # 记录无结果
            if run_manager:
                run_manager.on_text("No documents retrieved")
        
        return {"retriever_result": retriever_result, "raw_results": results}

# 兼容旧代码的函数
def retriever_tool(state: Dict[str, Any]) -> Dict[str, Any]:
    """A retriever that searches health records in PostgreSQL (legacy API)"""
    query = state.get("query", "")
    
    tool = RetrieverTool()
    result = tool.invoke(query)
    
    return {"retriever_result": result["retriever_result"]}