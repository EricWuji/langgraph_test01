import os
import argparse
import psycopg2
from dotenv import load_dotenv
from src.graph.flow import build_graph
from src.utils.ingest import ingest_health_records
from src.tools.retriever import PostgresRetriever  # 添加此导入
from config.settings import (
    OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL, OPENAI_EMBEDDING_MODEL,
    POSTGRES_HOST, POSTGRES_DB, POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_PORT
)

# Load environment variables from .env file
load_dotenv()

def check_database_table():
    """Check the structure of the health_records table"""
    try:
        conn = psycopg2.connect(
            host=POSTGRES_HOST,
            database=POSTGRES_DB,
            user=POSTGRES_USER,
            password=POSTGRES_PASSWORD,
            port=POSTGRES_PORT
        )
        
        cur = conn.cursor()
        
        # Check if table exists
        cur.execute("SELECT to_regclass('public.health_records');")
        table_exists = cur.fetchone()[0]
        
        if not table_exists:
            print("Health records table does not exist. Creating...")
            return False
        
        # Check vector dimension
        cur.execute("""
            SELECT a.atttypmod
            FROM pg_attribute a
            JOIN pg_class c ON a.attrelid = c.oid
            JOIN pg_type t ON a.atttypid = t.oid
            WHERE c.relname = 'health_records' 
            AND a.attname = 'embedding'
            AND t.typname = 'vector';
        """)
        
        result = cur.fetchone()
        if result:
            dimension = result[0]
            print(f"Vector dimension: {dimension}")
            return True
        else:
            print("Could not determine vector dimension")
            return False
    
    except Exception as e:
        print(f"Database check error: {e}")
        return False
    finally:
        if conn:
            conn.close()

# 添加新函数用于直接搜索
def direct_search(query, top_k=5):
    """直接执行向量相似度搜索并返回结果"""
    print(f"\n执行向量相似度搜索: \"{query}\"")
    print(f"检索前 {top_k} 个最相关的结果...\n")
    
    retriever = PostgresRetriever()
    results = retriever.similarity_search(query, top_k)
    
    if not results:
        print("未找到相关健康记录。")
        return
    
    print(f"找到 {len(results)} 条相关记录:\n")
    for i, result in enumerate(results):
        print(f"结果 {i+1} (相似度: {result['similarity']:.4f}):")
        print(f"内容: {result['content']}")
        
        # 解析并显示元数据
        metadata = result.get('metadata', {})
        if isinstance(metadata, str):
            import json
            try:
                metadata = json.loads(metadata)
            except:
                metadata = {}
        
        source = metadata.get('source', '未知')
        page = metadata.get('page', '未知')
        print(f"来源: {source}, 页码: {page}")
        print("-" * 80)

def main():
    parser = argparse.ArgumentParser(description="Health Records LangGraph Agent")
    parser.add_argument("--ingest", action="store_true", help="Ingest health records into database")
    parser.add_argument("--query", type=str, help="Query to process", 
                        default="What's in 张三九 health records about blood pressure?")
    parser.add_argument("--check-db", action="store_true", help="Check database structure")
    # 添加新参数
    parser.add_argument("--search", type=str, help="Direct vector similarity search")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results to return")
    args = parser.parse_args()
    
    # Check if API key is set
    if not OPENAI_API_KEY:
        print("Warning: OPENAI_API_KEY environment variable is not set.")
        print("Please set it or create a .env file with OPENAI_API_KEY.")
    
    # Print configuration
    print(f"Using OpenAI configuration:")
    print(f"- Base URL: {OPENAI_BASE_URL}")
    print(f"- Model: {OPENAI_MODEL}")
    print(f"- Embedding Model: {OPENAI_EMBEDDING_MODEL}")
    print(f"- Using database: {POSTGRES_DB} on {POSTGRES_HOST}")
    
    # Check database if requested
    if args.check_db:
        check_database_table()
        return
    
    # Ingest health records if requested
    if args.ingest:
        print("Ingesting health records...")
        if ingest_health_records():
            print("Health records ingested successfully.")
        else:
            print("Failed to ingest health records.")
            return
    
    # 执行直接搜索（如果指定）
    if args.search:
        direct_search(args.search, args.top_k)
        return
    
    # Build the graph
    graph = build_graph()
    
    # Run the graph with the provided query
    print(f"\nProcessing query: {args.query}")
    result = graph.invoke({
        "query": args.query
    })
    
    print("\nFinal Result:")
    print(result.get("final_output"))

if __name__ == "__main__":
    main()