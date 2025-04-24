import os
import psycopg2
import uuid
import json
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config.settings import (
    POSTGRES_HOST, POSTGRES_DB, POSTGRES_USER, 
    POSTGRES_PASSWORD, POSTGRES_PORT,
    OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_EMBEDDING_MODEL
)

class PostgresIngestor:
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
        self.vector_dim = 1536  # Update to match existing database
    
    def connect(self):
        """Establish connection to PostgreSQL"""
        try:
            conn = psycopg2.connect(**self.db_params)
            return conn
        except Exception as e:
            print(f"Error connecting to PostgreSQL: {e}")
            return None
    
    def check_table_exists(self):
        """Check if the table already exists and get its vector dimension"""
        conn = self.connect()
        if not conn:
            return False, 0
        
        try:
            cur = conn.cursor()
            
            # Check if table exists
            cur.execute(f"SELECT to_regclass('public.{self.table_name}');")
            table_exists = cur.fetchone()[0]
            
            if table_exists:
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
                    print(f"Found existing table with vector dimension: {dimension}")
                    return True, dimension
            
            return False, 0
        except Exception as e:
            print(f"Error checking table: {e}")
            return False, 0
        finally:
            conn.close()
    
    def create_tables(self):
        """Create the necessary tables if they don't exist"""
        # First check if table exists and get its dimension
        table_exists, dimension = self.check_table_exists()
        if table_exists:
            self.vector_dim = dimension
            print(f"Using existing table with dimension {dimension}")
            return True
        
        conn = self.connect()
        if not conn:
            return False
        
        try:
            cur = conn.cursor()
            
            # Check if pgvector extension exists
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            
            # Create table with vector support
            cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.table_name} (
                id TEXT PRIMARY KEY,
                content TEXT,
                metadata JSONB,
                embedding vector({self.vector_dim})
            );
            """)
            
            # Create index for faster similarity search
            cur.execute(f"""
            CREATE INDEX IF NOT EXISTS {self.table_name}_embedding_idx 
            ON {self.table_name} USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 100);
            """)
            
            conn.commit()
            print(f"Created new table with dimension {self.vector_dim}")
            return True
        except Exception as e:
            print(f"Error creating tables: {e}")
            return False
        finally:
            conn.close()
    
    def ingest_document(self, file_path):
        """Ingest a document into the database"""
        # Create tables if they don't exist
        if not self.create_tables():
            return False
        
        # Initialize conn as None so it's always defined
        conn = None
        
        try:
            # Load and split PDF
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=100
            )
            splits = text_splitter.split_documents(documents)
            
            # Get connection
            conn = self.connect()
            if not conn:
                return False
            
            # Process and insert each chunk
            cur = conn.cursor()
            for doc in splits:
                # Generate embedding
                embedding = self.embeddings.embed_documents([doc.page_content])[0]
                
                # Check embedding dimension
                if len(embedding) != self.vector_dim:
                    print(f"Warning: Embedding dimension mismatch. Got {len(embedding)}, expected {self.vector_dim}")
                    
                    # Adjust embedding dimension if needed
                    if len(embedding) < self.vector_dim:
                        # Pad with zeros
                        embedding = list(embedding) + [0.0] * (self.vector_dim - len(embedding))
                    else:
                        # Truncate
                        embedding = embedding[:self.vector_dim]
                
                # Create metadata
                metadata = {
                    "source": file_path,
                    "page": doc.metadata.get("page", 0)
                }
                
                # Format embedding as a PostgreSQL array string with square brackets
                embedding_str = '[' + ','.join([str(x) for x in embedding]) + ']'
                
                # Insert into database
                cur.execute(
                    f"""
                    INSERT INTO {self.table_name} (id, content, metadata, embedding)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (id) DO UPDATE
                    SET content = EXCLUDED.content,
                        metadata = EXCLUDED.metadata,
                        embedding = EXCLUDED.embedding
                    """,
                    (
                        str(uuid.uuid4()),
                        doc.page_content,
                        json.dumps(metadata),
                        embedding_str
                    )
                )
            
            conn.commit()
            print(f"Successfully ingested {len(splits)} chunks from {file_path}")
            return True
        except Exception as e:
            print(f"Error ingesting document: {e}")
            if conn:
                conn.rollback()
            return False
        finally:
            if conn:
                conn.close()

def ingest_health_records():
    """Ingest health records from the input directory"""
    ingestor = PostgresIngestor()
    
    # Get the correct project root path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "../.."))
    
    # Path to the health records
    health_records_path = os.path.join(project_root, "input", "健康档案.pdf")
    
    print(f"Looking for health records at: {health_records_path}")
    
    if not os.path.exists(health_records_path):
        print(f"Error: Health records file not found at {health_records_path}")
        return False
    
    return ingestor.ingest_document(health_records_path)