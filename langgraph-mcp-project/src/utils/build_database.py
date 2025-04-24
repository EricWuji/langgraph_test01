import os
import psycopg2
import uuid
import json
import sys
from pathlib import Path
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from config.settings import (
    POSTGRES_HOST, POSTGRES_DB, POSTGRES_USER, 
    POSTGRES_PASSWORD, POSTGRES_PORT,
    OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_EMBEDDING_MODEL
)

class DatabaseBuilder:
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
        
    def connect(self):
        """Establish connection to PostgreSQL"""
        try:
            conn = psycopg2.connect(**self.db_params)
            return conn
        except Exception as e:
            print(f"Error connecting to PostgreSQL: {e}")
            return None
    
    def setup_database(self):
        """Set up the database and create necessary tables"""
        conn = self.connect()
        if not conn:
            return False
        
        try:
            cur = conn.cursor()
            
            # Check if pgvector extension exists
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            
            # Drop existing table if requested
            cur.execute(f"DROP TABLE IF EXISTS {self.table_name};")
            
            # Create table with vector support
            cur.execute(f"""
            CREATE TABLE {self.table_name} (
                id TEXT PRIMARY KEY,
                content TEXT,
                metadata JSONB,
                embedding VECTOR(1024)
            );
            """)
            
            # Create index for faster similarity search
            cur.execute(f"""
            CREATE INDEX {self.table_name}_embedding_idx 
            ON {self.table_name} USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 100);
            """)
            
            conn.commit()
            print(f"Database setup complete. Table '{self.table_name}' created successfully.")
            return True
        except Exception as e:
            print(f"Error setting up database: {e}")
            conn.rollback()
            return False
        finally:
            conn.close()
    
    def process_pdf(self, pdf_path):
        """Process a PDF file and extract text chunks"""
        try:
            print(f"Loading PDF from {pdf_path}...")
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            
            print(f"Splitting document into chunks...")
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            splits = text_splitter.split_documents(documents)
            
            print(f"Created {len(splits)} text chunks from the PDF.")
            return splits
        except Exception as e:
            print(f"Error processing PDF: {e}")
            return []
    
    def embed_and_store(self, documents):
        """Generate embeddings and store documents in PostgreSQL"""
        if not documents:
            print("No documents to embed.")
            return False
        
        conn = self.connect()
        if not conn:
            return False
        
        try:
            cur = conn.cursor()
            
            print(f"Generating embeddings and storing documents in PostgreSQL...")
            for i, doc in enumerate(documents):
                if (i+1) % 5 == 0 or i+1 == len(documents):
                    print(f"Processing document {i+1}/{len(documents)}...")
                
                # Generate embedding
                embedding = self.embeddings.embed_documents([doc.page_content])[0]
                
                # Create metadata
                metadata = {
                    "source": doc.metadata.get("source", "unknown"),
                    "page": doc.metadata.get("page", 0)
                }
                
                # Insert into database
                cur.execute(
                    f"""
                    INSERT INTO {self.table_name} (id, content, metadata, embedding)
                    VALUES (%s, %s, %s, %s)
                    """,
                    (
                        str(uuid.uuid4()),
                        doc.page_content,
                        json.dumps(metadata),
                        embedding
                    )
                )
            
            conn.commit()
            print(f"Successfully ingested {len(documents)} documents into PostgreSQL.")
            return True
        except Exception as e:
            print(f"Error storing documents: {e}")
            conn.rollback()
            return False
        finally:
            conn.close()

def build_health_records_database():
    """Build the health records database from PDF file"""
    builder = DatabaseBuilder()
    
    # Get absolute path to PDF file
    input_dir = os.path.join(project_root, "input")
    pdf_path = os.path.join(input_dir, "健康档案.pdf")
    
    if not os.path.exists(pdf_path):
        print(f"Error: Health records file not found at {pdf_path}")
        print(f"Current directory: {os.getcwd()}")
        print(f"Looking for file at: {pdf_path}")
        return False
    
    # Set up the database
    print("Setting up database...")
    if not builder.setup_database():
        return False
    
    # Process the PDF
    documents = builder.process_pdf(pdf_path)
    
    # Embed and store documents
    return builder.embed_and_store(documents)

if __name__ == "__main__":
    print("Building health records database...")
    success = build_health_records_database()
    if success:
        print("Database build completed successfully.")
    else:
        print("Database build failed.")