import os
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Body
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import tempfile
import shutil
import uvicorn
from src.utils.build_database import DatabaseBuilder
from src.graph.flow import build_graph
from src.tools.retriever import PostgresRetriever
from config.settings import (
    OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL, OPENAI_EMBEDDING_MODEL,
    POSTGRES_HOST, POSTGRES_DB
)

app = FastAPI(
    title="Health Records LangGraph API",
    description="API for health records ingestion and querying using LangGraph",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Create an instance of the graph at startup
@app.on_event("startup")
async def startup_event():
    print(f"Using OpenAI configuration:")
    print(f"- Base URL: {OPENAI_BASE_URL}")
    print(f"- Model: {OPENAI_MODEL}")
    print(f"- Embedding Model: {OPENAI_EMBEDDING_MODEL}")
    print(f"- Using database: {POSTGRES_DB} on {POSTGRES_HOST}")
    
    # Ensure table exists
    retriever = PostgresRetriever()
    retriever.ensure_table_exists()

class QueryRequest(BaseModel):
    query: str

class SearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = 3

class IngestResponse(BaseModel):
    success: bool
    message: str
    document_count: Optional[int] = None

class QueryResponse(BaseModel):
    query: str
    result: str

class SearchResult(BaseModel):
    content: str
    similarity: float
    page: Optional[int] = None
    source: Optional[str] = None

class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]

@app.post("/ingest", response_model=IngestResponse)
async def ingest_document(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """Upload and ingest a PDF document into the vector database."""
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
        # Copy uploaded file to the temporary file
        shutil.copyfileobj(file.file, tmp)
        temp_path = tmp.name
    
    try:
        # Process the document
        db_builder = DatabaseBuilder()
        
        # Ensure database is set up
        if not db_builder.setup_database():
            raise HTTPException(status_code=500, detail="Failed to set up database")
        
        # Process PDF
        documents = db_builder.process_pdf(temp_path)
        
        # Embed and store
        success = db_builder.embed_and_store(documents)
        
        if success:
            return IngestResponse(
                success=True, 
                message=f"Document '{file.filename}' successfully ingested",
                document_count=len(documents)
            )
        else:
            raise HTTPException(status_code=500, detail="Failed to ingest document")
    
    finally:
        # Clean up the temporary file in the background
        background_tasks.add_task(os.unlink, temp_path)

@app.post("/query", response_model=QueryResponse)
async def query_health_records(request: QueryRequest):
    """Query the health records using the LangGraph agent."""
    if not request.query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    # Initialize the graph
    graph = build_graph()
    
    # Process the query
    result = graph.invoke({
        "query": request.query
    })
    
    return QueryResponse(
        query=request.query,
        result=result.get("final_output", "No result generated")
    )

@app.post("/search", response_model=SearchResponse)
async def search_documents(request: SearchRequest):
    """Direct search in the vector database."""
    if not request.query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    retriever = PostgresRetriever()
    results = retriever.similarity_search(request.query, request.top_k)
    
    if not results:
        return SearchResponse(
            query=request.query,
            results=[]
        )
    
    formatted_results = []
    for result in results:
        metadata = result.get("metadata", {})
        if isinstance(metadata, str):
            import json
            try:
                metadata = json.loads(metadata)
            except:
                metadata = {}
        
        formatted_results.append(
            SearchResult(
                content=result.get("content", ""),
                similarity=result.get("similarity", 0),
                page=metadata.get("page"),
                source=metadata.get("source")
            )
        )
    
    return SearchResponse(
        query=request.query,
        results=formatted_results
    )

@app.get("/health")
async def health_check():
    """API health check endpoint."""
    return {"status": "healthy"}

@app.get("/config")
async def get_config():
    """Get the current API configuration."""
    return {
        "openai_base_url": OPENAI_BASE_URL,
        "openai_model": OPENAI_MODEL,
        "openai_embedding_model": OPENAI_EMBEDDING_MODEL,
        "postgres_host": POSTGRES_HOST,
        "postgres_db": POSTGRES_DB,
    }

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)