import os
import json
import uuid
import logging
from typing import Any, Dict, List, Optional, Tuple, Sequence, Iterable, AsyncIterable
import psycopg2
from psycopg2.extras import Json, execute_batch
from langgraph.store.base import BaseStore
from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)

class PostgreSQLStore(BaseStore):
    """PostgreSQL implementation of the BaseStore."""

    def __init__(
        self, 
        connection_string: str,
        index: Optional[Dict[str, Any]] = None,
        table_name: str = "langgraph_store"
    ):
        """Initialize with PostgreSQL connection parameters.
        
        Args:
            connection_string: PostgreSQL connection string
            index: Optional dictionary with embedding configuration
            table_name: Name of the table to use for storage
        """
        self.connection_string = connection_string
        self.table_name = table_name
        self._index = index
        self._setup_tables()

    def _setup_tables(self) -> None:
        """Create the necessary tables if they don't exist."""
        conn = None
        try:
            conn = psycopg2.connect(self.connection_string)
            cursor = conn.cursor()
            
            # Create vector extension if not exists
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            
            # Create the main storage table
            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    namespace TEXT,
                    key TEXT,
                    value JSONB,
                    embedding VECTOR(1024),
                    PRIMARY KEY (namespace, key)
                );
            """)
            
            # Create index for efficient embedding similarity search
            if self._index:
                cursor.execute(f"""
                    CREATE INDEX IF NOT EXISTS {self.table_name}_embedding_idx 
                    ON {self.table_name} USING ivfflat (embedding vector_cosine_ops)
                    WITH (lists = 100);
                """)
            
            conn.commit()
            logger.info(f"PostgreSQL tables setup complete for {self.table_name}")
        except Exception as e:
            logger.error(f"Failed to setup PostgreSQL tables: {e}")
            if conn:
                conn.rollback()
            raise
        finally:
            if conn:
                conn.close()

    def get(self, namespace: Tuple[str, ...], key: str) -> Optional[Dict[str, Any]]:
        """Get a value from the store."""
        conn = None
        try:
            conn = psycopg2.connect(self.connection_string)
            cursor = conn.cursor()
            
            ns = "/".join(namespace)
            cursor.execute(
                f"SELECT value FROM {self.table_name} WHERE namespace = %s AND key = %s",
                (ns, key)
            )
            result = cursor.fetchone()
            
            if result:
                return result[0]
            return None
        except Exception as e:
            logger.error(f"Error retrieving from PostgreSQL: {e}")
            return None
        finally:
            if conn:
                conn.close()

    def put(self, namespace: Tuple[str, ...], key: str, value: Dict[str, Any]) -> None:
        """Put a value into the store."""
        conn = None
        try:
            conn = psycopg2.connect(self.connection_string)
            cursor = conn.cursor()
            
            ns = "/".join(namespace)
            embedding = None
            
            # Generate embedding if configured
            if self._index and "embed" in self._index:
                embedder: Embeddings = self._index["embed"]
                if isinstance(value.get("data", ""), str):
                    embedding = embedder.embed_query(value.get("data", ""))
            
            # Upsert the value
            cursor.execute(
                f"""
                INSERT INTO {self.table_name} (namespace, key, value, embedding)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (namespace, key) DO UPDATE
                SET value = EXCLUDED.value, embedding = EXCLUDED.embedding
                """,
                (ns, key, Json(value), embedding)
            )
            
            conn.commit()
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Error storing in PostgreSQL: {e}")
            raise
        finally:
            if conn:
                conn.close()

    def batch(self, ops: List[Dict[str, Any]]) -> List[Optional[Dict[str, Any]]]:
        """Execute multiple operations in batch."""
        conn = None
        results: List[Optional[Dict[str, Any]]] = []
        
        try:
            conn = psycopg2.connect(self.connection_string)
            cursor = conn.cursor()
            
            for op in ops:
                op_type = op.get("op")
                namespace = op.get("namespace", tuple())
                ns = "/".join(namespace)
                key = op.get("key", "")
                
                if op_type == "get":
                    cursor.execute(
                        f"SELECT value FROM {self.table_name} WHERE namespace = %s AND key = %s",
                        (ns, key)
                    )
                    result = cursor.fetchone()
                    results.append(result[0] if result else None)
                
                elif op_type == "put":
                    value = op.get("value", {})
                    embedding = None
                    
                    # Generate embedding if configured
                    if self._index and "embed" in self._index:
                        embedder: Embeddings = self._index["embed"]
                        if isinstance(value.get("data", ""), str):
                            embedding = embedder.embed_query(value.get("data", ""))
                    
                    cursor.execute(
                        f"""
                        INSERT INTO {self.table_name} (namespace, key, value, embedding)
                        VALUES (%s, %s, %s, %s)
                        ON CONFLICT (namespace, key) DO UPDATE
                        SET value = EXCLUDED.value, embedding = EXCLUDED.embedding
                        """,
                        (ns, key, Json(value), embedding)
                    )
                    results.append(None)  # put operations don't return values
                
                elif op_type == "delete":
                    cursor.execute(
                        f"DELETE FROM {self.table_name} WHERE namespace = %s AND key = %s",
                        (ns, key)
                    )
                    results.append(None)  # delete operations don't return values
            
            conn.commit()
            return results
        
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Error executing batch operations in PostgreSQL: {e}")
            raise
        
        finally:
            if conn:
                conn.close()

    async def abatch(self, ops: List[Dict[str, Any]]) -> List[Optional[Dict[str, Any]]]:
        """Execute multiple operations asynchronously in batch.
        
        Since psycopg2 doesn't support async operations directly, we just call the synchronous version.
        For true async, you'd need to use an async PostgreSQL driver like asyncpg.
        """
        return self.batch(ops)

    def delete(self, namespace: Tuple[str, ...], key: str) -> None:
        """Delete a value from the store."""
        conn = None
        try:
            conn = psycopg2.connect(self.connection_string)
            cursor = conn.cursor()
            
            ns = "/".join(namespace)
            cursor.execute(
                f"DELETE FROM {self.table_name} WHERE namespace = %s AND key = %s",
                (ns, key)
            )
            
            conn.commit()
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Error deleting from PostgreSQL: {e}")
            raise
        finally:
            if conn:
                conn.close()

    def search(
        self, namespace: Tuple[str, ...], query: str, limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Search for similar items using vector embeddings."""
        if not self._index or "embed" not in self._index:
            logger.warning("Search attempted but no embeddings configured")
            return []
            
        conn = None
        try:
            conn = psycopg2.connect(self.connection_string)
            cursor = conn.cursor()
            
            embedder: Embeddings = self._index["embed"]
            query_embedding = embedder.embed_query(query)
            
            ns = "/".join(namespace)
            
            # Convert the embedding to the proper PostgreSQL vector format
            embedding_str = f"[{','.join(map(str, query_embedding))}]"
            
            # Use the vector constructor syntax
            cursor.execute(
                f"""
                SELECT key, value, embedding <=> '{embedding_str}'::vector AS distance
                FROM {self.table_name}
                WHERE namespace = %s AND embedding IS NOT NULL
                ORDER BY distance
                LIMIT %s
                """,
                (ns, limit)
            )
            
            results = []
            for key, value, distance in cursor.fetchall():
                results.append({
                    "id": key,
                    "value": value,
                    "metadata": {"distance": distance}
                })
            
            return results
        except Exception as e:
            logger.error(f"Error searching in PostgreSQL: {e}")
            return []
        finally:
            if conn:
                conn.close()
                
    def clear(self, namespace: Optional[Tuple[str, ...]] = None) -> None:
        """Clear data from the store."""
        conn = None
        try:
            conn = psycopg2.connect(self.connection_string)
            cursor = conn.cursor()
            
            if namespace:
                ns = "/".join(namespace)
                cursor.execute(
                    f"DELETE FROM {self.table_name} WHERE namespace = %s",
                    (ns,)
                )
            else:
                cursor.execute(f"TRUNCATE TABLE {self.table_name}")
            
            conn.commit()
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Error clearing PostgreSQL store: {e}")
            raise
        finally:
            if conn:
                conn.close()