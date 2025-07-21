import psycopg2
from psycopg2.extensions import register_adapter, AsIs
import json
import numpy as np
import logging
import traceback
from typing import List, Dict, Any, Optional, Tuple

# Configure logger
logger = logging.getLogger('DatabaseManager')
logging.basicConfig(level=logging.INFO)

def adapt_numpy_array(arr: np.ndarray) -> AsIs:
    """Adapt numpy array to a format that psycopg2 can understand."""
    embedding_str = "[" + ",".join(map(str, arr.tolist())) + "]"
    return AsIs(f"'{embedding_str}'")

def adapt_list(lst: List[float]) -> AsIs:
    """Adapt list to a format that psycopg2 can understand."""
    embedding_str = "[" + ",".join(map(str, lst)) + "]"
    return AsIs(f"'{embedding_str}'")

register_adapter(np.ndarray, adapt_numpy_array)
register_adapter(list, adapt_list)  # Add adapter for Python lists

class DatabaseManager:
    """A context manager for handling database connections and operations."""

    def __init__(self, db_params: Dict[str, str]):
        """
        Initialize the DatabaseManager.

        Args:
            db_params: A dictionary of database connection parameters.
        """
        self.db_params = db_params
        self.conn = None
        self.cursor = None

    def __enter__(self) -> 'DatabaseManager':
        """
        Enter the context manager, establishing a database connection.

        Returns:
            The DatabaseManager instance.
        """
        try:
            self.conn = psycopg2.connect(**self.db_params)
            self.cursor = self.conn.cursor()
            logger.info("Database connection established")
            self._setup_database()
            return self
        except Exception as e:
            logger.error(f"Error connecting to database: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context manager, closing the database connection."""
        if self.conn:
            if exc_type:
                self.conn.rollback()
            else:
                self.conn.commit()
            self.cursor.close()
            self.conn.close()
            logger.info("Database connection closed")

    def _setup_database(self):
        """Set up database tables and extensions."""
        self.cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id SERIAL PRIMARY KEY,
                context TEXT,
                embedding vector(384),
                timestamp TIMESTAMP DEFAULT NOW(),
                source VARCHAR(255)
            );
        """)
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS tools_embeddings (
                id SERIAL PRIMARY KEY,
                tool_name VARCHAR(100) UNIQUE,
                description TEXT,
                embedding vector(384),
                updated_at TIMESTAMP DEFAULT NOW()
            );
        """)
        self.cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_memories_embedding 
            ON memories 
            USING ivfflat (embedding vector_l2_ops) 
            WITH (lists = 100);
        """)
        self.cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_tools_embedding 
            ON tools_embeddings 
            USING ivfflat (embedding vector_l2_ops) 
            WITH (lists = 100);
        """)
        self.cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_memories_timestamp 
            ON memories (timestamp);
        """)
        self.conn.commit()

    def insert_memory(self, context: str, embedding: List[float], source: str):
        """Insert a memory into the database."""
        self.cursor.execute(
            """INSERT INTO memories 
               (context, embedding, source) 
               VALUES (%s, %s::vector, %s);""",
            (context, embedding, source)
        )

    def retrieve_memories(self, query_embedding: List[float], top_k: int = 3, category: Optional[str] = None) -> List[Tuple[str, float]]:
        """Retrieve relevant memories from the database."""
        sql_query = """
            SELECT context, 
            embedding <-> %s::vector AS distance 
            FROM memories
        """
        params = [query_embedding]
        
        if category:
            sql_query += " WHERE source = %s"
            params.append(category)
        
        sql_query += " ORDER BY distance LIMIT %s;"
        params.append(top_k)
        
        self.cursor.execute(sql_query, params)
        return self.cursor.fetchall()

    def get_unique_sources(self) -> List[str]:
        """Retrieve unique sources from the database."""
        self.cursor.execute("SELECT DISTINCT source FROM memories;")
        return [row[0] for row in self.cursor.fetchall()]

    def delete_memories_by_source(self, source: str):
        """Delete memories from the database by source."""
        self.cursor.execute("DELETE FROM memories WHERE source = %s;", (source,))

    def clear_memories(self):
        """Delete all memories from the database."""
        self.cursor.execute("DELETE FROM memories;")

    def store_tool_embedding(self, tool_name: str, description: str, embedding: List[float]):
        """Store tool information and embedding in the database."""
        self.cursor.execute(
            """INSERT INTO tools_embeddings 
               (tool_name, description, embedding, updated_at) 
               VALUES (%s, %s, %s::vector, NOW())
               ON CONFLICT (tool_name) 
               DO UPDATE SET 
                   description = EXCLUDED.description,
                   embedding = EXCLUDED.embedding,
                   updated_at = NOW();""",
            (tool_name, description, embedding)
        )

    def find_similar_tools(self, query_embedding: List[float], similarity_threshold: float = 0.8, top_k: int = 3) -> List[Tuple[str, str, float]]:
        """Find tools similar to the query embedding."""
        self.cursor.execute(
            """SELECT tool_name, description, 2 - (embedding <-> %s::vector) AS similarity
               FROM tools_embeddings
               WHERE 2 - (embedding <-> %s::vector) > %s
               ORDER BY similarity DESC
               LIMIT %s;""",
            (query_embedding, query_embedding, similarity_threshold, top_k)
        )
        return self.cursor.fetchall()

    def get_all_tools(self) -> List[Tuple[str, str]]:
        """Retrieve all stored tools."""
        self.cursor.execute("SELECT tool_name, description FROM tools_embeddings;")
        return self.cursor.fetchall()

    def delete_tool(self, tool_name: str):
        """Remove a tool from the database."""
        self.cursor.execute("DELETE FROM tools_embeddings WHERE tool_name = %s;", (tool_name,))

    def find_most_similar_tool(self, query_embedding: List[float], similarity_threshold: float = 0.75) -> Optional[Tuple[str, float]]:
        """Find the most similar tool to the query embedding."""
        similar_tools = self.find_similar_tools(query_embedding, similarity_threshold, top_k=1)
        if similar_tools:
            tool_name, description, similarity = similar_tools[0]
            return (tool_name, similarity)
        return None

