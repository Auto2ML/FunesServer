import psycopg2
from psycopg2.extensions import register_adapter, AsIs
import json
import numpy as np

def adapt_numpy_array(arr):
    embedding_str = "[" + ",".join(map(str, arr.tolist())) + "]"
    return AsIs(f"'{embedding_str}'")

register_adapter(np.ndarray, adapt_numpy_array)

class DatabaseManager:
    def __init__(self, db_params):
        self.db_params = db_params
        self._setup_database()
    
    def _setup_database(self):
        """Set up database connection and tables"""
        self.conn = psycopg2.connect(**self.db_params)
        self.cursor = self.conn.cursor()
        
        # Create vector extension if it doesn't exist
        self.cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
        
        # Create memories table without metadata
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id SERIAL PRIMARY KEY,
                context TEXT,
                embedding vector(384),
                timestamp TIMESTAMP DEFAULT NOW(),
                source VARCHAR(255)
            );
        """)
        
        # Create index on embedding for faster similarity search
        self.cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_memories_embedding 
            ON memories 
            USING ivfflat (embedding vector_l2_ops) 
            WITH (lists = 100);
        """)
        
        # Create index on timestamp for faster range queries
        self.cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_memories_timestamp 
            ON memories (timestamp);
        """)
        
      
        self.conn.commit()
    
    def insert_memory(self, context, embedding, source):
        """Insert a memory into the database"""
        self.cursor.execute(
            """INSERT INTO memories 
               (context, embedding, source) 
               VALUES (%s, %s::vector, %s);""",
            (context, embedding, source)
        )
        self.conn.commit()
    
    def retrieve_memories(self, query_embedding, top_k=3):
        """Retrieve relevant memories from the database"""
        sql_query = """
            SELECT context, 
            embedding <-> %s::vector AS distance 
            FROM memories
        """
        params = [query_embedding]
        
        sql_query += " ORDER BY distance LIMIT %s;"
        params.append(top_k)
        
        self.cursor.execute(sql_query, params)
        return self.cursor.fetchall()
    
    def get_unique_sources(self):
        """Retrieve unique sources from the database"""
        self.cursor.execute("SELECT DISTINCT source FROM memories;")
        return [row[0] for row in self.cursor.fetchall()]

    def delete_memories_by_source(self, source):
        """Delete memories from the database by source"""
        self.cursor.execute("DELETE FROM memories WHERE source = %s;", (source,))
        self.conn.commit()

