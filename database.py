import psycopg2
from psycopg2.extensions import register_adapter, AsIs
import json
import numpy as np

def adapt_numpy_array(arr):
    embedding_str = "[" + ",".join(map(str, arr.tolist())) + "]"
    return AsIs(f"'{embedding_str}'")

def adapt_list(lst):
    embedding_str = "[" + ",".join(map(str, lst)) + "]"
    return AsIs(f"'{embedding_str}'")

register_adapter(np.ndarray, adapt_numpy_array)
register_adapter(list, adapt_list)  # Add adapter for Python lists

class DatabaseManager:
    def __init__(self, db_params):
        self.db_params = db_params
        self._setup_database()
    
    def _setup_database(self):
        """Set up database connection and tables"""
        try:
            print(f"[DatabaseManager] Connecting to database with params: {self.db_params}")
            self.conn = psycopg2.connect(**self.db_params)
            self.cursor = self.conn.cursor()
            print("[DatabaseManager] Database connection established")
            
            # Create vector extension if it doesn't exist
            print("[DatabaseManager] Creating vector extension if needed")
            self.cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
            
            # Create memories table without metadata
            print("[DatabaseManager] Creating memories table if needed")
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id SERIAL PRIMARY KEY,
                    context TEXT,
                    embedding vector(384),
                    timestamp TIMESTAMP DEFAULT NOW(),
                    source VARCHAR(255)
                );
            """)
            
            # Create tools_embeddings table for tool information and embeddings
            print("[DatabaseManager] Creating tools_embeddings table if needed")
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS tools_embeddings (
                    id SERIAL PRIMARY KEY,
                    tool_name VARCHAR(100) UNIQUE,
                    description TEXT,
                    embedding vector(384),
                    updated_at TIMESTAMP DEFAULT NOW()
                );
            """)
            
            # Create index on embedding for faster similarity search
            print("[DatabaseManager] Creating embedding index if needed")
            self.cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_memories_embedding 
                ON memories 
                USING ivfflat (embedding vector_l2_ops) 
                WITH (lists = 100);
            """)
            
            # Create index on tool embeddings for faster similarity search
            print("[DatabaseManager] Creating tool embedding index if needed")
            self.cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_tools_embedding 
                ON tools_embeddings 
                USING ivfflat (embedding vector_l2_ops) 
                WITH (lists = 100);
            """)
            
            # Create index on timestamp for faster range queries
            print("[DatabaseManager] Creating timestamp index if needed")
            self.cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_memories_timestamp 
                ON memories (timestamp);
            """)
            
            self.conn.commit()
            print("[DatabaseManager] Database setup completed successfully")
        except Exception as e:
            import traceback
            print(f"[DatabaseManager] Error setting up database: {str(e)}")
            print(f"[DatabaseManager] Traceback: {traceback.format_exc()}")
            raise
    
    def insert_memory(self, context, embedding, source):
        """Insert a memory into the database"""
        try:
            print(f"[DatabaseManager] Inserting memory from source: {source}")
            self.cursor.execute(
                """INSERT INTO memories 
                   (context, embedding, source) 
                   VALUES (%s, %s::vector, %s);""",
                (context, embedding, source)
            )
            self.conn.commit()
            print("[DatabaseManager] Memory inserted successfully")
        except Exception as e:
            import traceback
            print(f"[DatabaseManager] Error inserting memory: {str(e)}")
            print(f"[DatabaseManager] Traceback: {traceback.format_exc()}")
            self.conn.rollback()  # Rollback in case of error
            raise
    
    def retrieve_memories(self, query_embedding, top_k=3, category=None):
        """Retrieve relevant memories from the database"""
        try:
            print(f"[DatabaseManager] Retrieving top {top_k} memories")
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
            results = self.cursor.fetchall()
            print(f"[DatabaseManager] Retrieved {len(results)} memories")
            return results
        except Exception as e:
            import traceback
            print(f"[DatabaseManager] Error retrieving memories: {str(e)}")
            print(f"[DatabaseManager] Traceback: {traceback.format_exc()}")
            return []  # Return empty list in case of error
    
    def get_unique_sources(self):
        """Retrieve unique sources from the database"""
        try:
            print("[DatabaseManager] Getting unique sources")
            self.cursor.execute("SELECT DISTINCT source FROM memories;")
            sources = [row[0] for row in self.cursor.fetchall()]
            print(f"[DatabaseManager] Found {len(sources)} unique sources")
            return sources
        except Exception as e:
            import traceback
            print(f"[DatabaseManager] Error getting unique sources: {str(e)}")
            print(f"[DatabaseManager] Traceback: {traceback.format_exc()}")
            return []  # Return empty list in case of error

    def delete_memories_by_source(self, source):
        """Delete memories from the database by source"""
        try:
            print(f"[DatabaseManager] Deleting memories for source: {source}")
            self.cursor.execute("DELETE FROM memories WHERE source = %s;", (source,))
            self.conn.commit()
            print("[DatabaseManager] Memories deleted successfully")
        except Exception as e:
            import traceback
            print(f"[DatabaseManager] Error deleting memories: {str(e)}")
            print(f"[DatabaseManager] Traceback: {traceback.format_exc()}")
            self.conn.rollback()  # Rollback in case of error
            raise
    
    def clear_memories(self):
        """Delete all memories from the database"""
        try:
            print("[DatabaseManager] Clearing all memories")
            self.cursor.execute("DELETE FROM memories;")
            self.conn.commit()
            print("[DatabaseManager] All memories cleared successfully")
        except Exception as e:
            import traceback
            print(f"[DatabaseManager] Error clearing memories: {str(e)}")
            print(f"[DatabaseManager] Traceback: {traceback.format_exc()}")
            self.conn.rollback()  # Rollback in case of error
            raise
            
    def store_tool_embedding(self, tool_name, description, embedding):
        """Store tool information and embedding in the database"""
        try:
            print(f"[DatabaseManager] Storing embedding for tool: {tool_name}")
            # Use UPSERT (INSERT ... ON CONFLICT UPDATE) pattern for tools
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
            self.conn.commit()
            print(f"[DatabaseManager] Tool embedding stored successfully for: {tool_name}")
        except Exception as e:
            import traceback
            print(f"[DatabaseManager] Error storing tool embedding: {str(e)}")
            print(f"[DatabaseManager] Traceback: {traceback.format_exc()}")
            self.conn.rollback()
            raise
            
    def find_similar_tools(self, query_embedding, similarity_threshold=0.75, top_k=3):
        """Find tools similar to the query embedding"""
        try:
            print(f"[DatabaseManager] Finding similar tools with threshold {similarity_threshold}")
            # Lower distance = higher similarity, so we invert the comparison
            # Use a combination of similarity threshold and top_k
            self.cursor.execute(
                """SELECT tool_name, description, 1 - (embedding <-> %s::vector) AS similarity
                   FROM tools_embeddings
                   WHERE 1 - (embedding <-> %s::vector) > %s
                   ORDER BY similarity DESC
                   LIMIT %s;""",
                (query_embedding, query_embedding, similarity_threshold, top_k)
            )
            results = self.cursor.fetchall()
            print(f"[DatabaseManager] Found {len(results)} similar tools")
            return results  # Returns [(tool_name, description, similarity), ...]
        except Exception as e:
            import traceback
            print(f"[DatabaseManager] Error finding similar tools: {str(e)}")
            print(f"[DatabaseManager] Traceback: {traceback.format_exc()}")
            return []  # Return empty list in case of error
            
    def get_all_tools(self):
        """Retrieve all stored tools"""
        try:
            print("[DatabaseManager] Retrieving all tools")
            self.cursor.execute("SELECT tool_name, description FROM tools_embeddings;")
            results = self.cursor.fetchall()
            print(f"[DatabaseManager] Found {len(results)} tools")
            return results
        except Exception as e:
            import traceback
            print(f"[DatabaseManager] Error retrieving all tools: {str(e)}")
            print(f"[DatabaseManager] Traceback: {traceback.format_exc()}")
            return []  # Return empty list in case of error
            
    def delete_tool(self, tool_name):
        """Remove a tool from the database"""
        try:
            print(f"[DatabaseManager] Deleting tool: {tool_name}")
            self.cursor.execute("DELETE FROM tools_embeddings WHERE tool_name = %s;", (tool_name,))
            self.conn.commit()
            print(f"[DatabaseManager] Tool {tool_name} deleted successfully")
        except Exception as e:
            import traceback
            print(f"[DatabaseManager] Error deleting tool: {str(e)}")
            print(f"[DatabaseManager] Traceback: {traceback.format_exc()}")
            self.conn.rollback()
            raise
            
    def __del__(self):
        """Clean up database connection on object destruction"""
        try:
            if hasattr(self, 'cursor') and self.cursor:
                self.cursor.close()
            if hasattr(self, 'conn') and self.conn:
                self.conn.close()
            print("[DatabaseManager] Database connection closed")
        except Exception as e:
            print(f"[DatabaseManager] Error closing database connection: {str(e)}")

