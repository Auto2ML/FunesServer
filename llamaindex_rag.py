"""
LlamaIndex RAG integration for Funes

This module provides integration between Funes' existing RAG system and LlamaIndex.
"""

import os
import traceback
from typing import List, Dict, Any, Optional
from config import DB_CONFIG, EMBEDDING_CONFIG, LLAMAINDEX_CONFIG
import docling

from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.core.storage import StorageContext
from llama_index.core.schema import TextNode
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


class LlamaIndexRAGSystem:
    """LlamaIndex-based RAG system for Funes"""
    
    def __init__(self, db_params=None):
        """
        Initialize the RAG system with LlamaIndex components
        
        Args:
            db_params: Database connection parameters. If None, uses DB_CONFIG from config.py
        """
        self.db_params = db_params or DB_CONFIG
        
        # Initialize embedding model from config
        model_name = EMBEDDING_CONFIG.get('model_name', 'all-MiniLM-L6-v2')
        Settings.embed_model = HuggingFaceEmbedding(model_name=model_name)
        
        # Set up PostgreSQL vector store for LlamaIndex
        self._setup_vector_store()
        
    def _setup_vector_store(self):
        """Set up the PostgreSQL vector store for LlamaIndex"""
        try:
            # Connect to PostgreSQL
            conn_str = f"postgresql://{self.db_params['user']}:{self.db_params['password']}@{self.db_params['host']}/{self.db_params['dbname']}"
            
            # Try different approaches to create the PGVectorStore
            try:
                # First approach: use direct constructor
                self.vector_store = PGVectorStore(
                    connection_string=conn_str,
                    table_name="memories",
                    embed_dim=384
                )
                print("[LlamaIndexRAG] Created vector store using direct constructor")
            except Exception as e1:
                print(f"[LlamaIndexRAG] First approach failed: {str(e1)}")
                
                try:
                    # Second approach: use from_params but with minimal parameters
                    self.vector_store = PGVectorStore.from_params(
                        database=self.db_params['dbname'],
                        host=self.db_params['host'],
                        password=self.db_params['password'],
                        port=5432,  # Default PostgreSQL port
                        user=self.db_params['user'],
                        table_name="memories",  # Use existing table
                        embed_dim=384,  # Dimension of the embedding model
                    )
                    print("[LlamaIndexRAG] Created vector store using from_params with minimal parameters")
                except Exception as e2:
                    print(f"[LlamaIndexRAG] Second approach failed: {str(e2)}")
                    
                    try:
                        # Third approach: use from_connection_string
                        self.vector_store = PGVectorStore.from_connection_string(
                            connection_string=conn_str,
                            table_name="memories",
                            embed_dim=384,
                        )
                        print("[LlamaIndexRAG] Created vector store using from_connection_string")
                    except Exception as e3:
                        print(f"[LlamaIndexRAG] All approaches failed, last error: {str(e3)}")
                        raise e3
            
            # Set up the storage context
            storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
            
            # Initialize an empty index with our vector store
            self.index = VectorStoreIndex.from_vector_store(
                vector_store=self.vector_store,
            )
            
            print("[LlamaIndexRAG] Successfully connected to PostgreSQL vector store")
        except Exception as e:
            print(f"[LlamaIndexRAG] Error setting up vector store: {str(e)}")
            print(f"[LlamaIndexRAG] Traceback: {traceback.format_exc()}")
            # Create an in-memory fallback index
            self.index = VectorStoreIndex([])
            print("[LlamaIndexRAG] Using fallback in-memory vector store")
    
    def process_file(self, file_path: str):
        """
        Process a file and add it to the RAG system
        
        Args:
            file_path: Path to the file to process
        """
        try:
            # Convert document to text using docling (keep existing functionality)
            converter = docling.document_converter.DocumentConverter()
            result = converter.convert(file_path)
            content = result.document.export_to_markdown()
            
            # Create LlamaIndex document
            source_name = os.path.basename(file_path)
            doc = Document(
                text=content,
                metadata={"source": source_name}
            )
            
            # Create nodes from the document with source info
            nodes = TextNode.from_text(
                content,
                metadata={"source": source_name}
            )
            
            # Add to the index
            self.index.insert_nodes([nodes])
            
            print(f"[LlamaIndexRAG] Successfully processed file: {file_path}")
            return True
        except Exception as e:
            print(f"[LlamaIndexRAG] Error processing file {file_path}: {str(e)}")
            print(f"[LlamaIndexRAG] Traceback: {traceback.format_exc()}")
            return False
    
    def query(self, query_text: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Query the RAG system
        
        Args:
            query_text: Query text
            top_k: Number of results to return
            
        Returns:
            List of retrieval results
        """
        try:
            # Create a retriever from the index with top_k setting
            retriever = self.index.as_retriever(similarity_top_k=top_k)
            
            # Retrieve nodes
            nodes = retriever.retrieve(query_text)
            
            # Format the results in the same structure as the current system returns
            results = []
            for node in nodes:
                results.append((node.text, node.metadata.get("source", "unknown")))
            
            return results
        except Exception as e:
            print(f"[LlamaIndexRAG] Error querying: {str(e)}")
            print(f"[LlamaIndexRAG] Traceback: {traceback.format_exc()}")
            return []
    
    def query_engine(self, query_text: str):
        """
        Get a query engine for the RAG system
        
        This creates a more powerful query engine that can synthesize responses
        from the retrieved documents.
        
        Args:
            query_text: Query text
            
        Returns:
            Response from the query engine
        """
        try:
            # Create a query engine from the index
            query_engine = self.index.as_query_engine()
            
            # Query the engine
            response = query_engine.query(query_text)
            
            return str(response)
        except Exception as e:
            print(f"[LlamaIndexRAG] Error with query engine: {str(e)}")
            print(f"[LlamaIndexRAG] Traceback: {traceback.format_exc()}")
            return f"Error querying the knowledge base: {str(e)}"
            
    def store_memory(self, text: str, source: str = "chat"):
        """
        Store a memory in the RAG system
        
        Args:
            text: Text to store
            source: Source of the text
        """
        try:
            # Create a node for the memory
            node = TextNode(
                text=text,
                metadata={"source": source}
            )
            
            # Add to the index
            self.index.insert_nodes([node])
            
            print(f"[LlamaIndexRAG] Successfully stored memory from source: {source}")
            return True
        except Exception as e:
            print(f"[LlamaIndexRAG] Error storing memory: {str(e)}")
            print(f"[LlamaIndexRAG] Traceback: {traceback.format_exc()}")
            return False

    def get_unique_sources(self) -> List[str]:
        """
        Get a list of unique sources in the vector store
        
        Returns:
            List of unique sources
        """
        try:
            # This requires executing a direct SQL query on the PostgreSQL database
            # We'll need to use the underlying PostgreSQL connection to do this
            import psycopg2
            
            conn = psycopg2.connect(
                dbname=self.db_params['dbname'],
                user=self.db_params['user'],
                password=self.db_params['password'],
                host=self.db_params['host']
            )
            
            cursor = conn.cursor()
            table_name = LLAMAINDEX_CONFIG.get('vector_store_table', 'memories')
            
            # Query for unique sources - this assumes the source is stored in the metadata field
            # and that the metadata field is a JSON column or contains source information
            cursor.execute(f"SELECT DISTINCT metadata FROM {table_name} WHERE metadata IS NOT NULL;")
            results = cursor.fetchall()
            
            # Extract source values from metadata
            sources = []
            for result in results:
                if result[0] and isinstance(result[0], dict) and 'source' in result[0]:
                    sources.append(result[0]['source'])
            
            cursor.close()
            conn.close()
            
            return list(set(sources)) if sources else ["chat"]  # Return at least "chat" as a default
        except Exception as e:
            print(f"[LlamaIndexRAG] Error getting unique sources: {str(e)}")
            print(f"[LlamaIndexRAG] Traceback: {traceback.format_exc()}")
            return ["chat"]  # Return a default value in case of error
    
    def delete_memories_by_source(self, source: str) -> bool:
        """
        Delete memories from a specific source
        
        Args:
            source: Source name to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # This requires executing a direct SQL query on the PostgreSQL database
            import psycopg2
            
            conn = psycopg2.connect(
                dbname=self.db_params['dbname'],
                user=self.db_params['user'],
                password=self.db_params['password'],
                host=self.db_params['host']
            )
            
            cursor = conn.cursor()
            table_name = LLAMAINDEX_CONFIG.get('vector_store_table', 'memories')
            
            # Delete records where the metadata contains the specified source
            cursor.execute(f"DELETE FROM {table_name} WHERE metadata->>'source' = %s;", (source,))
            count = cursor.rowcount
            conn.commit()
            
            cursor.close()
            conn.close()
            
            print(f"[LlamaIndexRAG] Deleted {count} memories from source: {source}")
            return True
        except Exception as e:
            print(f"[LlamaIndexRAG] Error deleting memories by source: {str(e)}")
            print(f"[LlamaIndexRAG] Traceback: {traceback.format_exc()}")
            return False
    
    def clear_memories(self) -> bool:
        """
        Clear all memories from the vector store
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # This requires executing a direct SQL query on the PostgreSQL database
            import psycopg2
            
            conn = psycopg2.connect(
                dbname=self.db_params['dbname'],
                user=self.db_params['user'],
                password=self.db_params['password'],
                host=self.db_params['host']
            )
            
            cursor = conn.cursor()
            table_name = LLAMAINDEX_CONFIG.get('vector_store_table', 'memories')
            
            # Delete all records
            cursor.execute(f"DELETE FROM {table_name};")
            count = cursor.rowcount
            conn.commit()
            
            cursor.close()
            conn.close()
            
            print(f"[LlamaIndexRAG] Cleared {count} memories from the vector store")
            return True
        except Exception as e:
            print(f"[LlamaIndexRAG] Error clearing memories: {str(e)}")
            print(f"[LlamaIndexRAG] Traceback: {traceback.format_exc()}")
            return False