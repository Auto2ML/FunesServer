import os
import json
import threading
import queue
from collections import deque
from datetime import datetime, timedelta
from sentence_transformers import SentenceTransformer
import ollama
from database import DatabaseManager

class DualMemoryManager:
    def __init__(self, short_term_capacity=10, short_term_ttl_minutes=30):
        # Database connection parameters
        self.db_params = {
            'dbname': 'funes',
            'user': 'llm',
            'password': 'llm',
            'host': 'localhost'
        }
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Short-term memory setup
        self.short_term_memory = deque(maxlen=short_term_capacity)
        self.short_term_ttl = timedelta(minutes=short_term_ttl_minutes)
        
        # Initialize database connection
        self.db_manager = DatabaseManager(self.db_params)
        
        # Chat history for UI
        self.chat_history = []
        
        # Thread-safe queue for processing
        self.message_queue = queue.Queue()
    
    def _add_to_short_term(self, role, content):
        """Add a message to short-term memory with timestamp"""
        self.short_term_memory.append({
            'role': role,
            'content': content,
            'timestamp': datetime.now()
        })
    
    def _clean_short_term(self):
        """Remove expired messages from short-term memory"""
        current_time = datetime.now()
        self.short_term_memory = deque(
            [msg for msg in self.short_term_memory 
             if current_time - msg['timestamp'] <= self.short_term_ttl],
            maxlen=self.short_term_memory.maxlen
        )
    
    def store_memory(self, context, source='chat'):
        """Store memory in long-term storage (PostgreSQL)"""
        embedding = self.embedding_model.encode(context)
        self.db_manager.insert_memory(context, embedding, source)
    
    def retrieve_relevant_memories(self, query, top_k=3):
        """Retrieve relevant memories from long-term storage"""
        query_embedding = self.embedding_model.encode(query)
        return self.db_manager.retrieve_memories(query_embedding.tolist(), top_k)
 
    def _build_context(self, user_message):
        """Build context from both short-term and long-term memory"""
        self._clean_short_term()  # Remove expired messages
        
        # Get relevant long-term memories
        long_term_memories = self.retrieve_relevant_memories(user_message)
        
        # Construct context
        context = "Current conversation:\n"
        for msg in self.short_term_memory:
            context += f"{msg['role']}: {msg['content']}\n"
        
        if long_term_memories:
            context += "\nRelevant past memories:\n"
            for memory in long_term_memories:
                context += f"- {memory[0]}\n"
        
        return context
    
    def process_chat(self, user_message):
        """Process chat with both short-term and long-term memory"""
        try:
            # Retrieve relevant memories from long-term storage
            query_embedding = self.embedding_model.encode(user_message)
            relevant_memories = self.db_manager.retrieve_memories(query_embedding.tolist(), top_k=2)
            
            # Build context from both short-term and long-term memory
            context = self._build_context(user_message)
            
            # Add relevant file content to context
            if relevant_memories:
                context += "\nRelevant file content:\n"
                for memory in relevant_memories:
                    context += f"- {memory[0]}\n"
            
            # Prepare messages for LLM
            messages = [
                {
                    'role': 'system',
                    'content': "You are Funes, a helpful assistant. Use your recent conversation history and relevant memories stored in your databas to provide more informed and consistent responses, only if it is relevant for the conversation. Do not repeat unnecesary information. If you don't find relevant information in history or memories, use your training data"
                },
                {
                    'role': 'user',
                    'content': f"{context}\n\nCurrent user message: {user_message}"
                }
            ]
            
            # Get LLM response
            response = ollama.chat(model='llama3.2:latest', messages=messages)
            llm_response = response['message']['content']
            
            # Store in both memory systems
            self._add_to_short_term('user', user_message)
            self._add_to_short_term('assistant', llm_response)
            
            # Store important interactions in long-term memory
            self.store_memory(user_message)
            self.store_memory(llm_response)
            
            # Update chat history for UI
            self.chat_history.append((user_message, llm_response))
            
            return llm_response
                
        except Exception as e:
            error_msg = f"Error processing request: {str(e)}"
            print(error_msg)  # For logging
            return error_msg
    
    def clear_memories(self):
        """Clear all memories"""
        self.short_term_memory.clear()
        self.db_manager.clear_memories()
        return "All memories cleared successfully."
