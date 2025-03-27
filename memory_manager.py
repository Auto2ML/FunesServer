import os
import json
import threading
import queue
from collections import deque
from datetime import datetime, timedelta
from database import DatabaseManager
from llm_handler import LLMHandler
from config import MEMORY_CONFIG, DB_CONFIG

class DualMemoryManager:
    def __init__(self, short_term_capacity=None, short_term_ttl_minutes=None):
        # Use config values with optional overrides
        self.db_params = DB_CONFIG
        
        # Get memory settings from config with optional overrides
        short_term_capacity = short_term_capacity or MEMORY_CONFIG['short_term_capacity']
        short_term_ttl_minutes = short_term_ttl_minutes or MEMORY_CONFIG['short_term_ttl_minutes']
        
        # Initialize LLM handler
        self.llm_handler = LLMHandler()
        
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
        embedding = self.llm_handler.get_single_embedding(context)
        self.db_manager.insert_memory(context, embedding, source)
    
    def retrieve_relevant_memories(self, query, top_k=None):
        """Retrieve relevant memories from long-term storage"""
        top_k = top_k or MEMORY_CONFIG['default_top_k']
        query_embedding = self.llm_handler.get_single_embedding(query)
        return self.db_manager.retrieve_memories(query_embedding, top_k)
 
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
            # Get relevant long-term memories
            long_term_memories = self.retrieve_relevant_memories(user_message)
            
            # Convert short-term memory to the format expected by LLMHandler
            conversation_history = []
            for msg in self.short_term_memory:
                if msg['role'] in ['user', 'assistant', 'system']:
                    conversation_history.append({
                        'role': msg['role'],
                        'content': msg['content']
                    })
            
            # Build additional context string containing only long-term memories
            additional_context = ""
            if long_term_memories:
                additional_context = "Relevant past memories:\n"
                for memory in long_term_memories:
                    additional_context += f"- {memory[0]}\n"
            
            # Get LLM response using the LLM handler
            llm_response = self.llm_handler.generate_response(
                user_input=user_message,
                conversation_history=conversation_history,
                additional_context=additional_context if additional_context else None
            )
            
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
