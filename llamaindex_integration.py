"""
LlamaIndex integration for Funes

This module provides integration helpers to gradually migrate Funes to use LlamaIndex.
"""

import os
from typing import Dict, Any, Optional, List
from config import LLM_CONFIG, DB_CONFIG, MEMORY_CONFIG

# Import both old and new implementations
from memory_manager import DualMemoryManager
from llm_handler import LLMHandler
from rag_system import RAGSystem

# Import the new LlamaIndex implementations
from llamaindex_llm import LlamaIndexLLMHandler
from llamaindex_rag import LlamaIndexRAGSystem
from llamaindex_tools import get_all_tools_as_llamaindex

# Global flag to control which implementation to use
USE_LLAMAINDEX = os.environ.get('USE_LLAMAINDEX', 'false').lower() == 'true'


class IntegratedMemoryManager:
    """
    Integration wrapper around memory management functionality
    
    This class provides a unified interface that can switch between the original
    implementation and the LlamaIndex-based one.
    """
    
    def __init__(self, use_llamaindex=None):
        """
        Initialize the memory manager with either original or LlamaIndex implementation
        
        Args:
            use_llamaindex: Override the global flag for this instance
        """
        self.use_llamaindex = use_llamaindex if use_llamaindex is not None else USE_LLAMAINDEX
        
        # Make db_params accessible at instance level
        self.db_params = DB_CONFIG
        
        if self.use_llamaindex:
            print("[Integration] Using LlamaIndex-based memory manager")
            self.llm_handler = LlamaIndexLLMHandler()
            self.rag_system = LlamaIndexRAGSystem(self.db_params)
            # The original memory manager is still needed for some functionality
            # that hasn't been migrated yet
            self.legacy_memory_manager = DualMemoryManager()
        else:
            print("[Integration] Using original memory manager")
            self.legacy_memory_manager = DualMemoryManager()
            self.llm_handler = self.legacy_memory_manager.llm_handler
        
        # Properties needed for compatibility
        self.db_manager = self.legacy_memory_manager.db_manager
        self.short_term_memory = self.legacy_memory_manager.short_term_memory
        self.chat_history = self.legacy_memory_manager.chat_history

    def process_chat(self, user_message):
        """Process chat with either original or LlamaIndex implementation"""
        if self.use_llamaindex:
            # Convert short-term memory to format expected by LlamaIndex
            conversation_history = []
            for msg in self.legacy_memory_manager.short_term_memory:
                if msg['role'] in ['user', 'assistant', 'system']:
                    conversation_history.append({
                        'role': msg['role'],
                        'content': msg['content']
                    })
            
            # Get relevant memories from RAG
            try:
                memories = self.rag_system.query(user_message, 
                                              top_k=MEMORY_CONFIG.get('default_top_k', 3))
                additional_context = ""
                if memories:
                    additional_context = "Relevant past memories:\n"
                    for memory in memories:
                        additional_context += f"- {memory[0]}\n"
            except Exception as e:
                print(f"[Integration] Error retrieving memories: {str(e)}")
                additional_context = None
                memories = []
            
            # Get response from LlamaIndex LLM
            response = self.llm_handler.generate_response(
                user_message, 
                conversation_history=conversation_history,
                additional_context=additional_context
            )
            
            # Update the original memory manager's short-term memory
            self.legacy_memory_manager._add_to_short_term('user', user_message)
            self.legacy_memory_manager._add_to_short_term('assistant', response)
            self.legacy_memory_manager.chat_history.append((user_message, response))
            
            # Store in RAG system
            try:
                self.rag_system.store_memory(user_message, source="chat")
                self.rag_system.store_memory(response, source="chat")
            except Exception as e:
                print(f"[Integration] Error storing memory: {str(e)}")
            
            return response
        else:
            # Use the original implementation
            return self.legacy_memory_manager.process_chat(user_message)
    
    def store_memory(self, context, source='chat'):
        """Store memory using either original or LlamaIndex implementation"""
        if self.use_llamaindex:
            # Store using both implementations during migration
            self.legacy_memory_manager.store_memory(context, source)
            self.rag_system.store_memory(context, source)
        else:
            self.legacy_memory_manager.store_memory(context, source)
    
    def retrieve_relevant_memories(self, query, top_k=None):
        """Retrieve memories using either original or LlamaIndex implementation"""
        if self.use_llamaindex:
            top_k = top_k or MEMORY_CONFIG.get('default_top_k', 3)
            return self.rag_system.query(query, top_k=top_k)
        else:
            return self.legacy_memory_manager.retrieve_relevant_memories(query, top_k)


class IntegratedRAGSystem:
    """
    Integration wrapper around RAG functionality
    
    This class provides a unified interface that can switch between the original
    implementation and the LlamaIndex-based one.
    """
    
    def __init__(self, db_params=None, use_llamaindex=None):
        """
        Initialize the RAG system with either original or LlamaIndex implementation
        
        Args:
            db_params: Database connection parameters
            use_llamaindex: Override the global flag for this instance
        """
        self.use_llamaindex = use_llamaindex if use_llamaindex is not None else USE_LLAMAINDEX
        self.db_params = db_params or DB_CONFIG
        
        if self.use_llamaindex:
            print("[Integration] Using LlamaIndex-based RAG system")
            self.rag_system = LlamaIndexRAGSystem(self.db_params)
        else:
            print("[Integration] Using original RAG system")
            self.rag_system = RAGSystem(self.db_params)
    
    def process_file(self, file_path):
        """Process file using either original or LlamaIndex implementation"""
        return self.rag_system.process_file(file_path)


# Helper function to enable LlamaIndex implementation globally
def enable_llamaindex():
    """Enable LlamaIndex implementation globally"""
    global USE_LLAMAINDEX
    USE_LLAMAINDEX = True
    os.environ['USE_LLAMAINDEX'] = 'true'
    print("[Integration] LlamaIndex implementation enabled globally")


# Helper function to disable LlamaIndex implementation globally
def disable_llamaindex():
    """Disable LlamaIndex implementation globally"""
    global USE_LLAMAINDEX
    USE_LLAMAINDEX = False
    os.environ['USE_LLAMAINDEX'] = 'false'
    print("[Integration] LlamaIndex implementation disabled globally")