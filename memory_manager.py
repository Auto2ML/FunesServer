import os
import json
import threading
import queue
import re  # Add missing import for regular expressions
from collections import deque
from datetime import datetime, timedelta
from database import DatabaseManager
from llm_handler import LLMHandler
from config import MEMORY_CONFIG, DB_CONFIG
import tools  # Import tools module to check tool properties

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
        
        # Initialize database connection with error checking
        try:
            print("[DualMemoryManager] Initializing database connection with params:", self.db_params)
            self.db_manager = DatabaseManager(self.db_params)
            print("[DualMemoryManager] Database connection initialized successfully")
            # Test the connection with a simple query
            try:
                sources = self.db_manager.get_unique_sources()
                print(f"[DualMemoryManager] Database connection test successful. Found {len(sources)} sources.")
            except Exception as e:
                print(f"[DualMemoryManager] Database connection test failed: {str(e)}")
        except Exception as e:
            import traceback
            print(f"[DualMemoryManager] Error initializing database connection: {str(e)}")
            print(f"[DualMemoryManager] Traceback: {traceback.format_exc()}")
            # Continue without crashing, but flag the database as unavailable
            self.db_manager = None
            print("[DualMemoryManager] Continuing with database features disabled")
        
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
        try:
            # Check if database connection is available
            if self.db_manager is None:
                print("[store_memory] Database connection not available, skipping memory storage")
                return
                
            print(f"[store_memory] Generating embedding for context: {context[:30]}...")
            embedding = self.llm_handler.get_single_embedding(context)
            print(f"[store_memory] Embedding generated, storing in database with source: {source}")
            self.db_manager.insert_memory(context, embedding, source)
            print("[store_memory] Memory stored successfully")
        except Exception as e:
            import traceback
            print(f"[store_memory] Error storing memory: {str(e)}")
            print(f"[store_memory] Traceback: {traceback.format_exc()}")
            # Continue without crashing
    
    def should_store_tool_response(self, tool_name):
        """Check if a tool response should be stored in long-term memory"""
        # Get the tool by name
        tool = tools.get_tool(tool_name)
        if tool:
            # Check if the tool wants its responses stored in memory
            return tool.store_in_memory
        # Default to False if tool not found
        return False
    
    def retrieve_relevant_memories(self, query, top_k=None):
        """Retrieve relevant memories from long-term storage"""
        try:
            # Check if database connection is available
            if self.db_manager is None:
                print("[retrieve_relevant_memories] Database connection not available, returning empty list")
                return []
                
            top_k = top_k or MEMORY_CONFIG['default_top_k']
            print(f"[retrieve_relevant_memories] Generating embedding for query: {query[:30]}...")
            query_embedding = self.llm_handler.get_single_embedding(query)
            print(f"[retrieve_relevant_memories] Embedding generated, retrieving top {top_k} memories")
            memories = self.db_manager.retrieve_memories(query_embedding, top_k)
            print(f"[retrieve_relevant_memories] Retrieved {len(memories)} memories")
            return memories
        except Exception as e:
            import traceback
            print(f"[retrieve_relevant_memories] Error retrieving memories: {str(e)}")
            print(f"[retrieve_relevant_memories] Traceback: {traceback.format_exc()}")
            # Return empty list in case of error
            return []
 
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
            print("Starting process_chat with message:", user_message[:30] + "..." if len(user_message) > 30 else user_message)
            
            # Get relevant long-term memories
            print("Retrieving relevant memories...")
            try:
                long_term_memories = self.retrieve_relevant_memories(user_message)
                print(f"Retrieved {len(long_term_memories)} memories")
            except Exception as e:
                print(f"Error retrieving memories: {str(e)}")
                long_term_memories = []
            
            # Convert short-term memory to the format expected by LLMHandler
            print("Building conversation history...")
            conversation_history = []
            try:
                for msg in self.short_term_memory:
                    if msg['role'] in ['user', 'assistant', 'system', 'tool']:
                        entry = {
                            'role': msg['role'],
                            'content': msg['content']
                        }
                        # Include tool_calls if present
                        if 'tool_calls' in msg:
                            entry['tool_calls'] = msg['tool_calls']
                        
                        # Include tool call ID if present
                        if 'tool_call_id' in msg:
                            entry['tool_call_id'] = msg['tool_call_id']
                        
                        # Include name if present (for tool responses)
                        if 'name' in msg:
                            entry['name'] = msg['name']
                            
                        conversation_history.append(entry)
                print(f"Built conversation history with {len(conversation_history)} messages")
            except Exception as e:
                print(f"Error building conversation history: {str(e)}")
                conversation_history = []
            
            # Build additional context string containing only long-term memories
            print("Building additional context...")
            additional_context = ""
            if long_term_memories:
                additional_context = "Relevant past memories:\n"
                for memory in long_term_memories:
                    additional_context += f"- {memory[0]}\n"
            
            # Get LLM response using the LLM handler
            print("Generating LLM response...")
            try:
                llm_response = self.llm_handler.generate_response(
                    user_input=user_message,
                    conversation_history=conversation_history,
                    additional_context=additional_context if additional_context else None
                )
                print(f"Received response of type: {type(llm_response)}")
            except Exception as e:
                import traceback
                print(f"Error generating LLM response: {str(e)}")
                print(f"Traceback: {traceback.format_exc()}")
                raise
            
            # Handle response based on whether it's a simple string or a dict with tool calls
            print("Processing LLM response...")
            if isinstance(llm_response, dict) and 'tool_calls' in llm_response:
                print(f"Response contains tool calls: {llm_response.get('tool_calls')}")
                # Add the assistant message with tool calls
                self._add_to_short_term_with_tool_calls('assistant', llm_response['content'], llm_response['tool_calls'])
                
                # For UI display, we'll just show the content with a note about tool usage
                display_response = llm_response['content']
                if llm_response['tool_calls']:
                    display_response += "\n[Tool usage detected: Processing tool request]"
                
                self.chat_history.append((user_message, display_response))
                return display_response
            else:
                print("Response is a simple text response")
                # Just a regular text response
                self._add_to_short_term('assistant', llm_response)
                self.chat_history.append((user_message, llm_response))
                
                # Store user message in long-term memory
                print("Storing user message memory...")
                try:
                    self.store_memory(user_message)
                    print("User message stored successfully")
                except Exception as e:
                    print(f"Error storing user message: {str(e)}")
                
                # Store assistant response in long-term memory
                # For regular text responses, we always store them
                print("Storing assistant response memory...")
                try:
                    # Check if this is a tool response by looking for common patterns
                    is_tool_response = False
                    tool_name = None
                    
                    # Look for patterns indicating this is a tool response
                    tool_patterns = [
                        r"Tool response from (\w+):",
                        r"I used the (\w+) tool",
                        r"According to the (\w+) tool",
                        r"Based on the (\w+) tool",
                        r"The (\w+) tool returned"
                    ]
                    
                    for pattern in tool_patterns:
                        match = re.search(pattern, llm_response)
                        if match:
                            is_tool_response = True
                            tool_name = match.group(1)
                            print(f"Detected tool response from {tool_name}")
                            break
                    
                    # If this is a tool response, check if we should store it
                    if is_tool_response and tool_name:
                        should_store = self.should_store_tool_response(tool_name)
                        if should_store:
                            print(f"Tool {tool_name} allows storing responses, storing in memory")
                            self.store_memory(llm_response)
                        else:
                            print(f"Tool {tool_name} does not allow storing responses, skipping memory storage")
                    else:
                        # It's a regular assistant response, store it
                        self.store_memory(llm_response)
                        print("Memory stored successfully")
                except Exception as e:
                    print(f"Error storing memory: {str(e)}")
                
                return llm_response
                
        except Exception as e:
            import traceback
            error_msg = f"Error processing request: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)  # For logging
            return error_msg
    
    def _add_to_short_term_with_tool_calls(self, role, content, tool_calls):
        """Add a message with tool calls to short-term memory"""
        self.short_term_memory.append({
            'role': role,
            'content': content,
            'tool_calls': tool_calls,
            'timestamp': datetime.now()
        })
