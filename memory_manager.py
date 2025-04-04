import os
import json
import threading
import queue
import re  # Add missing import for regular expressions
from collections import deque
from datetime import datetime, timedelta
from database import DatabaseManager
from config import MEMORY_CONFIG, DB_CONFIG, EMBEDDING_CONFIG
import tools  # Import tools module to check tool properties
import logging
import traceback
from sentence_transformers import SentenceTransformer

# Get logger for this module
logger = logging.getLogger('MemoryManager')

class EmbeddingManager:
    """
    Manages embedding operations for text-to-vector conversion.
    This class centralizes all embedding functionality.
    """
    
    def __init__(self):
        # Initialize logger
        self.logger = logging.getLogger('EmbeddingManager')
        self.logger.info("Initializing embedding manager")
        
        # Initialize configuration parameters
        self.embedding_config = EMBEDDING_CONFIG
        
        # Initialize the embedding model
        self.embedding_model = None
        self.initialize_embedding_model()
    
    def initialize_embedding_model(self):
        """Initialize the embedding model based on configuration"""
        try:
            model_name = self.embedding_config.get('model', 'all-MiniLM-L6-v2')
            self.logger.info(f"Loading embedding model: {model_name}")
            self.embedding_model = SentenceTransformer(model_name)
            self.logger.info("Embedding model loaded successfully")
        except Exception as e:
            self.logger.error(f"Error loading embedding model: {str(e)}")
            self.logger.error("Embeddings will not be available!")
            self.embedding_model = None
    
    def get_embedding(self, text):
        """Generate an embedding vector for a single text string"""
        if self.embedding_model is None:
            self.logger.error("Cannot generate embedding: model not available")
            return [0.0] * 384  # Return a zero vector of default size
            
        try:
            self.logger.debug(f"Generating embedding for text: {text[:30]}...")
            embedding = self.embedding_model.encode(text)
            return embedding.tolist()
        except Exception as e:
            self.logger.error(f"Error generating embedding: {str(e)}")
            return [0.0] * 384  # Return a zero vector of default size
    
    def get_batch_embeddings(self, texts):
        """Generate embedding vectors for multiple text strings"""
        if self.embedding_model is None:
            self.logger.error("Cannot generate embeddings: model not available")
            return [[0.0] * 384 for _ in texts]  # Return zero vectors
            
        try:
            self.logger.debug(f"Generating batch embeddings for {len(texts)} texts")
            embeddings = self.embedding_model.encode(texts)
            return [embedding.tolist() for embedding in embeddings]
        except Exception as e:
            self.logger.error(f"Error generating batch embeddings: {str(e)}")
            return [[0.0] * 384 for _ in texts]  # Return zero vectors
            
    def store_tool_embeddings(self, tools, db_manager):
        """
        Initialize or update embeddings for all available tools
        
        Args:
            tools: List of tool definitions
            db_manager: Database manager to store embeddings
        """
        self.logger.info("Initializing tool embeddings...")
        
        for tool in tools:
            if "function" in tool:
                function = tool["function"]
                name = function.get("name", "").lower()
                
                if name:
                    # Extract all relevant information about the tool
                    description = function.get("description", "")
                    parameters = function.get("parameters", {})
                    
                    if description and self.embedding_model is not None:
                        try:
                            # Create an enhanced description that includes parameter information
                            enhanced_description = description + "\n"
                            if parameters and isinstance(parameters, dict) and "properties" in parameters:
                                enhanced_description += "Parameters:\n"
                                for param_name, param_info in parameters.get("properties", {}).items():
                                    param_desc = param_info.get("description", "")
                                    enhanced_description += f"- {param_name}: {param_desc}\n"
                            
                            # Generate embedding for the enhanced description
                            self.logger.info(f"Generating embedding for tool: {name}")
                            embedding = self.embedding_model.encode(enhanced_description)
                            
                            # Store in database
                            db_manager.store_tool_embedding(name, enhanced_description, embedding)
                            self.logger.info(f"Stored embedding for tool: {name}")
                        except Exception as e:
                            self.logger.error(f"Error generating/storing embedding for tool {name}: {str(e)}")
        
        self.logger.info("Tool embeddings initialization complete")

# Create singleton instance for global use
embedding_manager = EmbeddingManager()

# Expose direct function for ease of use
def get_embedding(text):
    """Global function to get embeddings from the embedding manager"""
    return embedding_manager.get_embedding(text)

# Expose function for initializing tool embeddings
def initialize_tool_embeddings(tools, db_manager):
    """Initialize embeddings for tools using the embedding manager"""
    embedding_manager.store_tool_embeddings(tools, db_manager)

class DualMemoryManager:
    def __init__(self, short_term_capacity=None, short_term_ttl_minutes=None):
        # Use config values with optional overrides
        self.db_params = DB_CONFIG
        
        # Get memory settings from config with optional overrides
        short_term_capacity = short_term_capacity or MEMORY_CONFIG['short_term_capacity']
        short_term_ttl_minutes = short_term_ttl_minutes or MEMORY_CONFIG['short_term_ttl_minutes']
        
        # Use the embedding manager
        self.embedding_manager = embedding_manager
        
        # Initialize database connection with error checking
        try:
            logger.info("[DualMemoryManager] Initializing database connection with params: %s", self.db_params)
            self.db_manager = DatabaseManager(self.db_params)
            logger.info("[DualMemoryManager] Database connection initialized successfully")
            # Test the connection with a simple query
            try:
                sources = self.db_manager.get_unique_sources()
                logger.info(f"[DualMemoryManager] Database connection test successful. Found {len(sources)} sources.")
            except Exception as e:
                logger.error(f"[DualMemoryManager] Database connection test failed: {str(e)}")
        except Exception as e:
            logger.error(f"[DualMemoryManager] Error initializing database connection: {str(e)}")
            logger.error(f"[DualMemoryManager] Traceback: {traceback.format_exc()}")
            # Continue without crashing, but flag the database as unavailable
            self.db_manager = None
            logger.warning("[DualMemoryManager] Continuing with database features disabled")
        
        # Chat history for UI
        self.chat_history = []
        
        # Thread-safe queue for processing
        self.message_queue = queue.Queue()
        
        # Import LLM handler here to avoid circular imports
        from llm_handler import LLMHandler
        self.llm_handler = LLMHandler()
        
        # Short-term memory setup
        self.short_term_memory = deque(maxlen=short_term_capacity)
        self.short_term_ttl = timedelta(minutes=short_term_ttl_minutes)
    
    def _add_to_short_term(self, role, content, **kwargs):
        """Add a message to short-term memory with timestamp and optional additional fields"""
        message = {
            'role': role,
            'content': content,
            'timestamp': datetime.now()
        }
        
        # Add any additional fields (like tool_call_id, name, etc.)
        for key, value in kwargs.items():
            message[key] = value
            
        self.short_term_memory.append(message)
    
    def _add_to_short_term_with_tool_calls(self, role, content, tool_calls):
        """Add a message with tool calls to short-term memory"""
        self.short_term_memory.append({
            'role': role,
            'content': content,
            'tool_calls': tool_calls,
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
                logger.warning("[store_memory] Database connection not available, skipping memory storage")
                return
                
            logger.info(f"[store_memory] Generating embedding for context: {context[:30]}...")
            embedding = self.embedding_manager.get_embedding(context)
            logger.info(f"[store_memory] Embedding generated, storing in database with source: {source}")
            self.db_manager.insert_memory(context, embedding, source)
            logger.info("[store_memory] Memory stored successfully")
        except Exception as e:
            logger.error(f"[store_memory] Error storing memory: {str(e)}")
            logger.error(f"[store_memory] Traceback: {traceback.format_exc()}")
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
                logger.warning("[retrieve_relevant_memories] Database connection not available, returning empty list")
                return []
                
            top_k = top_k or MEMORY_CONFIG['default_top_k']
            logger.info(f"[retrieve_relevant_memories] Generating embedding for query: {query[:30]}...")
            query_embedding = self.embedding_manager.get_embedding(query)
            logger.info(f"[retrieve_relevant_memories] Embedding generated, retrieving top {top_k} memories")
            memories = self.db_manager.retrieve_memories(query_embedding, top_k)
            logger.info(f"[retrieve_relevant_memories] Retrieved {len(memories)} memories")
            return memories
        except Exception as e:
            logger.error(f"[retrieve_relevant_memories] Error retrieving memories: {str(e)}")
            logger.error(f"[retrieve_relevant_memories] Traceback: {traceback.format_exc()}")
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
            logger.info("Starting process_chat with message: %s", user_message[:30] + "..." if len(user_message) > 30 else user_message)
            
            # Check if this is a tool-related query using vector embedding system
            is_tool_query = False
            suggested_tool = None
            
            # Try to use vector-based tool selection if available
            try:
                # Use the centralized function from llm_utilities
                from llm_utilities import should_use_tools_vector
                if self.db_manager:
                    is_tool_query, suggested_tool = should_use_tools_vector(
                        user_message,
                        None,  # We'll use the existing get_embedding function
                        self.db_manager,
                        similarity_threshold=0.75
                    )
                    logger.info(f"Vector-based tool selection result: is_tool_query={is_tool_query}, tool={suggested_tool or 'None'}")
            except Exception as e:
                logger.error(f"Error in vector-based tool selection: {str(e)}")
                # Fall back to keyword-based detection in case of error
                is_tool_query = False
            
            # Get relevant long-term memories (only if this isn't a tool query)
            logger.info("Retrieving relevant memories...")
            long_term_memories = []
            try:
                if not is_tool_query:
                    # Only retrieve memories if this is NOT a tool query
                    long_term_memories = self.retrieve_relevant_memories(user_message)
                    logger.info(f"Retrieved {len(long_term_memories)} memories")
                else:
                    logger.info("Skipping memory retrieval for tool query - using tools instead")
            except Exception as e:
                logger.error(f"Error retrieving memories: {str(e)}")
            
            # Convert short-term memory to the format expected by LLMHandler
            logger.info("Building conversation history...")
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
                logger.info(f"Built conversation history with {len(conversation_history)} messages")
            except Exception as e:
                logger.error(f"Error building conversation history: {str(e)}")
                conversation_history = []
            
            # Build additional context string containing only long-term memories
            logger.info("Building additional context...")
            additional_context = ""
            if long_term_memories:
                additional_context = "Relevant past memories:\n"
                for memory in long_term_memories:
                    additional_context += f"- {memory[0]}\n"
            
            # Add the user message to short-term memory
            logger.info("Adding user message to short-term memory...")
            self._add_to_short_term('user', user_message)
            
            # Pass tool information to the LLM handler if a tool was selected
            specific_tool = None
            if is_tool_query and suggested_tool:
                # Get actual tool details to pass to LLM handler
                tool_info = tools.get_tool(suggested_tool)
                if tool_info:
                    specific_tool = suggested_tool
                    tool_note = f"Selected tool: {suggested_tool} - {tool_info.description}"
                    logger.info(tool_note)
                    
                    # We might add this to the system message
                    if additional_context:
                        additional_context += f"\n{tool_note}"
                    else:
                        additional_context = tool_note
            
            # Get LLM response using the LLM handler - pass the specific_tool if we have one
            logger.info("Generating LLM response...")
            try:
                # Make sure the LLM handler has access to the database manager for vector tool selection
                if not hasattr(self.llm_handler, 'db_manager') or self.llm_handler.db_manager is None:
                    self.llm_handler.db_manager = self.db_manager
                    logger.info("Set db_manager reference in LLM handler")
                
                # Pass specific_tool to generate_response if we identified one
                llm_response = self.llm_handler.generate_response(
                    user_input=user_message,
                    conversation_history=conversation_history,
                    additional_context=additional_context if additional_context else None,
                    include_tools=True,  # Always include tools, the LLM handler will filter as needed
                    specific_tool=specific_tool  # Pass the identified tool if any
                )
                logger.info(f"Received response of type: {type(llm_response)}")
            except Exception as e:
                logger.error(f"Error generating LLM response: {str(e)}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                raise
            
            # Handle response based on whether it's a simple string or a dict with tool calls
            logger.info("Processing LLM response...")
            if isinstance(llm_response, dict) and 'tool_calls' in llm_response and llm_response['tool_calls']:
                logger.info(f"Response contains tool calls: {llm_response.get('tool_calls')}")
                # Add the assistant message with tool calls
                self._add_to_short_term_with_tool_calls('assistant', llm_response['content'], llm_response['tool_calls'])
                
                # Process each tool call
                tool_results = []
                for tool_call in llm_response['tool_calls']:
                    try:
                        # Execute the tool
                        tool_result = tools.execute_tool_call(tool_call)
                        
                        # Get the tool name for more context
                        tool_name = tool_call['function']['name'] if 'function' in tool_call else "unknown_tool"
                        
                        # Enhance the raw tool response with natural language
                        from llm_utilities import enhance_tool_response
                        enhanced_response = enhance_tool_response(user_message, tool_name, tool_result)
                        
                        # Store the tool result in short-term memory
                        self._add_to_short_term('tool', enhanced_response, tool_call_id=tool_call.get('id', '0'), name=tool_name)
                        
                        # Add to results list for display
                        tool_results.append(enhanced_response)
                        logger.info(f"Tool '{tool_name}' executed successfully")
                    except Exception as e:
                        error_msg = f"Error executing tool: {str(e)}"
                        logger.error(error_msg)
                        tool_results.append(error_msg)
                
                # Create a combined display response
                if len(tool_results) == 1:
                    display_response = tool_results[0]
                else:
                    display_response = "Multiple tool results:\n" + "\n---\n".join(tool_results)
                
                # Add to chat history for UI
                self.chat_history.append((user_message, display_response))
                
                # Don't store tool-related interactions in long-term memory
                logger.info("Skipping memory storage for tool interaction")
                
                return display_response
            else:
                logger.info("Response is a simple text response")
                # Just a regular text response (or the LLM didn't make a tool call even though it should have)
                response_content = llm_response['content'] if isinstance(llm_response, dict) else llm_response
                self._add_to_short_term('assistant', response_content)
                self.chat_history.append((user_message, response_content))
                
                # Only store in long-term memory if this is NOT a tool interaction
                if not is_tool_query:
                    # Store user message in long-term memory
                    logger.info("Storing user message memory...")
                    try:
                        self.store_memory(user_message)
                        logger.info("User message stored successfully")
                    except Exception as e:
                        logger.error(f"Error storing user message: {str(e)}")
                    
                    # Store assistant response in long-term memory (if not a tool response)
                    logger.info("Storing assistant response memory...")
                    try:
                        # Check if this looks like a tool response by analyzing content
                        tool_response_patterns = [
                            r"Tool response from (\w+):",
                            r"I used the (\w+) tool",
                            r"According to the (\w+) tool",
                            r"Based on the (\w+) tool",
                            r"The (\w+) tool returned"
                        ]
                        
                        looks_like_tool_response = any(re.search(pattern, response_content) 
                                                       for pattern in tool_response_patterns)
                        
                        if not looks_like_tool_response:
                            self.store_memory(response_content)
                            logger.info("Memory stored successfully")
                        else:
                            logger.info("Response looks like tool output, skipping memory storage")
                    except Exception as e:
                        logger.error(f"Error storing memory: {str(e)}")
                else:
                    logger.info("Skipping memory storage for tool query response")
                
                return response_content
                
        except Exception as e:
            error_msg = f"Error processing request: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)  # For logging
            return error_msg
