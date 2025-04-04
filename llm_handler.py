import ollama
from config import LLM_CONFIG, EMBEDDING_CONFIG, LOGGING_CONFIG
import abc
import json
import requests
import re
from typing import List, Dict, Any, Optional, Tuple, Union
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import llama_cpp
import tools  # Import the new tools package
from llm_utilities import format_messages, extract_tool_information, should_use_tools, extract_tool_calls_from_response, enhance_tool_response, should_use_tools_vector
import logging
import datetime

# Configure logging system
def setup_logging(level=logging.INFO, enable_logging=True):
    """
    Setup logging with customized format including timestamps.
    
    Args:
        level: The logging level (default: INFO)
        enable_logging: Whether to enable logging (default: True)
    """
    if not enable_logging:
        logging.disable(logging.CRITICAL)
        return
    
    # Reset any existing handlers to avoid duplicates
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Configure logging format with timestamp, level, and module/class info
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

# Initialize logging with settings from LOGGING_CONFIG
setup_logging(
    level=getattr(logging, LOGGING_CONFIG.get('level', logging.INFO)),
    enable_logging=LOGGING_CONFIG.get('enable', True)
)

# Create logger instances for different components
llm_logger = logging.getLogger('LLMHandler')
backend_logger = logging.getLogger('LLMBackend')

class LLMHandler:
    """
    Main handler class for LLM operations, including response generation.
    This class interfaces with various backends based on configuration.
    """
    
    def __init__(self):
        # Initialize logger
        self.logger = llm_logger
        self.logger.info("[LLMHandler] Initializing...")
        
        # Initialize configuration parameters
        self.config = LLM_CONFIG
        
        # Optional references for vector-based tool selection
        self.vector_tool_selection = self.config.get('vector_tool_selection', False)
        self.db_manager = None  # Will be set externally by memory_manager
        
        # Set up the LLM backend
        self.backend = self._initialize_backend()
    
    def _initialize_backend(self) -> 'LLMBackend':
        """Initialize the appropriate LLM backend based on configuration"""
        backend_name = self.config.get('backend', 'ollama')
        self.logger.info(f"[LLMHandler] Initializing backend: {backend_name}")
        
        
        if backend_name.lower() == 'ollama':
            model_name = self.config.get('model_name', 'llama3')
            backend = OllamaBackend(model_name)
            backend._llm_handler = self  # Set reference back to this handler
            return backend
            
            # Commenting out all non-Ollama backends
            # elif backend_name.lower() == 'llamacpp':
            #     model_path = self.config.get('model_path', 'models/llama.gguf')
            #     context_size = self.config.get('context_size', 4096)
            #     temperature = self.config.get('temperature', 0.7)
            #     max_tokens = self.config.get('max_tokens', 1024)
            #     backend = LlamaCppBackend(model_path, context_size, temperature, max_tokens)
            #     backend._llm_handler = self  # Set reference back to this handler
            #     return backend
            #     
            # elif backend_name.lower() == 'llamafile':
            #     model_name = self.config.get('model_name', 'LLaMA_CPP')
            #     api_url = self.config.get('api_url', 'http://localhost:8080/v1')
            #     backend = LlamafileBackend(model_name, api_url)
            #     backend._llm_handler = self  # Set reference back to this handler
            #     return backend
                
        else:
            self.logger.error(f"[LLMHandler] Unknown backend type: {backend_name}")
            self.logger.info("[LLMHandler] Falling back to Ollama backend")
            backend = OllamaBackend(self.config.get('model_name', 'llama3'))
            backend._llm_handler = self  # Set reference back to this handler
            return backend
    
    def generate_response(self, user_input: str, conversation_history: List[Dict[str, Any]], 
                          additional_context: str = None,
                          include_tools: bool = True,
                          specific_tool: str = None) -> Union[str, Dict[str, Any]]:
        """Generate a response using the configured LLM backend"""
        try:
            # Create a new copy of the conversation history
            messages = conversation_history.copy() if conversation_history else []
            
            # Check if there's already a system message
            has_system = False
            for msg in messages:
                if msg.get('role') == 'system':
                    has_system = True
                    # If additional context is provided, append it to existing system message
                    if additional_context:
                        msg['content'] += f"\n\nAdditional context: {additional_context}"
                    
                    # If a specific tool is requested, explicitly instruct to use it
                    if specific_tool:
                        msg['content'] += f"\n\nIMPORTANT: For this query, you MUST use the '{specific_tool}' tool."
                    break
            
            # Add system message with system prompt from config if none exists
            if not has_system:
                system_content = self.config.get('system_prompt', "You are a helpful assistant.")
                if additional_context:
                    system_content += f"\n\nAdditional context: {additional_context}"
                
                # If a specific tool is requested, explicitly instruct to use it
                if specific_tool:
                    system_content += f"\n\nIMPORTANT: For this query, you MUST use the '{specific_tool}' tool."
                
                messages.insert(0, {
                    'role': 'system',
                    'content': system_content
                })
            
            # Add the latest user message if not already in conversation history
            if user_input and (not messages or messages[-1].get('role') != 'user'):
                messages.append({
                    'role': 'user',
                    'content': user_input
                })
            
            # Get available tools if requested
            available_tools = None
            if include_tools:
                try:
                    if specific_tool:
                        # Only get the specific tool that was requested
                        all_tools = tools.get_available_tools()
                        available_tools = []
                        for tool in all_tools:
                            if 'function' in tool and tool['function'].get('name') == specific_tool:
                                available_tools = [tool]
                                self.logger.info(f"[LLMHandler] Including only the specified tool: {specific_tool}")
                                break
                        if not available_tools:
                            self.logger.warning(f"[LLMHandler] Specified tool {specific_tool} not found in available tools")
                            available_tools = all_tools
                    else:
                        # Get all tools
                        available_tools = tools.get_available_tools()
                    self.logger.info(f"[LLMHandler] Retrieved {len(available_tools)} available tools")
                except Exception as e:
                    self.logger.warning(f"[LLMHandler] Error getting tools: {str(e)}")
            
            # Generate the response
            self.logger.info("[LLMHandler] Generating response...")
            response = self.backend.generate(messages, available_tools, specific_tool)
            
            # Check if this is a tool call that needs processing
            if response.get('tool_calls'):
                self.logger.info(f"[LLMHandler] Response contains {len(response['tool_calls'])} tool calls")
                # Return the full response including tool calls for further processing
                return response
            else:
                # Return just the content for simple text responses
                return response.get('content', "I couldn't generate a response.")
                
        except Exception as e:
            self.logger.error(f"[LLMHandler] Error generating response: {str(e)}")
            return f"Sorry, I encountered an error: {str(e)}"

# Utility functions for backend implementations have been moved to llm_utilities.py

class LLMBackend(abc.ABC):
    """Abstract base class for LLM backends"""
    
    def __init__(self):
        self.logger = logging.getLogger(f'LLMBackend.{self.__class__.__name__}')
        self._llm_handler = None  # Reference to the LLM handler, set externally
    
    @abc.abstractmethod
    def generate(self, messages: List[Dict[str, str]], tools: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """Generate a response from the model using the provided messages"""
        pass
    
    @abc.abstractmethod
    def supports_tool_use(self) -> bool:
        """Check if this backend/model supports tool use"""
        pass

# Commented out non-Ollama backend implementations
"""
class LlamafileBackend(LLMBackend):
    # Llamafile backend implementation using OpenAI-compatible API
    
    def __init__(self, model_name: str = None, api_url: str = None):
        super().__init__()
        self.api_url = api_url or "http://localhost:8080/v1"
        self.api_key = "sk-no-key-required"  # Llamafile doesn't require a real API key
        self.model_name = model_name or "LLaMA_CPP"  # Default model name for Llamafile
        
    def supports_tool_use(self) -> bool:
        # Most Llamafile models support tool use via the OpenAI function calling API
        return True
    
    def generate(self, messages: List[Dict[str, str]], tools: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        # Make sure we have a clean messages structure using the shared function
        formatted_messages = format_messages(messages)
        
        # Prepare the request parameters for OpenAI-compatible API
        params = {
            "model": self.model_name,
            "messages": formatted_messages,
            "temperature": 0.7,
            "max_tokens": 1024
        }
        
        # Add functions/tools if provided (using OpenAI format)
        if tools is not None:
            # Format tools for OpenAI-compatible function calling
            openai_functions = []
            for tool in tools:
                if "function" in tool:
                    openai_functions.append(tool["function"])
            
            if openai_functions:
                params["functions"] = openai_functions
                # Optional: You can set function_call to "auto" if you want the model to decide when to call functions
                params["function_call"] = "auto"
                self.logger.info(f"Including {len(openai_functions)} functions in request")
            else:
                self.logger.info("No valid functions found in tools")
        
        # Send to Llamafile OpenAI-compatible API
        try:
            # Make the HTTP POST request to the Llamafile API endpoint
            self.logger.info(f"Sending request to {self.api_url}/chat/completions")
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            response = requests.post(
                f"{self.api_url}/chat/completions", 
                headers=headers,
                json=params
            )
            response.raise_for_status()  # Raise exception for HTTP errors
            
            response_data = response.json()
            self.logger.info(f"Response received: {json.dumps(response_data)[:100]}...")
            
            # Extract the response content and tool calls using the shared function
            content = ""
            if "choices" in response_data and len(response_data["choices"]) > 0:
                message = response_data["choices"][0].get("message", {})
                content = message.get("content", "")
            
            # Use shared function to extract tool calls
            tool_calls = extract_tool_calls_from_response(response_data)
            
            return {
                'content': content,
                'tool_calls': tool_calls
            }
            
        except Exception as e:
            self.logger.error(f"Error: {str(e)}", exc_info=True)
            return {
                'content': f"Error generating response: {str(e)}",
                'tool_calls': []
            }
"""

class OllamaBackend(LLMBackend):
    """Ollama backend implementation"""
    
    def __init__(self, model_name: str):
        super().__init__()
        self.model_name = model_name
        # Extract base model name for feature detection
        self.base_model = model_name.split(':')[0] if ':' in model_name else model_name
    
    def supports_tool_use(self) -> bool:
        """All models are assumed to support tool use"""
        return True
    
    def generate(self, messages: List[Dict[str, str]], tools: Optional[List[Dict[str, Any]]] = None, specific_tool: Optional[str] = None) -> Dict[str, Any]:
        # Format messages using the shared utility function
        formatted_messages = []
        
        # Process and clean each message for Ollama's specific format
        for msg in format_messages(messages):
            if msg["role"] in ["system", "user", "assistant"]:
                # Keep the original format for system, user, and assistant messages
                formatted_messages.append(msg)
            elif msg["role"] == "tool":
                # For Ollama, convert tool responses to assistant messages
                formatted_messages.append({
                    "role": "assistant", 
                    "content": f"Tool response from {msg.get('name', 'unknown')}: {msg['content']}"
                })
        
        # Prepare the base request parameters
        params = {"model": self.model_name, "messages": formatted_messages, "stream": False}
        
        # Only continue with tool processing if tools are provided
        if tools is not None and len(tools) > 0:
            # Get the latest user message for tool selection
            latest_user_msg = ""
            for msg in reversed(messages):
                if msg["role"] == "user" and msg.get("content"):
                    latest_user_msg = msg["content"]
                    break
            
            if not latest_user_msg:
                self.logger.info("No user message found, skipping tool selection")
            else:
                # Check if we're in a tool conversation (if there's a previous tool response)
                in_tool_conversation = any(
                    msg.get("role") == "assistant" and "Tool response from" in msg.get("content", "")
                    for msg in formatted_messages
                )
                
                # If a specific tool is requested, override any other tool selection logic
                should_use_tool = False
                suggested_tool = None
                
                if specific_tool:
                    should_use_tool = True
                    suggested_tool = specific_tool
                    self.logger.info(f"Using specifically requested tool: {specific_tool}")
                # Otherwise, perform normal tool selection if needed
                elif not in_tool_conversation:
                    # Get reference to the LLM handler for vector tool selection
                    llm_handler = getattr(self, "_llm_handler", None)
                    
                    # Only use vector-based tool selection if we have the necessary components
                    if llm_handler and hasattr(llm_handler, "db_manager") and llm_handler.db_manager:
                        try:
                            # Use the centralized function from llm_utilities
                            from llm_utilities import should_use_tools_vector
                            # Use vector-based tool selection with the embedding from memory_manager
                            should_use_tool, suggested_tool = should_use_tools_vector(
                                latest_user_msg, 
                                None,  # We'll use memory_manager.get_embedding directly
                                llm_handler.db_manager,
                                similarity_threshold=0.70  # Adjust threshold to be more permissive
                            )
                            self.logger.info(f"Vector-based tool selection result: use={should_use_tool}, tool={suggested_tool or 'None'}")
                            
                            # Log the input that led to this decision for debugging
                            self.logger.debug(f"Tool decision based on user message: '{latest_user_msg[:50]}...'")
                        except Exception as e:
                            self.logger.error(f"Error in vector-based tool selection: {str(e)}")
                            should_use_tool = False
                
                # Add tools to the request when appropriate
                if should_use_tool or in_tool_conversation:
                    # If we have a specific suggested tool, only include that one tool
                    if should_use_tool and suggested_tool:
                        # Filter to only include the suggested tool
                        selected_tool_list = []
                        for tool in tools:
                            if "function" in tool and tool["function"].get("name") == suggested_tool:
                                selected_tool_list.append(tool)
                                params_schema = tool["function"].get("parameters", {})
                                self.logger.info(f"Enforcing use of tool: {suggested_tool}")
                                self.logger.debug(f"Tool parameters schema: {json.dumps(params_schema)}")
                                break
                        
                        if selected_tool_list:
                            # Only send the selected tool to the LLM
                            params["tools"] = selected_tool_list
                            
                            # Modify the system message to explicitly instruct the LLM to use the selected tool
                            system_message_modified = False
                            for idx, msg in enumerate(formatted_messages):
                                if msg["role"] == "system":
                                    # Add instruction to use the specified tool
                                    tool_instruction = f"\nFor this query, you MUST use the '{suggested_tool}' tool to get the information. " \
                                                      f"Please use this tool properly with appropriate parameters to answer the user's question."
                                    msg["content"] = msg["content"] + tool_instruction
                                    formatted_messages[idx] = msg
                                    system_message_modified = True
                                    break
                            
                            # If no system message exists, create one with the tool instruction
                            if not system_message_modified:
                                tool_instruction = f"You are a helpful assistant. For this query, you MUST use the '{suggested_tool}' tool " \
                                                 f"to get the information. Please use this tool properly with appropriate parameters to answer the user's question."
                                formatted_messages.insert(0, {"role": "system", "content": tool_instruction})
                            
                            # Update the messages in the parameters
                            params["messages"] = formatted_messages
                            self.logger.info(f"Enforcing tool use for '{suggested_tool}' via system message")
                            
                            # DEBUGGING: Print the full context being sent to the LLM
                            self.logger.debug("--- DEBUG: FULL CONTEXT BEING SENT TO LLM ---")
                            for i, msg in enumerate(params["messages"]):
                                self.logger.debug(f"Message {i} - Role: {msg['role']}")
                                self.logger.debug(f"Content: {msg['content'][:100]}...")
                            
                            self.logger.debug("--- Tool information being sent: ---")
                            for tool in params["tools"]:
                                if "function" in tool:
                                    self.logger.debug(f"Tool name: {tool['function'].get('name')}")
                        else:
                            # If we couldn't find the suggested tool, fall back to including all tools
                            params["tools"] = tools
                            self.logger.warning(f"Suggested tool '{suggested_tool}' not found in available tools, including all tools")
                    else:
                        # In a tool conversation or no specific tool suggested, include all tools
                        params["tools"] = tools
                        self.logger.info("Including all tools in request")
        
        # Send to Ollama API
        try:
            # Use the Ollama Python client to make the API call
            self.logger.info(f"Sending request to Ollama with model: {self.model_name}")
            if "tools" in params:
                self.logger.info(f"Request includes {len(params['tools'])} tools")
            
            response = ollama.chat(**params)
            content = response['message']['content']
            
            # DEBUGGING: Print the LLM's response
            self.logger.info("--- DEBUG: LLM RESPONSE ---")
            self.logger.info(f"Response content: {content[:500]}...")
            if len(content) > 500:
                self.logger.info("(Response content truncated for log)")
            self.logger.info("--- END LLM RESPONSE ---")
            
            # Check for tool calls in the response
            tool_calls = []
            
            # Extract tool calls from Ollama response
            if 'tool_calls' in response['message']:
                raw_tool_calls = response['message']['tool_calls']
                self.logger.info(f"Response contains {len(raw_tool_calls)} tool calls")
                
                # Process each tool call and ensure proper format
                for tool_call in raw_tool_calls:
                    # Helper function to safely convert non-serializable objects to dictionaries
                    def tool_call_to_dict(obj):
                        if hasattr(obj, '__dict__'):
                            return {k: tool_call_to_dict(v) for k, v in obj.__dict__.items() 
                                   if not k.startswith('_')}
                        elif isinstance(obj, dict):
                            return {k: tool_call_to_dict(v) for k, v in obj.items()}
                        elif isinstance(obj, (list, tuple)):
                            return [tool_call_to_dict(x) for x in obj]
                        else:
                            return obj
                    
                    try:
                        # Convert complex objects to dictionaries for logging
                        tool_call_dict = tool_call_to_dict(tool_call)
                        self.logger.debug(f"Processing tool call: {json.dumps(tool_call_dict)}")
                    except (TypeError, ValueError):
                        self.logger.debug(f"Processing tool call: {str(tool_call)}")
                    
                    # Standardize tool call format
                    formatted_tool_call = None
                    
                    # Case 1: Dictionary format with function field
                    if isinstance(tool_call, dict) and 'function' in tool_call:
                        # Already in the correct format
                        formatted_tool_call = tool_call.copy()
                        
                    # Case 2: Dictionary format with flat structure (name and arguments at top level)
                    elif isinstance(tool_call, dict) and 'name' in tool_call:
                        tool_name = tool_call.get('name')
                        tool_args = tool_call.get('arguments', {})
                        tool_call_id = tool_call.get('id', f"call_{len(tool_calls)}")
                        
                        # Parse string arguments to JSON if needed
                        if isinstance(tool_args, str):
                            try:
                                tool_args = json.loads(tool_args)
                            except json.JSONDecodeError:
                                self.logger.warning(f"Failed to parse tool arguments as JSON: {tool_args}")
                                tool_args = {}
                        
                        formatted_tool_call = {
                            'id': tool_call_id,
                            'function': {
                                'name': tool_name,
                                'arguments': tool_args
                            }
                        }
                        
                    # Case 3: Object-style tool calls (from newer Ollama versions)
                    else:
                        try:
                            tool_name = None
                            tool_args = {}
                            tool_id = None
                            
                            # Extract name
                            if hasattr(tool_call, 'name'):
                                tool_name = tool_call.name() if callable(getattr(tool_call, 'name', None)) else tool_call.name
                            
                            # Extract arguments
                            if hasattr(tool_call, 'arguments'):
                                tool_args = tool_call.arguments() if callable(getattr(tool_call, 'arguments', None)) else tool_call.arguments
                            
                            # Extract ID
                            if hasattr(tool_call, 'id'):
                                tool_id = tool_call.id() if callable(getattr(tool_call, 'id', None)) else tool_call.id
                            else:
                                tool_id = f"call_{len(tool_calls)}"
                            
                            # Parse string arguments if needed
                            if isinstance(tool_args, str):
                                try:
                                    tool_args = json.loads(tool_args)
                                except json.JSONDecodeError:
                                    self.logger.warning(f"Failed to parse tool arguments as JSON: {tool_args}")
                                    tool_args = {}
                            
                            # Special case: Handle date/time related queries
                            if tool_name in ["unknown_tool", None] and "time" in latest_user_msg.lower() or "date" in latest_user_msg.lower():
                                tool_name = "get_date_time"
                                self.logger.info(f"Resolving unknown tool to 'get_date_time' based on query context")
                            
                            # Only add to formatted list if we have a tool name
                            if tool_name:
                                formatted_tool_call = {
                                    'id': tool_id,
                                    'function': {
                                        'name': tool_name,
                                        'arguments': tool_args
                                    }
                                }
                            else:
                                self.logger.warning(f"Skipping tool call due to missing tool name")
                                
                        except Exception as e:
                            self.logger.error(f"Error processing object-style tool call: {str(e)}")
                            self.logger.error(f"Tool call type: {type(tool_call)}")
                            # Continue with next tool call
                            continue
                    
                    # Add the formatted tool call if valid
                    if formatted_tool_call:
                        # Log the tool call before adding
                        tool_name = formatted_tool_call['function'].get('name', 'unknown') if 'function' in formatted_tool_call else 'unknown'
                        tool_args = formatted_tool_call['function'].get('arguments', {}) if 'function' in formatted_tool_call else {}
                        self.logger.info(f"Adding tool call: {tool_name} with arguments: {json.dumps(tool_args)}")
                        
                        # Add to the list of tool calls
                        tool_calls.append(formatted_tool_call)
                
                # Log the extracted parameters for verification
                for idx, tool_call in enumerate(tool_calls):
                    if 'function' in tool_call:
                        func_name = tool_call['function'].get('name', 'unknown')
                        func_args = tool_call['function'].get('arguments', {})
                        self.logger.info(f"Tool call {idx+1}: {func_name} with arguments: {json.dumps(func_args)}")
            else:
                self.logger.warning("LLM response did not contain any tool_calls")
            
            # If we have an enforced tool but no tool calls were made, try to extract parameters from content
            # and create a synthetic tool call
            if should_use_tool and suggested_tool and not tool_calls:
                self.logger.warning(f"LLM didn't use the enforced tool '{suggested_tool}'. Attempting to extract parameters from content.")
                
                # Implement a simple parameter extraction for date/time tool - this can be enhanced later
                if suggested_tool == "get_date_time":
                    try:
                        # Create a synthetic tool call with default parameters
                        synthetic_tool_call = {
                            'id': 'synthetic_call_0',
                            'function': {
                                'name': 'get_date_time',
                                'arguments': {}  # Empty arguments will use defaults
                            }
                        }
                        
                        self.logger.info("Created synthetic tool call for get_date_time with default parameters")
                        tool_calls.append(synthetic_tool_call)
                    except Exception as e:
                        self.logger.error(f"Error creating synthetic tool call: {str(e)}")
            
            return {
                'content': content,
                'tool_calls': tool_calls
            }
        except Exception as e:
            self.logger.error(f"Error generating response: {str(e)}", exc_info=True)
            return {
                'content': f"Error generating response: {str(e)}",
                'tool_calls': []
            }

"""
class LlamaCppBackend(LLMBackend):
    #llama.cpp backend implementation
    
    def __init__(self, model_path: str, context_size: int = 4096, 
                 temperature: float = 0.7, max_tokens: int = 1024):
        super().__init__()
        self.model_path = model_path
        self.temperature = temperature  # Store temperature as an instance variable
        self.context_size = context_size
        self.max_tokens = max_tokens
        
        # We'll load models with different formats depending on the request
        self._llm_instances = {}
        
        # Try to extract model name from path for feature detection
        path_parts = model_path.split('/')
        filename = path_parts[-1] if path_parts else ""
        self.base_model = filename.split('.')[0] if '.' in filename else filename
        
        self.logger.info(f"Initialized for model: {model_path}")
        
    def _get_llm(self, chat_format="chatml"):
        # Get or create a llama.cpp instance with the specified chat format
        if chat_format not in self._llm_instances:
            self.logger.info(f"Creating new llm instance with format: {chat_format}")
            try:
                self._llm_instances[chat_format] = llama_cpp.Llama(
                    model_path=self.model_path,
                    n_ctx=self.context_size,
                    temperature=self.temperature,
                    chat_format=chat_format
                )
                self.logger.info(f"Model loaded successfully with {chat_format} format")
            except Exception as e:
                self.logger.error(f"Error loading model with {chat_format} format: {str(e)}", exc_info=True)
                raise
        return self._llm_instances[chat_format]
        
    def supports_tool_use(self) -> bool:
        # All models are assumed to support tool use
        return True
    
    def generate(self, messages: List[Dict[str, str]], tools: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        # Make sure we have clean messages using the shared utility function
        formatted_messages = []
        for msg in format_messages(messages):
            if msg["role"] in ["system", "user", "assistant"]:
                formatted_messages.append(msg)
            elif msg["role"] == "tool":
                # For tool responses, convert to llama.cpp format
                formatted_messages.append({
                    "role": "function",
                    "name": msg.get("name", "unknown_tool"),
                    "content": msg["content"]
                })
        
        try:
            self.logger.info(f"Processing {len(formatted_messages)} messages")
            # Convert tools to llama.cpp format if provided
            llama_tools = None
            tool_choice = None
            
            if tools is not None and len(tools) > 0:
                self.logger.info(f"Processing {len(tools)} tools")
                llama_tools = []
                for tool in tools:
                    # Convert from our format to llama.cpp expected format
                    if "function" in tool:
                        llama_tools.append({
                            "type": "function",
                            "function": tool["function"]
                        })
                self.logger.info(f"Converted {len(llama_tools)} tools for llama.cpp")
            
            # Get the latest user message
            latest_user_msg = ""
            for msg in reversed(formatted_messages):
                if msg["role"] == "user" and msg.get("content"):
                    latest_user_msg = msg["content"].lower()
                    break
            
            # Try to get LLM handler reference to use vector-based tool selection
            llm_handler = getattr(self, "_llm_handler", None)
            if hasattr(llm_handler, "db_manager") and llm_handler.db_manager and latest_user_msg:
                # Use vector-based tool selection
                should_use_function_calling, suggested_tool = should_use_tools_vector(
                    latest_user_msg,
                    None,  # Use memory_manager.get_embedding directly
                    llm_handler.db_manager
                )
                self.logger.info(f": use={should_use_function_calling}, tool={suggested_tool or 'None'}")
            else:
                # Fall back to keyword-based tool selection
                should_use_function_calling, suggested_tool = should_use_tools(messages, tools)
                self.logger.info(f"Keyword-based tool selection result: use={should_use_function_calling}, tool={suggested_tool or 'None'}")
            
            # Check if a specific tool is directly mentioned to force its use
            if should_use_function_calling and llama_tools:
                # Look for explicit mentions of tools
                for tool in tools:
                    if "function" in tool:
                        tool_name = tool["function"].get("name", "").lower()
                        if tool_name and tool_name in latest_user_msg:
                            # Force the model to use this specific tool
                            tool_choice = {
                                "type": "function",
                                "function": {
                                    "name": tool_name
                                }
                            }
                            self.logger.info(f"Forcing use of tool: {tool_name}")
                            break
                
                # If a specific tool was suggested, use it
                if suggested_tool:
                    for tool in tools:
                        if "function" in tool and tool["function"].get("name") == suggested_tool:
                            tool_choice = {
                                "type": "function",
                                "function": {
                                    "name": suggested_tool
                                }
                            }
                            self.logger.info(f"Using suggested tool: {suggested_tool}")
                            break
            
            # Try function calling first if needed, otherwise use standard chat
            response = None
            if should_use_function_calling:
                self.logger.info("Using function calling mode (chatml-function-calling)")
                try:
                    llm_func = self._get_llm("chatml-function-calling")
                    response = llm_func.create_chat_completion(
                        messages=formatted_messages,
                        tools=llama_tools,
                        tool_choice=tool_choice,
                        max_tokens=self.max_tokens,
                        temperature=self.temperature
                    )
                    self.logger.info("Function calling response received")
                    
                    # Check if the response actually contains tool calls or is empty
                    if "choices" in response and len(response["choices"]) > 0:
                        message = response["choices"][0].get("message", {})
                        content = message.get("content", "")
                        tool_calls = message.get("tool_calls", [])
                        
                        # If no tool calls and empty/short content, fall back to standard chat
                        if not tool_calls and (not content or len(content.strip()) < 10):
                            self.logger.info("Function calling resulted in empty response, falling back to standard chat")
                            response = None  # Clear response to trigger fallback
                    else:
                        self.logger.info("Function calling returned invalid response, falling back to standard chat")
                        response = None  # Clear response to trigger fallback
                        
                except Exception as e:
                    self.logger.error(f"Error with function calling: {str(e)}", exc_info=True)
                    response = None  # Clear response to trigger fallback
            
            # Fall back to standard chat if function calling wasn't used or failed
            if response is None:
                self.logger.info("Using standard chat mode (chatml)")
                try:
                    llm_std = self._get_llm("chatml")
                    response = llm_std.create_chat_completion(
                        messages=formatted_messages,
                        max_tokens=self.max_tokens,
                        temperature=self.temperature
                    )
                    self.logger.info("Standard chat response received")
                except Exception as e:
                    self.logger.error(f"Error with standard chat: {str(e)}", exc_info=True)
                    # If even the fallback fails, return an error
                    return {
                        'content': f"Error generating response: {str(e)}",
                        'tool_calls': []
                    }
            
            # Extract the content and any tool calls from the response
            if "choices" not in response or len(response["choices"]) == 0:
                self.logger.info("No choices in response!")
                return {
                    'content': "I'm sorry, I couldn't generate a proper response. Please try again.",
                    'tool_calls': []
                }
                
            message = response["choices"][0].get("message", {})
            content = message.get("content", "")
            
            if not content or content.strip() == "":
                self.logger.info("Empty content in response!")
                content = "I'm sorry, I couldn't generate a proper response. Please try again."
            
            # Use shared function to extract tool calls
            tool_calls = extract_tool_calls_from_response(response)
            
            self.logger.info(f"Final content length: {len(content)}")
            self.logger.info(f"Final tool calls: {len(tool_calls)}")
            return {
                'content': content,
                'tool_calls': tool_calls
            }
        except Exception as e:
            import traceback
            self.logger.error(f"Error in generate method: {str(e)}", exc_info=True)
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return {
                'content': f"Error generating response: {str(e)}",
                'tool_calls': []
            }
"""