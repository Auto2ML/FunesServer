import ollama
from sentence_transformers import SentenceTransformer
from config import LLM_CONFIG, EMBEDDING_CONFIG, TOOL_CONFIG
import abc
import json
import requests
import re
from typing import List, Dict, Any, Optional, Tuple, Union
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import llama_cpp
import tools  # Import the new tools package

# Utility functions for backend implementations
def format_messages(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Format messages consistently for all backends"""
    formatted_messages = []
    
    # Process and clean each message
    for msg in messages:
        if msg["role"] in ["system", "user", "assistant"]:
            message_obj = {
                "role": msg["role"],
                "content": msg["content"].strip() if msg["content"] is not None else ""
            }
            # Include tool_calls if present in the message
            if "tool_calls" in msg and msg["tool_calls"]:
                message_obj["tool_calls"] = msg["tool_calls"]
            formatted_messages.append(message_obj)
        elif msg["role"] == "tool":
            # For tool responses, format for various backends
            formatted_messages.append({
                "role": "tool",
                "tool_call_id": msg.get("tool_call_id", "unknown"),
                "name": msg.get("name", "unknown_tool"),
                "content": msg["content"]
            })
    
    return formatted_messages

def extract_tool_information(tools: List[Dict[str, Any]]) -> Tuple[List[str], List[str]]:
    """Extract tool names and relevant keywords from tools definition"""
    tool_names = []
    tool_relevant_keywords = []
    
    # Extract tool names and keywords from function descriptions
    for tool in tools:
        if "function" in tool:
            name = tool["function"].get("name", "").lower()
            if name:
                tool_names.append(name)
                # Add keywords from the function description
                desc = tool["function"].get("description", "").lower()
                if desc:
                    for keyword in ["weather", "time", "date", "day", "extract", "get", "search"]:
                        if keyword in desc and keyword not in tool_relevant_keywords:
                            tool_relevant_keywords.append(keyword)
    
    return tool_names, tool_relevant_keywords

def should_use_tools(messages: List[Dict[str, str]], tools: List[Dict[str, Any]]) -> Tuple[bool, Optional[str]]:
    """Determine if tools should be used and which specific tool might be needed"""
    if not tools or len(tools) == 0:
        return False, None
        
    # Default keywords that suggest tool use
    tool_relevant_keywords = [
        "weather", "temperature", "forecast",  # Weather tool
        "time", "date", "today", "now", "current time", "current date",  # Date/time tool
        "latest", "current", "right now", "today's",  # Real-time info
        "extract", "parse", "get", "find", "search",  # Extraction verbs
        "convert", "calculate", "compute"  # Calculation verbs
    ]
    
    # Extraction patterns that suggest tool use
    extraction_patterns = ["what is", "how many", "who is", "when is", "extract", "parse"]
    
    # Get the latest user message
    latest_user_msg = ""
    for msg in reversed(messages):
        if msg["role"] == "user" and msg.get("content"):
            latest_user_msg = msg["content"].lower()
            break
    
    # Get tool names and additional keywords
    tool_names, additional_keywords = extract_tool_information(tools)
    tool_relevant_keywords.extend(additional_keywords)
    
    # Check if any tool names are directly mentioned
    for name in tool_names:
        if name in latest_user_msg:
            return True, name
    
    # Check for relevant keywords that suggest tool use
    for keyword in tool_relevant_keywords:
        if keyword in latest_user_msg:
            # Attempt to match specific tools to keywords
            if keyword in ["weather", "temperature", "forecast", "rain", "sunny"]:
                return True, "get_weather"
            elif keyword in ["time", "date", "today", "now", "current time", "current date"]:
                return True, "get_date_time"
            return True, None
    
    # Check for extraction patterns
    for pattern in extraction_patterns:
        if pattern in latest_user_msg:
            return True, None
    
    # Default to not using tools
    return False, None

def extract_tool_calls_from_response(response_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract tool calls from various response formats"""
    tool_calls = []
    
    # Try to get message from the response
    message = {}
    if "choices" in response_data and len(response_data["choices"]) > 0:
        message = response_data["choices"][0].get("message", {})
    else:
        message = response_data.get("message", {})
    
    # Check for tool_calls in the message
    if "tool_calls" in message:
        # Handle OpenAI/LlamaCpp format
        raw_tool_calls = message["tool_calls"]
        for tool_call in raw_tool_calls:
            if "type" in tool_call and tool_call["type"] == "function":
                tool_calls.append({
                    "function": {
                        "name": tool_call["function"]["name"],
                        "arguments": tool_call["function"]["arguments"]
                    },
                    "id": tool_call.get("id", f"call_{len(tool_calls)}")
                })
            elif "function" in tool_call:
                tool_calls.append({
                    "function": tool_call["function"],
                    "id": tool_call.get("id", f"call_{len(tool_calls)}")
                })
    
    # Check for function_call in the message (older OpenAI format)
    elif "function_call" in message:
        function_call = message["function_call"]
        tool_calls.append({
            "function": {
                "name": function_call.get("name", ""),
                "arguments": function_call.get("arguments", "{}")
            },
            "id": f"call_0"
        })
    
    return tool_calls

class LLMBackend(abc.ABC):
    """Abstract base class for LLM backends"""
    
    @abc.abstractmethod
    def generate(self, messages: List[Dict[str, str]], tools: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """Generate a response from the model using the provided messages"""
        pass
    
    @abc.abstractmethod
    def supports_tool_use(self) -> bool:
        """Check if this backend/model supports tool use"""
        pass

class LlamafileBackend(LLMBackend):
    """Llamafile backend implementation using OpenAI-compatible API"""
    
    def __init__(self, model_name: str = None, api_url: str = None):
        self.api_url = api_url or "http://localhost:8080/v1"
        self.api_key = "sk-no-key-required"  # Llamafile doesn't require a real API key
        self.model_name = model_name or "LLaMA_CPP"  # Default model name for Llamafile
        
    def supports_tool_use(self) -> bool:
        """Most Llamafile models support tool use via the OpenAI function calling API"""
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
                print(f"[LlamafileBackend] Including {len(openai_functions)} functions in request")
            else:
                print("[LlamafileBackend] No valid functions found in tools")
        
        # Send to Llamafile OpenAI-compatible API
        try:
            # Make the HTTP POST request to the Llamafile API endpoint
            print(f"[LlamafileBackend] Sending request to {self.api_url}/chat/completions")
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
            print(f"[LlamafileBackend] Response received: {json.dumps(response_data)[:100]}...")
            
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
            print(f"[LlamafileBackend] Error: {str(e)}")
            return {
                'content': f"Error generating response: {str(e)}",
                'tool_calls': []
            }

class OllamaBackend(LLMBackend):
    """Ollama backend implementation"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        # Extract base model name for feature detection
        self.base_model = model_name.split(':')[0] if ':' in model_name else model_name
    
    def supports_tool_use(self) -> bool:
        """All models are assumed to support tool use"""
        return True
    
    def generate(self, messages: List[Dict[str, str]], tools: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
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
        
        # Prepare the request parameters
        params = {"model": self.model_name, "messages": formatted_messages, "stream": False}
        
        # Only add tools if provided AND the request seems appropriate for tool use
        if tools is not None:
            # Use the shared function to determine if tools should be used
            should_use_tool, suggested_tool = should_use_tools(messages, tools)
            
            # Always include tools when in a tool conversation (if there's a previous tool response)
            in_tool_conversation = any(msg.get("role") == "assistant" and "Tool response from" in msg.get("content", "") 
                                   for msg in formatted_messages)
            
            # Add tools to the request when appropriate
            if should_use_tool or in_tool_conversation:
                params["tools"] = tools
                print("[OllamaBackend] Including tools in request based on content analysis")
                if suggested_tool:
                    print(f"[OllamaBackend] Suggested tool: {suggested_tool}")
            else:
                print("[OllamaBackend] Omitting tools from request - query appears to be general knowledge")
        
        # Send to Ollama API
        try:
            # Use the Ollama Python client to make the API call
            response = ollama.chat(**params)
            content = response['message']['content']
            
            # Check for tool calls in the response
            tool_calls = []
            
            # For newer Ollama versions that support tools natively
            if 'tool_calls' in response['message']:
                tool_calls = response['message']['tool_calls']
                
                # Ensure the tool_calls are in the correct format
                for tool_call in tool_calls:
                    # Make sure each tool call has a function field with name and arguments
                    if 'function' not in tool_call and 'name' in tool_call:
                        # Convert from flat format to nested format
                        tool_call['function'] = {
                            'name': tool_call.pop('name'),
                            'arguments': tool_call.pop('arguments', {})
                        }
            
            return {
                'content': content,
                'tool_calls': tool_calls
            }
            
        except Exception as e:
            return {
                'content': f"Error generating response: {str(e)}",
                'tool_calls': []
            }

class LlamaCppBackend(LLMBackend):
    """llama.cpp backend implementation"""
    
    def __init__(self, model_path: str, context_size: int = 4096, 
                 temperature: float = 0.7, max_tokens: int = 1024):
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
        
        print(f"[LlamaCppBackend] Initialized for model: {model_path}")
        
    def _get_llm(self, chat_format="chatml"):
        """Get or create a llama.cpp instance with the specified chat format"""
        if chat_format not in self._llm_instances:
            print(f"[LlamaCppBackend] Creating new llm instance with format: {chat_format}")
            try:
                self._llm_instances[chat_format] = llama_cpp.Llama(
                    model_path=self.model_path,
                    n_ctx=self.context_size,
                    temperature=self.temperature,
                    chat_format=chat_format
                )
                print(f"[LlamaCppBackend] Model loaded successfully with {chat_format} format")
            except Exception as e:
                print(f"[LlamaCppBackend] Error loading model with {chat_format} format: {str(e)}")
                raise
        return self._llm_instances[chat_format]
        
    def supports_tool_use(self) -> bool:
        """All models are assumed to support tool use"""
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
            print(f"[LlamaCppBackend] Processing {len(formatted_messages)} messages")
            # Convert tools to llama.cpp format if provided
            llama_tools = None
            tool_choice = None
            
            if tools is not None and len(tools) > 0:
                print(f"[LlamaCppBackend] Processing {len(tools)} tools")
                llama_tools = []
                for tool in tools:
                    # Convert from our format to llama.cpp expected format
                    if "function" in tool:
                        llama_tools.append({
                            "type": "function",
                            "function": tool["function"]
                        })
                print(f"[LlamaCppBackend] Converted {len(llama_tools)} tools for llama.cpp")
            
            # Use shared utility to determine if we should use function calling
            should_use_function_calling, suggested_tool = should_use_tools(messages, tools)
            
            # Get the latest user message to check for specific tool mentions
            latest_user_msg = ""
            for msg in reversed(formatted_messages):
                if msg["role"] == "user" and msg.get("content"):
                    latest_user_msg = msg["content"].lower()
                    break
            
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
                            print(f"[LlamaCppBackend] Forcing use of tool: {tool_name}")
                            break
                
                # If time/date is mentioned, force the datetime tool
                if any(keyword in latest_user_msg for keyword in ["time", "date", "today", "now", "current time", "current date"]):
                    for tool in tools:
                        if "function" in tool and tool["function"].get("name") == "get_date_time":
                            tool_choice = {
                                "type": "function",
                                "function": {
                                    "name": "get_date_time"
                                }
                            }
                            print("[LlamaCppBackend] Forcing use of get_date_time tool")
                            break
                            
                # If weather is mentioned, force the weather tool
                if any(keyword in latest_user_msg for keyword in ["weather", "temperature", "forecast", "rain", "sunny"]):
                    for tool in tools:
                        if "function" in tool and tool["function"].get("name") == "get_weather":
                            tool_choice = {
                                "type": "function",
                                "function": {
                                    "name": "get_weather"
                                }
                            }
                            print("[LlamaCppBackend] Forcing use of get_weather tool")
                            break
                
                # If a specific tool was suggested by the shared function, use it
                if suggested_tool:
                    for tool in tools:
                        if "function" in tool and tool["function"].get("name") == suggested_tool:
                            tool_choice = {
                                "type": "function",
                                "function": {
                                    "name": suggested_tool
                                }
                            }
                            print(f"[LlamaCppBackend] Using suggested tool: {suggested_tool}")
                            break
            
            # Try function calling first if needed, otherwise use standard chat
            response = None
            if should_use_function_calling:
                print("[LlamaCppBackend] Using function calling mode (chatml-function-calling)")
                try:
                    llm_func = self._get_llm("chatml-function-calling")
                    response = llm_func.create_chat_completion(
                        messages=formatted_messages,
                        tools=llama_tools,
                        tool_choice=tool_choice,
                        max_tokens=self.max_tokens,
                        temperature=self.temperature
                    )
                    print(f"[LlamaCppBackend] Function calling response received")
                    
                    # Check if the response actually contains tool calls or is empty
                    if "choices" in response and len(response["choices"]) > 0:
                        message = response["choices"][0].get("message", {})
                        content = message.get("content", "")
                        tool_calls = message.get("tool_calls", [])
                        
                        # If no tool calls and empty/short content, fall back to standard chat
                        if not tool_calls and (not content or len(content.strip()) < 10):
                            print("[LlamaCppBackend] Function calling resulted in empty response, falling back to standard chat")
                            response = None  # Clear response to trigger fallback
                    else:
                        print("[LlamaCppBackend] Function calling returned invalid response, falling back to standard chat")
                        response = None  # Clear response to trigger fallback
                        
                except Exception as e:
                    print(f"[LlamaCppBackend] Error with function calling: {str(e)}")
                    response = None  # Clear response to trigger fallback
            
            # Fall back to standard chat if function calling wasn't used or failed
            if response is None:
                print("[LlamaCppBackend] Using standard chat mode (chatml)")
                try:
                    llm_std = self._get_llm("chatml")
                    response = llm_std.create_chat_completion(
                        messages=formatted_messages,
                        max_tokens=self.max_tokens,
                        temperature=self.temperature
                    )
                    print(f"[LlamaCppBackend] Standard chat response received")
                except Exception as e:
                    print(f"[LlamaCppBackend] Error with standard chat: {str(e)}")
                    # If even the fallback fails, return an error
                    return {
                        'content': f"Error generating response: {str(e)}",
                        'tool_calls': []
                    }
            
            # Extract the content and any tool calls from the response
            if "choices" not in response or len(response["choices"]) == 0:
                print("[LlamaCppBackend] No choices in response!")
                return {
                    'content': "I'm sorry, I couldn't generate a proper response. Please try again.",
                    'tool_calls': []
                }
                
            message = response["choices"][0].get("message", {})
            content = message.get("content", "")
            
            if not content or content.strip() == "":
                print("[LlamaCppBackend] Empty content in response!")
                content = "I'm sorry, I couldn't generate a proper response. Please try again."
            
            # Use shared function to extract tool calls
            tool_calls = extract_tool_calls_from_response(response)
            
            print(f"[LlamaCppBackend] Final content length: {len(content)}")
            print(f"[LlamaCppBackend] Final tool calls: {len(tool_calls)}")
            return {
                'content': content,
                'tool_calls': tool_calls
            }
        except Exception as e:
            import traceback
            print(f"[LlamaCppBackend] Error in generate method: {str(e)}")
            print(f"[LlamaCppBackend] Traceback: {traceback.format_exc()}")
            return {
                'content': f"Error generating response: {str(e)}",
                'tool_calls': []
            }

class HuggingFaceBackend(LLMBackend):
    """HuggingFace backend implementation"""
    
    def __init__(self, model_name: str, device: str = "cpu"):
        self.model_name = model_name
        self.base_model = model_name.split('/')[-1] if '/' in model_name else model_name
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            device_map=device,
            torch_dtype="auto"
        )
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer
        )
    
    def supports_tool_use(self) -> bool:
        """All models are assumed to support tool use"""
        return True
    
    def generate(self, messages: List[Dict[str, str]], tools: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        # Use the shared function to format messages
        formatted_messages = format_messages(messages)
        
        # Convert messages to a prompt format
        prompt = ""
        for msg in formatted_messages:
            role = msg["role"]
            content = msg["content"]
            
            # If this is the system message and tools are provided, add tool descriptions
            if role == "system" and tools is not None:
                # Check if we should use tools for this request
                should_use_tool, suggested_tool = should_use_tools(messages, tools)
                if should_use_tool:
                    tools_json = json.dumps(tools, indent=2)
                    content += f"\n\nYou have access to the following tools:\n{tools_json}\n"
                    content += "\nWhen a user request requires using these tools, respond with a JSON object containing 'tool_calls' with the format: [{\"name\": \"tool_name\", \"arguments\": {...}}]."
                    
                    if suggested_tool:
                        content += f"\n\nThe tool '{suggested_tool}' might be especially helpful for this request."
            
            if role == "system":
                prompt += f"<|system|>\n{content}\n"
            elif role == "user":
                prompt += f"<|user|>\n{content}\n"
            elif role == "assistant":
                prompt += f"<|assistant|>\n{content}\n"
        
        prompt += "<|assistant|>\n"
        
        try:
            result = self.pipe(
                prompt,
                max_new_tokens=1024,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )[0]["generated_text"]
            
            # Extract only the assistant's response
            assistant_response = result.split("<|assistant|>\n")[-1].strip()
            
            # Try to extract any tool calls from the text format
            tool_calls = []
            
            # Look for JSON-like tool call format in the response
            tool_call_match = re.search(r'\{\s*"tool_calls"\s*:\s*(\[.*?\])\s*\}', assistant_response, re.DOTALL)
            if tool_call_match:
                try:
                    tool_calls_json = tool_call_match.group(1)
                    # Parse the extracted JSON
                    parsed_tool_calls = json.loads(tool_calls_json)
                    
                    # Convert to standard format
                    for i, call in enumerate(parsed_tool_calls):
                        if "name" in call and "arguments" in call:
                            tool_calls.append({
                                "function": {
                                    "name": call["name"],
                                    "arguments": json.dumps(call["arguments"]) if isinstance(call["arguments"], dict) else call["arguments"]
                                },
                                "id": f"call_{i}"
                            })
                except:
                    # If JSON parsing fails, leave tool_calls empty
                    pass
            
            # Also check for function_call format
            function_call_match = re.search(r'\{\s*"function"\s*:\s*\{\s*"name"\s*:\s*"([^"]+)"\s*,\s*"arguments"\s*:\s*(\{.*?\})\s*\}\s*\}', assistant_response, re.DOTALL)
            if function_call_match and not tool_calls:
                try:
                    function_name = function_call_match.group(1)
                    function_args = function_call_match.group(2)
                    
                    tool_calls.append({
                        "function": {
                            "name": function_name,
                            "arguments": function_args
                        },
                        "id": "call_0"
                    })
                except:
                    # If extraction fails, leave tool_calls empty
                    pass
            
            # Clean the response if we found tool calls
            if tool_calls:
                # Remove the JSON part from the response
                assistant_response = re.sub(r'\{\s*"tool_calls"\s*:\s*\[.*?\]\s*\}', '', assistant_response, flags=re.DOTALL)
                assistant_response = re.sub(r'\{\s*"function"\s*:\s*\{.*?\}\s*\}', '', assistant_response, flags=re.DOTALL)
                assistant_response = assistant_response.strip()
            
            return {
                'content': assistant_response,
                'tool_calls': tool_calls
            }
        except Exception as e:
            return {
                'content': f"Error generating response: {str(e)}",
                'tool_calls': []
            }

class LLMHandler:
    def __init__(self, embedding_model=None, llm_model=None, backend_type=None):
        # Initialize embedding model with config or override
        embedding_model_name = embedding_model or EMBEDDING_CONFIG['model_name']
        self.embedding_model = SentenceTransformer(embedding_model_name)
        
        # Initialize LLM model with config or override
        llm_model_name = llm_model or LLM_CONFIG['model_name']
        backend_type = backend_type or LLM_CONFIG.get('backend_type', 'ollama')  # Get from config or default to ollama
        
        # Create the appropriate backend
        if backend_type.lower() == "ollama":
            self.llm_backend = OllamaBackend(llm_model_name)
        elif backend_type.lower() == "llamacpp":
            self.llm_backend = LlamaCppBackend(llm_model_name)
        elif backend_type.lower() == "huggingface":
            self.llm_backend = HuggingFaceBackend(llm_model_name)
        elif backend_type.lower() == "llamafile":
            # Get API URL from config or use default
            api_url = LLM_CONFIG.get('llamafile_api_url', "http://localhost:8080/api")
            self.llm_backend = LlamafileBackend(llm_model_name, api_url)
        else:
            raise ValueError(f"Unsupported backend type: {backend_type}")
        
        # System prompt from config
        self.system_prompt = LLM_CONFIG.get('system_prompt', '')
        
        # Tool use is always enabled
        self.enable_tools = True
        
        # Load tools
        self.tools = tools.list_tools()
        
        # Print model information
        print(f"Using model: {llm_model_name} with {backend_type} backend")
        print(f"Tools are always enabled in Funes")
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts"""
        return self.embedding_model.encode(texts).tolist()
    
    def get_single_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text"""
        return self.embedding_model.encode(text).tolist()
    
    def generate_response(self, 
                         user_input: str, 
                         conversation_history: Optional[List[Dict[str, str]]] = None, 
                         additional_context: Optional[str] = None) -> Union[str, Dict[str, Any]]:
        """
        Generate a response using the LLM
        
        Args:
            user_input: The user's query
            conversation_history: Optional list of previous conversation messages
            additional_context: Optional context information to add to the system prompt
            
        Returns:
            Either a string response or a dict with 'content' and 'tool_calls' if tools are used
        """
        try:
            print("[LLMHandler] Starting generate_response...")
            
            # Initialize messages with system prompt
            messages = []
            
            # Add system message with additional context if provided
            system_message = self.system_prompt
            if additional_context:
                print(f"[LLMHandler] Adding additional context of length {len(additional_context)}")
                system_message += f"\n\nContext: {additional_context}"
            
            # Always add tool information to system prompt
            tool_use_prompt = LLM_CONFIG.get('tool_use_prompt', '')
            if tool_use_prompt:
                print("[LLMHandler] Adding tool use prompt...")
                try:
                    tools_description = tools.get_tools_description()
                    print(f"[LLMHandler] Tools description: {tools_description[:100]}...")
                    
                    # Safely format the prompt with the tools description
                    try:
                        formatted_prompt = tool_use_prompt.format(tools_description=tools_description)
                        system_message += "\n\n" + formatted_prompt
                        print("[LLMHandler] Tool use prompt added successfully")
                    except KeyError as e:
                        # Handle string formatting errors specifically
                        print(f"[LLMHandler] Error formatting tool use prompt: {str(e)}")
                        # Add the tool use prompt without formatting as a fallback
                        system_message += "\n\n" + tool_use_prompt.replace("{tools_description}", tools_description)
                        print("[LLMHandler] Added tool use prompt with manual replacement instead")
                except Exception as e:
                    import traceback
                    print(f"[LLMHandler] Error getting tools description: {str(e)}")
                    print(f"[LLMHandler] Traceback: {traceback.format_exc()}")
                    # Add a generic tool prompt as fallback
                    tools_list = [tool.get("function", {}).get("name", "unknown") for tool in self.tools] if self.tools else []
                    system_message += f"\n\nYou have access to these tools: {', '.join(tools_list)}. Use them when appropriate."
                    print("[LLMHandler] Added generic tool prompt as fallback")
            
            print(f"[LLMHandler] System message prepared, length: {len(system_message)}")
            
            messages.append({"role": "system", "content": system_message})
            
            # Add conversation history if provided
            if conversation_history:
                print(f"[LLMHandler] Adding {len(conversation_history)} conversation history messages")
                messages.extend(conversation_history)
            
            # Add the current user input
            print(f"[LLMHandler] Adding user message: {user_input[:30]}...")
            messages.append({"role": "user", "content": user_input})
            
            # Generate response using the backend, always with tools
            print(f"[LLMHandler] Calling backend.generate with {len(messages)} messages")
            try:
                print(f"[LLMHandler] Tools count: {len(self.tools) if self.tools else 0}")
                response = self.llm_backend.generate(messages, tools=self.tools)
                print(f"[LLMHandler] Backend response received: {type(response)}")
                print(f"[LLMHandler] Response content length: {len(response.get('content', ''))}")
                print(f"[LLMHandler] Response has tool_calls: {'tool_calls' in response}")
            except Exception as e:
                import traceback
                print(f"[LLMHandler] Error in backend.generate: {str(e)}")
                print(f"[LLMHandler] Traceback: {traceback.format_exc()}")
                raise
            
            # Process any tool calls in the response
            tool_calls = response.get('tool_calls', [])
            if tool_calls:
                print(f"[LLMHandler] Processing {len(tool_calls)} tool calls")
                # Execute each tool call and add the results
                for i, tool_call in enumerate(tool_calls):
                    print(f"[LLMHandler] Executing tool call {i+1}: {tool_call.get('name') or tool_call.get('function', {}).get('name')}")
                    try:
                        tool_result = self._execute_tool_call(tool_call)
                        print(f"[LLMHandler] Tool result received, length: {len(tool_result) if tool_result else 0}")
                    except Exception as e:
                        print(f"[LLMHandler] Error executing tool call: {str(e)}")
                        tool_result = f"Error executing tool: {str(e)}"
                    
                    # Add the tool call and result to the conversation
                    messages.append({
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [tool_call]
                    })
                    
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.get("id", f"call_{i}"),
                        "name": tool_call.get("name", tool_call.get("function", {}).get("name")),
                        "content": tool_result
                    })
                
                # Generate a final response that includes the tool results
                print("[LLMHandler] Generating final response with tool results...")
                try:
                    final_response = self.llm_backend.generate(messages)
                    final_content = final_response.get('content', '')
                    print(f"[LLMHandler] Final response received, length: {len(final_content)}")
                    
                    # Check if the response is empty and provide a fallback
                    if not final_content or len(final_content.strip()) == 0:
                        print("[LLMHandler] Received empty response, generating fallback response")
                        # Create a fallback response using the tool results
                        tool_results_summary = []
                        for msg in messages:
                            if msg.get('role') == 'tool':
                                tool_name = msg.get('name', 'unknown tool')
                                tool_result = msg.get('content', 'No result')
                                tool_results_summary.append(f"I used the {tool_name} tool and got: {tool_result}")
                        
                        if tool_results_summary:
                            fallback_response = "\n".join(tool_results_summary)
                            print(f"[LLMHandler] Generated fallback response: {fallback_response[:100]}...")
                            return fallback_response
                        else:
                            return "I used tools to answer your question, but couldn't generate a proper response. Here's what I found: " + str(tool_result)
                    
                    return final_content
                except Exception as e:
                    print(f"[LLMHandler] Error generating final response: {str(e)}")
                    # Provide a fallback even when an exception occurs
                    return f"I retrieved this information for you: {tool_result}"
            else:
                print("[LLMHandler] No tool calls in response, returning content directly")
                # No tool calls, just return the content
                return response.get('content')
        
        except Exception as e:
            import traceback
            print(f"[LLMHandler] Uncaught exception in generate_response: {str(e)}")
            print(f"[LLMHandler] Traceback: {traceback.format_exc()}")
            raise

    def _execute_tool_call(self, tool_call: Dict[str, Any]) -> str:
        """Execute a tool call and return the result"""
        return tools.execute_tool_call(tool_call)
    
    def get_model_info(self) -> str:
        """Get information about the current model and tools configuration"""
        model_name = LLM_CONFIG.get('model_name', 'unknown')
        backend_type = LLM_CONFIG.get('backend_type', 'unknown')
        message = f"Using {model_name} with {backend_type} backend.\n"
        message += "Tools are always enabled in Funes. "
        message += "If you encounter issues with tool use, please try a different model from the following options:\n"
        message += "- llama3.2:latest (recommended for best tool support)\n"
        message += "- mistral:latest\n"
        message += "- mixtral:latest\n"
        message += "- llama2:latest\n"
        message += "You can change the model in config.py by setting LLM_CONFIG['model_name']"
        return message