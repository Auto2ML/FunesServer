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
        # Make sure we have a clean messages structure
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
                # For Llamafile/OpenAI format, include the tool response
                formatted_messages.append({
                    "role": "tool",
                    "tool_call_id": msg.get("tool_call_id", "unknown"),
                    "name": msg.get("name", "unknown_tool"),
                    "content": msg["content"]
                })
        
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
            
            # Extract the response content from the OpenAI-compatible format
            if "choices" in response_data and len(response_data["choices"]) > 0:
                message = response_data["choices"][0].get("message", {})
                content = message.get("content", "")
                
                # Handle function calls if present (convert to tool_calls format)
                tool_calls = []
                if "function_call" in message:
                    # Handle single function call format
                    function_call = message["function_call"]
                    tool_calls.append({
                        "function": {
                            "name": function_call.get("name", ""),
                            "arguments": function_call.get("arguments", "{}")
                        }
                    })
                elif "tool_calls" in message:
                    # Handle multiple tool calls format
                    for tool_call in message["tool_calls"]:
                        if "function" in tool_call:
                            tool_calls.append({
                                "function": tool_call["function"]
                            })
                
                return {
                    'content': content,
                    'tool_calls': tool_calls
                }
            else:
                return {
                    'content': "No valid response from the model",
                    'tool_calls': []
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
        # Make sure we have a clean messages structure
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
                # Add tool responses as assistant messages according to Ollama format
                formatted_messages.append({
                    "role": "assistant",
                    "content": f"Tool response from {msg.get('name', 'unknown')}: {msg['content']}"
                })
        
        # Prepare the request parameters
        params = {"model": self.model_name, "messages": formatted_messages, "stream": False}
        
        # Only add tools if provided AND the request seems appropriate for tool use
        # This check allows the model to respond normally for general knowledge questions
        if tools is not None:
            # Look for specific markers that might indicate tool use is beneficial
            # Check the latest user message
            user_message = next((msg["content"] for msg in reversed(formatted_messages) 
                              if msg["role"] == "user"), "")
            
            # These keywords might indicate when tools would be useful
            tool_relevant_keywords = [
                "weather", "temperature", "forecast",  # Weather tool
                "time", "date", "today", "now", "current time", "current date",  # Date/time tool
                "latest", "current", "right now", "today's"  # Real-time info
            ]
            
            # Check for tool names in the user message
            tool_names = [t.get("function", {}).get("name", "") for t in tools]
            tool_name_mentioned = any(name in user_message.lower() for name in tool_names if name)
            
            # Check for keywords that suggest tool use would be helpful
            keywords_present = any(keyword in user_message.lower() for keyword in tool_relevant_keywords)
            
            # Always include tools when in a tool conversation (if there's a previous tool response)
            in_tool_conversation = any(msg.get("role") == "assistant" and "Tool response from" in msg.get("content", "") 
                                   for msg in formatted_messages)
            
            # Add tools to the request only when appropriate
            if tool_name_mentioned or keywords_present or in_tool_conversation:
                params["tools"] = tools
                print("[OllamaBackend] Including tools in request based on content analysis")
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
        self.llm = llama_cpp.Llama(
            model_path=model_path,
            n_ctx=context_size,
            temperature=temperature
        )
        self.max_tokens = max_tokens
        
        # Try to extract model name from path for feature detection
        path_parts = model_path.split('/')
        filename = path_parts[-1] if path_parts else ""
        self.base_model = filename.split('.')[0] if '.' in filename else filename
    
    def supports_tool_use(self) -> bool:
        """All models are assumed to support tool use"""
        return True
    
    def generate(self, messages: List[Dict[str, str]], tools: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        # Make sure we have clean messages
        formatted_messages = []
        for msg in messages:
            if msg["role"] in ["system", "user", "assistant"]:
                formatted_messages.append({
                    "role": msg["role"],
                    "content": msg["content"].strip()
                })
        
        # Always include tool information in system messages
        # Convert messages to llama.cpp format
        prompt = ""
        for msg in formatted_messages:
            role = msg["role"]
            content = msg["content"]
            
            # If this is the system message and tools are provided, add tool descriptions
            if role == "system" and tools is not None:
                tools_json = json.dumps(tools, indent=2)
                content += f"\n\nYou have access to the following tools:\n{tools_json}\n"
                content += "\nWhen a user request requires using these tools, respond with a JSON object containing 'tool_calls' with the format: [{\"name\": \"tool_name\", \"arguments\": {...}}]."
            
            if role == "system":
                prompt += f"<s>[INST] <<SYS>>\n{content}\n<</SYS>>\n\n"
            elif role == "user":
                if not prompt:
                    prompt += f"<s>[INST] {content} [/INST]"
                else:
                    prompt += f"{content} [/INST]"
            elif role == "assistant":
                prompt += f" {content} </s><s>[INST] "
        
        # Handle case where we might end with an unclosed instruction tag
        if prompt.endswith("[INST] "):
            prompt = prompt[:-7]  # Remove the trailing "[INST] "
        
        try:
            response = self.llm(
                prompt=prompt,
                max_tokens=self.max_tokens,
                echo=False
            )
            content = response["choices"][0]["text"]
            
            # For llama.cpp, we need to rely on native tool handling
            # No more parsing from text responses since we only support models with native tool capability
            return {
                'content': content,
                'tool_calls': []  # Tool calls will only be supported natively in future llama.cpp versions
            }
        except Exception as e:
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
        # Make sure we have clean messages
        formatted_messages = []
        for msg in messages:
            if msg["role"] in ["system", "user", "assistant"]:
                formatted_messages.append({
                    "role": msg["role"],
                    "content": msg["content"].strip()
                })
        
        # Convert messages to a prompt format
        prompt = ""
        for msg in formatted_messages:
            role = msg["role"]
            content = msg["content"]
            
            # If this is the system message and tools are provided, add tool descriptions
            if role == "system" and tools is not None:
                tools_json = json.dumps(tools, indent=2)
                content += f"\n\nYou have access to the following tools:\n{tools_json}\n"
                content += "\nWhen a user request requires using these tools, respond with a JSON object containing 'tool_calls' with the format: [{\"name\": \"tool_name\", \"arguments\": {...}}]."
            
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
            
            # For HuggingFace models, we'll rely on the API's native tool support in future
            # No parsing from text since we're only supporting models with native tool capability
            return {
                'content': assistant_response,
                'tool_calls': []  # Tool calls will be handled natively in future HuggingFace versions
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