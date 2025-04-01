"""
LlamaIndex LLM integration for Funes

This module provides integration between Funes' existing LLM handling and LlamaIndex's LLM interfaces.
"""

import asyncio
from typing import List, Dict, Any, Optional, Union
from config import LLM_CONFIG
from llama_index.llms.ollama import Ollama
from llama_index.core import Settings
from llama_index.core.agent import FunctionCallingAgentWorker
from llama_index.core.tools import BaseTool
from llamaindex_tools import get_all_tools_as_llamaindex, native_llamaindex_tools


class LlamaIndexLLMHandler:
    """LlamaIndex-based LLM handler for Funes"""
    
    def __init__(self, model_name=None, system_prompt=None):
        """
        Initialize the LLM handler with LlamaIndex components
        
        Args:
            model_name: Name of the LLM model to use
            system_prompt: System prompt for the agent
        """
        # Get config values with optional overrides
        self.model_name = model_name or LLM_CONFIG.get('model_name', 'llama3:latest')
        self.system_prompt = system_prompt or LLM_CONFIG.get('system_prompt', '')
        self.backend_type = LLM_CONFIG.get('backend_type', 'ollama')
        
        # Set up the LLM based on the backend type
        self._setup_llm()
        
        # Set up tools
        self.tools = self._setup_tools()
        
        # Create an agent
        self.agent = self._create_agent()
        
        print(f"[LlamaIndexLLM] Initialized with model: {self.model_name}")
        print(f"[LlamaIndexLLM] Using {len(self.tools)} tools")
    
    def _setup_llm(self):
        """Set up the LLM based on the backend type"""
        if self.backend_type.lower() == 'ollama':
            self.llm = Ollama(model=self.model_name, request_timeout=360.0)
            Settings.llm = self.llm
            print(f"[LlamaIndexLLM] Using Ollama with model: {self.model_name}")
        else:
            # Default to Ollama for now, we can add more backends later
            self.llm = Ollama(model=self.model_name, request_timeout=360.0)
            Settings.llm = self.llm
            print(f"[LlamaIndexLLM] Unsupported backend type: {self.backend_type}, using Ollama")

    def _setup_tools(self) -> List[BaseTool]:
        """Set up tools for the agent"""
        # First try to convert existing Funes tools to LlamaIndex tools
        tools = get_all_tools_as_llamaindex()
        
        # If there are no converted tools, use the native LlamaIndex tool implementations
        if not tools:
            print("[LlamaIndexLLM] No existing tools found, using native LlamaIndex tools")
            tools = native_llamaindex_tools
        else:
            print(f"[LlamaIndexLLM] Converted {len(tools)} existing tools to LlamaIndex format")
        
        return tools
    
    def _create_agent(self):
        """Create an agent with the LLM and tools"""
        agent = FunctionCallingAgentWorker.from_tools(
            tools=self.tools,
            llm=self.llm,
            system_prompt=self.system_prompt,
            verbose=True
        )
        return agent
    
    async def agenerate_response(self, 
                               user_input: str, 
                               conversation_history: Optional[List[Dict[str, str]]] = None, 
                               additional_context: Optional[str] = None) -> str:
        """
        Generate a response using the LlamaIndex agent asynchronously
        
        Args:
            user_input: The user's query
            conversation_history: Optional list of previous conversation messages
            additional_context: Optional context information to add to the system prompt
            
        Returns:
            The response from the agent
        """
        try:
            # Create a full prompt with additional context if provided
            full_prompt = user_input
            if additional_context:
                full_prompt = f"Context: {additional_context}\n\nQuestion: {user_input}"
            
            # Initialize chat history in the agent if provided
            if conversation_history:
                # Reset the agent's chat history
                self.agent.reset()
                
                # We'll simplify the chat history handling for now to avoid potential issues
                # Just use the most recent messages
                recent_messages = conversation_history[-5:] if len(conversation_history) > 5 else conversation_history
                
                # Convert to a simplified format
                chat_history_str = ""
                for msg in recent_messages:
                    role = msg.get("role", "")
                    content = msg.get("content", "")
                    if role and content:
                        chat_history_str += f"{role.capitalize()}: {content}\n"
                
                # Add chat history to the prompt if available
                if chat_history_str:
                    full_prompt = f"Previous conversation:\n{chat_history_str}\n\nCurrent query: {full_prompt}"
            
            # Generate response
            try:
                # Check for available async methods in the agent
                if hasattr(self.agent, 'achat'):
                    response = await self.agent.achat(full_prompt)
                elif hasattr(self.agent, 'astream_chat'):
                    response = await self.agent.astream_chat(full_prompt)
                elif hasattr(self.agent, 'ahandle_async'):
                    response = await self.agent.ahandle_async(full_prompt)
                elif hasattr(self.agent, 'acomplete'):
                    response = await self.agent.acomplete(full_prompt)
                else:
                    # If no async methods are available, fall back to synchronous methods
                    print("[LlamaIndexLLM] No async methods available, falling back to sync methods")
                    if hasattr(self.agent, 'chat'):
                        response = self.agent.chat(full_prompt)
                    elif hasattr(self.agent, 'complete'):
                        response = self.agent.complete(full_prompt)
                    else:
                        # Last resort: direct LLM call
                        response = self.llm.complete(full_prompt)
                
                return str(response)
            except Exception as e:
                # If the async chat fails, try a synchronous approach as fallback
                import traceback
                print(f"[LlamaIndexLLM] Error in async chat: {str(e)}")
                print(f"[LlamaIndexLLM] Traceback: {traceback.format_exc()}")
                
                # Try a simple completion as fallback
                try:
                    # Try synchronous agent chat if available
                    if hasattr(self.agent, 'chat'):
                        response = self.agent.chat(full_prompt)
                    else:
                        # Direct LLM call as last resort
                        response = self.llm.complete(full_prompt)
                    return str(response)
                except:
                    return f"Error generating response: {str(e)}"
        except Exception as e:
            import traceback
            print(f"[LlamaIndexLLM] Error generating response: {str(e)}")
            print(f"[LlamaIndexLLM] Traceback: {traceback.format_exc()}")
            return f"Error generating response: {str(e)}"
    
    def generate_response(self, 
                        user_input: str, 
                        conversation_history: Optional[List[Dict[str, str]]] = None, 
                        additional_context: Optional[str] = None) -> str:
        """
        Generate a response using the LlamaIndex agent (synchronous wrapper)
        
        Args:
            user_input: The user's query
            conversation_history: Optional list of previous conversation messages
            additional_context: Optional context information to add to the system prompt
            
        Returns:
            The response from the agent
        """
        # Create a new event loop if needed and run the async function
        try:
            # Try getting the current event loop
            loop = asyncio.get_event_loop()
        except RuntimeError:
            # If there isn't one, create a new loop and set it as the current one for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        # Now run the async function
        try:
            task = self.agenerate_response(user_input, conversation_history, additional_context)
            result = loop.run_until_complete(task)
            return result
        except Exception as e:
            import traceback
            print(f"[LlamaIndexLLM] Error in run_until_complete: {str(e)}")
            print(f"[LlamaIndexLLM] Traceback: {traceback.format_exc()}")
            
            # If asyncio approach failed, try direct sync call to the LLM as fallback
            try:
                print("[LlamaIndexLLM] Trying direct synchronous approach as fallback")
                full_prompt = user_input
                if additional_context:
                    full_prompt = f"Context: {additional_context}\n\nQuestion: {user_input}"
                    
                # Try synchronous agent chat if available
                if hasattr(self.agent, 'chat'):
                    print("[LlamaIndexLLM] Using agent.chat")
                    response = self.agent.chat(full_prompt)
                else:
                    # Direct LLM call as last resort
                    print("[LlamaIndexLLM] Using direct LLM call")
                    response = self.llm.complete(full_prompt)
                    
                return str(response)
            except Exception as inner_e:
                print(f"[LlamaIndexLLM] Fallback also failed: {str(inner_e)}")
                return f"Error generating response: {str(e)}. Fallback also failed: {str(inner_e)}"