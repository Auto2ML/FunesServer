import ollama
from ollama import ChatResponse, chat
from config import LLM_CONFIG, LOGGING_CONFIG
import json
import logging
from typing import List, Dict, Any, Optional, Union
import tools  # Import the tools package

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

# Create logger instance
llm_logger = logging.getLogger('LLMHandler')

class LLMHandler:
    """
    Simplified LLM handler that only supports Ollama.
    """
    
    def __init__(self):
        self.logger = llm_logger
        self.logger.info("Initializing LLM Handler")
        self.config = LLM_CONFIG
        self.model_name = self.config.get('model_name', 'llama3')
        self.db_manager = None  # Will be set externally by memory_manager
        
    def format_messages(self, messages):
        """Format messages to ensure they have the correct structure"""
        formatted_messages = []
        
        for msg in messages:
            if not isinstance(msg, dict):
                self.logger.warning(f"Skipping non-dict message: {msg}")
                continue
                
            role = msg.get('role', '').lower()
            content = msg.get('content', '')
            
            if role in ['system', 'user', 'assistant']:
                formatted_messages.append({'role': role, 'content': content})
            elif role == 'tool':
                # For tool responses, format appropriately
                formatted_messages.append({
                    'role': 'tool',
                    'content': content,
                    'name': msg.get('name', 'unknown_tool')
                })
                
        return formatted_messages
    
    def generate_response(self, user_input: str, conversation_history: List[Dict[str, Any]], 
                          additional_context: str = None,
                          include_tools: bool = True) -> Union[str, Dict[str, Any]]:
        """Generate a response using Ollama"""
        try:
            # Create a new copy of the conversation history
            messages = self.format_messages(conversation_history.copy() if conversation_history else [])
            
            # Get available tools if requested
            available_tools = None
            available_functions = {}
            
            # Only process tools if include_tools is True
            if include_tools:
                try:
                    available_tools = tools.get_available_tools()
                    available_functions = tools.get_available_functions()
                    self.logger.info(f"Retrieved {len(available_tools)} available tools")
                except Exception as e:
                    self.logger.warning(f"Error getting tools: {str(e)}")
            else:
                self.logger.info("Tools are disabled for this request")
            
            # Check if there's already a system message
            has_system = False
            for msg in messages:
                if msg.get('role') == 'system':
                    has_system = True
                    # If additional context is provided, append it to existing system message
                    if additional_context:
                        msg['content'] += f"\n\nAdditional context: {additional_context}"
                    break
            
            # Add system message with system prompt from config if none exists
            if not has_system:
                system_content = self.config.get('system_prompt', "You are a helpful assistant.")
                
                # Add tool usage instructions if tools are available AND include_tools is True
                if include_tools and available_tools and 'tool_use_prompt' in self.config:
                    tools_description = tools.get_tools_description()
                    
                    # Format and append the tool use instructions
                    tool_instructions = self.config['tool_use_prompt'].format(
                        tools_description=tools_description
                    )
                    system_content += f"\n\n{tool_instructions}"
                
                if additional_context:
                    system_content += f"\n\nAdditional context: {additional_context}"
                
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
            
            # Generate the response
            self.logger.info(f"Sending request to Ollama with model: {self.model_name}")
            
            # Simple chat call to Ollama - only pass tools if they're enabled
            response: ChatResponse = chat(
                self.model_name,
                messages=messages,
                tools=available_tools if include_tools and available_tools else None,
            )
            
            # Process tool calls if present
            tool_calls = []
            tool_response_content = None
            
            if response.message.tool_calls:
                self.logger.info(f"Response contains {len(response.message.tool_calls)} tool calls")
                
                # Process each tool call
                for tool in response.message.tool_calls:
                    tool_calls.append({
                        'id': getattr(tool, 'id', f"call_{len(tool_calls)}"),
                        'function': {
                            'name': tool.function.name,
                            'arguments': tool.function.arguments
                        }
                    })
                    
                    # Execute the tool if available
                    if function_to_call := available_functions.get(tool.function.name):
                        try:
                            self.logger.info(f"Calling function: {tool.function.name}")
                            self.logger.info(f"Arguments: {tool.function.arguments}")
                            
                            # Call the function with its arguments
                            output = function_to_call(**tool.function.arguments)
                            self.logger.info(f"Function output: {output}")
                            
                            # Add the function response to messages for the model
                            messages.append(response.message)
                            messages.append({
                                'role': 'tool', 
                                'content': str(output), 
                                'name': tool.function.name
                            })
                            
                            # Get final response from model with function outputs
                            final_response = chat(self.model_name, messages=messages)
                            tool_response_content = final_response.message.content
                        except Exception as e:
                            self.logger.error(f"Error executing tool {tool.function.name}: {str(e)}")
                            tool_response_content = f"Error executing tool: {str(e)}"
                    else:
                        self.logger.warning(f"Function {tool.function.name} not found")
                
                # If we processed tools, return the final result
                if tool_response_content:
                    return {
                        'content': tool_response_content,
                        'tool_calls': tool_calls
                    }
                        
            # Return the response
            return {
                'content': response.message.content,
                'tool_calls': tool_calls
            }
                
        except Exception as e:
            self.logger.error(f"Error generating response: {str(e)}")
            return {
                'content': f"Sorry, I encountered an error: {str(e)}",
                'tool_calls': []
            }