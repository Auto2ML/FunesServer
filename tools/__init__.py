"""
Tools package for Funes

This package contains all the tools available for use by Funes.
Each tool should be defined in its own file and registered in this __init__.py file.
"""

import importlib
import os
import inspect
from typing import Dict, List, Any, Optional, Type
from .generic_tool import GenericTool

# Dictionary to store all registered tools
_registered_tools: Dict[str, GenericTool] = {}

def register_tool(tool_instance: GenericTool) -> None:
    """
    Register a tool with the system.
    
    Args:
        tool_instance: An instance of a class derived from GenericTool
    """
    if tool_instance.name in _registered_tools:
        print(f"Warning: Tool '{tool_instance.name}' is being re-registered")
    _registered_tools[tool_instance.name] = tool_instance
    print(f"Registered tool: {tool_instance.name}")

def get_tool(name: str) -> Optional[GenericTool]:
    """
    Get a tool by name.
    
    Args:
        name: The name of the tool
        
    Returns:
        The tool instance or None if not found
    """
    return _registered_tools.get(name)

def get_available_tools() -> List[Dict[str, Any]]:
    """
    Get a list of all registered tools in a format suitable for LLM function calling.
    This function is an alias for list_tools() for compatibility.
    
    Returns:
        List of tool definitions in OpenAI function calling format
    """
    return list_tools()

def list_tools() -> List[Dict[str, Any]]:
    """
    List all registered tools in a format suitable for LLM function calling.
    
    Returns:
        List of tool definitions in OpenAI function calling format
    """
    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters
            }
        }
        for tool in _registered_tools.values()
    ]

def get_tools_description() -> str:
    """
    Get a human-readable description of all registered tools.
    
    Returns:
        A formatted string describing all tools
    """
    import json
    
    if not _registered_tools:
        return "No tools available."
    
    descriptions = ["Available tools:"]
    for tool in _registered_tools.values():
        param_desc = json.dumps(tool.parameters, indent=2)
        descriptions.append(f"- {tool.name}: {tool.description}")
        descriptions.append(f"  Parameters: {param_desc}")
    
    return "\n".join(descriptions)

def execute_tool_call(tool_call: Dict[str, Any]) -> str:
    """
    Execute a tool call based on the format provided by the LLM.
    
    Args:
        tool_call: Dictionary containing tool call information
        
    Returns:
        The result of the tool execution
    """
    # Extract tool name from various possible structures
    tool_name = None
    
    # Handle different formats of tool calls
    if "function" in tool_call:
        if isinstance(tool_call["function"], dict) and "name" in tool_call["function"]:
            tool_name = tool_call["function"]["name"]
        elif hasattr(tool_call["function"], "name"):
            tool_name = tool_call["function"].name
    elif "name" in tool_call:
        tool_name = tool_call["name"]
    
    if not tool_name:
        return "Error: Tool name not specified"
    
    # Get the tool
    tool = get_tool(tool_name)
    if not tool:
        return f"Error: Unknown tool '{tool_name}'"
    
    # Extract arguments
    arguments = {}
    if "arguments" in tool_call:
        arguments = tool_call["arguments"]
    elif "function" in tool_call and isinstance(tool_call["function"], dict) and "arguments" in tool_call["function"]:
        arguments = tool_call["function"]["arguments"]
    elif "function" in tool_call and hasattr(tool_call["function"], "arguments"):
        arguments = tool_call["function"].arguments
    
    # Parse arguments if they're a string
    if isinstance(arguments, str):
        try:
            import json
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return f"Error: Invalid JSON in arguments: {arguments}"
    
    # Execute the tool
    try:
        return tool.execute(**arguments)
    except Exception as e:
        return f"Error executing tool '{tool_name}': {str(e)}"

# Store embeddings for tools in database
def store_tools_in_database() -> None:
    """
    Store all registered tools in the database with their embeddings.
    This allows for vector-based tool selection.
    """
    try:
        # Import here to avoid circular imports
        from config import DB_CONFIG, EMBEDDING_CONFIG
        from database import DatabaseManager
        from sentence_transformers import SentenceTransformer
        import json
        
        print("Initializing tool embeddings in database...")
        
        # Load embedding model directly instead of importing from memory_manager
        model_name = EMBEDDING_CONFIG.get('model_name', 'all-MiniLM-L6-v2')
        embedding_model = SentenceTransformer(model_name)
        
        # Connect to the database
        db_manager = DatabaseManager(DB_CONFIG)
        
        # Process each registered tool
        for tool_name, tool in _registered_tools.items():
            # Use only the tool name and description for embeddings
            description = tool.description
            
            # Generate embedding for the tool name and description only
            print(f"Generating embedding for tool: {tool_name}")
            embedding = embedding_model.encode(description)
            
            # Store in database
            db_manager.store_tool_embedding(tool_name, description, embedding.tolist())
            print(f"Stored embedding for tool: {tool_name}")
            
        print("Tool embeddings stored in database successfully")
    except Exception as e:
        print(f"Error storing tool embeddings: {str(e)}")
        import traceback
        print(traceback.format_exc())
        # Continue without crashing - this is non-critical functionality

# Auto-discover and load tools from this directory
def _discover_tools() -> None:
    """Discover and import all tools in the tools directory"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Skip these files
    skip_files = ['__init__.py', 'generic_tool.py']
    
    for filename in os.listdir(current_dir):
        if filename.endswith('.py') and filename not in skip_files:
            module_name = filename[:-3]  # Remove .py extension
            try:
                # Import the module
                module = importlib.import_module(f"tools.{module_name}")
                
                # Find all classes that inherit from GenericTool
                for name, obj in inspect.getmembers(module):
                    if (inspect.isclass(obj) and 
                        issubclass(obj, GenericTool) and 
                        obj is not GenericTool):
                        # Create an instance and register it
                        try:
                            tool_instance = obj()
                            register_tool(tool_instance)
                        except Exception as e:
                            print(f"Error registering tool from {filename}: {str(e)}")
            except Exception as e:
                print(f"Error loading tool module {module_name}: {str(e)}")
    
    # After registering all tools, store them in database
    store_tools_in_database()

# Call the discovery function when the module is imported
_discover_tools()