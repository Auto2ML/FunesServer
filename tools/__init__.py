"""
Tools package for Funes

This package contains all the tools available for use by Funes.
Each tool should be defined in its own file and registered in this __init__.py file.
"""

import importlib
import os
import inspect
import json
import logging
import traceback
from typing import Dict, List, Any, Optional, Type
from .generic_tool import GenericTool

# Configure logger
logger = logging.getLogger('FunesTools')

# Dictionary to store all registered tools
_registered_tools: Dict[str, GenericTool] = {}

def register_tool(tool_instance: GenericTool) -> None:
    """
    Register a tool with the system.
    
    Args:
        tool_instance: An instance of a class derived from GenericTool
    """
    if tool_instance.name in _registered_tools:
        logger.warning(f"Tool '{tool_instance.name}' is being re-registered")
    _registered_tools[tool_instance.name] = tool_instance
    logger.info(f"Registered tool: {tool_instance.name}")

def get_tool(name: str) -> Optional[GenericTool]:
    """
    Get a tool by name with robust name resolution.
    
    This function uses multiple strategies to find a tool:
    1. Direct lookup by exact name
    2. Known common name variations (e.g., "date_time" → "get_date_time")
    3. Prefix handling (adding/removing "get_")
    4. Basic fuzzy matching for similar tool names
    
    Args:
        name: The name of the tool
        
    Returns:
        The tool instance or None if not found
    """
    # Direct lookup first (most efficient)
    if name in _registered_tools:
        return _registered_tools[name]
    
    # Common known variations dictionary 
    # This can be expanded as new tools are added and common variations are discovered
    variation_map = {
        # Date/time variations
        "unknown_tool": "get_date_time",  # Special case for Ollama's default
        "date_time": "get_date_time",
        "datetime": "get_date_time",
        "time": "get_date_time",
        # Weather variations
        "weather": "get_weather",
        "forecast": "get_weather",
        # Add more mappings as needed for future tools
    }
    
    # Check if this is a known variation
    if name.lower() in variation_map:
        mapped_name = variation_map[name.lower()]
        if mapped_name in _registered_tools:
            logger.info(f"Resolving '{name}' to '{mapped_name}' using variation map")
            return _registered_tools[mapped_name]
    
    # Try prefix handling (works for any tool, not just specific ones)
    name_lower = name.lower()
    
    # Try without 'get_' prefix if present
    if name_lower.startswith('get_') and name_lower[4:] in _registered_tools:
        return _registered_tools[name_lower[4:]]
    
    # Try adding 'get_' prefix if not present
    if not name_lower.startswith('get_') and f"get_{name_lower}" in _registered_tools:
        return _registered_tools[f"get_{name_lower}"]
    
    # Try case-insensitive matching (for any tool)
    for registered_name in _registered_tools:
        if registered_name.lower() == name_lower:
            logger.info(f"Resolving '{name}' to '{registered_name}' using case-insensitive match")
            return _registered_tools[registered_name]
    
    # Handle commands that might contain the tool name
    # For example, "get the weather for Paris" → "get_weather"
    if _registered_tools:  # Only try this if we have tools registered
        for tool_name in _registered_tools:
            # Extract the meaningful part of the tool name without "get_" prefix
            core_name = tool_name[4:] if tool_name.startswith("get_") else tool_name
            # If the core name is in the provided name, it might be the right tool
            if len(core_name) > 3 and core_name in name_lower:
                logger.info(f"Resolving '{name}' to '{tool_name}' using partial name match")
                return _registered_tools[tool_name]
                
    # No match found
    return None

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
    # Debug logging to see what's being received
    try:
        logger.debug(f"Executing tool call: {json.dumps(tool_call)}")
    except:
        logger.debug(f"Executing tool call: {str(tool_call)}")
    
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
        logger.error("Error: Tool name not specified in tool call")
        return "Error: Tool name not specified"
    
    # Get the tool
    tool = get_tool(tool_name)
    if not tool:
        logger.error(f"Error: Unknown tool '{tool_name}'")
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
            logger.debug(f"Parsing JSON arguments: {arguments}")
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            logger.error(f"Error: Invalid JSON in arguments: {arguments}")
            return f"Error: Invalid JSON in arguments: {arguments}"
    
    # Log the arguments we extracted
    logger.info(f"Tool: {tool_name}, Arguments: {arguments}")
    
    # Validate required parameters
    required_params = []
    if "parameters" in tool.parameters and "required" in tool.parameters:
        required_params = tool.parameters["required"]
    
    for param in required_params:
        if param not in arguments:
            error_msg = f"Error: Missing required parameter '{param}' for tool '{tool_name}'"
            logger.error(error_msg)
            return error_msg
    
    # Execute the tool
    try:
        logger.info(f"Executing tool '{tool_name}' with arguments: {arguments}")
        result = tool.execute(**arguments)
        logger.info(f"Tool '{tool_name}' execution completed successfully")
        return result
    except Exception as e:
        error_msg = f"Error executing tool '{tool_name}': {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        return error_msg

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
        
        logger.info("Initializing tool embeddings in database...")
        
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
            logger.info(f"Generating embedding for tool: {tool_name}")
            embedding = embedding_model.encode(description)
            
            # Store in database
            db_manager.store_tool_embedding(tool_name, description, embedding.tolist())
            logger.info(f"Stored embedding for tool: {tool_name}")
            
        logger.info("Tool embeddings stored in database successfully")
    except Exception as e:
        logger.error(f"Error storing tool embeddings: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
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
                            logger.error(f"Error registering tool from {filename}: {str(e)}")
            except Exception as e:
                logger.error(f"Error loading tool module {module_name}: {str(e)}")
    
    # After registering all tools, try to store them in database
    try:
        store_tools_in_database()
    except Exception as e:
        logger.error(f"Error in store_tools_in_database: {str(e)}")

# Call the discovery function when the module is imported
_discover_tools()