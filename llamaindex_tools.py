"""
LlamaIndex tools integration for Funes

This module provides integration between Funes' existing tool system and LlamaIndex's function calling.
"""

import asyncio
from typing import Dict, List, Any, Callable, Optional
from tools import get_tool, list_tools, execute_tool_call
from llama_index.core.tools import FunctionTool


def convert_funes_tool_to_llamaindex(tool_name: str) -> Optional[FunctionTool]:
    """
    Convert a Funes tool to a LlamaIndex FunctionTool
    
    Args:
        tool_name: Name of the Funes tool to convert
        
    Returns:
        A LlamaIndex FunctionTool or None if the tool is not found
    """
    funes_tool = get_tool(tool_name)
    if not funes_tool:
        return None
    
    # Create a wrapper function that will execute the Funes tool
    def tool_wrapper(**kwargs) -> str:
        # Create a tool call structure compatible with execute_tool_call
        tool_call = {
            "function": {
                "name": tool_name,
                "arguments": kwargs
            }
        }
        
        # Execute the tool call using Funes' existing system
        return execute_tool_call(tool_call)
    
    # Set the wrapper function's name and docstring
    tool_wrapper.__name__ = tool_name
    tool_wrapper.__doc__ = funes_tool.description
    
    # Create a LlamaIndex FunctionTool
    return FunctionTool.from_defaults(
        name=tool_name,
        description=funes_tool.description,
        fn=tool_wrapper
    )


def get_all_tools_as_llamaindex() -> List[FunctionTool]:
    """
    Convert all registered Funes tools to LlamaIndex FunctionTools
    
    Returns:
        List of LlamaIndex FunctionTools
    """
    funes_tools = list_tools()
    llamaindex_tools = []
    
    for tool in funes_tools:
        if "function" in tool and "name" in tool["function"]:
            tool_name = tool["function"]["name"]
            llamaindex_tool = convert_funes_tool_to_llamaindex(tool_name)
            if llamaindex_tool:
                llamaindex_tools.append(llamaindex_tool)
    
    return llamaindex_tools


# Define direct LlamaIndex implementations of the built-in tools
# These will eventually replace the Funes tool implementations
def weather_tool(location: str, format: str = "celsius") -> str:
    """Get the current weather for a specified location"""
    # For now, this just calls the existing Funes tool
    weather_tool = get_tool("get_weather")
    if weather_tool:
        return weather_tool.execute(location=location, format=format)
    return "Weather tool not available"


def datetime_tool(timezone: Optional[str] = None, format: str = "full") -> str:
    """Get the current date and time, optionally in a specified timezone"""
    # For now, this just calls the existing Funes tool
    datetime_tool = get_tool("get_date_time")
    if datetime_tool:
        return datetime_tool.execute(timezone=timezone, format=format)
    return "Date/time tool not available"


# Define the native LlamaIndex tools
native_llamaindex_tools = [
    FunctionTool.from_defaults(
        name="get_weather",
        description="Get the current weather for a specified location",
        fn=weather_tool
    ),
    FunctionTool.from_defaults(
        name="get_date_time",
        description="Get the current date and time, optionally in a specified timezone",
        fn=datetime_tool
    )
]