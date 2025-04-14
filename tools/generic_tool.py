"""
Generic tool template for Funes

This file serves as a template for creating new tools for Funes.
Each tool should follow this basic structure and be placed in the tools directory.

To create a new tool:
1. Create a new file in the tools directory (e.g., my_tool.py)
2. Import GenericTool from generic_tool.py
3. Create a class that inherits from GenericTool (e.g., class MyTool(GenericTool))
4. Implement all required methods and properties
5. Register the tool in the tools/__init__.py file
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import logging

# Logger setup example for tool implementations
# logger = logging.getLogger('MyToolName')

class GenericTool(ABC):
    """
    Abstract base class for all Funes tools.
    
    This class defines the interface that all tools must implement.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """
        The name of the tool. This is used to identify the tool in function calls.
        Should be a simple string without spaces, like 'get_weather' or 'search_web'.
        
        Example:
            return "my_tool_name"
        """
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """
        A human-readable description of what the tool does.
        This will be shown to the LLM to help it decide when to use this tool.
        
        Example:
            return "Get information about X based on Y parameter"
        """
        pass
    
    @property
    @abstractmethod
    def parameters(self) -> Dict[str, Any]:
        """
        Define the parameters this tool accepts in JSON Schema format.
        This schema is used to validate inputs and generate documentation.
        
        Example:
        {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to look up"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of results to return",
                    "default": 5
                },
                "format": {
                    "type": "string",
                    "enum": ["compact", "detailed"],
                    "description": "Response format style",
                    "default": "compact"
                }
            },
            "required": ["query"]
        }
        """
        pass
    
    @property
    def store_in_memory(self) -> bool:
        """
        Whether responses from this tool should be stored in long-term memory.
        Default is False since most tool responses are time-dependent or transient.
        Override this method in tool implementations that should be stored.
        
        Example for tool with persistent results:
            return True
            
        Example for time-sensitive tool:
            return False
        
        Returns:
            Boolean indicating if tool responses should be stored in the database
        """
        return False
    
    @abstractmethod
    def execute(self, **kwargs) -> str:
        """
        Execute the tool with the provided parameters.
        This method will be called when the LLM decides to use this tool.
        
        The parameters passed to this method will match those defined in the
        parameters property schema. Include appropriate type hints based on
        your parameter definitions.
        
        Example implementation:
        
        def execute(self, query: str, limit: int = 5, format: str = "compact") -> str:
            try:
                # Your tool implementation logic goes here
                # ...
                
                # Return the result as a string
                return "Tool execution result"
                
            except Exception as e:
                # Always include error handling
                return f"Error executing tool: {str(e)}"
        
        Args:
            **kwargs: Keyword arguments matching the parameters schema
            
        Returns:
            A string containing the results of the tool execution
        """
        pass
    
    def validate_parameters(self, **kwargs) -> bool:
        """
        Optional method to validate parameters before execution.
        Default implementation just returns True.
        
        You can override this method to implement custom validation logic:
        
        Example:
            def validate_parameters(self, query: str, limit: int = 5) -> bool:
                if not query or not isinstance(query, str):
                    return False
                if limit < 1 or limit > 100:
                    return False
                return True
        
        Returns:
            Boolean indicating if parameters are valid
        """
        return True