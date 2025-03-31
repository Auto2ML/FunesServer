"""
Generic tool template for Funes

This file serves as a template for creating new tools for Funes.
Each tool should follow this basic structure and be placed in the tools directory.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

class GenericTool(ABC):
    """
    Abstract base class for all Funes tools.
    
    To create a new tool:
    1. Create a new file in the tools directory
    2. Create a class that inherits from GenericTool
    3. Implement all required methods and properties
    4. Register the tool in the __init__.py file
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """
        The name of the tool. This is used to identify the tool in function calls.
        Should be a simple string without spaces, like 'get_weather' or 'search_web'.
        """
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """
        A human-readable description of what the tool does.
        This will be shown to the LLM to help it decide when to use this tool.
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
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA"
                },
                "format": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "Temperature format"
                }
            },
            "required": ["location"]
        }
        """
        pass
    
    @abstractmethod
    def execute(self, **kwargs) -> str:
        """
        Execute the tool with the provided parameters.
        This method will be called when the LLM decides to use this tool.
        
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
        
        Returns:
            Boolean indicating if parameters are valid
        """
        return True