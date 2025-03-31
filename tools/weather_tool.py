"""
Weather tool for Funes

This tool provides weather information for a specified location.
"""

import random
from typing import Dict, Any
from .generic_tool import GenericTool

class WeatherTool(GenericTool):
    """Weather information tool implementation"""
    
    @property
    def name(self) -> str:
        return "get_weather"
    
    @property
    def description(self) -> str:
        return "Get the current weather for a specified location"
    
    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state/country, e.g. 'San Francisco, CA' or 'Paris, France'"
                },
                "format": {
                    "type": "string",
                    "description": "Temperature format to use",
                    "enum": ["celsius", "fahrenheit"],
                    "default": "celsius"
                }
            },
            "required": ["location"]
        }
    
    def execute(self, location: str, format: str = "celsius") -> str:
        """
        Get the weather for the specified location.
        
        In a real implementation, this would call a weather API.
        This is just a mockup that returns random weather data.
        
        Args:
            location: City and state/country
            format: Temperature format ('celsius' or 'fahrenheit')
            
        Returns:
            Weather information as a string
        """
        # In a real implementation, you would call a weather API here
        # This is just a mockup that returns random weather data
        
        # Random temperature between 0-30°C or 32-86°F
        temp = random.randint(0, 30)
        if format.lower() == "fahrenheit":
            temp_display = f"{int(temp * 9/5 + 32)}°F"
        else:
            temp_display = f"{temp}°C"
            
        # Random conditions
        conditions = random.choice([
            "Sunny", "Partly Cloudy", "Cloudy", 
            "Light Rain", "Heavy Rain", "Thunderstorms",
            "Snowy", "Foggy", "Clear"
        ])
        
        # Random humidity
        humidity = random.randint(30, 90)
        
        # Random wind speed
        wind = random.randint(0, 30)
        wind_unit = "km/h" if format.lower() == "celsius" else "mph"
        
        return (
            f"Weather for {location}:\n"
            f"Temperature: {temp_display}\n"
            f"Conditions: {conditions}\n"
            f"Humidity: {humidity}%\n"
            f"Wind: {wind} {wind_unit}"
        )