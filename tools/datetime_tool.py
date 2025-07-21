"""
Date and time tool for Funes

This tool provides current date and time information based on a specific location and timezone
"""

import datetime
from typing import Dict, Any, Optional
from .generic_tool import GenericTool

class DateTimeTool(GenericTool):
    """Date and time tool implementation"""
    
    @property
    def name(self) -> str:
        return "get_date_time"
    
    @property
    def description(self) -> str:
        return "Get the current date and time for a specified location. Provides current time, today's date, what time is it now, current time in different cities and timezones around the world including Madrid, London, New York, Tokyo, etc."
    
    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "Location to get the date/time for (e.g. 'New York', 'London', 'Tokyo'). Expected to be extracted from user query."
                },
                "timezone": {
                    "type": "string",
                    "description": "Timezone identifier (e.g. 'America/New_York', 'Europe/London', 'Asia/Tokyo'). Expected to be defined by Funes based on the location.",
                },
                "format": {
                    "type": "string",
                    "description": "Date/time format (e.g. 'full', 'date', 'time', 'iso')",
                    "enum": ["full", "date", "time", "iso"],
                    "default": "full"
                }
            },
            "required": ["location", "timezone"]
        }
    
    @property
    def store_in_memory(self) -> bool:
        """
        Date and time responses should not be stored in memory as they are time-dependent.
        """
        return False
    
    def execute(self, location: str, timezone: str, format: str = "full") -> str:
        """
        Get the current date and time for a specified location.
        
        Args:
            location: Location name extracted from user query
            timezone: Timezone identifier that the LLM knows based on the location
            format: Format to return ('full', 'date', 'time', or 'iso')
            
        Returns:
            Current date and time as a formatted string
        """
        try:
            # Get the current time
            now = datetime.datetime.now()
            
            # Try to adjust for timezone if specified
            if timezone:
                try:
                    import pytz
                    tz = pytz.timezone(timezone)
                    now = datetime.datetime.now(tz)
                    location_info = f" in {location} ({timezone})"
                except (ImportError, pytz.exceptions.UnknownTimeZoneError):
                    # If pytz is not installed or timezone is invalid, use system time
                    location_info = f" (system time - could not determine timezone for {location})"
            else:
                location_info = f" (system time - no timezone provided for {location})"
            
            # Format the response based on the requested format
            if format.lower() == "date":
                return f"Today is {now.strftime('%A')}, {now.strftime('%B')} {now.strftime('%d').lstrip('0')}, {now.strftime('%Y')}{location_info}."
            elif format.lower() == "time":
                return f"The current time is {now.strftime('%I:%M:%S %p')}{location_info}."
            elif format.lower() == "iso":
                return f"{now.isoformat()}"
            else:  # "full" is the default
                return f"It is currently {now.strftime('%I:%M:%S %p')} on {now.strftime('%A')}, {now.strftime('%B')} {now.strftime('%d').lstrip('0')}, {now.strftime('%Y')}{location_info}."
        except Exception as e:
            return f"Error retrieving date/time information: {str(e)}"