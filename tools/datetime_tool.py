"""
Date and time tool for Funes

This tool provides current date and time information, optionally in a specific timezone.
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
        return "Get the current date and time, optionally in a specified timezone"
    
    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "timezone": {
                    "type": "string",
                    "description": "Timezone to return the date/time in (e.g. 'UTC', 'US/Pacific', 'Europe/London'). If not specified, local system time is used."
                },
                "format": {
                    "type": "string",
                    "description": "Date/time format (e.g. 'full', 'date', 'time', 'iso')",
                    "enum": ["full", "date", "time", "iso"],
                    "default": "full"
                }
            },
            "required": []
        }
    
    def execute(self, timezone: Optional[str] = None, format: str = "full") -> str:
        """
        Get the current date and time.
        
        Args:
            timezone: Optional timezone name (e.g. 'UTC', 'US/Pacific')
            format: Format to return ('full', 'date', 'time', or 'iso')
            
        Returns:
            Current date and time as a string
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
                    tz_info = f" ({timezone})"
                except (ImportError, pytz.exceptions.UnknownTimeZoneError):
                    # If pytz is not installed or timezone is invalid, use system time
                    tz_info = " (system timezone - pytz not available or invalid timezone)"
            else:
                tz_info = " (system timezone)"
            
            # Format the time according to the format parameter
            if format.lower() == "date":
                result = now.strftime("%Y-%m-%d")
                return f"Current date: {result}{tz_info}"
            elif format.lower() == "time":
                result = now.strftime("%H:%M:%S")
                return f"Current time: {result}{tz_info}"
            elif format.lower() == "iso":
                result = now.isoformat()
                return result
            else:  # "full" is the default
                result = now.strftime("%Y-%m-%d %H:%M:%S")
                return f"Current date and time: {result}{tz_info}"
        except Exception as e:
            return f"Error retrieving date/time information: {str(e)}"