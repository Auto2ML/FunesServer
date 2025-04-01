import json
import re
from typing import List, Dict, Any, Optional, Tuple

def format_messages(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Format messages consistently for all backends"""
    formatted_messages = []
    
    # Process and clean each message
    for msg in messages:
        if msg["role"] in ["system", "user", "assistant"]:
            message_obj = {
                "role": msg["role"],
                "content": msg["content"].strip() if msg["content"] is not None else ""
            }
            # Include tool_calls if present in the message
            if "tool_calls" in msg and msg["tool_calls"]:
                message_obj["tool_calls"] = msg["tool_calls"]
            formatted_messages.append(message_obj)
        elif msg["role"] == "tool":
            # For tool responses, format for various backends
            formatted_messages.append({
                "role": "tool",
                "tool_call_id": msg.get("tool_call_id", "unknown"),
                "name": msg.get("name", "unknown_tool"),
                "content": msg["content"]
            })
    
    return formatted_messages

def extract_tool_information(tools: List[Dict[str, Any]]) -> Tuple[List[str], List[str]]:
    """Extract tool names and relevant keywords from tools definition"""
    tool_names = []
    tool_relevant_keywords = []
    
    # Extract tool names and keywords from function descriptions
    for tool in tools:
        if "function" in tool:
            name = tool["function"].get("name", "").lower()
            if name:
                tool_names.append(name)
                # Add keywords from the function description
                desc = tool["function"].get("description", "").lower()
                if desc:
                    for keyword in ["weather", "time", "date", "day", "extract", "get", "search"]:
                        if keyword in desc and keyword not in tool_relevant_keywords:
                            tool_relevant_keywords.append(keyword)
    
    return tool_names, tool_relevant_keywords

def should_use_tools(messages: List[Dict[str, str]], tools: List[Dict[str, Any]]) -> Tuple[bool, Optional[str]]:
    """Determine if tools should be used and which specific tool might be needed"""
    if not tools or len(tools) == 0:
        return False, None
        
    # Default keywords that suggest tool use
    tool_relevant_keywords = [
        "weather", "temperature", "forecast",  # Weather tool
        "time", "date", "today", "now", "current time", "current date",  # Date/time tool
        "latest", "current", "right now", "today's",  # Real-time info
        "extract", "parse", "get", "find", "search",  # Extraction verbs
        "convert", "calculate", "compute"  # Calculation verbs
    ]
    
    # Extraction patterns that suggest tool use
    extraction_patterns = ["what is", "how many", "who is", "when is", "extract", "parse"]
    
    # Get the latest user message
    latest_user_msg = ""
    for msg in reversed(messages):
        if msg["role"] == "user" and msg.get("content"):
            latest_user_msg = msg["content"].lower()
            break
    
    # Get tool names and additional keywords
    tool_names, additional_keywords = extract_tool_information(tools)
    tool_relevant_keywords.extend(additional_keywords)
    
    # Check if any tool names are directly mentioned
    for name in tool_names:
        if name in latest_user_msg:
            return True, name
    
    # Check for relevant keywords that suggest tool use
    for keyword in tool_relevant_keywords:
        if keyword in latest_user_msg:
            # Attempt to match specific tools to keywords
            if keyword in ["weather", "temperature", "forecast", "rain", "sunny"]:
                return True, "get_weather"
            elif keyword in ["time", "date", "today", "now", "current time", "current date"]:
                return True, "get_date_time"
            return True, None
    
    # Check for extraction patterns
    for pattern in extraction_patterns:
        if pattern in latest_user_msg:
            return True, None
    
    # Default to not using tools
    return False, None

def extract_tool_calls_from_response(response_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract tool calls from various response formats"""
    tool_calls = []
    
    # Try to get message from the response
    message = {}
    if "choices" in response_data and len(response_data["choices"]) > 0:
        message = response_data["choices"][0].get("message", {})
    else:
        message = response_data.get("message", {})
    
    # Check for tool_calls in the message
    if "tool_calls" in message:
        # Handle OpenAI/LlamaCpp format
        raw_tool_calls = message["tool_calls"]
        for tool_call in raw_tool_calls:
            if "type" in tool_call and tool_call["type"] == "function":
                tool_calls.append({
                    "function": {
                        "name": tool_call["function"]["name"],
                        "arguments": tool_call["function"]["arguments"]
                    },
                    "id": tool_call.get("id", f"call_{len(tool_calls)}")
                })
            elif "function" in tool_call:
                tool_calls.append({
                    "function": tool_call["function"],
                    "id": tool_call.get("id", f"call_{len(tool_calls)}")
                })
    
    # Check for function_call in the message (older OpenAI format)
    elif "function_call" in message:
        function_call = message["function_call"]
        tool_calls.append({
            "function": {
                "name": function_call.get("name", ""),
                "arguments": function_call.get("arguments", "{}")
            },
            "id": f"call_0"
        })
    
    return tool_calls

def enhance_tool_response(user_query: str, tool_name: str, tool_result: str) -> str:
    """
    Create a more natural, conversational response based on tool results
    
    Args:
        user_query: Original user question that triggered the tool
        tool_name: Name of the tool that was executed
        tool_result: Raw result returned from the tool
        
    Returns:
        A conversational response that naturally incorporates the tool result
    """
    # Pattern-based templates for common tool responses
    response_templates = {
        "get_weather": [
            "Based on the weather data, {result}",
            "I checked the weather for you. {result}",
            "The weather forecast shows {result}"
        ],
        "get_date_time": [
            "Right now it's {result}",
            "The current date and time is {result}",
            "According to my clock, it's {result}"
        ],
        # Default templates for any tool
        "default": [
            "I found this information for you: {result}",
            "Here's what I discovered: {result}",
            "The {tool_name} tool returned: {result}"
        ]
    }
    
    # Extract key information from the tool result if it's in JSON format
    try:
        result_data = json.loads(tool_result)
        # If it's JSON, we could extract specific fields based on the tool type
        if tool_name == "get_weather" and isinstance(result_data, dict):
            if "temperature" in result_data and "conditions" in result_data:
                temp = result_data.get("temperature")
                cond = result_data.get("conditions", "").lower()
                location = result_data.get("location", "the requested location")
                
                # More natural response for weather
                return f"I checked the weather in {location}. It's currently {temp}°C with {cond} conditions."
        
        # For other JSON results, format them nicely
        tool_result = json.dumps(result_data, indent=2)
    except (json.JSONDecodeError, TypeError):
        # Not JSON, use as is
        pass
    
    # Select appropriate templates
    templates = response_templates.get(tool_name, response_templates["default"])
    
    # Choose a template based on simple hashing of the query for variety
    template_index = hash(user_query) % len(templates)
    template = templates[template_index]
    
    # Format the response
    return template.format(result=tool_result, tool_name=tool_name)