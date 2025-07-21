import json
import re
import logging
from typing import List, Dict, Any, Optional, Tuple

# Get logger for this module
logger = logging.getLogger('LLMUtilities')

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

def should_use_tools(messages: List[Dict[str, str]], tools: List[Dict[str, Any]]) -> Tuple[bool, Optional[str]]:
    """
    Determine if tools should be used and which specific tool might be needed
    using keyword-based matching.
    
    Args:
        messages: Conversation history
        tools: Available tools
        
    Returns:
        Tuple of (should_use, tool_name) where tool_name can be None if no specific tool is identified
    """
    # Exit early if no tools are available
    if not tools or len(tools) == 0:
        return False, None
        
    # Get the latest user message
    latest_user_msg = ""
    for msg in reversed(messages):
        if msg["role"] == "user" and msg.get("content"):
            latest_user_msg = msg["content"].lower()
            break
    
    if not latest_user_msg:
        return False, None
    
    # Direct tool mention check - if user explicitly mentions a tool by name
    for tool in tools:
        if tool['function']['name'] in latest_user_msg:
            return True, tool['function']['name']
    
    # These patterns suggest factual or dynamic information needs that tools could address
    info_seeking_patterns = [
        r"what is (?:the|current) (.+)\??",
        r"how (?:many|much) (.+)\??",
        r"when is (.+)\??",
        r"where is (.+)\??",
        r"(?:can you )?(?:tell|show) me (?:the|about) (.+)\??",
        r"(?:can you )?(?:get|fetch|find|search for) (.+)\??",
        r"(current|latest|today's) (.+)",
        r"(.+) right now\??",
    ]
    
    # Check if the message contains information-seeking patterns 
    contains_info_seeking = any(re.search(pattern, latest_user_msg) for pattern in info_seeking_patterns)
    
    if contains_info_seeking:
        return True, None
    
    # Default to not using tools
    return False, None

def should_use_tools_vector(user_message: str, embedding_function, db_manager, similarity_threshold: float = 0.75) -> Tuple[bool, Optional[str]]:
    """
    Determine if tools should be used and which specific tool might be needed
    using vector embedding similarity matching.
    
    Args:
        user_message: The user's input message
        embedding_function: Function to generate embeddings (can be None, will use default)
        db_manager: Database manager instance for querying tool embeddings
        similarity_threshold: Minimum similarity score to consider a tool match
        
    Returns:
        Tuple of (should_use, tool_name) where tool_name can be None if no specific tool is identified
    """
    try:
        logger.info(f"Vector-based tool selection for query: {user_message[:50]}...")
        
        # Generate embedding for the user message
        if embedding_function:
            query_embedding = embedding_function(user_message)
        else:
            # Use the global embedding manager
            from memory_manager import embedding_manager
            query_embedding = embedding_manager.get_embedding(user_message)
        
        # Find the most similar tool using the database manager
        similar_tool = db_manager.find_most_similar_tool(query_embedding, similarity_threshold)
        
        if similar_tool:
            tool_name, similarity_score = similar_tool
            logger.info(f"Found similar tool: {tool_name} with similarity {similarity_score:.3f}")
            return True, tool_name
        else:
            logger.info(f"No tool found above similarity threshold {similarity_threshold}")
            return False, None
            
    except Exception as e:
        logger.error(f"Error in vector-based tool selection: {str(e)}")
        # Fall back to keyword-based detection
        logger.info("Falling back to keyword-based tool detection")
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
    
    This function dynamically adapts to custom tools by analyzing tool names and contents.
    
    Args:
        user_query: Original user question that triggered the tool
        tool_name: Name of the tool that was executed
        tool_result: Raw result returned from the tool
        
    Returns:
        A conversational response that naturally incorporates the tool result
    """
    # Extract the tool's general purpose from its name
    tool_category = None
    
    # Check for common tool categories by name pattern
    if "weather" in tool_name:
        tool_category = "weather"
    elif any(term in tool_name for term in ["time", "date", "calendar"]):
        tool_category = "datetime"
    elif any(term in tool_name for term in ["search", "find", "query", "lookup"]):
        tool_category = "search"
    elif any(term in tool_name for term in ["calculate", "compute", "math"]):
        tool_category = "calculation"
    
    # Try to parse the tool result as JSON
    try:
        result_data = json.loads(tool_result)
        is_json = True
    except (json.JSONDecodeError, TypeError):
        result_data = None
        is_json = False
    
    # Handle natural datetime formatting
    if tool_category == "datetime" and is_json and isinstance(result_data, dict):
        # Check if this is our enhanced datetime format with detailed time components
        if "type" in result_data and "weekday" in result_data and "month" in result_data:
            # This is our enhanced datetime format
            data_type = result_data.get("type", "full")
            tz_info = f" in {result_data.get('timezone')}" if result_data.get('timezone') != "system" else ""
            
            # Build natural language response based on data type
            if data_type == "date":
                # Format: "Today is Monday, April 15, 2025"
                return f"Today is {result_data['weekday']}, {result_data['month']} {result_data['day']}, {result_data['year']}{tz_info}."
            
            elif data_type == "time":
                # Format: "It's currently 2:30 PM"
                if int(result_data['minute']) == 0:
                    # For even hours, use simpler format
                    return f"It's {result_data['hour']} {result_data['am_pm']}{tz_info}."
                else:
                    return f"It's {result_data['hour']}:{result_data['minute']} {result_data['am_pm']}{tz_info}."
            
            else:  # full format
                # Create more natural date/time expressions
                weekday = result_data['weekday']
                month = result_data['month']
                day = result_data['day']
                year = result_data['year']
                hour = result_data['hour']
                minute = result_data['minute']
                am_pm = result_data['am_pm']
                
                # Select different formats randomly based on query hash for variety
                formats = [
                    f"It's {hour}:{minute} {am_pm} on {weekday}, {month} {day}, {year}{tz_info}.",
                    f"The current time is {hour}:{minute} {am_pm} on {weekday}, {month} {day}{tz_info}.",
                    f"Right now it's {hour}:{minute} {am_pm}, {month} {day}, {year}{tz_info}."
                ]
                
                # Choose format based on hash of query
                format_index = hash(user_query) % len(formats)
                return formats[format_index]
                
    # Category-specific formatting for other JSON data
    if tool_category == "weather" and is_json:
        # Format weather results in a natural way
        if isinstance(result_data, dict):
            # Common fields in weather APIs
            temp = result_data.get("temperature") or result_data.get("temp")
            cond = (result_data.get("conditions") or 
                   result_data.get("description") or 
                   result_data.get("weather"))
            location = (result_data.get("location") or 
                      result_data.get("city") or 
                      result_data.get("place") or 
                      "the requested location")
            
            if temp and cond:
                units = result_data.get("units", "°C")
                if not isinstance(units, str):
                    units = "°"
                return f"I checked the weather in {location}. It's currently {temp}{units} with {cond.lower() if isinstance(cond, str) else ''} conditions."
    
    elif tool_category == "datetime" and is_json:
        # Format datetime results naturally (old format handling)
        if isinstance(result_data, dict):
            date = result_data.get("date") or result_data.get("current_date")
            time = result_data.get("time") or result_data.get("current_time")
            
            if date and time:
                return f"It's currently {time} on {date}."
            elif time:
                return f"The current time is {time}."
            elif date:
                return f"Today's date is {date}."
    
    # Category-specific templates when we can't extract structured data
    templates = {
        "weather": [
            "Based on the weather data, {result}",
            "I checked the weather for you. {result}",
            "The weather forecast shows {result}"
        ],
        "datetime": [
            "Right now it's {result}",
            "The current date and time is {result}",
            "According to my clock, it's {result}"
        ],
        "search": [
            "Here's what I found: {result}",
            "I searched for that information. {result}",
            "The search returned: {result}"
        ],
        "calculation": [
            "I calculated that {result}",
            "The result is {result}",
            "Based on my calculations: {result}"
        ],
        # Default templates for any tool
        "default": [
            "I found this information for you: {result}",
            "Here's what I discovered: {result}",
            "Based on the {tool_name}: {result}"
        ]
    }
    
    # Format the result for JSON data if we couldn't do specific formatting
    if is_json and result_data:
        # For JSON results, format them nicely
        tool_result = json.dumps(result_data, indent=2)
        
        # Try to extract a key piece of information if there's just one main field
        if isinstance(result_data, dict) and len(result_data) == 1:
            key = list(result_data.keys())[0]
            value = result_data[key]
            if isinstance(value, (str, int, float, bool)):
                tool_result = f"{key}: {value}"
    
    # Choose appropriate templates
    category_templates = templates.get(tool_category, templates["default"])
    
    # Choose a template based on simple hashing of the query for variety
    template_index = hash(user_query) % len(category_templates)
    template = category_templates[template_index]
    
    # Extract key verb from the user query to create more natural responses
    query_verbs = ["show", "tell", "get", "find", "what", "how", "when", "where", "is", "are"]
    user_verb = next((verb for verb in query_verbs if verb in user_query.lower()), None)
    
    # Format the response
    response = template.format(result=tool_result, tool_name=tool_name.replace("_", " "))
    
    # Make first character uppercase if it's not already
    if response and len(response) > 0 and not response[0].isupper():
        response = response[0].upper() + response[1:]
    
    return response
