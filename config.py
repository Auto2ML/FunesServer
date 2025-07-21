"""
Configuration settings for Funes server
"""
import logging
from typing import Dict, List, Union

# LLM model configuration
LLM_CONFIG: Dict[str, Union[str, bool, List[str]]] = {
    'model_name': 'llama3.2:latest',  # Default model name
    'backend_type': 'ollama',  # May add other options in the future
     'system_prompt': "Your name is Funes, a general purpose AI assistant with memory capabilities. IMPORTANT: When responding to user queries, first carefully review any 'Relevant past memories' section that appears at the end of this system message - these are memories retrieved from your database that are most relevant to the current query. Incorporate insights from these memories to provide more informed responses. Only use these memories when they are directly relevant to the query, and do not repeat unnecessary information. If no relevant memories are provided or they don't apply to the current query, simply use your training data to respond. When using tools, prefer natural language formatting for responses by setting response_format to 'natural' when available to provide more conversational interactions.",
    'tool_use_prompt': "You have access to the following tools: {tools_description}. Only use these tools when necessary, such as for getting real-time information or performing specific actions that cannot be answered using your existing knowledge or memory context. For questions about general knowledge, facts, concepts, or information included in your training data, respond directly without using tools. If the user's query can be answered using context from memory or your training data, prefer those sources over tools.",
    'vector_tool_selection': True,  # Enable vector-based tool selection using database similarity search
    # Models with known good tool support - this is for documentation purposes only
    'recommended_models': [
        'llama3.2:latest',  # Best tool support
        'mistral:latest',
        'mixtral:latest',
        'gemma',
        'llama2:latest',
        # 'llamafile with llama-3-8b or llama-3-70b'  # Llamafile models also support tools
    ]
}

# Embedding model configuration
EMBEDDING_CONFIG: Dict[str, str] = {
    'model_name': 'all-MiniLM-L6-v2',  # Default embedding model
}

# Memory configuration
MEMORY_CONFIG: Dict[str, int] = {
    'short_term_capacity': 10,
    'short_term_ttl_minutes': 30,
    'default_top_k': 3
}

# Database configuration
DB_CONFIG: Dict[str, str] = {
    'dbname': 'funes',
    'user': 'llm',
    'password': 'llm',
    'host': 'localhost'
}

# Logging configuration
LOGGING_CONFIG: Dict[str, Union[str, bool, None]] = {
    'level': 'DEBUG',  # Default log level
    'format': '%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    'date_format': '%Y-%m-%d %H:%M:%S',
    'file_path': None,  # Set to a file path to enable logging to a file
    'enable': True  # Master switch to enable/disable all logging
}

# Note: Funes always uses tools. If you encounter issues with tool functionality, 
# try changing to a model with better tool support like 'llama3.2:latest'
# Commented out Llamafile references:
# # For Llamafile support, make sure to run your llamafile with proper API access:
# # ./your-llamafile --api-on
# # The Llamafile backend uses the OpenAI-compatible API format
