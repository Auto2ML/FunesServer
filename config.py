"""
Configuration settings for Funes server
"""
import logging

# LLM model configuration
LLM_CONFIG = {
    'model_name': 'llama3.2:1b',
    'backend_type': 'ollama',  # Options: 'ollama', 'llamacpp', 'huggingface', 'llamafile'
    #'model_name': '/home/julio/.ollama/models/blobs/sha256-dde5aa3fc5ffc17176b5e8bdc82f587b24b2678c6c66101bf7da77af9f7ccdff',
    'llamafile_api_url': "http://localhost:8080/v1",  # Default URL for Llamafile API (OpenAI-compatible format)
    'system_prompt': "Your name is Funes, a general purpose AI assistant. Use your recent conversation history and relevant memories stored in your database to provide more informed and consistent responses, only if it is relevant for the conversation. Do not repeat unnecessary information. If you don't find relevant information in history or memories, use your training data",
    'tool_use_prompt': "You have access to the following tools: {tools_description}. Only use these tools when necessary, such as for getting real-time information or performing specific actions that cannot be answered using your existing knowledge or memory context. For questions about general knowledge, facts, concepts, or information included in your training data, respond directly without using tools. If the user's query can be answered using context from memory or your training data, prefer those sources over tools.",
    # Models with known good tool support - this is for documentation purposes only
    'recommended_models': [
        'llama3.2:latest',  # Best tool support
        'mistral:latest',
        'mixtral:latest',
        'gemma',
        'llama2:latest',
        'llamafile with llama-3-8b or llama-3-70b'  # Llamafile models also support tools
    ],
    # Logging configuration
    'log_level': 'INFO',  # Options: DEBUG, INFO, WARNING, ERROR, CRITICAL
    'enable_logging': True,  # Set to False to disable all logging
    'log_format': '%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    'log_date_format': '%Y-%m-%d %H:%M:%S'
}

# Embedding model configuration
EMBEDDING_CONFIG = {
    'model_name': 'all-MiniLM-L6-v2',  # Default embedding model
}

# Memory configuration
MEMORY_CONFIG = {
    'short_term_capacity': 10,
    'short_term_ttl_minutes': 30,
    'default_top_k': 3
}

# Database configuration
DB_CONFIG = {
    'dbname': 'funes',
    'user': 'llm',
    'password': 'llm',
    'host': 'localhost'
}

# Logging configuration
LOGGING_CONFIG = {
    'level': logging.INFO,  # Default log level
    'format': '%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    'date_format': '%Y-%m-%d %H:%M:%S',
    'file_path': None,  # Set to a file path to enable logging to a file
    'enable': True  # Master switch to enable/disable all logging
}

# Note: Funes always uses tools. If you encounter issues with tool functionality, 
# try changing to a model with better tool support like 'llama3.2:latest'
# For Llamafile support, make sure to run your llamafile with proper API access:
# ./your-llamafile --api-on
# The Llamafile backend uses the OpenAI-compatible API format
