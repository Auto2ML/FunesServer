"""
Configuration settings for Funes server
"""
# LLM model configuration
LLM_CONFIG = {
    'model_name': 'llama3.2:latest',
    'backend_type': 'ollama',  # Options: 'ollama', 'llamacpp', 'huggingface'
    #'model_name': '/home/julio/.ollama/models/blobs/sha256-dde5aa3fc5ffc17176b5e8bdc82f587b24b2678c6c66101bf7da77af9f7ccdff',
    'system_prompt': "You are Funes, a helpful assistant. Use your recent conversation history and relevant memories stored in your database to provide more informed and consistent responses, only if it is relevant for the conversation. Do not repeat unnecessary information. If you don't find relevant information in history or memories, use your training data",
    'tool_use_prompt': "You have access to the following tools: {tools_description}. Only use these tools when necessary, such as for getting real-time information or performing specific actions that cannot be answered using your existing knowledge or memory context. For questions about general knowledge, facts, concepts, or information included in your training data, respond directly without using tools. If the user's query can be answered using context from memory or your training data, prefer those sources over tools.",
    # Models with known good tool support - this is for documentation purposes only
    'recommended_models': [
        'llama3.2:latest',  # Best tool support
        'mistral:latest',
        'mixtral:latest',
        'llama2:latest'
    ]
}

# Tool configuration
TOOL_CONFIG = {
    'enabled_tools': ['get_weather', 'get_date_time'],  # Tool names to enable
    'custom_tools_dir': None,  # Directory with custom tool implementations
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

# Note: Funes always uses tools. If you encounter issues with tool functionality, 
# try changing to a model with better tool support like 'llama3.2:latest'
