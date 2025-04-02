"""
Configuration settings for Funes server
"""
# LLM model configuration
LLM_CONFIG = {
    'model_name': 'llama3.2:latest',
    'backend_type': 'ollama',  # Currently, only 'ollama' is fully implemented in the LlamaIndex integration
    'system_prompt': "You are Funes, an AI assistant. Use your recent conversation history and relevant memories stored in your database to provide more informed and consistent responses, only if it is relevant for the conversation. Do not repeat unnecessary information. If you don't find relevant information in history or memories, use your training data",
    # Models with known good tool support - this is for documentation purposes only
    'recommended_models': [
        'llama3.2:latest',  # Best tool support
        'mistral:latest',
        'mixtral:latest',
        'gemma',
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

# LlamaIndex configuration
LLAMAINDEX_CONFIG = {
    'enabled': True,  # This should always be true now that we've fully migrated to LlamaIndex
    'vector_store_table': 'memories',  # Table name for vector storage
    'agent_verbose': True,  # Enable verbose output from LlamaIndex agent
    'storage_dir': 'llamaindex_storage',  # Directory for persisting indices
}

# Note: If you encounter issues with tool functionality, try changing to a model with better function calling support.
# The current integration only fully supports the Ollama backend.
# To add support for other backends (llamafile, llamacpp, huggingface), you'll need to update the _setup_llm() method
# in llamaindex_llm.py
