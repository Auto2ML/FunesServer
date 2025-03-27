"""
Configuration settings for Funes server
"""

# LLM model configuration
LLM_CONFIG = {
    'model_name': 'llama3.2:latest',
    'backend_type': 'ollama',  # Options: 'ollama', 'llamacpp', 'huggingface'
    #'model_name': '/home/julio/.ollama/models/blobs/sha256-dde5aa3fc5ffc17176b5e8bdc82f587b24b2678c6c66101bf7da77af9f7ccdff',
    'system_prompt': "You are Funes, a helpful assistant. Use your recent conversation history and relevant memories stored in your database to provide more informed and consistent responses, only if it is relevant for the conversation. Do not repeat unnecessary information. If you don't find relevant information in history or memories, use your training data"
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
