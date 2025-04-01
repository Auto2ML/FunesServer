# Funes: Enhanced LLM Memory System with RAG Pipeline

## Overview

Funes is a system that enhances local Large Language Models with persistent memory capabilities and a Retrieval-Augmented Generation (RAG) pipeline, inspired by Jorge Luis Borges' short story "Funes the Memorious" (Funes el Memorioso). Just as the character Funes remembered everything he experienced, our system provides local LLMs with a persistent and contextually relevant memory system.

The system supports multiple LLM backends including Ollama, llama.cpp, and HuggingFace, while maintaining a persistent memory of previous interactions and knowledge in a PostgreSQL database with vector storage capabilities for semantic search.

## Prerequisites

- Linux-based system (Ubuntu/Debian recommended)
- Sudo privileges (non-root user)
- Internet connection for downloading dependencies

## Quick Installation

The entire installation process has been simplified with a bash script that handles all the necessary setup:

```bash
# Clone the repository
git clone https://github.com/Auto2ML/FunesServer.git
cd FunesServer

# Make the install script executable
chmod +x install.sh

# Run the installation script
./install.sh
```

The installation script automatically:
- Installs system dependencies
- Sets up LLM backends (Ollama, required libraries for llama.cpp and HuggingFace)
- Configures PostgreSQL with pgvector extension
- Creates the required database and user
- Sets up Python virtual environment
- Downloads the required LLM models
- Creates a launcher script for easy execution

## Running Funes

After installation, you can run Funes with the provided launcher script:

```bash
./run_funes.sh
```

The Gradio interface will be available at `http://localhost:7860`

## Configuration

Funes can be configured through the `config.py` file to use different LLM backends and models:

```python
# LLM model configuration
LLM_CONFIG = {
    'model_name': 'llama3.2:latest',         # The model name or path
    'backend_type': 'ollama',                # Options: 'ollama', 'llamacpp', 'huggingface'
    'system_prompt': "You are Funes..."      # System prompt for the LLM
}

# Embedding model configuration
EMBEDDING_CONFIG = {
    'model_name': 'all-MiniLM-L6-v2',        # Embedding model name
}
```

### LLM Backend Options:

1. **Ollama Backend**:
   - Set `'backend_type': 'ollama'`
   - Use any model available in Ollama for `model_name`

2. **llama.cpp Backend**:
   - Set `'backend_type': 'llamacpp'`
   - For `model_name`, provide the path to your GGUF model file

3. **HuggingFace Backend**:
   - Set `'backend_type': 'huggingface'`
   - Use any model identifier from HuggingFace for `model_name`

### Memory Configuration:

```python
# Memory configuration
MEMORY_CONFIG = {
    'short_term_capacity': 10,               # Number of messages to keep in short-term memory
    'short_term_ttl_minutes': 30,            # Time-to-live for short-term memory items
    'default_top_k': 3                       # Default number of relevant memories to retrieve
}
```

## Tools System

Funes includes a robust tools system that allows the LLM to interact with external functionalities. This system enables the LLM to:

1. **Access Real-time Information**: Get current date/time information and weather conditions
2. **Process and Format Data**: Extract, transform, and present data in natural language
3. **Extend Capabilities**: The framework allows for easy addition of new tools

### Available Tools:

#### 1. DateTime Tool
- Provides current date and time information
- Supports multiple timezones
- Configurable output formats (full, date, time, iso)
- Example usage: "What's the current time in Tokyo?"

#### 2. Weather Tool
- Retrieves weather information for specified locations
- Supports different temperature formats (Celsius/Fahrenheit)
- Provides details like temperature, conditions, humidity, and wind speed
- Example usage: "What's the weather like in Paris today?"

### Tools Architecture:

The tools system uses a flexible architecture based on the GenericTool class, making it easy to implement new tools. Each tool defines:
- A unique name
- A descriptive explanation of its functionality
- Required and optional parameters
- An execution method that performs the actual functionality

### Tool Response Enhancement:

The system includes a response enhancement mechanism that converts raw tool outputs into natural, conversational responses. This makes the LLM's responses feel more human-like when using tools.

### Adding Custom Tools:

To create a new tool, extend the GenericTool class and implement the required methods:
1. Define properties: name, description, and parameters
2. Implement the execute method that performs the tool's functionality
3. Place the tool in the tools/ directory
4. The tool will be automatically detected and made available to the LLM

## Manual Installation (Alternative)

If you prefer to install components manually, follow these steps:

### 1. Install Ollama

```bash
# For macOS/Linux (using curl)
curl -fsSL https://ollama.ai/install.sh | sh

# For Windows
# Download the installer from https://ollama.ai/download
```

### 2. PostgreSQL Setup with pgvector

```bash
# Install PostgreSQL and development libraries
sudo apt update
sudo apt install postgresql postgresql-contrib postgresql-server-dev-all

# Install pgvector
git clone https://github.com/pgvector/pgvector.git
cd pgvector
make
sudo make install

# Create database and user
sudo -u postgres psql
postgres=# CREATE DATABASE funes;
postgres=# CREATE USER llm WITH PASSWORD 'llm';
postgres=# GRANT ALL PRIVILEGES ON DATABASE funes TO llm;
postgres=# \c funes
postgres=# CREATE EXTENSION vector;
postgres=# \q
```

### 3. Python Environment Setup

```bash
# Create a virtual environment
python3 -m venv funes-env
source funes-env/bin/activate

# Install required packages
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Environment Configuration

Create a `.env` file in the project root:

```plaintext
DATABASE_URL=postgresql://llm:llm@localhost:5432/funes
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_USER=llm
POSTGRES_PASSWORD=llm
POSTGRES_DB=funes
OLLAMA_HOST=localhost
OLLAMA_PORT=11434
```

## Funes RAG Architecture

Funes implements a Retrieval-Augmented Generation (RAG) pipeline that enhances LLM capabilities by:

1. **Dual Memory System**: 
   - Short-term memory for recent conversation context
   - Long-term memory stored as vector embeddings in PostgreSQL
2. **Embedding Storage**: Converting previous conversations and knowledge into vector embeddings stored in PostgreSQL using pgvector
3. **Semantic Retrieval**: Finding contextually relevant information from past conversations using vector similarity search
4. **Context Enhancement**: Augmenting LLM prompts with retrieved context to provide more informed responses
5. **Persistent Memory**: Maintaining knowledge across sessions for continuous learning and improvement

This architecture allows Funes to provide more accurate, contextual responses based on conversation history and stored knowledge.

## Usage

1. Open your web browser and navigate to `http://localhost:7860`
2. The interface provides:
   - A chat interface for interacting with the LLM
   - Memory management options
   - Context visualization tools
   - RAG pipeline configuration

## Troubleshooting

### Common Issues:

1. **PostgreSQL Connection Issues:**
   - Verify PostgreSQL is running:
     ```bash
     sudo systemctl status postgresql
     ```
   - Check database credentials in `config.py`

2. **LLM Backend Connection:**
   - For Ollama backend:
     ```bash
     ollama list
     ```
   - For llama.cpp: Ensure the model path in `config.py` is correct
   - For HuggingFace: Verify internet connection or local model availability

3. **Python Dependencies:**
   - If you encounter module not found errors:
     ```bash
     pip install -r requirements.txt --upgrade
     ```

4. **Model Behavior Issues:**
   - If the model seems to be "talking to itself" or duplicating context, try:
     - Clearing the conversation history
     - Checking your `config.py` settings
     - Verifying that your LLM model is properly installed

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[Add your license information here]

## Acknowledgments

- Inspired by Jorge Luis Borges' "Funes the Memorious"
- Built with PostgreSQL, pgvector, and Gradio
- Supports multiple LLM backends: Ollama, llama.cpp, and HuggingFace
- Special thanks to the open-source community
