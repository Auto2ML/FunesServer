# Funes: Enhanced LLM Memory System with RAG Pipeline

## Overview

Funes is a system that enhances local Large Language Models with persistent memory capabilities and a Retrieval-Augmented Generation (RAG) pipeline, inspired by Jorge Luis Borges' short story "Funes the Memorious" (Funes el Memorioso). Just as the character Funes remembered everything he experienced, our system provides local LLMs with a persistent and contextually relevant memory system.

The system integrates with Ollama for local LLM execution while maintaining a persistent memory of previous interactions and knowledge in a PostgreSQL database with vector storage capabilities for semantic search.

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
- Sets up Ollama
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

1. **Embedding Storage**: Converting previous conversations and knowledge into vector embeddings stored in PostgreSQL using pgvector
2. **Semantic Retrieval**: Finding contextually relevant information from past conversations using vector similarity search
3. **Context Enhancement**: Augmenting LLM prompts with retrieved context to provide more informed responses
4. **Persistent Memory**: Maintaining knowledge across sessions for continuous learning and improvement

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
   - Check database credentials in `.env`

2. **Ollama Connection:**
   - Ensure Ollama is running:
     ```bash
     ollama list
     ```
   - Verify the OLLAMA_HOST and OLLAMA_PORT in `.env`

3. **Python Dependencies:**
   - If you encounter module not found errors:
     ```bash
     pip install -r requirements.txt --upgrade
     ```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[Add your license information here]

## Acknowledgments

- Inspired by Jorge Luis Borges' "Funes the Memorious"
- Built with Ollama, PostgreSQL, pgvector, and Gradio
- Special thanks to the open-source community
