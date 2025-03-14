# Funes: Enhanced LLM Memory System

## Overview

Funes is a system that enhances local Large Language Models with persistent memory capabilities, inspired by Jorge Luis Borges' short story "Funes the Memorious" (Funes el Memorioso). Just as the character Funes possessed an infinite capacity for memory, this system extends an LLM's capabilities by providing external memory storage through PostgreSQL.

The system integrates with Ollama for local LLM execution while maintaining a persistent memory of previous interactions and knowledge in a PostgreSQL database.

## Prerequisites

- Python 3.8 or higher
- PostgreSQL 13.0 or higher
- Ollama (for running local LLMs)
- Basic familiarity with terminal/command line operations

## Detailed Installation Guide

### 1. Install Ollama

First, you need to install Ollama to run the LLM locally. Visit [Ollama's official website](https://ollama.ai) and follow the installation instructions for your operating system:

```bash
# For macOS/Linux (using curl)
curl -fsSL https://ollama.ai/install.sh | sh

# For Windows
# Download the installer from https://ollama.ai/download
```

### 2. Clone the Repository

```bash
git clone https://github.com/yourusername/funes.git
cd funes
```

### 3. Set Up Python Environment

```bash
# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
# On Linux/macOS:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

### 4. PostgreSQL Setup

#### Install PostgreSQL:
- **Ubuntu/Debian:**
  ```bash
  sudo apt update
  sudo apt install postgresql postgresql-contrib
  ```
- **macOS (using Homebrew):**
  ```bash
  brew install postgresql
  ```
- **Windows:** Download and install from [PostgreSQL website](https://www.postgresql.org/download/windows/)

#### Configure Database:
```bash
# Start PostgreSQL service
# Linux:
sudo service postgresql start
# macOS:
brew services start postgresql
# Windows: PostgreSQL is installed as a service and should start automatically

# Create database and user
sudo -u postgres psql

postgres=# CREATE DATABASE funes;
postgres=# CREATE USER llm WITH PASSWORD 'llm';
postgres=# GRANT ALL PRIVILEGES ON DATABASE funes TO llm;
postgres=# \q
```

### Install pgvector (vector database extension for Postgres)

```bash
sudo apt-get install postgresql-server-dev-all
```

If you are a Windows user, you can download the PostgreSQL installer from the official website.

#### Clone the pgvector GitHub repository

```bash
git clone https://github.com/pgvector/pgvector.git
```

#### Build and install the pgvector extension:

```bash
cd pgvector
make
sudo make install
```

If you are a Windows user, ensure you have C++ support in Visual Studio Code installed. The official installation documentation provides a step-by-step process.

#### Connect to your PostgreSQL database

You have several options for connecting and interacting with the PostgreSQL database: pgAdmin is one of the most commonly used interfaces. Alternatively, you can use pSQL (PostgreSQL command line interface) or even a VS Code extension for PostgreSQL.

#### After connecting to your PostgreSQL database, create the extension:

```bash
CREATE EXTENSION vector;
```

### 5. Environment Configuration

Create a `.env` file in the project root:

```plaintext
DATABASE_URL=postgresql://llm:llm@localhost:5432/funes
OLLAMA_BASE_URL=http://localhost:11434
```

### 6. Initialize Ollama 
### HERE WE NEED TO ADD THE FINAL LIST OF MODELS USED BY FUNES

#### Start ollama server
ollama serve


### 7. Start the Application

```bash
python Funes.py
```

The Gradio interface will be available at `http://localhost:7860`

## Usage

1. Open your web browser and navigate to `http://localhost:7860`
2. The interface provides:
   - A chat interface for interacting with the LLM
   - Memory management options
   - Context visualization tools

## Troubleshooting

### Common Issues:

1. **PostgreSQL Connection Issues:**
   - Verify PostgreSQL is running:
     ```bash
     # Linux
     sudo service postgresql status
     # macOS
     brew services list
     ```
   - Check database credentials in `.env`

2. **Ollama Connection:**
   - Ensure Ollama is running:
     ```bash
     ollama list
     ```
   - Verify the OLLAMA_BASE_URL in `.env`

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
- Built with Ollama, PostgreSQL, and Gradio
- Special thanks to the open-source community
