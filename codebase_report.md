# Funes Codebase Report

## Project Overview

**Funes** is a Python-based system designed to enhance Large Language Models (LLMs) with a persistent, dual-memory system and a Retrieval-Augmented Generation (RAG) pipeline. The name is inspired by Jorge Luis Borges' short story "Funes the Memorious."

The system's core functionality is to provide LLMs with both short-term and long-term memory, allowing them to have more contextually aware and informed conversations. It supports multiple LLM backends (Ollama, llama.cpp, and HuggingFace) and uses a PostgreSQL database with the `pgvector` extension for efficient semantic search of memories.

## Key Components

### 1. `funes.py` (Main Entry Point)

*   **Purpose:** This is the main script that initializes and runs the Funes server.
*   **Functionality:**
    *   Sets up the logging configuration for the entire application based on `config.py`.
    *   Initializes and launches the Gradio web interface for user interaction.

### 2. `interface.py` (Gradio Web Interface)

*   **Purpose:** Defines the user interface using the Gradio library.
*   **Functionality:**
    *   Provides a chat interface for users to interact with the LLM.
    *   Displays "Memory Insights," showing memories that were relevant to the last query.
    *   Includes a "Memory Management" tab for:
        *   Uploading documents to be added to the long-term memory.
        *   Clearing all memories or deleting memories from a specific source.
    *   Features a "Tools" tab that lists the available tools and explains how they are used.
    *   An "About" tab gives an overview of the Funes project.

### 3. `memory_manager.py` (Core Memory Logic)

*   **Purpose:** This is the heart of the Funes system, managing both short-term and long-term memory.
*   **Key Classes:**
    *   `EmbeddingManager`: Handles the creation of vector embeddings for text using the `sentence-transformers` library.
    *   `DualMemoryManager`:
        *   **Short-Term Memory:** A `deque` (a list-like container with fast appends and pops from both ends) that stores the recent conversation history. It has a configurable capacity and a Time-to-Live (TTL) for messages.
        *   **Long-Term Memory:** Interacts with the `DatabaseManager` to store and retrieve memories from the PostgreSQL database.
        *   **Chat Processing:** When a user sends a message, it builds a context from both short-term and long-term memories, sends it to the LLM, and stores the new interaction.
        *   **Tool Detection:** It uses a vector-based approach to determine if a user's query should trigger a tool.

### 4. `llm_handler.py` (LLM Interaction)

*   **Purpose:** Manages all communication with the configured LLM backend.
*   **Functionality:**
    *   Currently supports `ollama` as the primary backend.
    *   Formats the conversation history and context before sending it to the LLM.
    *   Handles tool calls. If the LLM decides to use a tool, this handler executes the tool and sends the result back to the LLM to generate a final response.

### 5. `database.py` (Database Management)

*   **Purpose:** Handles all interactions with the PostgreSQL database.
*   **Functionality:**
    *   Connects to the database and creates the necessary tables (`memories` and `tools_embeddings`) if they don't exist.
    *   Uses the `pgvector` extension to store and search for vector embeddings.
    *   Provides methods to insert, retrieve, and delete memories and tool embeddings.

### 6. `rag_system.py` (Retrieval-Augmented Generation)

*   **Purpose:** Implements the RAG pipeline, which allows Funes to incorporate information from external documents into its knowledge base.
*   **Functionality:**
    *   Uses the `docling` library to convert uploaded documents (PDF, DOCX, TXT, etc.) into plain text.
    *   Splits the text into chunks.
    *   Generates embeddings for each chunk and stores them in the long-term memory (database).

### 7. `config.py` (Configuration)

*   **Purpose:** Centralized configuration for the entire application.
*   **Settings:**
    *   `LLM_CONFIG`: LLM model name, backend type, and system prompts.
    *   `EMBEDDING_CONFIG`: Embedding model name.
    *   `MEMORY_CONFIG`: Short-term memory capacity, TTL, and default number of memories to retrieve.
    *   `DB_CONFIG`: Database connection parameters.
    *   `LOGGING_CONFIG`: Logging level, format, and file path.

### 8. `tools/` (Directory for Tools)

*   **Purpose:** Contains the implementation of external tools that the LLM can use.
*   **Examples (from `README.md`):**
    *   `DateTime Tool`: Gets the current date and time.
    *   `Weather Tool`: Retrieves weather information.
*   **Architecture:** The system is designed to be extensible, allowing new tools to be added easily by placing them in this directory.

## Dependencies

The project relies on the following key Python libraries (from `requirements.txt`):

*   `gradio`: For the web interface.
*   `sentence-transformers`: For creating text embeddings.
*   `psycopg2`: For connecting to the PostgreSQL database.
*   `ollama`: For interacting with the Ollama LLM backend.
*   `docling`: For converting various document formats to text.
*   `transformers`, `torch`, `llama-cpp-python`: For supporting different LLM backends.

## Installation and Execution

*   `install.sh`: A shell script that automates the installation of all dependencies, including setting up PostgreSQL and downloading LLM models.
*   `run_funes.sh`: A launcher script to run the Funes server.

## Overall Architecture

Funes is a well-structured and modular application. The separation of concerns is clear:

*   **`interface.py`** handles the user-facing part.
*   **`memory_manager.py`** is the central orchestrator.
*   **`llm_handler.py`** deals with the LLM.
*   **`database.py`** manages data persistence.
*   **`rag_system.py`** handles external knowledge.
*   **`config.py`** centralizes all settings.

The use of a dual-memory system combined with a RAG pipeline and an extensible tool system makes Funes a powerful and flexible platform for building advanced conversational AI applications. The code is well-documented and follows good practices, such as using a dedicated logging system and providing automated installation scripts.
