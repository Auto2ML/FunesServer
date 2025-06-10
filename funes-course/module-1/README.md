# Module 1: Foundation and Architecture

## Overview
This module introduces the concepts behind memory-enhanced LLMs and sets up the development environment for building Funes from scratch.

**Duration:** 45 minutes  
**Prerequisites:** Basic Python knowledge, familiarity with virtual environments

## Learning Objectives
By the end of this module, you will:
- Understand why stateless LLMs need memory enhancement
- Know the architecture of Funes' dual memory system
- Have a working development environment set up
- Understand vector embeddings and their role in semantic search

## Module Structure
```
module-1/
├── README.md                 # This file
├── setup/
│   ├── environment_setup.py  # Automated environment setup
│   ├── requirements.txt      # Python dependencies
│   └── test_setup.py        # Environment validation
├── exercises/
│   ├── 01_memory_concepts.py # Memory system concepts
│   ├── 02_vector_basics.py   # Vector embedding basics
│   └── 03_architecture.py    # Funes architecture overview
└── solutions/
    ├── exercise_01_solution.py
    ├── exercise_02_solution.py
    └── exercise_03_solution.py
```

## Getting Started

1. **Set up your environment:**
   ```bash
   cd module-1/setup
   python environment_setup.py
   ```

2. **Verify installation:**
   ```bash
   python test_setup.py
   ```

3. **Complete exercises:**
   ```bash
   cd ../exercises
   python 01_memory_concepts.py
   python 02_vector_basics.py
   python 03_architecture.py
   ```

## Key Concepts

### 1. The Problem with Stateless LLMs
- LLMs don't retain information between conversations
- Context window limitations
- No persistent learning from interactions

### 2. Memory-Enhanced Architecture
- **Short-term memory**: Recent conversation context
- **Long-term memory**: Persistent knowledge storage
- **Vector embeddings**: Semantic similarity search

### 3. Funes Architecture Components
- Database layer (PostgreSQL + pgvector)
- Memory manager (dual memory system)
- LLM handler (conversation management)
- Tools system (external capabilities)
- RAG pipeline (document processing)

## Next Steps
After completing this module, proceed to Module 2: Database Layer and Vector Storage.
