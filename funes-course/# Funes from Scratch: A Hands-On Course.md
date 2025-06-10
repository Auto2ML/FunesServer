# Funes from Scratch: A Hands-On Course
## Building an Enhanced LLM Memory System with RAG Pipeline

### Course Overview

This hands-on course will guide you through building Funes, an enhanced Large Language Model system with persistent memory capabilities and Retrieval-Augmented Generation (RAG) pipeline, from the ground up. Named after Jorge Luis Borges' character "Funes the Memorious," this system provides LLMs with contextual memory that persists across sessions.

**Duration:** 8-10 hours  
**Level:** Intermediate to Advanced  
**Prerequisites:** Python programming, basic SQL knowledge, familiarity with LLMs

---

## Module 1: Foundation and Architecture (45 minutes)

### Learning Objectives
- Understand the concept of memory-enhanced LLMs
- Learn about RAG (Retrieval-Augmented Generation) architecture
- Set up the development environment

### Topics Covered
1. **Introduction to Memory-Enhanced LLMs**
   - The problem with stateless LLMs
   - Short-term vs long-term memory concepts
   - Vector embeddings for semantic search

2. **Funes Architecture Overview**
   - Dual memory system design
   - Database-backed persistent memory
   - Tool integration framework

3. **Development Environment Setup**
   - Python virtual environment creation
   - Installing core dependencies
   - PostgreSQL with pgvector setup

### Hands-On Exercise
- Set up project structure
- Create virtual environment
- Install basic dependencies: `psycopg2`, `sentence-transformers`, `numpy`

### Deliverable
- Working development environment with all dependencies installed

---

## Module 2: Database Layer and Vector Storage (90 minutes)

### Learning Objectives
- Implement PostgreSQL database with vector extensions
- Create database schemas for memory storage
- Build comprehensive database management layer
- Understand vector similarity search operations

### Topics Covered
1. **PostgreSQL and pgvector Deep Dive**
   - Vector database fundamentals
   - pgvector extension installation and configuration
   - Distance metrics (cosine, euclidean, inner product)

2. **Database Schema Design**
   - Memory table structure with vector embeddings
   - Tool embeddings for vector-based selection
   - Document storage for RAG pipeline
   - Indexing strategies (HNSW, IVFFlat) for performance

3. **DatabaseManager Implementation**
   - Connection pooling and management
   - CRUD operations for memories and tools
   - Vector similarity search with filtering
   - Error handling and transaction management

4. **Performance Optimization**
   - Index configuration and tuning
   - Query optimization strategies
   - Batch operations and maintenance

### Hands-On Exercise
- Build complete `DatabaseManager` class with connection pooling
- Implement memory insertion and retrieval with vector search
- Create tool embedding storage and retrieval
- Test database operations with comprehensive test suite
- Performance benchmark similarity search operations

### Code Implementation
```python
class DatabaseManager:
    def __init__(self, db_params, pool_size=10):
        # Initialize connection pool and prepare statements
        
    def insert_memory(self, content, embedding, source='chat', **kwargs):
        # Store memory with vector embedding and metadata
        
    def retrieve_memories(self, query_embedding, top_k=5, **filters):
        # Semantic search with filtering options
        
    def store_tool_embedding(self, tool_name, description, parameters, embedding):
        # Store tool information for vector-based selection
        
    def find_relevant_tools(self, query_embedding, top_k=3):
        # Find most relevant tools using vector similarity
```

### Advanced Features
- Hybrid search combining semantic and keyword matching
- Memory importance scoring and cleanup strategies
- Performance monitoring and health checks
- Concurrent access testing and optimization

### Deliverable
- Complete `database.py` module with all CRUD operations
- Database schema with proper indexing
- Comprehensive test suite with performance benchmarks
- Documentation of vector operations and best practices

---

## Module 3: Embedding System and Memory Manager (75 minutes)

### Learning Objectives
- Implement text-to-vector embedding system
- Build memory management with dual storage
- Create embedding-based retrieval system

### Topics Covered
1. **Embedding Generation**
   - Sentence transformers for embeddings
   - Batch processing for efficiency
   - Embedding model management

2. **Dual Memory System**
   - Short-term memory (in-memory deque)
   - Long-term memory (database storage)
   - Memory lifecycle management

3. **Context Building**
   - Combining short and long-term memories
   - Relevance scoring and filtering
   - Context window management

### Hands-On Exercise
- Build `EmbeddingManager` class
- Implement `DualMemoryManager` with both memory types
- Create context building algorithms
- Test memory retrieval with various queries

### Code Implementation
```python
class EmbeddingManager:
    def __init__(self):
        # Initialize sentence transformer model
        
    def get_embedding(self, text):
        # Generate embedding for single text
        
    def get_batch_embeddings(self, texts):
        # Batch embedding generation

class DualMemoryManager:
    def __init__(self):
        # Initialize both memory systems
        
    def store_memory(self, context, source='chat'):
        # Store in long-term memory
        
    def retrieve_relevant_memories(self, query, top_k=3):
        # Semantic retrieval from database
        
    def _build_context(self, user_message):
        # Combine memories for context
```

### Deliverable
- Complete `memory_manager.py` with dual memory system

---

## Module 4: LLM Integration and Handler (60 minutes)

### Learning Objectives
- Integrate multiple LLM backends (Ollama, llama.cpp, HuggingFace)
- Implement conversation management
- Handle tool calling and function execution

### Topics Covered
1. **Multi-Backend LLM Support**
   - Ollama integration
   - llama.cpp integration
   - HuggingFace transformers integration

2. **Conversation Management**
   - Message formatting for different backends
   - Context injection strategies
   - Response processing

3. **Tool Integration**
   - Function calling protocols
   - Tool response processing
   - Error handling and fallbacks

### Hands-On Exercise
- Build `LLMHandler` class with backend abstraction
- Implement conversation flow management
- Test with different LLM backends
- Create tool calling mechanism

### Code Implementation
```python
class LLMHandler:
    def __init__(self, backend_type='ollama'):
        # Initialize selected backend
        
    def generate_response(self, user_input, conversation_history, 
                         additional_context=None, include_tools=True):
        # Generate response with context and tools
        
    def format_messages(self, messages):
        # Format for specific backend
        
    def _handle_tool_calls(self, response):
        # Process and execute tool calls
```

### Deliverable
- Complete `llm_handler.py` with multi-backend support

---

## Module 5: Tools Framework and Implementation (90 minutes)

### Learning Objectives
- Design extensible tool architecture
- Implement core tools (DateTime, Weather)
- Create tool discovery and execution system

### Topics Covered
1. **Tool Architecture Design**
   - Abstract base class for tools
   - Parameter validation
   - Response formatting

2. **Core Tool Implementation**
   - DateTime tool with timezone support
   - Weather tool with API integration
   - Tool response enhancement

3. **Tool Management System**
   - Automatic tool discovery
   - Vector-based tool selection
   - Tool embedding storage

### Hands-On Exercise
- Create `GenericTool` abstract base class
- Implement `DateTimeTool` and `WeatherTool`
- Build tool discovery system
- Create vector-based tool selection
- Test tool execution and response formatting

### Code Implementation
```python
class GenericTool(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        pass
        
    @property
    @abstractmethod
    def description(self) -> str:
        pass
        
    @property
    @abstractmethod
    def parameters(self) -> Dict[str, Any]:
        pass
        
    @abstractmethod
    def execute(self, **kwargs) -> str:
        pass

class DateTimeTool(GenericTool):
    # Implementation of datetime functionality
    
class WeatherTool(GenericTool):
    # Implementation of weather functionality
```

### Deliverable
- Complete `tools/` package with extensible framework

---

## Module 6: RAG Pipeline and Document Processing (75 minutes)

### Learning Objectives
- Implement document ingestion pipeline
- Create chunking and embedding strategies
- Build retrieval-augmented generation system

### Topics Covered
1. **Document Processing**
   - Multi-format document support (PDF, DOCX, TXT, MD)
   - Text extraction and cleaning
   - Document chunking strategies

2. **RAG Pipeline**
   - Document embedding and storage
   - Query-document similarity matching
   - Context injection for generation

3. **Advanced Retrieval**
   - Hybrid search (semantic + keyword)
   - Re-ranking strategies
   - Context window optimization

### Hands-On Exercise
- Build `RAGSystem` class
- Implement document processing pipeline
- Create chunking algorithms
- Test document ingestion and retrieval

### Code Implementation
```python
class RAGSystem:
    def __init__(self, db_params):
        # Initialize document processing components
        
    def process_file(self, file_path):
        # Process and store document
        
    def _convert_to_text(self, file_path):
        # Extract text from various formats
        
    def _chunk_document(self, text, chunk_size=512):
        # Split document into manageable chunks
```

### Deliverable
- Complete `rag_system.py` with document processing

---

## Module 7: User Interface and Web Application (60 minutes)

### Learning Objectives
- Build interactive web interface with Gradio
- Create memory visualization tools
- Implement file upload and management

### Topics Covered
1. **Gradio Interface Design**
   - Chat interface components
   - Memory management panels
   - Tool information displays

2. **Interactive Features**
   - Real-time chat with memory context
   - Memory visualization and insights
   - Document upload and processing

3. **User Experience Optimization**
   - Response streaming
   - Error handling and feedback
   - Mobile-responsive design

### Hands-On Exercise
- Build complete Gradio interface
- Implement chat functionality
- Create memory management tools
- Add document upload capability
- Test full user workflow

### Code Implementation
```python
def setup_gradio():
    with gr.Blocks() as demo:
        # Chat interface
        chatbot = gr.Chatbot()
        msg = gr.Textbox()
        
        # Memory management
        memory_display = gr.HTML()
        file_upload = gr.File()
        
        # Event handlers
        msg.submit(chat_interface, [msg, chatbot], chatbot)
        
    return demo
```

### Deliverable
- Complete `interface.py` with full web application

---

## Module 8: Configuration and Deployment (45 minutes)

### Learning Objectives
- Create flexible configuration system
- Implement logging and monitoring
- Deploy and test complete system

### Topics Covered
1. **Configuration Management**
   - Environment-specific settings
   - LLM backend configuration
   - Memory and embedding parameters

2. **Logging and Monitoring**
   - Structured logging implementation
   - Performance monitoring
   - Error tracking and debugging

3. **Deployment and Testing**
   - Installation script creation
   - End-to-end testing
   - Performance optimization

### Hands-On Exercise
- Create comprehensive `config.py`
- Implement logging throughout system
- Build installation script
- Perform full system testing
- Deploy and verify functionality

### Code Implementation
```python
# config.py
LLM_CONFIG = {
    'model_name': 'llama3.2:latest',
    'backend_type': 'ollama',
    'system_prompt': "You are Funes...",
    'tool_use_prompt': "You have access to tools..."
}

MEMORY_CONFIG = {
    'short_term_capacity': 10,
    'short_term_ttl_minutes': 30,
    'default_top_k': 3
}
```

### Deliverable
- Complete system ready for production use

---

## Module 9: Advanced Features and Optimization (60 minutes)

### Learning Objectives
- Implement advanced memory management
- Optimize performance and scalability
- Add advanced tool capabilities

### Topics Covered
1. **Advanced Memory Features**
   - Memory importance scoring
   - Automatic memory cleanup
   - Cross-session memory persistence

2. **Performance Optimization**
   - Database query optimization
   - Embedding caching strategies
   - Batch processing improvements

3. **Advanced Tool Integration**
   - Custom tool development
   - Tool chaining and workflows
   - API integration patterns

### Hands-On Exercise
- Implement memory importance scoring
- Add database indexing optimizations
- Create custom tool examples
- Benchmark and optimize performance

### Deliverable
- Enhanced system with advanced features

---

## Module 10: Testing and Quality Assurance (45 minutes)

### Learning Objectives
- Implement comprehensive testing strategy
- Create integration tests
- Ensure system reliability

### Topics Covered
1. **Unit Testing**
   - Database operations testing
   - Memory management testing
   - Tool execution testing

2. **Integration Testing**
   - End-to-end workflow testing
   - LLM backend integration testing
   - Error scenario testing

3. **Performance Testing**
   - Memory retrieval benchmarks
   - Embedding generation performance
   - Concurrent user testing

### Hands-On Exercise
- Write unit tests for core components
- Create integration test suite
- Perform performance benchmarking
- Document test results

### Deliverable
- Complete test suite with documentation

---

## Final Project: Custom Enhancement

### Project Options (Choose One)

1. **Advanced Document Processing**
   - Add support for additional file formats
   - Implement advanced chunking strategies
   - Create document relationship mapping

2. **Custom Tool Development**
   - Build domain-specific tools
   - Create tool workflow system
   - Implement API integrations

3. **Memory Enhancement**
   - Add memory categorization
   - Implement memory relationship graphs
   - Create advanced retrieval algorithms

4. **Multi-User Support**
   - Add user authentication
   - Implement user-specific memories
   - Create sharing and collaboration features

### Final Deliverable
- Enhanced Funes system with custom features
- Documentation of enhancements
- Presentation of implementation

---

## Course Resources

### Required Software
- Python 3.8+
- PostgreSQL 12+
- Git
- Text editor/IDE

### Recommended Reading
- "Funes the Memorious" by Jorge Luis Borges
- RAG system design patterns
- Vector database optimization techniques

### Additional Resources
- Sentence Transformers documentation
- Ollama API documentation
- PostgreSQL pgvector extension guide
- Gradio interface development guide

### Code Repository Structure
```
funes-from-scratch/
├── database.py
├── memory_manager.py
├── llm_handler.py
├── rag_system.py
├── interface.py
├── config.py
├── funes.py
├── tools/
│   ├── __init__.py
│   ├── generic_tool.py
│   ├── datetime_tool.py
│   └── weather_tool.py
├── tests/
├── requirements.txt
├── install.sh
└── README.md
```

### Assessment Criteria
- Code quality and organization
- Functionality completeness
- Performance optimization
- Documentation quality
- Creative enhancements in final project

---

## Course Completion Certificate

Upon successful completion of all modules and the final project, participants will receive a certificate demonstrating proficiency in:

- RAG system architecture and implementation
- Vector database design and optimization
- LLM integration and tool development
- Memory-enhanced AI system development
- Full-stack AI application development

**Total Course Duration:** 8-10 hours  
**Hands-on Coding Time:** ~6 hours  
**Theory and Discussion:** ~2-4 hours