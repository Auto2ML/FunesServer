#!/usr/bin/env python3
"""
Funes Course - Module 1, Exercise 3: Funes Architecture Overview
Understanding the complete system architecture and component interactions
"""

import json
from datetime import datetime
from collections import deque
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
from enum import Enum

class ComponentStatus(Enum):
    """Status of system components"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"
    LOADING = "loading"

@dataclass
class MemoryEntry:
    """Data structure for memory entries"""
    id: str
    content: str
    timestamp: datetime
    source: str
    embedding_summary: str  # Simplified representation of embedding
    relevance_score: float = 0.0

@dataclass
class ToolCall:
    """Data structure for tool calls"""
    tool_name: str
    parameters: Dict[str, Any]
    result: str
    timestamp: datetime

class MockComponent:
    """Base class for mock system components"""
    
    def __init__(self, name: str):
        self.name = name
        self.status = ComponentStatus.INACTIVE
        self.last_activity = None
    
    def activate(self):
        self.status = ComponentStatus.ACTIVE
        self.last_activity = datetime.now()
    
    def get_status(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'status': self.status.value,
            'last_activity': self.last_activity.isoformat() if self.last_activity else None
        }

class MockDatabaseManager(MockComponent):
    """Mock database manager to demonstrate database layer"""
    
    def __init__(self):
        super().__init__("Database Manager")
        self.memories = []
        self.tools_embeddings = []
        self.connection_pool_size = 5
    
    def store_memory(self, content: str, source: str) -> str:
        """Store a memory entry"""
        self.activate()
        memory_id = f"mem_{len(self.memories) + 1:04d}"
        
        memory = MemoryEntry(
            id=memory_id,
            content=content,
            timestamp=datetime.now(),
            source=source,
            embedding_summary=f"vec[{len(content.split())}_words]"
        )
        
        self.memories.append(memory)
        return memory_id
    
    def retrieve_memories(self, query: str, top_k: int = 3) -> List[MemoryEntry]:
        """Retrieve relevant memories (simplified similarity)"""
        self.activate()
        # Simple keyword-based similarity for demonstration
        query_words = set(query.lower().split())
        
        scored_memories = []
        for memory in self.memories:
            memory_words = set(memory.content.lower().split())
            overlap = len(query_words.intersection(memory_words))
            relevance = overlap / len(query_words) if query_words else 0
            
            memory.relevance_score = relevance
            if relevance > 0:
                scored_memories.append(memory)
        
        # Sort by relevance and return top_k
        scored_memories.sort(key=lambda x: x.relevance_score, reverse=True)
        return scored_memories[:top_k]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        return {
            'total_memories': len(self.memories),
            'memory_sources': list(set(m.source for m in self.memories)),
            'oldest_memory': min(self.memories, key=lambda x: x.timestamp).timestamp.isoformat() if self.memories else None,
            'connection_pool_size': self.connection_pool_size
        }

class MockMemoryManager(MockComponent):
    """Mock memory manager demonstrating dual memory system"""
    
    def __init__(self, db_manager: MockDatabaseManager):
        super().__init__("Memory Manager")
        self.db_manager = db_manager
        self.short_term_memory = deque(maxlen=10)
        self.context_window_size = 2048
    
    def add_to_short_term(self, content: str, role: str):
        """Add entry to short-term memory"""
        self.activate()
        entry = {
            'content': content,
            'role': role,
            'timestamp': datetime.now()
        }
        self.short_term_memory.append(entry)
    
    def store_long_term(self, content: str, source: str = "conversation"):
        """Store entry in long-term memory"""
        return self.db_manager.store_memory(content, source)
    
    def build_context(self, user_query: str) -> Dict[str, Any]:
        """Build context from both memory systems"""
        self.activate()
        
        # Get recent short-term memory
        recent_context = list(self.short_term_memory)[-5:]  # Last 5 entries
        
        # Get relevant long-term memories
        relevant_memories = self.db_manager.retrieve_memories(user_query)
        
        return {
            'short_term_context': recent_context,
            'long_term_context': [asdict(m) for m in relevant_memories],
            'context_tokens_estimate': len(user_query.split()) + sum(len(entry['content'].split()) for entry in recent_context)
        }

class MockLLMHandler(MockComponent):
    """Mock LLM handler demonstrating conversation management"""
    
    def __init__(self, memory_manager: MockMemoryManager):
        super().__init__("LLM Handler")
        self.memory_manager = memory_manager
        self.model_name = "mock-llama3.2:1b"
        self.tools_available = ["datetime", "weather", "calculator"]
    
    def generate_response(self, user_input: str, use_tools: bool = True) -> Dict[str, Any]:
        """Generate response using memory context"""
        self.activate()
        
        # Build context from memory
        context = self.memory_manager.build_context(user_input)
        
        # Add user input to short-term memory
        self.memory_manager.add_to_short_term(user_input, "user")
        
        # Simulate LLM processing
        response_parts = []
        tool_calls = []
        
        # Check if tools should be used
        if use_tools and any(keyword in user_input.lower() for keyword in ['time', 'weather', 'calculate']):
            if 'time' in user_input.lower():
                tool_calls.append(ToolCall(
                    tool_name="datetime",
                    parameters={"format": "full"},
                    result="It's currently 2:30 PM on Tuesday, March 15, 2024",
                    timestamp=datetime.now()
                ))
        
        # Generate response based on context
        if context['long_term_context']:
            response_parts.append(f"Based on our previous conversations, I remember that {context['long_term_context'][0]['content'][:50]}...")
        
        if tool_calls:
            response_parts.append(f"Using tools: {tool_calls[0].result}")
        else:
            response_parts.append(f"I can help with: {user_input}")
        
        response = " ".join(response_parts)
        
        # Add response to short-term memory
        self.memory_manager.add_to_short_term(response, "assistant")
        
        # Store significant interactions in long-term memory
        if len(user_input.split()) > 5:  # Only store substantial interactions
            self.memory_manager.store_long_term(user_input, "conversation")
        
        return {
            'response': response,
            'context_used': context,
            'tool_calls': [asdict(tc) for tc in tool_calls],
            'tokens_used': len(response.split()) + context['context_tokens_estimate']
        }

class MockToolsSystem(MockComponent):
    """Mock tools system demonstrating tool integration"""
    
    def __init__(self):
        super().__init__("Tools System")
        self.available_tools = {
            'datetime': {
                'description': 'Get current date and time',
                'parameters': ['format', 'timezone'],
                'usage_count': 0
            },
            'weather': {
                'description': 'Get weather information',
                'parameters': ['location', 'format'],
                'usage_count': 0
            },
            'calculator': {
                'description': 'Perform mathematical calculations',
                'parameters': ['expression'],
                'usage_count': 0
            }
        }
    
    def execute_tool(self, tool_name: str, parameters: Dict[str, Any]) -> str:
        """Execute a tool with given parameters"""
        self.activate()
        
        if tool_name not in self.available_tools:
            return f"Tool '{tool_name}' not found"
        
        self.available_tools[tool_name]['usage_count'] += 1
        
        # Mock tool execution
        if tool_name == 'datetime':
            return f"Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        elif tool_name == 'weather':
            location = parameters.get('location', 'Unknown')
            return f"Weather in {location}: 22°C, Sunny"
        elif tool_name == 'calculator':
            expression = parameters.get('expression', '0')
            return f"Result: {expression} = 42"  # Mock result
        
        return "Tool executed successfully"
    
    def get_tool_stats(self) -> Dict[str, Any]:
        """Get tool usage statistics"""
        return {
            'total_tools': len(self.available_tools),
            'tool_usage': {name: info['usage_count'] for name, info in self.available_tools.items()},
            'most_used_tool': max(self.available_tools.items(), key=lambda x: x[1]['usage_count'])[0] if self.available_tools else "none"
        }

class MockRAGSystem(MockComponent):
    """Mock RAG system demonstrating document processing"""
    
    def __init__(self, db_manager: MockDatabaseManager):
        super().__init__("RAG System")
        self.db_manager = db_manager
        self.processed_documents = []
        self.supported_formats = ['.txt', '.md', '.pdf', '.docx']
    
    def process_document(self, file_path: str, document_content: str) -> Dict[str, Any]:
        """Process and store a document"""
        self.activate()
        
        # Simulate document processing
        chunks = self._chunk_document(document_content)
        stored_chunks = []
        
        for i, chunk in enumerate(chunks):
            chunk_id = self.db_manager.store_memory(
                content=chunk,
                source=f"document:{file_path}:chunk_{i}"
            )
            stored_chunks.append(chunk_id)
        
        doc_info = {
            'file_path': file_path,
            'chunks_count': len(chunks),
            'stored_chunk_ids': stored_chunks,
            'processed_at': datetime.now().isoformat()
        }
        
        self.processed_documents.append(doc_info)
        return doc_info
    
    def _chunk_document(self, content: str, chunk_size: int = 100) -> List[str]:
        """Split document into chunks"""
        words = content.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size):
            chunk = ' '.join(words[i:i + chunk_size])
            chunks.append(chunk)
        
        return chunks
    
    def get_document_stats(self) -> Dict[str, Any]:
        """Get document processing statistics"""
        return {
            'total_documents': len(self.processed_documents),
            'total_chunks': sum(doc['chunks_count'] for doc in self.processed_documents),
            'supported_formats': self.supported_formats
        }

class FunesArchitecture:
    """Main Funes architecture demonstrating component integration"""
    
    def __init__(self):
        # Initialize components in dependency order
        self.db_manager = MockDatabaseManager()
        self.memory_manager = MockMemoryManager(self.db_manager)
        self.llm_handler = MockLLMHandler(self.memory_manager)
        self.tools_system = MockToolsSystem()
        self.rag_system = MockRAGSystem(self.db_manager)
        
        self.components = [
            self.db_manager,
            self.memory_manager,
            self.llm_handler,
            self.tools_system,
            self.rag_system
        ]
    
    def process_user_input(self, user_input: str) -> Dict[str, Any]:
        """Process user input through the complete system"""
        
        # Check if this is a document upload simulation
        if user_input.startswith("UPLOAD:"):
            file_path = user_input.split(":", 1)[1]
            # Simulate document content
            doc_content = f"This is the content of {file_path}. It contains important information about various topics that the user wants to remember."
            doc_result = self.rag_system.process_document(file_path, doc_content)
            return {
                'type': 'document_upload',
                'result': doc_result,
                'system_status': self.get_system_status()
            }
        
        # Regular conversation processing
        response_data = self.llm_handler.generate_response(user_input)
        
        return {
            'type': 'conversation',
            'user_input': user_input,
            'response': response_data['response'],
            'context_used': response_data['context_used'],
            'tool_calls': response_data['tool_calls'],
            'system_status': self.get_system_status()
        }
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return {
            'components': [comp.get_status() for comp in self.components],
            'database_stats': self.db_manager.get_stats(),
            'tool_stats': self.tools_system.get_tool_stats(),
            'document_stats': self.rag_system.get_document_stats(),
            'memory_summary': {
                'short_term_entries': len(self.memory_manager.short_term_memory),
                'long_term_entries': len(self.db_manager.memories)
            }
        }

def demonstrate_architecture():
    """Demonstrate the complete Funes architecture"""
    print("🏗️ FUNES ARCHITECTURE DEMONSTRATION")
    print("=" * 60)
    
    # Initialize Funes system
    funes = FunesArchitecture()
    
    print("SYSTEM COMPONENTS INITIALIZED:")
    print("-" * 30)
    status = funes.get_system_status()
    for component in status['components']:
        print(f"  • {component['name']}: {component['status']}")
    
    print(f"\nDatabase: {status['database_stats']['total_memories']} memories")
    print(f"Tools: {status['tool_stats']['total_tools']} available")
    print(f"Documents: {status['document_stats']['total_documents']} processed")
    
    return funes

def demonstrate_conversation_flow():
    """Demonstrate how a conversation flows through the system"""
    print("\n💬 CONVERSATION FLOW DEMONSTRATION")
    print("=" * 60)
    
    funes = FunesArchitecture()
    
    # Simulate a conversation
    conversation = [
        "Hello, my name is Bob and I work as a software engineer.",
        "I'm interested in machine learning and AI.",
        "What time is it?",
        "Can you remember what I told you about my job?",
        "UPLOAD:my_resume.pdf"
    ]
    
    for i, user_input in enumerate(conversation, 1):
        print(f"\n{i}. User: {user_input}")
        
        result = funes.process_user_input(user_input)
        
        if result['type'] == 'conversation':
            print(f"   Funes: {result['response']}")
            
            # Show context usage
            context = result['context_used']
            if context['long_term_context']:
                print(f"   Memory used: {len(context['long_term_context'])} long-term memories")
            if context['short_term_context']:
                print(f"   Recent context: {len(context['short_term_context'])} recent messages")
            
            # Show tool usage
            if result['tool_calls']:
                for tool_call in result['tool_calls']:
                    print(f"   Tool called: {tool_call['tool_name']} -> {tool_call['result']}")
        
        elif result['type'] == 'document_upload':
            doc_result = result['result']
            print(f"   Document processed: {doc_result['chunks_count']} chunks stored")
        
        # Show system status after each interaction
        memory_summary = result['system_status']['memory_summary']
        print(f"   Memory state: {memory_summary['short_term_entries']} short-term, {memory_summary['long_term_entries']} long-term")

def demonstrate_component_interactions():
    """Show how components interact with each other"""
    print("\n🔄 COMPONENT INTERACTIONS")
    print("=" * 60)
    
    funes = FunesArchitecture()
    
    print("Data Flow Example: 'What's the weather like?'")
    print("-" * 40)
    
    # Trace the flow step by step
    user_input = "What's the weather like?"
    
    print("1. User Input → Memory Manager")
    print(f"   Input: '{user_input}'")
    
    print("\n2. Memory Manager → Database Manager")
    context = funes.memory_manager.build_context(user_input)
    print(f"   Retrieved {len(context['long_term_context'])} relevant memories")
    
    print("\n3. Memory Manager → LLM Handler")
    print(f"   Built context with {context['context_tokens_estimate']} estimated tokens")
    
    print("\n4. LLM Handler → Tools System")
    print("   Detected weather query, preparing tool call")
    
    print("\n5. Tools System → LLM Handler")
    weather_result = funes.tools_system.execute_tool("weather", {"location": "current"})
    print(f"   Tool result: {weather_result}")
    
    print("\n6. LLM Handler → Memory Manager")
    print("   Storing interaction in short-term memory")
    
    print("\n7. Memory Manager → Database Manager")
    print("   Significant interaction stored in long-term memory")
    
    print("\nComplete Response Generated ✓")

def architecture_quiz():
    """Interactive quiz about Funes architecture"""
    print("\n🎯 ARCHITECTURE QUIZ")
    print("=" * 60)
    
    questions = [
        {
            "question": "Which component is responsible for storing vector embeddings?",
            "options": ["A) Memory Manager", "B) Database Manager", "C) LLM Handler", "D) Tools System"],
            "correct": "B",
            "explanation": "The Database Manager handles PostgreSQL with pgvector for storing embeddings."
        },
        {
            "question": "What is the purpose of the dual memory system?",
            "options": ["A) Backup storage", "B) Short-term + Long-term memory", "C) Multiple users", "D) Error handling"],
            "correct": "B",
            "explanation": "Funes uses short-term memory for recent context and long-term memory for persistent storage."
        },
        {
            "question": "Which component decides when to use tools?",
            "options": ["A) Database Manager", "B) Memory Manager", "C) LLM Handler", "D) RAG System"],
            "correct": "C",
            "explanation": "The LLM Handler analyzes user input and decides when tools are needed."
        },
        {
            "question": "What does the RAG System primarily handle?",
            "options": ["A) Real-time data", "B) Document processing", "C) User authentication", "D) Error logging"],
            "correct": "B",
            "explanation": "RAG (Retrieval-Augmented Generation) processes documents into chunks for storage."
        }
    ]
    
    score = 0
    for i, q in enumerate(questions, 1):
        print(f"\nQuestion {i}: {q['question']}")
        for option in q['options']:
            print(f"  {option}")
        
        answer = input("Your answer (A/B/C/D): ").upper().strip()
        
        if answer == q['correct']:
            print("✓ Correct!")
            score += 1
        else:
            print(f"✗ Incorrect. The correct answer is {q['correct']}")
        
        print(f"Explanation: {q['explanation']}")
    
    print(f"\nQuiz complete! Score: {score}/{len(questions)}")
    
    if score == len(questions):
        print("🎉 Perfect score! You understand Funes architecture!")
    elif score >= len(questions) * 0.7:
        print("👍 Good job! You have a solid understanding.")
    else:
        print("📚 Consider reviewing the architecture concepts.")

def architecture_summary():
    """Provide a comprehensive architecture summary"""
    print("\n📋 FUNES ARCHITECTURE SUMMARY")
    print("=" * 60)
    
    architecture_info = {
        "Core Components": {
            "Database Manager": {
                "Purpose": "Vector storage and retrieval",
                "Technology": "PostgreSQL + pgvector",
                "Key Features": ["Vector similarity search", "Memory persistence", "Tool embeddings"]
            },
            "Memory Manager": {
                "Purpose": "Dual memory system management",
                "Technology": "Python deque + Database",
                "Key Features": ["Short-term context", "Long-term storage", "Context building"]
            },
            "LLM Handler": {
                "Purpose": "Conversation management",
                "Technology": "Ollama/llama.cpp/HuggingFace",
                "Key Features": ["Multi-backend support", "Tool integration", "Response generation"]
            },
            "Tools System": {
                "Purpose": "External capability integration",
                "Technology": "Plugin architecture",
                "Key Features": ["Dynamic tool discovery", "Parameter validation", "Vector-based selection"]
            },
            "RAG System": {
                "Purpose": "Document processing pipeline",
                "Technology": "Docling + embeddings",
                "Key Features": ["Multi-format support", "Chunking strategies", "Knowledge augmentation"]
            }
        },
        "Data Flow": [
            "User Input → Memory Manager",
            "Memory Manager ↔ Database Manager (context retrieval)",
            "Memory Manager → LLM Handler (context injection)",
            "LLM Handler ↔ Tools System (capability execution)",
            "LLM Handler → Memory Manager (response storage)",
            "Memory Manager → Database Manager (persistence)"
        ],
        "Key Innovations": [
            "Vector-based memory retrieval",
            "Dual memory architecture",
            "Tool selection via embeddings",
            "Cross-session persistence",
            "Multi-format document ingestion"
        ]
    }
    
    # Print formatted summary
    for section, content in architecture_info.items():
        print(f"\n{section.upper()}:")
        print("-" * (len(section) + 1))
        
        if isinstance(content, dict):
            for component, details in content.items():
                print(f"\n{component}:")
                for key, value in details.items():
                    if isinstance(value, list):
                        print(f"  {key}: {', '.join(value)}")
                    else:
                        print(f"  {key}: {value}")
        elif isinstance(content, list):
            for item in content:
                print(f"  • {item}")

def main():
    """Main function to run all architecture demonstrations"""
    print("🧠 FUNES COURSE - Module 1, Exercise 3")
    print("=" * 60)
    print("Understanding Funes Architecture and Component Interactions")
    print()
    
    # Run demonstrations
    funes = demonstrate_architecture()
    demonstrate_conversation_flow()
    demonstrate_component_interactions()
    
    # Ask if user wants to take the quiz
    print("\n" + "=" * 60)
    take_quiz = input("Would you like to take the architecture quiz? (y/n): ").lower().strip()
    if take_quiz == 'y':
        architecture_quiz()
    
    # Always show the summary
    architecture_summary()
    
    print("\n🎉 Module 1 Complete!")
    print("\nWhat you've learned:")
    print("• The problem with stateless LLMs and benefits of memory")
    print("• How vector embeddings enable semantic search")
    print("• Funes architecture and component interactions")
    print("• Data flow through the complete system")
    print("\nNext Steps:")
    print("• Module 2: Database Layer and Vector Storage")
    print("• Module 3: Embedding System and Memory Manager")
    print("• Module 4: LLM Integration and Handler")
    print("\nReady to build Funes from scratch! 🚀")

if __name__ == "__main__":
    main()
