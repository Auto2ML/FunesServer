#!/usr/bin/env python3
"""
Test script for LlamaIndex integration with Funes

This script provides a simple way to test the LlamaIndex integration components
without needing to run the full Funes server.
"""

import os
import sys

# Enable LlamaIndex by default for this test
os.environ['USE_LLAMAINDEX'] = 'true'

from llamaindex_integration import IntegratedMemoryManager, IntegratedRAGSystem
from llamaindex_llm import LlamaIndexLLMHandler
from llamaindex_rag import LlamaIndexRAGSystem
from config import DB_CONFIG

def test_rag_system():
    """Test the RAG system components"""
    print("Testing LlamaIndex RAG system...")
    try:
        # Initialize the RAG system
        rag_system = LlamaIndexRAGSystem(DB_CONFIG)
        print("✅ RAG system initialized successfully")
        
        # Test storing a memory
        result = rag_system.store_memory("This is a test memory from the test script", "test")
        print(f"✅ Store memory result: {result}")
        
        # Test querying
        result = rag_system.query("test memory", top_k=1)
        print(f"✅ Query result: {result}")
        
        return True
    except Exception as e:
        import traceback
        print(f"❌ Error testing RAG system: {str(e)}")
        print(traceback.format_exc())
        return False

def test_llm_handler():
    """Test the LLM handler components"""
    print("\nTesting LlamaIndex LLM handler...")
    try:
        # Initialize the LLM handler
        llm_handler = LlamaIndexLLMHandler()
        print("✅ LLM handler initialized successfully")
        
        # Test generating a response
        response = llm_handler.generate_response("Hello, what can you do?")
        print(f"✅ Response received: {response[:100]}...")
        
        return True
    except Exception as e:
        import traceback
        print(f"❌ Error testing LLM handler: {str(e)}")
        print(traceback.format_exc())
        return False

def test_integration():
    """Test the integration components"""
    print("\nTesting LlamaIndex integration...")
    try:
        # Initialize the integrated memory manager
        memory_manager = IntegratedMemoryManager(True)
        print("✅ Integrated memory manager initialized successfully")
        
        # Test processing a chat
        response = memory_manager.process_chat("Tell me about yourself")
        print(f"✅ Chat response: {response[:100]}...")
        
        return True
    except Exception as e:
        import traceback
        print(f"❌ Error testing integration: {str(e)}")
        print(traceback.format_exc())
        return False

if __name__ == "__main__":
    print("=== LLAMAINDEX INTEGRATION TEST ===\n")
    
    rag_success = test_rag_system()
    llm_success = test_llm_handler()
    integration_success = test_integration()
    
    if rag_success and llm_success and integration_success:
        print("\n✅ All tests passed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed")
        sys.exit(1)