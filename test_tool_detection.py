#!/usr/bin/env python3
"""
Test script to verify tool detection for time-related queries
"""

import logging
import sys
import os

# Add current directory to path to import modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import LOGGING_CONFIG
from memory_manager import DualMemoryManager

def setup_test_logging():
    """Setup logging for testing"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
    )

def test_time_query():
    """Test if time-related queries properly trigger tool usage"""
    print("Testing Funes tool detection for time queries...")
    
    # Initialize the memory manager
    try:
        memory_manager = DualMemoryManager()
        print("✓ Memory manager initialized")
    except Exception as e:
        print(f"✗ Error initializing memory manager: {e}")
        return False
    
    # Test queries that should trigger the datetime tool
    test_queries = [
        "What is the time in Madrid now?",
        "What time is it in Madrid?",
        "Current time in Madrid",
        "Tell me the time in Madrid",
        "Madrid time now",
    ]
    
    print("\nTesting queries:")
    for query in test_queries:
        print(f"\nQuery: {query}")
        try:
            response = memory_manager.process_chat(query)
            print(f"Response: {response[:100]}...")
            
            # Check if the response mentions using a tool or provides actual time info
            if any(keyword in response.lower() for keyword in ['tool', 'current', 'time', 'madrid']):
                if 'unfortunately' not in response.lower() and 'don\'t have access' not in response.lower():
                    print("✓ Tool appears to have been used successfully")
                else:
                    print("✗ Tool was not used - got generic 'no access' response")
            else:
                print("? Unclear if tool was used")
                
        except Exception as e:
            print(f"✗ Error processing query: {e}")
    
    return True

if __name__ == "__main__":
    setup_test_logging()
    test_time_query()
