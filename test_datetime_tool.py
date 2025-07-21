#!/usr/bin/env python3
"""
Test script to verify datetime tool functionality
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from memory_manager import DualMemoryManager

def test_madrid_time_query():
    """Test the Madrid time query that was failing"""
    print("Testing Madrid time query...")
    
    # Create memory manager
    memory_manager = DualMemoryManager()
    
    # Test the query
    query = "What is the time in Madrid now?"
    print(f"Query: {query}")
    
    try:
        response = memory_manager.process_chat(query)
        print(f"Response: {response}")
        
        # Check if response contains useful time information
        if "madrid" in response.lower() and any(word in response.lower() for word in ["time", "currently", "pm", "am"]):
            print("✅ SUCCESS: Response contains time information for Madrid")
        else:
            print("❌ FAIL: Response doesn't seem to contain proper time information")
            
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_madrid_time_query()
