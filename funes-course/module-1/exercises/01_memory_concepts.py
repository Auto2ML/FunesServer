#!/usr/bin/env python3
"""
Funes Course - Module 1, Exercise 1: Memory Concepts
Understanding the difference between stateless and stateful AI systems
"""

from collections import deque
from datetime import datetime, timedelta
import json

class StatelessLLM:
    """
    Simulates a stateless LLM that doesn't retain information between calls
    """
    
    def __init__(self, name="StatelessLLM"):
        self.name = name
        # Note: No persistent storage of conversations
    
    def generate_response(self, user_input, context=""):
        """
        Generate a response without memory of previous interactions
        """
        # Simulate LLM response based only on current input and optional context
        if "weather" in user_input.lower():
            return "I don't have access to current weather data. Please check a weather service."
        elif "my name" in user_input.lower():
            return "I don't know your name. You haven't told me in this conversation."
        elif "remember" in user_input.lower():
            return "I don't have the ability to remember information between conversations."
        else:
            return f"I can help with that based on the information you've provided: {user_input[:50]}..."

class MemoryEnhancedLLM:
    """
    Simulates a memory-enhanced LLM with short-term and long-term memory
    """
    
    def __init__(self, name="MemoryEnhancedLLM", short_term_capacity=5):
        self.name = name
        
        # Short-term memory (recent conversation)
        self.short_term_memory = deque(maxlen=short_term_capacity)
        
        # Long-term memory (persistent facts and preferences)
        self.long_term_memory = {}
        
        # User profile (extracted information about the user)
        self.user_profile = {}
    
    def _extract_user_info(self, user_input):
        """
        Extract and store user information from input
        """
        input_lower = user_input.lower()
        
        # Extract name
        if "my name is" in input_lower or "i'm" in input_lower or "i am" in input_lower:
            words = user_input.split()
            for i, word in enumerate(words):
                if word.lower() in ["is", "i'm", "am"] and i + 1 < len(words):
                    potential_name = words[i + 1].strip('.,!?')
                    if potential_name.isalpha():
                        self.user_profile['name'] = potential_name
                        break
        
        # Extract preferences
        if "i like" in input_lower or "i love" in input_lower:
            preference = user_input.split("like" if "like" in input_lower else "love")[1].strip()
            if 'preferences' not in self.user_profile:
                self.user_profile['preferences'] = []
            self.user_profile['preferences'].append(preference)
    
    def _search_memory(self, query):
        """
        Search both short-term and long-term memory for relevant information
        """
        relevant_info = []
        query_lower = query.lower()
        
        # Search short-term memory
        for entry in self.short_term_memory:
            if any(word in entry['content'].lower() for word in query_lower.split()):
                relevant_info.append(f"Recent: {entry['content']}")
        
        # Search long-term memory
        for key, value in self.long_term_memory.items():
            if query_lower in key.lower() or query_lower in str(value).lower():
                relevant_info.append(f"Stored: {key} -> {value}")
        
        return relevant_info
    
    def generate_response(self, user_input):
        """
        Generate a response using both short-term and long-term memory
        """
        # Extract and store user information
        self._extract_user_info(user_input)
        
        # Add current input to short-term memory
        self.short_term_memory.append({
            'timestamp': datetime.now(),
            'content': user_input,
            'type': 'user_input'
        })
        
        # Generate response based on input and memory
        input_lower = user_input.lower()
        
        if "my name" in input_lower:
            if 'name' in self.user_profile:
                response = f"Your name is {self.user_profile['name']}."
            else:
                response = "You haven't told me your name yet."
        
        elif "remember" in input_lower:
            relevant_memories = self._search_memory(user_input)
            if relevant_memories:
                response = f"I remember: {'; '.join(relevant_memories[:3])}"
            else:
                response = "I don't have specific memories about that topic."
        
        elif "weather" in input_lower:
            # Simulate checking if we have weather preferences
            if 'preferences' in self.user_profile:
                prefs = [p for p in self.user_profile['preferences'] if 'weather' in p or 'climate' in p]
                if prefs:
                    response = f"I don't have current weather, but I remember you mentioned: {prefs[0]}"
                else:
                    response = "I don't have access to current weather data."
            else:
                response = "I don't have access to current weather data."
        
        elif any(word in input_lower for word in ["hello", "hi", "hey"]):
            if 'name' in self.user_profile:
                response = f"Hello {self.user_profile['name']}! How can I help you today?"
            else:
                response = "Hello! How can I help you today?"
        
        else:
            # Use context from memory
            context_items = []
            if 'name' in self.user_profile:
                context_items.append(f"User: {self.user_profile['name']}")
            
            relevant_memories = self._search_memory(user_input)
            if relevant_memories:
                context_items.extend(relevant_memories[:2])
            
            if context_items:
                response = f"Based on our conversation ({'; '.join(context_items)}), I can help with: {user_input[:30]}..."
            else:
                response = f"I can help with that: {user_input[:50]}..."
        
        # Store response in short-term memory
        self.short_term_memory.append({
            'timestamp': datetime.now(),
            'content': response,
            'type': 'assistant_response'
        })
        
        return response
    
    def get_memory_summary(self):
        """
        Get a summary of the current memory state
        """
        return {
            'user_profile': self.user_profile,
            'short_term_entries': len(self.short_term_memory),
            'long_term_entries': len(self.long_term_memory),
            'recent_interactions': [
                entry['content'][:50] + "..." if len(entry['content']) > 50 else entry['content']
                for entry in list(self.short_term_memory)[-3:]
            ]
        }

def demonstrate_memory_difference():
    """
    Demonstrate the difference between stateless and memory-enhanced LLMs
    """
    print("🧠 FUNES COURSE - Module 1, Exercise 1")
    print("=" * 60)
    print("Demonstrating the difference between stateless and memory-enhanced LLMs")
    print()
    
    # Create both types of LLMs
    stateless_llm = StatelessLLM()
    memory_llm = MemoryEnhancedLLM()
    
    # Test conversation sequence
    conversation = [
        "Hi! My name is Alice.",
        "I love rainy weather and coffee.",
        "What's my name?",
        "What do I like?",
        "Can you remember our conversation?",
        "Tell me about the weather."
    ]
    
    print("CONVERSATION SIMULATION")
    print("-" * 30)
    
    for i, user_input in enumerate(conversation, 1):
        print(f"\n{i}. User: {user_input}")
        
        # Stateless response
        stateless_response = stateless_llm.generate_response(user_input)
        print(f"   Stateless LLM: {stateless_response}")
        
        # Memory-enhanced response
        memory_response = memory_llm.generate_response(user_input)
        print(f"   Memory LLM: {memory_response}")
        
        # Show memory state after key interactions
        if i in [2, 4]:
            memory_summary = memory_llm.get_memory_summary()
            print(f"   Memory State: {json.dumps(memory_summary, indent=2)}")
    
    print("\n" + "=" * 60)
    print("ANALYSIS")
    print("=" * 60)
    
    print("\n1. STATELESS LLM LIMITATIONS:")
    print("   • Cannot remember user's name across interactions")
    print("   • No retention of user preferences")
    print("   • Each response is independent of conversation history")
    print("   • Cannot build context or relationships")
    
    print("\n2. MEMORY-ENHANCED LLM ADVANTAGES:")
    print("   • Remembers user information (name, preferences)")
    print("   • Builds conversation context over time")
    print("   • Can reference previous interactions")
    print("   • Provides personalized responses")
    
    print("\n3. FUNES ARCHITECTURE PREVIEW:")
    print("   • Short-term memory: Recent conversation context")
    print("   • Long-term memory: Persistent facts and relationships")
    print("   • User profiling: Extracted information about users")
    print("   • Memory search: Finding relevant past information")

def interactive_exercise():
    """
    Interactive exercise for students to experiment with memory concepts
    """
    print("\n🎯 INTERACTIVE EXERCISE")
    print("=" * 60)
    print("Create your own conversation with the memory-enhanced LLM!")
    print("Type 'quit' to exit, 'memory' to see current memory state")
    print()
    
    memory_llm = MemoryEnhancedLLM("Your Personal Assistant")
    
    while True:
        user_input = input("You: ").strip()
        
        if user_input.lower() == 'quit':
            break
        elif user_input.lower() == 'memory':
            memory_summary = memory_llm.get_memory_summary()
            print("Current Memory State:")
            print(json.dumps(memory_summary, indent=2))
            continue
        elif not user_input:
            continue
        
        response = memory_llm.generate_response(user_input)
        print(f"Assistant: {response}")
        print()

def main():
    """
    Main function to run all exercises
    """
    demonstrate_memory_difference()
    
    # Ask if user wants to try interactive exercise
    print("\n" + "=" * 60)
    try_interactive = input("Would you like to try the interactive exercise? (y/n): ").lower().strip()
    if try_interactive == 'y':
        interactive_exercise()
    
    print("\n🎉 Exercise 1 Complete!")
    print("Key takeaways:")
    print("• Memory transforms AI from reactive to contextual")
    print("• Short-term + Long-term memory = Better user experience")
    print("• Funes implements this with PostgreSQL + vector embeddings")
    print("\nNext: Run exercise 02_vector_basics.py")

if __name__ == "__main__":
    main()
