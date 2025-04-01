#!/usr/bin/env python3
"""
Funes - Enhanced LLM Memory System with LlamaIndex-based RAG Pipeline

Main module for the Funes application.
"""

from interface import setup_gradio
from config import LLAMAINDEX_CONFIG

if __name__ == "__main__":
    # Set up and launch the Gradio interface
    print(f"LlamaIndex implementation {'enabled' if LLAMAINDEX_CONFIG['enabled'] else 'disabled'}")
    demo = setup_gradio()
    demo.launch(server_name="0.0.0.0")
