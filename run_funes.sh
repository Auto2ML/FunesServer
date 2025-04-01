#!/bin/bash
# Run script for Funes server with LlamaIndex implementation

# Activate the virtual environment if it exists
if [ -d "funes-env" ]; then
    source funes-env/bin/activate
elif [ -d "env" ]; then
    source env/bin/activate
elif [ -d "venv" ]; then
    source venv/bin/activate
fi

# Set environment variables
export PYTHONPATH=$PYTHONPATH:$(pwd)
export USE_LLAMAINDEX=true  # Enable LlamaIndex implementation by default

# Run the server
python funes.py

# Deactivate virtual environment
deactivate

