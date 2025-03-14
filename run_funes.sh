#!/bin/bash

# Run ollama server
ollama serve

# Activate virtual environment
source funes-env/bin/activate

# Run Funes
python /home/julio/Funes/funes.py > funes.log 2>&1

