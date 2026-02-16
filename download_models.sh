#!/bin/bash
# Pre-download required models before app starts
# This prevents timeouts during container startup

echo "Pre-downloading spaCy model..."
python -m spacy download en_core_web_sm

echo "All models ready!"
