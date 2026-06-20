#!/bin/bash
# DeepSeek Mobile Setup Script
# Run: bash setup.sh

echo "Installing DeepSeek mobile dependencies..."
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install transformers safetensors
echo "Done. Run: python generate.py"

