#!/bin/bash

# Setup script for Nairobi Information Collector
# This script automates the initial setup process

set -e  # Exit on error

echo "=================================="
echo "Nairobi Information Collector"
echo "Setup Script"
echo "=================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check Python version
echo -n "Checking Python version... "
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
    REQUIRED_VERSION="3.9"

    if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" = "$REQUIRED_VERSION" ]; then
        echo -e "${GREEN}✓ Python $PYTHON_VERSION${NC}"
    else
        echo -e "${RED}✗ Python 3.9+ required (found $PYTHON_VERSION)${NC}"
        exit 1
    fi
else
    echo -e "${RED}✗ Python 3 not found${NC}"
    exit 1
fi

# Create logs directory
echo -n "Creating logs directory... "
mkdir -p logs
echo -e "${GREEN}✓${NC}"

# Create virtual environment
if [ ! -d "venv" ]; then
    echo -n "Creating virtual environment... "
    python3 -m venv venv
    echo -e "${GREEN}✓${NC}"
else
    echo -e "${YELLOW}Virtual environment already exists${NC}"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo -n "Upgrading pip... "
pip install --upgrade pip > /dev/null 2>&1
echo -e "${GREEN}✓${NC}"

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Download spaCy model
echo -n "Downloading NLP model... "
python -m spacy download en_core_web_sm > /dev/null 2>&1
echo -e "${GREEN}✓${NC}"

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo -n "Creating .env file... "
    cp .env.example .env
    echo -e "${GREEN}✓${NC}"
    echo -e "${YELLOW}⚠ Please edit .env file with your API keys${NC}"
else
    echo -e "${YELLOW}.env file already exists${NC}"
fi

# Initialize database
echo -n "Initializing database... "
python cli.py init-db > /dev/null 2>&1
echo -e "${GREEN}✓${NC}"

# Make CLI executable
chmod +x cli.py

echo ""
echo "=================================="
echo -e "${GREEN}Setup completed successfully!${NC}"
echo "=================================="
echo ""
echo "Next steps:"
echo "1. Edit .env file with your API keys:"
echo "   nano .env"
echo ""
echo "2. Activate virtual environment:"
echo "   source venv/bin/activate"
echo ""
echo "3. Start the application:"
echo "   python -m app.main"
echo ""
echo "4. Or run a manual collection:"
echo "   python cli.py collect all"
echo ""
echo "5. Access the API:"
echo "   http://localhost:8000/docs"
echo ""
echo "For more information, see QUICKSTART.md"
echo ""
