#!/bin/bash
set -e

# Unlimited-OCR Setup & Startup Script
# Designed for clean local execution on macOS (with Apple Silicon/MPS) and Linux.

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== Unlimited-OCR Local Build & Launcher ===${NC}"

# 1. Python version check
echo -e "${YELLOW}[1/4] Checking Python environment...${NC}"
PYTHON_CMD=""
for cmd in "/opt/homebrew/bin/python3.10" "python3.10" "python3"; do
    if command -v $cmd &> /dev/null; then
        VER=$($cmd -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
        # Ensure it's python 3.10 or higher
        MAJ=$(echo $VER | cut -d. -f1)
        MIN=$(echo $VER | cut -d. -f2)
        if [ "$MAJ" -eq 3 ] && [ "$MIN" -ge 10 ]; then
            PYTHON_CMD=$cmd
            echo -e "${GREEN}Found compatible Python: $cmd (v$VER)${NC}"
            break
        fi
    fi
done

if [ -z "$PYTHON_CMD" ]; then
    echo -e "${RED}Error: Python 3.10 or higher is required. Please install it using Homebrew: 'brew install python@3.10'${NC}"
    exit 1
fi

# 2. Virtual Environment Setup
echo -e "${YELLOW}[2/4] Setting up Python virtual environment...${NC}"
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment in .venv..."
    $PYTHON_CMD -m venv .venv
else
    echo "Virtual environment already exists in .venv."
fi

# 3. Installing dependencies
echo -e "${YELLOW}[3/4] Installing / upgrading package dependencies...${NC}"
.venv/bin/python -m pip install --upgrade pip --quiet

echo "Installing requirements from requirements.txt..."
.venv/bin/pip install -r requirements.txt --quiet

echo "Installing core web and ML packages..."
.venv/bin/pip install "gradio>=6.0.0" fastapi torch torchvision "transformers<5" requests pytest --quiet

# Downgrade huggingface-hub to avoid import version checks in transformers
echo "Resolving package version checks..."
.venv/bin/pip install "huggingface-hub<1.0" --quiet

# 4. Run application
echo -e "${GREEN}[4/4] Starting Unlimited-OCR Server...${NC}"
echo -e "${BLUE}Opening server on local port 7860...${NC}"
exec .venv/bin/python app.py
