#!/bin/bash

# ============================================
# SARIKA AI - Automated Installation Script
# Tested on: Ubuntu 22.04, Windows WSL2
# Hardware: RTX 5060 Ti 16GB, 200GB SSD
# ============================================

echo "============================================"
echo "    SARIKA AI - Installation Starting      "
echo "============================================"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check Python version
echo -e "${YELLOW}[1/8] Checking Python version...${NC}"
python_version=$(python --version 2>&1 | awk '{print $2}')
required_version="3.11"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo -e "${RED}Error: Python 3.11+ required. You have $python_version${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Python $python_version detected${NC}"

# Check CUDA
echo -e "${YELLOW}[2/8] Checking CUDA...${NC}"
if command -v nvidia-smi &> /dev/null; then
    cuda_version=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}')
    echo -e "${GREEN}✓ CUDA $cuda_version detected${NC}"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo -e "${RED}⚠ NVIDIA GPU not detected. CPU training will be slow.${NC}"
fi

# Check disk space
echo -e "${YELLOW}[3/8] Checking disk space...${NC}"
available_space=$(df -BG . | awk 'NR==2 {print $4}' | sed 's/G//')
if [ "$available_space" -lt 100 ]; then
    echo -e "${RED}⚠ Warning: Only ${available_space}GB available. Recommended: 100GB+${NC}"
else
    echo -e "${GREEN}✓ ${available_space}GB available${NC}"
fi

# Create virtual environment
echo -e "${YELLOW}[4/8] Creating virtual environment...${NC}"
if [ ! -d "venv" ]; then
    python -m venv venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
else
    echo -e "${GREEN}✓ Virtual environment exists${NC}"
fi

# Activate virtual environment
echo -e "${YELLOW}[5/8] Activating virtual environment...${NC}"
source venv/bin/activate

# Upgrade pip
echo -e "${YELLOW}[6/8] Upgrading pip...${NC}"
pip install --upgrade pip setuptools wheel
echo -e "${GREEN}✓ Pip upgraded${NC}"

# Install ML dependencies
echo -e "${YELLOW}[7/8] Installing ML dependencies (this takes 10-15 minutes)...${NC}"
pip install -r ml/requirements.txt
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ ML dependencies installed${NC}"
else
    echo -e "${RED}✗ Failed to install ML dependencies${NC}"
    exit 1
fi

# Create .env from example
echo -e "${YELLOW}[8/8] Setting up configuration...${NC}"
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo -e "${GREEN}✓ Created .env file${NC}"
    echo -e "${YELLOW}⚠ Please edit .env and add your HuggingFace token${NC}"
else
    echo -e "${GREEN}✓ .env file exists${NC}"
fi

# Create .gitkeep files
touch data/raw/.gitkeep
touch data/processed/.gitkeep
touch data/synthetic/.gitkeep
touch models/teachers/.gitkeep
touch models/student/.gitkeep
touch models/final/.gitkeep
touch ml/checkpoints/.gitkeep

echo ""
echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}   Installation Complete! 🎉               ${NC}"
echo -e "${GREEN}============================================${NC}"
echo ""
echo "Next steps:"
echo "  1. Edit .env file with your HuggingFace token"
echo "  2. Run: source venv/bin/activate"
echo "  3. Test: python ml/scripts/verify_setup.py"
echo ""
