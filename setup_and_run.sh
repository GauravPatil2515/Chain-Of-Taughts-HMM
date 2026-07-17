#!/bin/bash
# ==============================================================================
# COT-HMM: Setup and Execution Script (macOS / MacBook optimized)
# ==============================================================================
# This script automates the installation of dependencies, runs verification
# unit tests, and guides you through running the CoT-HMM research pipeline.
# ==============================================================================

# Colors for terminal output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

clear
echo -e "${BLUE}======================================================================${NC}"
echo -e "${GREEN}                  CoT-HMM Pipeline setup & Runner                    ${NC}"
echo -e "${BLUE}======================================================================${NC}"
echo -e "Optimized for macOS / MacBook systems."
echo ""

# 1. System Check
echo -e "${BLUE}[1/5] Checking System Prerequisites...${NC}"
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: python3 is not installed. Please install Python 3.10+ via Homebrew or python.org.${NC}"
    exit 1
fi
PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo -e "✓ Found Python3 (version ${PYTHON_VERSION})"

if ! command -v git &> /dev/null; then
    echo -e "${YELLOW}Warning: git is not installed. Command Line Tools might be missing.${NC}"
fi

# 2. Virtual Environment Setup
echo -e "\n${BLUE}[2/5] Setting up Virtual Environment...${NC}"
if [ -d "venv" ]; then
    echo "✓ Existing virtual environment 'venv' found."
else
    echo "Creating virtual environment 'venv'..."
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        echo -e "${RED}Error: Failed to create virtual environment.${NC}"
        exit 1
    fi
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate
if [ $? -ne 0 ]; then
    echo -e "${RED}Error: Failed to activate virtual environment.${NC}"
    exit 1
fi

# Upgrade pip
pip install --upgrade pip

# 3. Installing Dependencies
echo -e "\n${BLUE}[3/5] Installing Dependencies...${NC}"
echo "Installing core packages from requirements.txt..."
# Install CPU-optimized PyTorch first to save disk space and prevent CUDA dependency downloads
echo "Installing CPU-optimized PyTorch..."
pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
echo "Installing other dependencies..."
pip install --no-cache-dir -r requirements.txt
if [ $? -ne 0 ]; then
    echo -e "${RED}Error: Package installation failed.${NC}"
    exit 1
fi
echo -e "${GREEN}✓ All dependencies installed successfully!${NC}"

# 4. Running Unit Tests
echo -e "\n${BLUE}[4/5] Running Verification Tests...${NC}"
echo "Running pytest..."
python3 -m pytest -p no:asyncio tests/
TEST_STATUS=$?

if [ $TEST_STATUS -ne 0 ]; then
    echo -e "${RED}❌ Some verification tests failed. Please check the test output above.${NC}"
else
    echo -e "${GREEN}✓ All verification tests passed successfully! Codebase is healthy.${NC}"
fi

# 5. Pipeline Demonstration Instructions
echo -e "\n${BLUE}[5/5] How to Run the CoT-HMM Pipeline${NC}"
echo -e "${YELLOW}======================================================================${NC}"
echo -e "Follow these steps to download the datasets and run experiments:"
echo -e "${YELLOW}======================================================================${NC}"
echo ""
echo -e "Step 1: Download Benchmark Datasets (GSM8K)"
echo -e "  ${BLUE}python3 -m src.data_collection.download_datasets --dataset gsm8k --samples 150${NC}"
echo ""
echo -e "Step 2: Generate CoT Reasoning traces (using Qwen-0.5B-Instruct or Qwen-1.5B-Instruct)"
echo -e "  ${BLUE}python3 src/data_collection/generate_batch.py \\"
echo -e "    --dataset data/raw/gsm8k.json \\"
echo -e "    --dataset-name gsm8k \\"
echo -e "    --domain math \\"
echo -e "    --model qwen-0.5b \\"
echo -e "    --output data/cot_outputs/qwen_0.5b_gsm8k.json \\"
echo -e "    --max-samples 150${NC}"
echo ""
echo -e "Step 3: Run the Complete Orchestration Pipeline (Tagging -> Training -> Evaluation)"
echo -e "  ${BLUE}python3 run_experiments.py \\"
echo -e "    --input data/cot_outputs/qwen_0.5b_gsm8k.json \\"
echo -e "    --output experiments/qwen_0.5b_results \\"
echo -e "    --stage all${NC}"
echo ""
echo -e "Step 4: View the Generated Analysis Report"
echo -e "  ${BLUE}cat experiments/qwen_0.5b_results/results_report.txt${NC}"
echo ""
echo -e "${BLUE}======================================================================${NC}"
echo -e "${GREEN}Setup completed successfully! Virtual environment 'venv' is active.${NC}"
echo -e "To deactivate it, type: ${BLUE}deactivate${NC}"
echo -e "${BLUE}======================================================================${NC}"
