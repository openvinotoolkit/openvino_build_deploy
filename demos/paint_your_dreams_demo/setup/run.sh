#!/bin/bash

# Enable error handling
set -e

# Get the current directory where the script is placed
DEMO_DIR="$(pwd)/openvino_build_deploy/demos/paint_your_dreams_demo"

# Navigate to the Paint Your Dreams Demo directory
cd "$DEMO_DIR"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "ERROR: Virtual environment not found! Please run ./install.sh first."
    exit 1
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Run the application
echo "Running Paint Your Dreams Demo..."
python main.py

# Final message
echo ""
echo "========================================"
echo "Paint Your Dreams Demo execution completed."
echo "========================================"
