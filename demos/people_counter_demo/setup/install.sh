#!/bin/bash

# Enable error handling
set -e

# Get the current directory where the script is placed
INSTALL_DIR="$(pwd)/openvino_build_deploy"

# Install dependencies
echo "Installing required packages..."
sudo apt update
sudo apt install -y git python3-venv python3-dev

# Clone the repository if it doesn't exist
if [ ! -d "$INSTALL_DIR" ]; then
    echo "Cloning repository..."
    git clone https://github.com/openvinotoolkit/openvino_build_deploy.git "$INSTALL_DIR"
else
    echo "Repository already exists. Skipping cloning..."
fi

# Navigate to the People Counter Demo directory
cd "$INSTALL_DIR/demos/people_counter_demo"

# Create a virtual environment
echo "Creating virtual environment..."
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate

# Upgrade pip and install dependencies
echo "Upgrading pip..."
python -m pip install --upgrade pip
pip install -r requirements.txt

# Final success message
echo ""
echo "========================================"
echo "All requirements installed for People Counter."
echo "You can now run the demo using ./run.sh"
echo "========================================"

