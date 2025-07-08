#!/bin/bash

# Enable error handling
set -e

# Detect OS
OS="$(uname -s)"
case "${OS}" in
    Linux*)     MACHINE=Linux;;
    Darwin*)    MACHINE=Mac;;
    *)          MACHINE="UNKNOWN:${OS}"
esac

# Get the current directory where the script is placed
INSTALL_DIR="$(pwd)/openvino_build_deploy"

# Install dependencies based on OS
echo "Installing required packages for ${MACHINE}..."

if [ "$MACHINE" = "Mac" ]; then
    # Check if Homebrew is installed
    if ! command -v brew &> /dev/null; then
        echo "Homebrew not found. Installing Homebrew..."
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    fi
    
    # Install dependencies (Python 3.10-3.13 required)
    echo "Installing required packages..."
    brew install git python@3.12
    
elif [ "$MACHINE" = "Linux" ]; then
    sudo apt update
    sudo apt install -y git python3.12 python3.12-venv python3.12-dev
else
    echo "Unsupported OS: ${MACHINE}"
    exit 1
fi

# Clone the repository if it doesn't exist
if [ ! -d "$INSTALL_DIR" ]; then
    echo "Cloning repository..."
    git clone https://github.com/openvinotoolkit/openvino_build_deploy.git "$INSTALL_DIR"
else
    echo "Repository already exists. Skipping cloning..."
fi

# Navigate to the Paint Your Dreams Demo directory
cd "$INSTALL_DIR/demos/paint_your_dreams_demo"

# Create a virtual environment
echo "Creating virtual environment..."
python3.12 -m venv venv

# Activate the virtual environment
source venv/bin/activate

# Upgrade pip and install dependencies
echo "Upgrading pip..."
python -m pip install --upgrade pip
pip install -r requirements.txt

# Final success message
echo ""
echo "========================================"
echo "All requirements installed for Paint Your Dreams."
echo "You can now run the demo using ./run.sh"
echo "========================================"

