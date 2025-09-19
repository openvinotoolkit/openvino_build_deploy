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
    
    # Install dependencies
    echo "Installing required packages..."
    brew install git python3
    
elif [ "$MACHINE" = "Linux" ]; then
    sudo apt update
    sudo apt install -y git python3-venv python3-dev
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

# Navigate to the Gesture Control Demo directory
cd "$INSTALL_DIR/demos/gesture_control_demo"

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
echo "All requirements installed for PalmPilot Gesture Control."
echo "You can now run the demo using ./run.sh"
echo "========================================"
