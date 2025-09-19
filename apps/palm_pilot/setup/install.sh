#!/bin/bash

# Enable error handling
set -e

# Get the current directory where the script is placed
INSTALL_DIR="$(pwd)/openvino_build_deploy"

# Clone the repository if it doesn't exist
if [ ! -d "$INSTALL_DIR" ]; then
    echo "Cloning repository..."
    git clone https://github.com/openvinotoolkit/openvino_build_deploy.git "$INSTALL_DIR"
else
    echo "Repository already exists. Skipping cloning..."
fi

# Navigate to Gesture Control Demo directory
cd "$INSTALL_DIR/demos/gesture_control_demo"

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
python -m pip install --upgrade pip

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Check for models directory
if [ ! -d "mediapipeModels" ]; then
    echo "Creating models directory..."
    mkdir -p mediapipeModels
    echo ""
    echo "⚠️ NOTE: Place your gesture recognition models in the mediapipeModels folder:"
    echo "   - hand_detector.xml/.bin"
    echo "   - hand_landmarks_detector.xml/.bin" 
    echo "   - gesture_embedder.xml/.bin"
    echo "   - canned_gesture_classifier.xml/.bin"
    echo ""
fi

# Final success message
echo ""
echo "========================================"
echo "PalmPilot - Gesture Control Demo Ready!"
echo "========================================"
echo ""
echo "✅ Virtual environment created"
echo "✅ Dependencies installed" 
echo "✅ Ready to run gesture control"
echo ""
echo "To start the demo:"
echo "  python main.py --stream 0"
echo "  python main.py --stream 0 --mode game_mode"
echo ""
echo "For full GUI:"
echo "  python src/gui_main.py"
echo ""
echo "========================================"
