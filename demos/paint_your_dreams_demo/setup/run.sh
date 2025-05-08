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
# Check for --menu option
if [ "$1" == "--menu" ]; then
    echo ""
    echo "Choose a model to run:"
    echo "[1] OpenVINO/LCM_Dreamshaper_v7-int8-ov"
    echo "[2] OpenVINO/LCM_Dreamshaper_v7-fp16-ov"
    echo "[3] dreamlike-art/dreamlike-anime-1.0"
    echo "[4] OpenVINO/FLUX.1-schnell-int4-ov"
    echo "[5] OpenVINO/FLUX.1-schnell-int8-ov"
    echo "[6] OpenVINO/FLUX.1-schnell-fp16-ov"
    echo -n "Enter model number: "
    read model_choice

    case "$model_choice" in
        1) MODEL="OpenVINO/LCM_Dreamshaper_v7-int8-ov" ;;
        2) MODEL="OpenVINO/LCM_Dreamshaper_v7-fp16-ov" ;;
        3) MODEL="dreamlike-art/dreamlike-anime-1.0" ;;
        4) MODEL="OpenVINO/FLUX.1-schnell-int4-ov" ;;
        5) MODEL="OpenVINO/FLUX.1-schnell-int8-ov" ;;
        6) MODEL="OpenVINO/FLUX.1-schnell-fp16-ov" ;;
        *) echo "Invalid option. Exiting."; exit 1 ;;
    esac

    echo "Running Paint Your Dreams Demo with model: $MODEL"
    python main.py --model_name "$MODEL" --public
else
    echo ""
    echo "Running Paint Your Dreams Demo with default model: OpenVINO/LCM_Dreamshaper_v7-fp16-ov"
    echo "To run with a different model, use: ./run.sh --menu"
    python main.py
fi

# Final message
echo ""
echo "========================================"
echo "Paint Your Dreams Demo execution completed."
echo "========================================"
