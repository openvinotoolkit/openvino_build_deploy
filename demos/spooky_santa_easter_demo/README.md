# Multi-Theme Demo with OpenVINO™

This demo detects people in front of a webcam and applies fun thematic overlays:

- **Spooky**: Turns people into skeletons with pumpkin heads.
- **Santa**: Adds Santa hats, beards, and reindeer accessories based on detected emotions.
- **Easter**: Places bunny ears on heads and eggs on torsos.

## Table of Contents

- [Installation](#installation)
- [Setup](#setup)
- [Running the Application](#running-the-application)
- [Customization](#customization)
- [Notes](#notes)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## Installation

### Step 0: Star the Repository

(Optional, but recommended :))

### Step 1: Install Python and Prerequisites

This project requires Python 3.10-3.12 and a few libraries. If you don’t have Python installed on your machine, go to [python.org](https://www.python.org/downloads/) and download the latest version for your operating system. Follow the prompts to install Python, making sure to check the option to add Python to your PATH environment variable.

#### Install Libraries and Tools

For Unix-based systems, run:
```shell
sudo apt install git python3-venv python3-dev
```

**Note:** If you are using Windows, you may need to install Microsoft Visual C++ Redistributable as well.

## Setup

### Step 2: Set Up the Environment

#### Clone the Repository
To clone the repository, run the following command:
```shell
git clone https://github.com/openvinotoolkit/openvino_build_deploy.git
```

Navigate into the directory:
```shell
cd openvino_build_deploy/demos/spooky_demo
```

#### Create a Virtual Environment
To create a virtual environment, run:
```shell
python3 -m venv venv
```

#### Activate the Environment
Activate the virtual environment using:
```shell
source venv/bin/activate   # For Unix-based operating systems
```

**Note:** If you are using Windows, use `venv\Scripts\activate` instead.

#### Install the Packages
To install the required packages, run:
```shell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Running the Application

### Step 3: Run the Application

To run the application with a specific theme, use one of the following commands (default uses webcam stream 0):

#### Spooky Theme (Skeletons with Pumpkin Heads)
```shell
python main.py --stream 0 --theme spooky
```

#### Santa Theme (Santa Hats and Reindeer Accessories)
```shell
python main.py --stream 0 --theme santa
```

#### Easter Theme (Bunny Ears and Eggs)
```shell
python main.py --stream 0 --theme easter
```

## Customization

To customize further (e.g., change device or precision), use:
```shell
python main.py --stream 0 --theme spooky --device AUTO --model_precision FP16-INT8
```

Run the following to see all available options:
```shell
python main.py --help
```

## Notes

- **Assets**: Ensure the `assets/` folder contains `pumpkin.png`, `santa_beard.png`, `santa_cap.png`, `reindeer_nose.png`, `reindeer_antlers.png`, `bunny_ears.png`, and `egg.png`. Missing files will be skipped.
- **Windows Commands**: The examples use `python`, but on Windows, you might need `python` or `py` depending on your setup. Adjust as needed (e.g., `py .\main.py --stream 0 --theme spooky`).
- **Requirements**: The `requirements.txt` should include `openvino`, `opencv-python`, `numpy`, and any other dependencies from the code.

## Troubleshooting

- If you encounter installation issues, ensure you have the correct Python version and necessary dependencies installed.
- Verify that your webcam is properly connected and accessible.
- If assets do not appear, confirm that the image files exist in the `assets/` directory.
- If OpenVINO fails to load models, check compatibility with your system and ensure the correct model precision is used.

## Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

