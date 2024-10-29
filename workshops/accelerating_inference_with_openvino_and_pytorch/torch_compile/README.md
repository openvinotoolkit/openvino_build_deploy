# Running PyTorch Compile Samples
These samples are designed to run on CPU with torch.compile. Here you will see 3 examples based on Latent Consistent Model, Stable Diffusion, and TorchVision. To use GPU, we recommand using OpenVINO optimized library such as [optimum-intel](https://github.com/huggingface/optimum-intel), and follow instructions and samples in [OpenVINO Notebooks](https://github.com/openvinotoolkit/openvino_notebooks/tree/latest). 

### Installation Instructions
- Create a virtual environment using 
  ```sh  
  python -m venv venv
  ```
- To activate the virtual environment (on Windows) use 
  ```sh
  .\venv\Scripts\activate
  ```
- Install the required dependencies via pip (this may take a little while)
  ```sh
  python -m pip install --upgrade pip
  pip install -r requirements.txt
  ```
- Now you only need a Jupyter server to start exploring the samples.
  ```sh
  jupyter lab .
  ```

Note: Please shutdown the kernel to free up memory between samples. This is especially critical for stable diffusion and LCM demo.
