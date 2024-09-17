# Introduction

Instead of using OpenCV for post-processing, OpenVINO can be utilized for optimized inference. Initially, a TensorFlow model is created, which is later converted to OpenVINO format using the provided script  ([create_model.py](create_model.py)). 

Model has two inputs: 
* Original Image to be blurred
* Mask that defines where the blur should be applied

It consists of four operations:
* 2D depthwise convolution
* multiplication
* addition
* subtraction 

![image](https://github.com/user-attachments/assets/46009aae-c681-469f-a322-a2317c3c4253)

# Testing

To verify the model's functionality, the testing script [test_model.py](test_model.py) can be used. First, a mask must be generated with the selfie_multiclass_256x256 model, using the script [generate_mask.py](generate_mask.py). The generated mask is then applied to blur the background of the original image.

![image](https://github.com/user-attachments/assets/ea03ff3d-0753-468f-af27-3f211d44e74b)
