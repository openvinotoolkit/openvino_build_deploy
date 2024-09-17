# Introduction

Instead of using OpenCV to postprocess the results, we can use OpenVINO. First, we need to create TensorFlow model, which later has to be converted to OpenVINO ([create_model.py](create_model.py). 

Model has two inputs: 
* Original Image, which is about to be blurred
* Mask, according to which the blur will be added

It consists of three operations:
* deepwise convolution 2D
* multiplixation
* addition and subtraction 

![image](https://github.com/user-attachments/assets/46009aae-c681-469f-a322-a2317c3c4253)

# Testing

You can check how does the model work using [test_model.py](test_model.py). First, we need to generate mask using selfie_multiclass_256x256 model ([generate_mask.py](generate_mask.py). Later we use the newly generated model to blur the background. 

![image](https://github.com/user-attachments/assets/ea03ff3d-0753-468f-af27-3f211d44e74b)
