# Introduction

Instead of using OpenCV to postprocess the results, we can use OpenVINO. First, we need to create TensorFlow model, which later has to be converted to OpenVINO (create_model.py). 

Model has two inputs: 
* Original Image, which is about to be blurred
* Mask, according to which the blur will be added

It consists of three operations:
* deepwise convolution 2D
* multiplixation
* addition and subtraction 

here model structure from netron

# Testing

You can check how does the model work using test_model.py. First, we need to generate mask using selfie_multiclass_256x256 model (generate_mask.py). Later we use the newly generated model to blur the background. 

here results 