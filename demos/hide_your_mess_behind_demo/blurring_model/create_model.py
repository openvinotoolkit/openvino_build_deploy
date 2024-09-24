import tensorflow as tf
import numpy as np
import openvino as ov
from pathlib import Path

def create_gaussian_kernel(size=25, sigma=10):
    kernel = np.fromfunction(
        lambda x, y: (1 / (2 * np.pi * sigma**2)) * np.exp(-((x - (size - 1) / 2)**2 + (y - (size - 1) / 2)**2) / (2 * sigma**2)),
        (size, size)
    )
    kernel = kernel / np.sum(kernel)
    return kernel.astype(np.float32)

def create_rgb_kernel(size=25, sigma=10):
    gaussian_kernel = create_gaussian_kernel(size, sigma)
    gaussian_kernel = gaussian_kernel[:, :, np.newaxis, np.newaxis]
    gaussian_kernel_rgb = np.tile(gaussian_kernel, [1, 1, 4, 1])
    return gaussian_kernel_rgb
    
class PostProcModel(tf.keras.Model):
    def __init__(self):
        super(PostProcModel, self).__init__()
        self.kernel = tf.convert_to_tensor(create_rgb_kernel(15,10))
        self.depthwise_conv = tf.keras.layers.DepthwiseConv2D(
            kernel_size=(15, 15),
            strides=(1, 1),
            padding='same',
            use_bias=False
        )
        self.depthwise_conv.build((None, None, None, 4))
        self.depthwise_conv.set_weights([self.kernel])

    def call(self, input_image, input_mask):
        # blurred = tf.nn.depthwise_conv2d(input_image, self.kernel, strides=[1, 1, 1, 1], padding='SAME')
        blurred = self.depthwise_conv(input_image)
        blurred_masked = blurred * (1 - input_mask)
        original_masked = input_image * input_mask
        return blurred_masked + original_masked

def main():
    model = PostProcModel()

    @tf.function
    def model_fn(input_image, input_mask):
        return model(input_image, input_mask)
    
    concrete_model = model_fn.get_concrete_function(tf.TensorSpec(shape=[1,None,None,4], dtype=tf.float32),tf.TensorSpec(shape=[1,None,None,1], dtype=tf.float32))
    ov_model = ov.convert_model(concrete_model)
    ov.save_model(ov_model, f"{Path(__file__).parent.parent}/models/postproc_model.xml", False)
    print("model saved")

if __name__ == "__main__":
    main()