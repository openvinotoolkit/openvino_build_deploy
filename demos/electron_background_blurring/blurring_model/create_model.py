import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import openvino as ov

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
    gaussian_kernel_rgb = np.tile(gaussian_kernel, [1, 1, 3, 1])
    return gaussian_kernel_rgb

kernel = create_rgb_kernel(25,10)

class BlurLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(BlurLayer, self).__init__(**kwargs)
        self.gaussian_kernel_rgb = kernel

    def build(self, input_shape):
        self.kernel = tf.constant(self.gaussian_kernel_rgb, dtype=tf.float32)
        
    def call(self, inputs):
        return tf.nn.depthwise_conv2d(inputs, self.kernel, strides=[1, 1, 1, 1], padding='SAME')

    def compute_output_shape(self, input_shape):
        return input_shape

def create_blur_model():
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(1, 256, 256, 3)),
        BlurLayer()
    ])
    return model

def load_image():
    # Wczytaj obraz
    image = tf.keras.preprocessing.image.load_img(tf.keras.utils.get_file('grace_hopper.jpg',
                                                                          'https://storage.googleapis.com/download.tensorflow.org/example_images/grace_hopper.jpg'))
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    
    # Zmień rozmiar obrazu
    img_array_resized = tf.image.resize(img_array, [256, 256])
    
    # Dodaj wymiar wsadowy i zmień typ na tf.float32
    img_array_resized = tf.expand_dims(img_array_resized, axis=0)
    img_array_resized = tf.cast(img_array_resized, dtype=tf.float32)
    
    return img_array_resized

def show_image(original, blurred):
    original = np.squeeze(original)
    blurred = np.squeeze(blurred)
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title('Oryginalny obraz')
    plt.imshow(original.astype(np.uint8))
    plt.subplot(1, 2, 2)
    plt.title('Rozmyty obraz')
    plt.imshow(blurred.astype(np.uint8))
    plt.show()

image = load_image()
image = tf.expand_dims(image, axis=0)

tf_model = create_blur_model()

blurred_image = tf_model(image)

show_image(image[0].numpy(), blurred_image[0].numpy())

ov_model = ov.convert_model(tf_model)
