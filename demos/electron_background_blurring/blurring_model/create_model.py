import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import openvino as ov
from PIL import Image

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
    
class PostProcModel(tf.keras.Model):
    def __init__(self):
        super(PostProcModel, self).__init__()
        self.kernel = tf.convert_to_tensor(create_rgb_kernel(15,10))

    def call(self, input_image, input_mask):
        blurred = tf.nn.depthwise_conv2d(input_image, self.kernel, strides=[1, 1, 1, 1], padding='SAME')
        blurred_masked = blurred * (1 - input_mask)
        original_masked = input_image * input_mask
        return tf.keras.layers.Add()([blurred_masked, original_masked])

def load_image(image_path, image_size=(128, 128)):
    img = Image.open(image_path).resize(image_size)
    img = np.array(img).astype(np.float32) / 255.0
    return img

def generate_mask(image_size=(128, 128)):
    input_mask = np.zeros(image_size, dtype=np.float32)
    input_mask[32:96, 32:96] = 1.0
    input_mask = np.expand_dims(input_mask, axis=-1)
    return input_mask

def visualize_results(original, mask, result):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 3, 1)
    plt.title("Oryginalny obraz")
    plt.imshow(original)
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("Maska")
    plt.imshow(mask.squeeze(), cmap='gray')  # squeeze() usuwa zbędny wymiar
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title("Wynik modelu")
    plt.imshow(result)
    plt.axis('off')
    
    plt.show()

def main():
    model = PostProcModel()
    input_image = load_image('/home/roszczyk-intel/Pictures/images.jpeg')
    input_mask = generate_mask(image_size=input_image.shape[:2])
    input_image_tensor = tf.convert_to_tensor(input_image[None, ...])
    input_mask_tensor = tf.convert_to_tensor(input_mask[None, ...])
    result_tensor = model(input_image_tensor, input_mask_tensor)

    @tf.function
    def model_fn(input_image, input_mask):
        return model(input_image, input_mask)
    
    concrete_model = model_fn.get_concrete_function(tf.TensorSpec(shape=input_image_tensor.shape, dtype=tf.float32),tf.TensorSpec(shape=input_mask_tensor.shape, dtype=tf.float32))
    ov_model = ov.convert_model(concrete_model)
    ov.save_model(ov_model, './postproc_model.xml', False)
    result_image = result_tensor[0].numpy()
    visualize_results(input_image, input_mask, result_image)

if __name__ == "__main__":
    main()