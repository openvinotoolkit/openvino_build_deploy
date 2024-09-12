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
    
class PostProcModel(tf.keras.Model):
    def __init__(self):
        super(PostProcModel, self).__init__()
        # self.kernel = tf.Variable(tf.random.normal([25, 25, 3, 1]), trainable=True)
        self.kernel = tf.convert_to_tensor(create_rgb_kernel(15,10))

    def call(self, input_image, input_mask):
        blurred = tf.nn.depthwise_conv2d(input_image, self.kernel, strides=[1, 1, 1, 1], padding='SAME')
        blurred_masked = blurred * (1 - input_mask)
        original_masked = input_image * input_mask
        return tf.keras.layers.Add()([blurred_masked, original_masked])

# Funkcja pomocnicza do wczytania obrazka z pliku
def load_image(image_path, image_size=(128, 128)):
    # Wczytanie obrazu i zmiana rozmiaru
    img = Image.open(image_path).resize(image_size)
    img = np.array(img).astype(np.float32) / 255.0  # Normalizacja do zakresu [0, 1]
    return img

# Funkcja pomocnicza do generowania maski
def generate_mask(image_size=(128, 128)):
    input_mask = np.zeros(image_size, dtype=np.float32)
    input_mask[32:96, 32:96] = 1.0  # kwadratowa maska w środku obrazu
    input_mask = np.expand_dims(input_mask, axis=-1)  # dodanie wymiaru do dopasowania do obrazu
    return input_mask

# Funkcja do wizualizacji wyników
def visualize_results(original, mask, result):
    plt.figure(figsize=(10, 5))
    
    # Oryginalny obraz
    plt.subplot(1, 3, 1)
    plt.title("Oryginalny obraz")
    plt.imshow(original)
    plt.axis('off')

    # Maska
    plt.subplot(1, 3, 2)
    plt.title("Maska")
    plt.imshow(mask.squeeze(), cmap='gray')  # squeeze() usuwa zbędny wymiar
    plt.axis('off')

    # Wynikowy obraz
    plt.subplot(1, 3, 3)
    plt.title("Wynik modelu")
    plt.imshow(result)
    plt.axis('off')
    
    plt.show()

# Główna funkcja testowa
def main():
    # Inicjalizacja modelu
    model = PostProcModel()
    
    # Wczytanie obrazka z pliku (zastąp 'path_to_image' swoją ścieżką)
    input_image = load_image('/home/roszczyk-intel/Pictures/images.jpeg')  # Podaj ścieżkę do obrazka

    # Generowanie maski
    input_mask = generate_mask(image_size=input_image.shape[:2])

    # Konwersja danych na tensory
    input_image_tensor = tf.convert_to_tensor(input_image[None, ...])  # dodanie batch dimension
    input_mask_tensor = tf.convert_to_tensor(input_mask[None, ...])    # dodanie batch dimension

    # Przeprowadzenie forward pass
    result_tensor = model(input_image_tensor, input_mask_tensor)

    # Usunięcie wymiaru batch i konwersja z powrotem na numpy
    result_image = result_tensor[0].numpy()

    # Wizualizacja wyników
    visualize_results(input_image, input_mask, result_image)

# Uruchomienie testu
if __name__ == "__main__":
    main()