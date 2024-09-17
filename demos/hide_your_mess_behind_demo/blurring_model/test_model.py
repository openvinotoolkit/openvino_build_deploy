import openvino as ov
import matplotlib.pyplot as plt
from PIL import Image
import requests
import numpy as np
from pathlib import Path
from generate_mask import generate_mask

def run_model(image, mask):
    core = ov.Core()
    read_model = core.read_model(f"{Path(__file__).parent.parent}/models/postproc_model.xml")
    compiled_model = core.compile_model(read_model, "AUTO")
    image = np.expand_dims(image, axis=0)
    out = compiled_model((image, mask))[0]
    return out


def view_results(image, mask, blurred):
    plt.figure(figsize=(20,10))

    plt.subplot(1, 3, 1)
    plt.title("Original image")
    plt.imshow(image)
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("Mask")
    plt.imshow(mask.squeeze(), cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title("Model result")
    plt.imshow(blurred)
    plt.axis('off')
    
    plt.show(block=True)


if __name__ == "__main__":
    import_image = np.array(Image.open(requests.get("https://user-images.githubusercontent.com/29454499/251036317-551a2399-303e-4a4a-a7d6-d7ce973e05c5.png", stream=True).raw))
    if import_image.shape[2] == 4:
        import_image = import_image[:, :, :3] # if RGBA, convert to RGB
    mask = generate_mask(import_image)
    result = run_model(import_image, mask)
    view_results(import_image, mask, result[0].astype(np.uint8))