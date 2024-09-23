import openvino as ov
from pathlib import Path
import cv2
import numpy as np

def generate_mask(image):
    core = ov.Core()
    read_model = core.read_model(f"{Path(__file__).parent.parent}/models/selfie_multiclass_256x256.xml")
    compiled_model = core.compile_model(read_model, "AUTO")
    if image.shape[2] == 4:
        image = image[:, :, :3] # if RGBA, convert to RGB
    original_image_shape = np.array(image).shape
    image = cv2.resize(image, (256,256))
    image = np.expand_dims(image.astype(np.float32) / 255, 0)
    mask = compiled_model(image)[0][0]
    mask = cv2.resize(mask, (original_image_shape[1], original_image_shape[0]), interpolation=cv2.INTER_NEAREST)
    mask = np.argmax(mask, axis=-1)
    mask = np.where(mask == 0, 0, 1)
    mask = np.expand_dims(mask, axis=-1)
    mask = np.expand_dims(mask, axis=0)
    return mask
