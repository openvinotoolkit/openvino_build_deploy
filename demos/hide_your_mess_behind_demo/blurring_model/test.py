import openvino as ov
from pathlib import Path
import cv2
import numpy as np

def generate_mask(image_size=(128, 128)):
    input_mask = np.zeros(image_size, dtype=np.float32)
    input_mask[32:96, 32:96] = 1.0
    input_mask = np.expand_dims(input_mask, axis=-1)
    return input_mask

core = ov.Core()
model = core.read_model(f"{Path(__file__).parent.parent}/models/postproc_model.xml")
compiled_model = core.compile_model(model, "AUTO")

image = cv2.imread(f"{Path(__file__).parent.parent}/assets/icons/icon.png")
result = compiled_model(image, generate_mask(image.shape[:2]))
print("akjKWJFN")
cv2.imshow("okienko", result)
cv2.waitKey()