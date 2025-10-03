# Mistral-7B-Instruct-v0.1 (INT8 OpenVINO)

This repository provides **instructions and configuration files** for running **Mistral-7B-Instruct-v0.1** in the **OpenVINO™ IR (Intermediate Representation)** format with weights compressed to **INT8** using **NNCF**.

⚠️ **Note**: This repository does **not** include the model weights.  
You must download the model files separately and place them in the folders below with the exact filenames.

---

## 📌 Model Information

- **Original Model**: [Mistral-7B-Instruct-v0.1](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1)
- **Format**: OpenVINO IR (INT8)
- **Quantization**: `INT8_ASYM` via `nncf.compress_weights`
- **License**: [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0)

---

## ✅ What files this project expects and where to put them

### A. Vision-Language Model (Mistral OpenVINO files)

The VLM code (`app/controllers/vlm_controller.py`) expects an OpenVINO VLM folder at:

Place the files inside `app/mistral_openvino_model/'
