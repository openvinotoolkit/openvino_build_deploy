# Mistral-7B-Instruct-v0.1 (INT8 OpenVINO)

This repository contains **Mistral-7B-Instruct-v0.1** converted to the **OpenVINO™ IR (Intermediate Representation)** format with weights compressed to **INT8** using **NNCF**.

---

## 📌 Model Information

- **Original Model**: [Mistral-7B-Instruct-v0.1](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1)
- **Format**: OpenVINO IR (INT8)
- **Quantization**: `INT8_ASYM` via `nncf.compress_weights`
- **License**: [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0)

---

## ✅ Compatibility

- **OpenVINO**: `2024.2.0` or higher
- **Optimum Intel**: `1.17.0` or higher

---

## 🚀 Getting Started

### 🔹 1. Run with Optimum Intel

Install dependencies:

```bash
pip install optimum[openvino]
```
