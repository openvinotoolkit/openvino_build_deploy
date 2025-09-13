import pandas as pd
import numpy as np

# Device/quantization/FPS limit combinations for demo data
devices = ["CPU", "GPU", "NPU", "IGPU", "AUTO"]
quants = ["FP32", "FP16", "INT-8 FP-16"]
fps_limits = [10, 25, None]  # None = Max

# Demo data generation ranges (tuned for realistic values)
metrics = {
    "FPS": (6, 32),
    "Latency (ms)": (3, 30),
    "CPU (%)": (1, 20),
    "GPU (%)": (5, 70),
    "NPU (%)": (5, 30),
    "RAM (MB)": (200, 600),
    "NPU Memory (MB)": (180, 220),
    "GPU Shared Memory (GB)": (2, 2.2),
    "Intel GPU (%)": (2, 30),
    "Intel GPU Memory (MB)": (280, 420),
}

rows = []
for device in devices:
    for quant in quants:
        for fps_limit in fps_limits:
            row = {
                "Quant": quant,
                "Device": device,
                "Limit": f"{fps_limit if fps_limit else 'Max'} - FPS",
                "FPS": round(np.random.uniform(*metrics["FPS"]), 1),
                "Latency (ms)": round(np.random.uniform(*metrics["Latency (ms)"]), 1),
                "CPU (%)": round(np.random.uniform(*metrics["CPU (%)"]), 1),
                "RAM (MB)": round(np.random.uniform(*metrics["RAM (MB)"]), 1),
            }
            # Add device-specific columns
            if device == "GPU":
                row["GPU (%)"] = round(np.random.uniform(*metrics["GPU (%)"]), 1)
                row["GPU Shared Memory (GB)"] = round(np.random.uniform(*metrics["GPU Shared Memory (GB)"]), 1)
            if device == "NPU":
                row["NPU (%)"] = round(np.random.uniform(*metrics["NPU (%)"]), 1)
                row["NPU Memory (MB)"] = round(np.random.uniform(*metrics["NPU Memory (MB)"]), 1)
            if device == "IGPU":
                row["Intel GPU (%)"] = round(np.random.uniform(*metrics["Intel GPU (%)"]), 1)
                row["Intel GPU Memory (MB)"] = round(np.random.uniform(*metrics["Intel GPU Memory (MB)"]), 1)
            if device == "AUTO":
                row["Intel GPU (%)"] = round(np.random.uniform(*metrics["Intel GPU (%)"]), 1)
                row["Intel GPU Memory (MB)"] = round(np.random.uniform(*metrics["Intel GPU Memory (MB)"]), 1)
            rows.append(row)

# Create DataFrame and save as CSV
df = pd.DataFrame(rows)
df.to_csv("visionguard_benchmark_demo.csv", index=False)

# Print as markdown tables (one per device)
for device in devices:
    print(f"\n### {device} Performance Table\n")
    print(df[df["Device"] == device].to_markdown(index=False))
