# PySide6 Traffic Monitoring Dashboard (Advanced)

## Features

- Real-time video detection (OpenVINO, YOLO)
- Drag-and-drop video/image, webcam, RTSP
- Live overlays (bounding boxes, labels, violations)
- Analytics: trends, histograms, summary cards
- Violations: searchable, filterable, snapshot preview
- Export: CSV/JSON, config editor, reload/apply
- Sidebar: device, thresholds, toggles, dark/light mode
- Performance overlay: CPU, RAM, FPS, backend
- Modern UI: QSS, icons, rounded corners, animations

## Structure

```
qt_app_pyside/
├── main.py
├── ui/
│   ├── main_window.py
│   ├── live_tab.py
│   ├── analytics_tab.py
│   ├── violations_tab.py
│   ├── export_tab.py
│   └── config_panel.py
├── controllers/
│   ├── video_controller.py
│   ├── analytics_controller.py
│   └── performance_overlay.py
├── utils/
│   ├── helpers.py
│   └── annotation_utils.py
├── resources/
│   ├── icons/
│   ├── style.qss
│   └── themes/
│       ├── dark.qss
│       └── light.qss
├── config.json
├── requirements.txt
```

## Usage

1. Install requirements: `pip install -r requirements.txt`

2. Run the application (several options):
   - **Recommended**: Use the enhanced controller: `python run_app.py`
   - Standard mode: `python main.py`

## Enhanced Features

The application now includes an enhanced video controller that is automatically activated at startup:

- ✅ **Async Inference Pipeline**: Better frame rate and responsiveness
- ✅ **FP16 Precision**: Optimized for CPU performance
- ✅ **Separate FPS Tracking**: UI and detection metrics are tracked separately
- ✅ **Auto Model Selection**: Uses optimal model based on device (yolo11n for CPU, yolo11x for GPU)
- ✅ **OpenVINO Embedder**: Optimized DeepSORT tracking with OpenVINO backend

## Integration

- Plug in your detection logic from `detection_openvino.py` and `violation_openvino.py` in the controllers.
- Use `config.json` for all parameters.
- Extend UI/controllers for advanced analytics, export, and overlays.

## Troubleshooting

If you encounter import errors:

- Try running with `python run_app.py` which handles import paths automatically
- Ensure you have all required dependencies installed
- Check that the correct model files exist in the openvino_models directory
