# Traffic Monitoring System - Project Documentation

## Overview

This document provides a comprehensive overview of the Traffic Monitoring System project, explaining the purpose and functionality of all files and directories in the project. The system uses computer vision and machine learning to detect traffic violations from video sources.

## Directory Structure

### Root Directory

- **main.py**: Application entry point that initializes the Qt application, shows the splash screen, creates the main window, and starts the event loop.
- **launch.py**: Alternative launcher with command-line argument support for configuring video sources, models, and detection settings.
- **run_app.py**: Production runner script with enhanced error handling and logging for deployment scenarios.
- **enhanced_main_window.py**: Extended version of the main window with additional features for traffic light and violation detection.
- **splash.py**: Creates an animated splash screen shown while the application is loading its components.
- **config.json**: Main configuration file containing settings for video sources, detection models, UI preferences, and violation detection parameters.
- **red_light_violation_pipeline.py**: Implementation of the complete pipeline for detecting red light violations at intersections.
- **requirements.txt**: Lists all Python package dependencies required to run the application.

### UI Directory (`/ui`)

- **main_window.py**: Core UI class that sets up the application window, tabs, toolbars, menus, and connects UI components to controllers.
- **fixed_live_tab.py**: Implements the live video monitoring tab with video display and control panel for real-time processing.
- **analytics_tab.py**: Implements the analytics tab showing statistical charts and metrics about traffic patterns and violations.
- **violations_tab.py**: Shows a list of detected violations with detailed information and evidence frames.
- **export_tab.py**: Provides functionality to export processed videos, report documents, and violation data.
- **config_panel.py**: Implements the settings panel for configuring detection parameters, UI preferences, and camera settings.
- **simple_live_display.py**: Basic video display component for showing frames without advanced overlay features.
- **enhanced_simple_live_display.py**: Enhanced version of the video display with overlay support and better performance.
- **temp_live_display.py**: Temporary implementation of the live display for development and testing purposes.

### Controllers Directory (`/controllers`)

- **video_controller_new.py**: Manages video processing workflow including reading frames, detection, tracking, and annotation in separate threads.
- **video_controller.py**: Original implementation of the video controller (superseded by video_controller_new.py).
- **enhanced_video_controller.py**: Extended version with traffic light detection and violation detection capabilities.
- **analytics_controller.py**: Collects and processes statistical data from video frames and detection results.
- **model_manager.py**: Handles loading, switching, and optimizing object detection models.
- **performance_overlay.py**: Creates performance metric overlays showing FPS, processing times, and memory usage.
- **red_light_violation_detector.py**: Specialized controller for detecting vehicles violating red traffic lights.

### Utils Directory (`/utils`)

- **annotation_utils.py**: Functions for drawing detection boxes, labels, and other overlays on video frames.
- **enhanced_annotation_utils.py**: Advanced visualization utilities with customizable styles and additional overlay types.
- **traffic_light_utils.py**: Specialized functions for traffic light detection, color state analysis, and visualization.
- **helpers.py**: General utility functions for file handling, configuration, and data formatting.
- **crosswalk_utils.py**: Functions for detecting and processing crosswalk areas in traffic scenes.
- **embedder_openvino.py**: Feature extraction utilities using OpenVINO framework for object tracking and recognition.
- ****init**.py**: Initialization file that makes the directory a Python package and exports common utilities.

### Violations Directory (`/violations`)

- ****init**.py**: Package initialization file that exports violation detection functions and classes.
- **red_light_violation.py**: Implements detection logic for red light violations at traffic signals.
- **speeding_violation.py**: Implements detection logic for vehicles exceeding speed limits.
- **wrong_direction_violation.py**: Detects vehicles traveling in the wrong direction on roads.
- **pedestrian_crossing_violation.py**: Detects unsafe interactions between vehicles and pedestrians at crossings.
- **crosswalk_blocking_violation.py**: Detects vehicles blocking pedestrian crosswalks.
- **helmet_seatbelt_violation.py**: Detects motorcyclists without helmets or vehicle occupants without seatbelts.
- **jaywalking_violation.py**: Detects pedestrians crossing roads illegally outside designated crossings.
- **segment_crosswalks.py**: Utility for segmenting and identifying crosswalk regions in images.
- **geometry_utils.py**: Geometric calculation utilities for violation detection (point-in-polygon, distance calculations, etc.).
- **camera_context_loader.py**: Loads camera-specific context information like regions of interest and calibration data.

#### OOP Modules Subdirectory (`/violations/oop_modules`)

- ****init**.py**: Package initialization for the object-oriented implementation of violation detectors.
- **violation_manager.py**: Central class coordinating multiple violation detectors and aggregating results.
- **red_light_violation_oop.py**: Object-oriented implementation of red light violation detection.
- **speeding_violation_oop.py**: Object-oriented implementation of speeding violation detection.
- **wrong_direction_violation_oop.py**: Object-oriented implementation for wrong direction detection.
- **test_oop_system.py**: Test script for verifying the OOP violation detection system.
- **usage_examples.py**: Example code demonstrating how to use the OOP violation detection system.

### Resources Directory (`/resources`)

- Contains UI assets including icons, images, style sheets, and other static resources.
- Organized into subdirectories for icons, logos, and UI themes.
- Includes sample configuration files and templates for report generation.

### Checkpoints Directory (`/Checkpoints`)

- Stores saved model weights and checkpoints for various detection models.
- Contains version history for models to allow rollback if needed.
- Includes configuration files specific to each model checkpoint.

### mobilenetv2_embedder Directory (`/mobilenetv2_embedder`)

- Contains implementation of the MobileNetV2-based feature embedder for object tracking.
- Includes model files (.bin and .xml) optimized for OpenVINO inference.
- Provides utilities for feature extraction from detected objects for re-identification.

## Key System Components

### Video Processing Pipeline

1. **Frame Acquisition**: Reading frames from video files or camera streams.
2. **Object Detection**: Detecting vehicles, pedestrians, traffic lights, and other relevant objects.
3. **Object Tracking**: Tracking detected objects across consecutive frames.
4. **Traffic Light Analysis**: Determining traffic light states (red, yellow, green).
5. **Violation Detection**: Applying rule-based logic to detect traffic violations.
6. **Annotation**: Adding visual indicators for detections, tracks, and violations.
7. **Display/Export**: Showing processed frames to the user or saving to files.

### Violation Types Supported

1. **Red Light Violations**: Vehicles crossing intersection during red light.
2. **Speeding**: Vehicles exceeding speed limits in monitored zones.
3. **Wrong Direction**: Vehicles traveling against designated direction of traffic.
4. **Pedestrian Crossing Violations**: Unsafe interaction between vehicles and pedestrians.
5. **Crosswalk Blocking**: Vehicles stopping on or blocking pedestrian crosswalks.
6. **Helmet/Seatbelt Violations**: Motorcycle riders without helmets or vehicle occupants without seatbelts.
7. **Jaywalking**: Pedestrians crossing outside designated crossing areas.

### User Interface Components

1. **Main Window**: Application shell with menu, toolbar, and status bar.
2. **Live Monitoring Tab**: Real-time video processing and visualization.
3. **Analytics Tab**: Statistical charts and metrics for traffic patterns.
4. **Violations Tab**: List and details of detected violations.
5. **Export Tab**: Tools for exporting data, videos, and reports.
6. **Configuration Panel**: Settings for all aspects of the system.

### Threading Model

1. **Main Thread**: Handles UI events and user interaction.
2. **Video Reader Thread**: Reads frames from source and buffers them.
3. **Processing Thread**: Performs detection, tracking, and violation analysis.
4. **Rendering Thread**: Prepares frames for display with annotations.
5. **Export Thread**: Handles saving outputs without blocking the UI.

## Integration Points

### Adding New Violation Types

1. Create a new violation detector module in the violations directory.
2. Implement the detection logic based on object detections and tracking data.
3. Register the new violation type in the violation manager.
4. Update UI components to display the new violation type.

### Switching Detection Models

1. Place new model files in the appropriate directory.
2. Register the model in the configuration file.
3. Use the model manager to load and switch to the new model.
4. Ensure any class mappings or preprocessing specific to the model are updated.

### Custom UI Extensions

1. Create new UI component classes extending the appropriate Qt classes.
2. Integrate the components into the main window or existing tabs.
3. Connect signals and slots to handle data flow and user interaction.
4. Update styles and resources as needed for consistent appearance.

## Configuration Options

The system can be configured through config.json and command-line parameters (when using launch.py) with options for:

1. **Video Sources**: File paths, camera IDs, streaming URLs.
2. **Detection Models**: Model paths, confidence thresholds, preprocessing options.
3. **UI Preferences**: Theme, layout, displayed metrics.
4. **Violation Detection**: Sensitivity settings, region definitions for each violation type.
5. **Export Settings**: Output formats, paths, and included data in exports.

## Performance Considerations

1. The application uses OpenVINO for optimized neural network inference.
2. Frame skipping and resolution scaling can be adjusted for lower-spec hardware.
3. Processing can be distributed across CPU, GPU, or specialized hardware accelerators.
4. The interface remains responsive due to the threaded processing architecture.
5. Configuration options allow tuning the performance-accuracy trade-off.
