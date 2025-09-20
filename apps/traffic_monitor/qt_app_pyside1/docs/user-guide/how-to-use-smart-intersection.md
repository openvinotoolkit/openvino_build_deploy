# How to Use Smart Intersection Features in Desktop App

This guide explains how to use the Smart Intersection analytics features integrated into the Traffic Monitoring Desktop Application.

## Overview

The Smart Intersection features provide advanced scene-based analytics for traffic monitoring, including multi-camera object tracking and region-of-interest analysis.

## Getting Started

### 1. Launch the Application
- Run `python main.py` from the qt_app_pyside1 directory
- The application will start with Intel Arc GPU acceleration enabled
- Wait for the splash screen to complete initialization

### 2. Access Smart Intersection Features

#### Configuration Panel
1. Navigate to the **Config** tab in the main window
2. Select **Smart Intersection** from the configuration categories
3. Configure the following settings:
   - **Camera Settings**: Add multiple camera sources for scene analytics
   - **Tracker Parameters**: Adjust tracking sensitivity and frame rate handling
   - **ROI Definition**: Define regions of interest for analytics
   - **Performance Settings**: Optimize for your Intel Arc GPU

#### Analytics Dashboard
1. Go to the **Analytics** tab
2. View real-time scene analytics including:
   - Multi-camera object tracking
   - Region-based event detection
   - Traffic flow analysis
   - Pedestrian safety monitoring

#### VLM Insights
1. The VLM insights panel provides AI-powered analysis
2. Enable scene-based insights for enhanced understanding
3. View contextual information about detected events

## Key Features

### Multi-Camera Scene Analytics
- **Object Tracking**: Track objects across multiple camera views
- **Scene Fusion**: Combine data from multiple cameras for comprehensive view
- **Real-time Processing**: All processing occurs locally with GPU acceleration

### Region of Interest (ROI) Analytics
- **Crosswalk Monitoring**: Detect pedestrians in crosswalk areas
- **Lane Analysis**: Monitor vehicle dwell time and traffic flow
- **Safety Zones**: Define areas for safety monitoring and alerts

### Performance Optimization
- **Intel Arc GPU**: Optimized for Intel Arc GPU acceleration
- **Local Processing**: No cloud dependency, all processing local
- **Real-time Feedback**: Immediate response to traffic events

## Configuration Options

### Tracker Settings
```json
{
    "max_unreliable_frames": 10,
    "non_measurement_frames_dynamic": 8,
    "non_measurement_frames_static": 16,
    "baseline_frame_rate": 30
}
```

### Camera Configuration
- **Camera ID**: Unique identifier for each camera
- **Position**: Physical position in the intersection
- **Calibration**: Camera calibration parameters
- **ROI Mapping**: Define which regions each camera monitors

## Troubleshooting

### Common Issues
1. **GPU Not Detected**: Ensure Intel Arc GPU drivers are installed
2. **Poor Tracking**: Adjust tracker parameters in configuration
3. **Performance Issues**: Check GPU utilization in Performance tab

### Performance Tips
- Use appropriate frame rates for your hardware
- Configure ROI regions efficiently
- Monitor GPU memory usage
- Adjust tracking parameters based on scene complexity

## Advanced Usage

### Custom ROI Definition
1. Load video or connect cameras
2. Pause on a representative frame
3. Use the ROI drawing tools to define areas
4. Save configuration for reuse

### Integration with Existing Workflows
- Export analytics data for further analysis
- Configure alerts for specific events
- Integrate with external systems via configuration

## Support

For additional help:
- Check the **Help** → **User Guide** menu
- Review system requirements
- Consult troubleshooting documentation
- Check performance metrics in the Performance tab
