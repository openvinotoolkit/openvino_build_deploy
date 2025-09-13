# Cross-Platform Build Guide for Traffic Monitor

This guide explains how to build the Traffic Monitor application for Windows and macOS using the new cross-platform build script.

## Prerequisites

### All Platforms
- Python 3.8 or higher
- pip package manager
- All dependencies listed in `requirements.txt`

### Windows
- Windows 10 or higher
- Visual Studio Build Tools (for some dependencies)
- Administrator privileges (for system-wide installation)

### macOS
- macOS 10.14 or higher
- Xcode Command Line Tools: `xcode-select --install`
- Administrator privileges (for system-wide installation)

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Build Script
```bash
python build_crossplatform.py
```

This will:
- Detect your current platform automatically
- Build both debug and release versions
- Create platform-specific installer scripts
- Generate all necessary files in the `dist/` directory

## Build Outputs

### Windows
- `dist/TrafficMonitor.exe` - Release version (no console)
- `dist/TrafficMonitorDebug.exe` - Debug version (with console)
- `install_trafficmonitor_windows.bat` - Windows installer script

### macOS
- `dist/TrafficMonitor.app` - Release version (app bundle)
- `dist/TrafficMonitorDebug.app` - Debug version (with console)
- `install_trafficmonitor_macos.sh` - macOS installer script

## Testing Your Build

### Windows
1. First, test the debug version: `dist\TrafficMonitorDebug.exe`
2. If it works correctly, test the release version: `dist\TrafficMonitor.exe`
3. For system-wide installation, run as administrator: `install_trafficmonitor_windows.bat`

### macOS
1. First, test the debug version: `open dist/TrafficMonitorDebug.app`
2. If it works correctly, test the release version: `open dist/TrafficMonitor.app`
3. For system-wide installation: `./install_trafficmonitor_macos.sh`

## Build Features

### Included Components
- **UI Components**: All Qt-based user interface elements
- **Controllers**: Video processing, analytics, model management
- **Utilities**: Helper functions, annotation utilities, data publishers
- **Models**: OpenVINO models and detection pipelines
- **Resources**: Icons, stylesheets, configuration files
- **Services**: MQTT, InfluxDB, Grafana integration
- **VLM Backend**: Vision-Language Model support
- **Detection Systems**: YOLO, OpenVINO, violation detection

### Hidden Imports
The build script automatically includes all necessary Python modules:
- PySide6 (Qt framework)
- OpenVINO (AI inference)
- OpenCV (computer vision)
- PyTorch (deep learning)
- NumPy, SciPy, Pandas (scientific computing)
- And many more...

## Customization

### Icons
- Windows: Place `icon.ico` in `resources/` directory
- macOS: Place `icon.icns` in `resources/` directory
- The script will automatically detect and use these icons

### Version Information
- Windows: Edit `version_info.txt` to customize version details
- macOS: Version info is embedded in the app bundle

### Build Options
You can modify `build_crossplatform.py` to:
- Change the application name
- Add or remove included files
- Modify PyInstaller options
- Customize installer scripts

## Troubleshooting

### Common Issues

#### "Module not found" errors
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Check that you're using the correct Python environment

#### Large executable size
- The executable includes all dependencies for standalone operation
- Size is typically 200-500MB depending on included models

#### Slow startup
- First launch may be slower due to model loading
- Debug version shows startup progress in console

#### Permission errors on macOS
- Run: `sudo xattr -rd com.apple.quarantine dist/TrafficMonitor.app`
- Or go to System Preferences > Security & Privacy and allow the app

### Build Optimization

#### Reducing Size
To create smaller executables, you can:
1. Remove unused model files before building
2. Use `--exclude-module` for unnecessary dependencies
3. Consider using `--onedir` instead of `--onefile`

#### Debug Mode
The debug versions include console output for troubleshooting:
- Windows: Shows console window with logs
- macOS: Run from terminal to see debug output

## Advanced Usage

### Manual PyInstaller Commands
If you need to customize the build further, you can run PyInstaller manually:

#### Windows
```bash
pyinstaller --name=TrafficMonitor --windowed --onefile --icon=resources/icon.ico --add-data="ui;ui" --add-data="controllers;controllers" main.py
```

#### macOS
```bash
pyinstaller --name=TrafficMonitor --windowed --onefile --icon=resources/icon.icns --add-data="ui:ui" --add-data="controllers:controllers" --target-arch=universal2 main.py
```

### Environment-Specific Builds
For different environments (development, staging, production), you can:
1. Modify `config.json` before building
2. Use different version numbers in `version_info.txt`
3. Include environment-specific resources

## Distribution

### Windows
- Distribute the `.exe` file directly
- Or provide the installer `.bat` script for easy installation
- Consider code signing for production releases

### macOS
- Distribute the `.app` bundle
- Or provide the installer `.sh` script
- Consider notarization for App Store or wider distribution

## Support

If you encounter issues:
1. Check the console output from the debug version
2. Verify all dependencies are correctly installed
3. Ensure you have the required system permissions
4. Check that all model files and resources are present

For development builds, use the debug versions to see detailed startup information and error messages.
