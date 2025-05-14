@echo off
setlocal enabledelayedexpansion

:: Create virtual environment
echo Creating virtual environment...
python -m venv venv

:: Activate virtual environment
call venv\Scripts\activate.bat

:: Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

:: Install dependencies
echo Installing dependencies...
pip install nncf==2.14.1 onnx==1.17.0 optimum-intel==1.22.0 huggingface-hub==0.31.1 openvino==2025.1 ^
openvino-tokenizers==2025.1 openvino-genai==2025.1 pyaudio librosa

:: Final success message
echo.
echo ========================================
echo All requirements installed!
echo ========================================
pause
exit
