@echo off
REM Mars Object Detection - Windows Environment Setup
REM This script installs all required Python packages for local inference testing

echo ============================================================
echo Mars Object Detection - Windows Setup
echo ============================================================
echo.

echo [1/4] Checking Python installation...
python --version
if errorlevel 1 (
    echo [ERROR] Python not found. Please install Python 3.8+ first.
    pause
    exit /b 1
)
echo [OK] Python found
echo.

echo [2/4] Upgrading pip...
python -m pip install --upgrade pip
echo.

echo [3/4] Installing required packages...
echo     - ultralytics (YOLO framework)
echo     - opencv-python (computer vision)
echo     - Pillow (image processing)
echo     - numpy (numerical computing)
echo.

python -m pip install ultralytics opencv-python Pillow numpy
if errorlevel 1 (
    echo [ERROR] Package installation failed
    pause
    exit /b 1
)
echo.

echo [4/4] Verifying installation...
python -c "import cv2, numpy, ultralytics; print('[OK] All packages installed successfully')"
if errorlevel 1 (
    echo [ERROR] Package verification failed
    pause
    exit /b 1
)
echo.

echo ============================================================
echo [SUCCESS] Setup complete!
echo ============================================================
echo.
echo Next steps:
echo   1. Run: python verify_model.py
echo   2. Run: python live_inference_windows.py
echo.
pause
