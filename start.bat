@echo off
title Federated Emotion Recognition - One-Click Launcher
echo =======================================================================
echo   FEDERATED EMOTION RECOGNITION - DEPLOYMENT LAUNCHER & EXPORTER
echo =======================================================================
echo.

:: 1. Check Python installation
echo [STEP 1] Checking Python installation...
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Python was not found in your system PATH!
    echo Please install Python 3.12 and add it to your environment variables.
    echo.
    pause
    exit /b 1
)

:: Prioritize stable Python 3.12 AppData path if available
set "PYTHON_EXE=python"
if exist "C:\Users\DELL\AppData\Local\Programs\Python\Python312\python.exe" (
    set "PYTHON_EXE=C:\Users\DELL\AppData\Local\Programs\Python\Python312\python.exe"
)
echo Using Python executable: %PYTHON_EXE%
echo.

:: 2. Create Virtual Environment if it does not exist
if not exist "venv" (
    echo [STEP 2] Virtual environment 'venv' not found. Creating it now...
    "%PYTHON_EXE%" -m venv venv
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to create virtual environment!
        pause
        exit /b 1
    )
    echo Virtual environment created successfully.
) else (
    echo [STEP 2] Virtual environment 'venv' found. Skipping creation.
)
echo.

:: 3. Upgrade pip and install/verify dependencies
echo [STEP 3] Activating virtual environment & installing/verifying requirements...
call venv\Scripts\activate.bat
python -m pip install --upgrade pip

echo Installing PyTorch CPU wheel...
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

echo Installing Flower, ONNX, Pandas, Matplotlib, and ONNX Script...
python -m pip install "flwr[simulation]" onnx pandas matplotlib onnxscript
echo.

:: 4. Run the Federated Learning Simulation
echo [STEP 4] Running Federated Learning Simulation (10 Rounds)...
echo Attacker Node 4 (Sign-Flip Attack) will be filtered and blocked by LBAAFedAvg.
python -m backend.simulation
if %errorlevel% neq 0 (
    echo [ERROR] Simulation training loop encountered an error!
    pause
    exit /b 1
)
echo.

:: 5. Export PyTorch weights to ONNX format
echo [STEP 5] Exporting PyTorch global model weights to ONNX...
python -m backend.export_onnx
if %errorlevel% neq 0 (
    echo [ERROR] ONNX model export failed!
    pause
    exit /b 1
)
echo.

:: 6. Compile quantitative research results & Matplotlib charts
echo [STEP 6] Compiling academic plots, CSV tables, and research report...
python -m backend.save_results
echo.

:: 7. Start local web server and launch default browser
echo [STEP 7] Starting local HTTP server on port 8000...
echo The interactive biometric scanner and research dashboard will load.
echo.
echo Launching default web browser pointing to http://localhost:8000/frontend/ ...
start "" "http://localhost:8000/frontend/"

:: Run the local web server
python -m http.server 8000
