@echo off
REM ============================================================
REM  Edge Thermal Guard — Windows Setup Script
REM ============================================================

echo.
echo  ==========================================
echo    Edge Thermal Guard — Environment Setup
echo  ==========================================
echo.

REM Check Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Please install Python 3.9+ and add it to PATH.
    pause
    exit /b 1
)

echo [1/4] Creating virtual environment (.venv)...
python -m venv .venv
if errorlevel 1 (
    echo [ERROR] Failed to create virtual environment.
    pause
    exit /b 1
)
echo       Done.

echo.
echo [2/4] Activating virtual environment...
call .venv\Scripts\activate.bat
if errorlevel 1 (
    echo [ERROR] Failed to activate virtual environment.
    pause
    exit /b 1
)
echo       Done.

echo.
echo [3/4] Upgrading pip...
python -m pip install --upgrade pip --quiet
echo       Done.

echo.
echo [4/4] Installing dependencies from requirements.txt...
pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] pip install failed. Check requirements.txt and your internet connection.
    pause
    exit /b 1
)
echo       Done.

echo.
echo  ==========================================
echo    Setup complete!
echo  ==========================================
echo.
echo  Next steps:
echo.
echo    1. Train the model (synthetic data, quick test):
echo       python ml\train_and_quantize.py --no-real-data --epochs 5
echo.
echo    2. Run the full demo:
echo       python dashboard\run_demo.py
echo.
echo    3. Open the dashboard:
echo       http://localhost:8501
echo.
echo  Happy hacking!
echo.
pause
