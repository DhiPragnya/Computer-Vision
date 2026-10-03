@echo off
REM Double-click this file to start the Computer Vision notes in your browser.
cd /d "%~dp0"

if not exist .venv (
    echo Creating a private Python environment in .venv ...
    python -m venv .venv
    if errorlevel 1 (
        echo.
        echo Python was not found. Install Python 3.10 or newer from https://www.python.org
        echo and tick "Add python.exe to PATH" during installation, then run this file again.
        pause
        exit /b 1
    )
)

call .venv\Scripts\activate.bat
echo Checking packages ...
python -m pip install --upgrade pip --quiet
python -m pip install -r requirements.txt --quiet

echo Starting the app - your browser will open at http://localhost:8501
python -m streamlit run app.py
pause
