@echo off
REM MoodSense Dashboard Launcher Script (Windows)

echo ==================================
echo    MoodSense Dashboard Launcher
echo ==================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo X Python is not installed. Please install Python 3.8 or higher.
    pause
    exit /b 1
)

echo + Python found
echo.

REM Check if pip is installed
pip --version >nul 2>&1
if %errorlevel% neq 0 (
    echo X pip is not installed. Please install pip.
    pause
    exit /b 1
)

echo + pip found
echo.

REM Check if streamlit is installed
python -c "import streamlit" >nul 2>&1
if %errorlevel% neq 0 (
    echo ! Streamlit not found. Installing dependencies...
    echo.
    pip install -r requirements.txt
    echo.
)

echo + All dependencies installed
echo.

REM Check for data files
echo Checking for data files...
if exist "Spotify_Tracks_Dataset.csv" (
    echo    + Found: Spotify_Tracks_Dataset.csv
) else (
    echo    ! Not found: Spotify_Tracks_Dataset.csv ^(will use sample data^)
)

if exist "spotify_millsongdata.csv" (
    echo    + Found: spotify_millsongdata.csv
) else (
    echo    ! Not found: spotify_millsongdata.csv ^(will use sample data^)
)

if exist "best_model.pkl" (
    echo    + Found: best_model.pkl
) else (
    echo    ! Not found: best_model.pkl ^(will use rule-based classification^)
)

echo.
echo Launching MoodSense Dashboard...
echo.
echo    The dashboard will open in your default browser.
echo    If it doesn't open automatically, navigate to:
echo    --^> http://localhost:8501
echo.
echo    Press Ctrl+C to stop the dashboard.
echo.
echo ==================================
echo.

REM Run streamlit
streamlit run streamlit_app.py
