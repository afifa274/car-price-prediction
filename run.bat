@echo off
echo ============================================
echo   Car Price Prediction - Auto Setup + Run
echo ============================================
echo.

echo [1/3] Installing Python dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: pip install failed. Make sure Python is installed and in PATH.
    pause
    exit /b 1
)

echo.
echo [2/3] Training the model (this may take ~30 seconds)...
python train_model.py
if errorlevel 1 (
    echo ERROR: Model training failed. Check train_model.py and the dataset.
    pause
    exit /b 1
)

echo.
echo [3/3] Launching the Streamlit app...
echo Open your browser at http://localhost:8501
echo Press Ctrl+C in this window to stop the app.
echo.
streamlit run app.py
pause
