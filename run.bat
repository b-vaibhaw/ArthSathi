@echo off
title ArthaSathi AI Web Dashboard
cls
echo ====================================================================
echo                   ArthaSathi AI - Money Coach
echo ====================================================================
echo.

:: 1. Check for Python
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in system PATH.
    echo Please install Python 3.11 or 3.12 and try again.
    pause
    exit /b 1
)

:: 2. Check for Uvicorn
python -c "import uvicorn" >nul 2>nul
if %errorlevel% neq 0 (
    echo [INFO] Installing required dependencies...
    pip install fastapi uvicorn pydantic httpx python-dotenv tokenizers sentence-transformers jinja2
)

:: 3. Check for Tokenizer
if not exist "arthasathi_tokenizer\tokenizer.json" (
    echo [INFO] Tokenizer not found. Training fast BPE tokenizer...
    python scratch\train_tokenizer_fast.py
)

:: 4. Check for Model Checkpoint
if not exist "checkpoints\finetune\final_ft.pt" (
    echo [INFO] Model checkpoint not found. Generating initial checkpoint...
    python scratch\train_mock_checkpoint.py
)

echo.
echo [INFO] Starting FastAPI Web Server...
echo [INFO] Exposing at http://127.0.0.1:8000/
echo [INFO] Press Ctrl+C in this terminal window to stop the server.
echo.

:: Open default browser
start http://127.0.0.1:8000/

:: Run FastAPI
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000

pause
