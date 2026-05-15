@echo off
setlocal EnableDelayedExpansion

echo.
echo ============================================================
echo  SAM3 x TouchDesigner Setup
echo ============================================================
echo.

set REPO_DIR=%~dp0
set ENV_NAME=sam3

REM ── 1. Find conda ──────────────────────────────────────────────────────────
echo [1/5] Looking for conda...

set CONDA_EXE=

REM Try PATH first
where conda >nul 2>&1
if %ERRORLEVEL% == 0 (
    set CONDA_EXE=conda
    goto :found_conda
)

REM Try common install locations
for %%P in (
    "%USERPROFILE%\anaconda3\Scripts\conda.exe"
    "%USERPROFILE%\miniconda3\Scripts\conda.exe"
    "%USERPROFILE%\Miniconda3\Scripts\conda.exe"
    "C:\ProgramData\anaconda3\Scripts\conda.exe"
    "C:\ProgramData\miniconda3\Scripts\conda.exe"
    "C:\tools\miniconda3\Scripts\conda.exe"
) do (
    if exist %%P (
        set CONDA_EXE=%%~P
        goto :found_conda
    )
)

echo [FAIL] conda not found.
echo.
echo Please install Miniconda or Anaconda first:
echo   https://docs.conda.io/en/latest/miniconda.html
echo.
pause
exit /b 1

:found_conda
echo [OK] Found conda: %CONDA_EXE%


REM ── 2. Create conda environment ────────────────────────────────────────────
echo.
echo [2/5] Creating conda environment "%ENV_NAME%" (Python 3.12)...

"%CONDA_EXE%" env list | findstr /C:"%ENV_NAME%" >nul 2>&1
if %ERRORLEVEL% == 0 (
    echo [OK] Environment already exists, skipping creation.
) else (
    "%CONDA_EXE%" create -n %ENV_NAME% python=3.12 -y
    if %ERRORLEVEL% neq 0 (
        echo [FAIL] Failed to create conda environment.
        pause & exit /b 1
    )
    echo [OK] Environment created.
)

REM Resolve python path from the environment
for /f "delims=" %%i in ('"%CONDA_EXE%" run -n %ENV_NAME% python -c "import sys; print(sys.executable)"') do set PYTHON_EXE=%%i
echo [OK] Python: %PYTHON_EXE%


REM ── 3. Install PyTorch with CUDA ───────────────────────────────────────────
echo.
echo [3/5] Installing PyTorch 2.7 + CUDA 12.6 (this may take a few minutes)...

"%CONDA_EXE%" run -n %ENV_NAME% python -c "import torch; print('PyTorch already installed:', torch.__version__)" >nul 2>&1
if %ERRORLEVEL% == 0 (
    echo [OK] PyTorch already installed, skipping.
) else (
    "%CONDA_EXE%" run -n %ENV_NAME% pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
    if %ERRORLEVEL% neq 0 (
        echo [FAIL] PyTorch installation failed.
        pause & exit /b 1
    )
    echo [OK] PyTorch installed.
)


REM ── 4. Install SAM3 and server dependencies ────────────────────────────────
echo.
echo [4/5] Installing SAM3 and server dependencies...

"%CONDA_EXE%" run -n %ENV_NAME% --no-capture-output pip install -e "%REPO_DIR%vendor\sam3\.[notebooks]"
if %ERRORLEVEL% neq 0 (
    echo [FAIL] SAM3 installation failed.
    pause & exit /b 1
)

"%CONDA_EXE%" run -n %ENV_NAME% --no-capture-output pip install fastapi "uvicorn[standard]" python-multipart
if %ERRORLEVEL% neq 0 (
    echo [FAIL] Server dependencies installation failed.
    pause & exit /b 1
)

echo [OK] All packages installed.


REM ── 5. HuggingFace authentication ─────────────────────────────────────────
echo.
echo [5/5] HuggingFace authentication (required for model weights)
echo.
echo  The SAM3 model weights are gated on HuggingFace.
echo  You need to:
echo    1. Request access at: https://huggingface.co/facebook/sam3
echo    2. Generate a token at: https://huggingface.co/settings/tokens
echo    3. Log in below when prompted.
echo.

"%CONDA_EXE%" run -n %ENV_NAME% --no-capture-output hf auth login
if %ERRORLEVEL% neq 0 (
    echo.
    echo [!] HuggingFace login skipped or failed.
    echo     Run this manually before starting the server:
    echo     conda run -n sam3 hf auth login
)


REM ── Done ───────────────────────────────────────────────────────────────────
echo.
echo ============================================================
echo  Setup complete!
echo.
echo  To start the server:   start_server.bat
echo  To run the test:       test.bat
echo ============================================================
echo.
pause
