@echo off
setlocal EnableDelayedExpansion

set REPO_DIR=%~dp0
set ENV_NAME=sam3

REM ── Find conda ─────────────────────────────────────────────────────────────
set CONDA_EXE=

where conda >nul 2>&1
if %ERRORLEVEL% == 0 ( set CONDA_EXE=conda & goto :start )

for %%P in (
    "%USERPROFILE%\anaconda3\Scripts\conda.exe"
    "%USERPROFILE%\miniconda3\Scripts\conda.exe"
    "%USERPROFILE%\Miniconda3\Scripts\conda.exe"
    "C:\ProgramData\anaconda3\Scripts\conda.exe"
    "C:\ProgramData\miniconda3\Scripts\conda.exe"
    "C:\tools\miniconda3\Scripts\conda.exe"
) do (
    if exist %%P ( set CONDA_EXE=%%~P & goto :start )
)

echo [FAIL] conda not found. Run setup.bat first.
pause & exit /b 1

:start
echo Starting SAM3 server at http://localhost:8765
echo Press Ctrl+C to stop.
echo.

"%CONDA_EXE%" run -n %ENV_NAME% --no-capture-output python -m uvicorn td_server:app --host 0.0.0.0 --port 8765 --app-dir "%REPO_DIR%"
pause
