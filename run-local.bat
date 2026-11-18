@echo off
rem ============================================================================
rem  Local Voice Bot - one-click start (Windows)
rem
rem  First run: creates the Python env, downloads llama.cpp, creates .env and
rem  installs the web UI (each step only if missing). Every run: builds the UI,
rem  starts the server and opens the browser when it is ready.
rem
rem  Set NO_BROWSER=1 to skip opening the browser.
rem ============================================================================
setlocal
cd /d "%~dp0"
title Local Voice Bot

rem ---- settings file ---------------------------------------------------------
if not exist ".env" (
    echo [setup] No .env found - creating it from .env.example
    copy /y ".env.example" ".env" >nul || goto :fail
)

rem Read the two values this script needs from .env (defaults match the app).
set "LLM_BACKEND=llamacpp"
set "WEB_PORT=8000"
for /f "usebackq tokens=1,* delims==" %%A in (`findstr /b /c:"LLM_BACKEND=" /c:"WEB_PORT=" ".env"`) do set "%%A=%%B"

rem ---- Python environment ----------------------------------------------------
if not exist ".venv\Scripts\python.exe" (
    echo [setup] Creating the Python environment - this takes a few minutes the first time
    powershell -NoProfile -ExecutionPolicy Bypass -File "scripts\setup_venv.ps1" || goto :fail
)

rem ---- llama.cpp server (only needed for LLM_BACKEND=llamacpp) ---------------
if /i "%LLM_BACKEND%"=="llamacpp" if not exist "tools\llama\llama-server.exe" (
    echo [setup] Downloading llama.cpp
    powershell -NoProfile -ExecutionPolicy Bypass -File "scripts\fetch_llama_server.ps1" || goto :fail
)

rem ---- web UI -----------------------------------------------------------------
where npm >nul 2>nul || (
    echo [error] npm was not found. Install Node.js 20 or newer from https://nodejs.org and run this again.
    goto :fail
)
if not exist "web\node_modules" (
    echo [setup] Installing web UI packages
    call npm --prefix web install || goto :fail
)
echo [build] Building the web UI
call npm --prefix web run build >nul || (
    echo [error] The web UI build failed. Run "npm --prefix web run build" to see why.
    goto :fail
)

rem ---- open the browser once the server answers ------------------------------
if not "%NO_BROWSER%"=="1" (
    start "" /min powershell -NoProfile -ExecutionPolicy Bypass -Command "$u='http://127.0.0.1:%WEB_PORT%'; for ($i = 0; $i -lt 1800; $i++) { try { Invoke-WebRequest -UseBasicParsing ($u + '/api/config') -TimeoutSec 2 | Out-Null; Start-Process $u; break } catch { Start-Sleep -Seconds 1 } }"
)

rem ---- run --------------------------------------------------------------------
echo.
echo  Starting the voice bot (LLM_BACKEND=%LLM_BACKEND%).
echo  Loading the models can take a minute; the browser opens at
echo  http://127.0.0.1:%WEB_PORT% when it is ready. Press Ctrl+C here to stop.
echo.
cd server
"..\.venv\Scripts\python.exe" -m voicebot.web
if errorlevel 1 goto :fail
endlocal
exit /b 0

:fail
echo.
echo [error] The voice bot did not start. See the messages above.
pause
endlocal
exit /b 1
