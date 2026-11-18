@echo off
rem Run the voice bot on this Windows PC.
rem   run-local.bat        production: builds the UI, starts the API, opens http://127.0.0.1:%WEB_PORT%
rem   run-local.bat dev    development: API in a new window plus the Vite dev server (hot reload) on :5173
rem First time only: powershell -ExecutionPolicy Bypass -File scripts\setup_venv.ps1
rem                  powershell -ExecutionPolicy Bypass -File scripts\fetch_llama_server.ps1
setlocal EnableExtensions
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo No .venv found. Run once: powershell -ExecutionPolicy Bypass -File scripts\setup_venv.ps1
  exit /b 1
)
if not exist ".env" (
  copy /y ".env.example" ".env" >nul
  echo Created .env from .env.example. Edit it to change settings.
)

rem Read WEB_PORT and LLM_BACKEND from .env (defaults: 8000, llamacpp).
set "PORT=8000"
set "BACKEND=llamacpp"
for /f "usebackq tokens=1,* delims==" %%A in (`findstr /b /i "WEB_PORT= LLM_BACKEND=" ".env"`) do (
  if /i "%%A"=="WEB_PORT" set "PORT=%%B"
  if /i "%%A"=="LLM_BACKEND" set "BACKEND=%%B"
)

if /i "%BACKEND%"=="llamacpp" if not exist "tools\llama\llama-server.exe" (
  echo llama-server.exe not found. Run once: powershell -ExecutionPolicy Bypass -File scripts\fetch_llama_server.ps1
  exit /b 1
)
if not exist "web\node_modules" (
  echo Installing web dependencies...
  call npm --prefix web install || exit /b 1
)

if /i "%~1"=="dev" goto dev

echo Building the web UI...
call npm --prefix web run build || exit /b 1
echo Starting the API on http://127.0.0.1:%PORT% (first start loads the models; this takes a while).
rem Open the browser once the API answers.
start "" /b powershell -NoProfile -Command "while ($true) { try { Invoke-WebRequest -UseBasicParsing -TimeoutSec 2 http://127.0.0.1:%PORT%/api/config | Out-Null; break } catch { Start-Sleep 2 } }; Start-Process http://127.0.0.1:%PORT%"
cd server
"..\.venv\Scripts\python.exe" -m voicebot.web
exit /b %ERRORLEVEL%

:dev
echo Starting the API in a new window (watch for "Uvicorn running")...
start "voicebot API" /d "%~dp0server" "%~dp0.venv\Scripts\python.exe" -m voicebot.web
echo Open http://localhost:5173 once the API is ready.
call npm --prefix web run dev
exit /b %ERRORLEVEL%
