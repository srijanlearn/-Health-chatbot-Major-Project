@echo off
:: HealthyPartner v2 — One-Click Installer (Windows)
::
:: Usage:
::   setup.bat             (fresh install)
::   setup.bat --update    (upgrade existing install)
::   setup.bat --dry-run   (pre-flight checks only)
::
setlocal EnableDelayedExpansion

:: ── Flags ────────────────────────────────────────────────────────────────────────
set "UPDATE_MODE=false"
set "DRY_RUN=false"

for %%a in (%*) do (
    if /i "%%a"=="--update"   set "UPDATE_MODE=true"
    if /i "%%a"=="--dry-run"  set "DRY_RUN=true"
    if /i "%%a"=="--help"     goto :show_help
    if /i "%%a"=="-h"         goto :show_help
)
goto :main

:show_help
echo Usage: setup.bat [--update] [--dry-run]
echo.
echo   (no flags)   Fresh install
echo   --update     Re-install deps and re-pull models without recreating venv
echo   --dry-run    Run pre-flight checks only -- make no changes
exit /b 0

:main
:: ── Resolve project root ──────────────────────────────────────────────────────
set "INSTALLER_DIR=%~dp0"
:: Remove trailing backslash from INSTALLER_DIR
if "%INSTALLER_DIR:~-1%"=="\" set "INSTALLER_DIR=%INSTALLER_DIR:~0,-1%"
for %%i in ("%INSTALLER_DIR%\..") do set "PROJECT_ROOT=%%~fi"
cd /d "%PROJECT_ROOT%"

echo.
echo ============================================
echo   HealthyPartner v2 -- Installer
echo   Privacy-first local healthcare AI
if "%UPDATE_MODE%"=="true" echo   [ UPDATE MODE ]
if "%DRY_RUN%"=="true"     echo   [ DRY RUN ]
echo ============================================
echo.

:: ── Track pre-flight errors ───────────────────────────────���───────────────────
set /a PREFLIGHT_ERRORS=0

echo -- Pre-flight checks --

:: 1. Python 3.10+
echo [INFO] Checking Python...
set "PYTHON_CMD="
set "PY_MAJOR=0"
set "PY_MINOR=0"

for %%p in (python3 python py) do (
    if not defined PYTHON_CMD (
        %%p --version >nul 2>&1 && (
            for /f "tokens=2" %%v in ('%%p --version 2^>^&1') do (
                set "PY_VER_FULL=%%v"
            )
            :: Extract major.minor
            for /f "tokens=1,2 delims=." %%a in ("!PY_VER_FULL!") do (
                set "PY_MAJOR=%%a"
                set "PY_MINOR=%%b"
            )
            if !PY_MAJOR! GEQ 3 if !PY_MINOR! GEQ 10 (
                set "PYTHON_CMD=%%p"
                echo [OK]   Python !PY_VER_FULL! ^(%%p^)
            ) else (
                echo [WARN] %%p is version !PY_VER_FULL! -- need 3.10+
            )
        )
    )
)

if not defined PYTHON_CMD (
    echo [FAIL] Python 3.10+ not found.
    echo        Download from: https://www.python.org/downloads/
    echo        During install, check "Add Python to PATH".
    set /a PREFLIGHT_ERRORS+=1
)

:: 2. Disk space -- need 5 GB free (models + venv)
echo [INFO] Checking disk space...
set "FREE_GB=0"
for /f "tokens=3" %%s in ('dir /-C "%PROJECT_ROOT%" 2^>nul ^| findstr /r "bytes free"') do (
    set "FREE_BYTES=%%s"
)
:: Convert bytes to GB (rough)
if defined FREE_BYTES (
    set "FREE_BYTES_LEN=0"
    for /l %%i in (1,1,20) do if not "!FREE_BYTES:~%%i,1!"=="" set /a FREE_BYTES_LEN=%%i+1
    :: FREE_BYTES_LEN is digit count; GB = bytes / 10^9, rough with string length
    if !FREE_BYTES_LEN! GEQ 10 (
        :: At least 1 GB range -- extract leading digits for rough GB count
        set "FREE_BYTES_TRUNC=!FREE_BYTES:~0,-9!"
        if defined FREE_BYTES_TRUNC set /a FREE_GB=!FREE_BYTES_TRUNC!+0 2>nul || set "FREE_GB=0"
    )
)
if !FREE_GB! GEQ 5 (
    echo [OK]   ~%FREE_GB% GB free
) else (
    echo [FAIL] Less than 5 GB free on disk. Free up space and re-run.
    set /a PREFLIGHT_ERRORS+=1
)

:: 3. Port 8000 availability
echo [INFO] Checking port 8000 ^(backend^)...
netstat -ano | findstr ":8000 " | findstr "LISTENING" >nul 2>&1
if not errorlevel 1 (
    echo [WARN] Port 8000 is already in use. Stop the process before starting HealthyPartner.
) else (
    echo [OK]   Port 8000 is free
)

:: 4. Port 8501 availability
echo [INFO] Checking port 8501 ^(frontend^)...
netstat -ano | findstr ":8501 " | findstr "LISTENING" >nul 2>&1
if not errorlevel 1 (
    echo [WARN] Port 8501 is already in use. The frontend may fail to start.
) else (
    echo [OK]   Port 8501 is free
)

:: 5. Ollama availability
echo [INFO] Checking Ollama...
where ollama >nul 2>&1
if errorlevel 1 (
    echo [WARN] Ollama is not installed -- you will be prompted to install it
) else (
    echo [OK]   Ollama found
)

:: 6. Internet connectivity
echo [INFO] Checking internet connectivity...
set "OFFLINE=false"
curl -sf --max-time 5 https://ollama.com >nul 2>&1
if errorlevel 1 (
    echo [WARN] Cannot reach ollama.com -- running in offline mode.
    echo        Model download will be skipped if models are already cached.
    set "OFFLINE=true"
) else (
    echo [OK]   Internet reachable
)

:: ── Abort on hard failures ──────────────────────────────────────────────────���─
if !PREFLIGHT_ERRORS! GTR 0 (
    echo.
    echo [ERROR] Pre-flight checks failed ^(!PREFLIGHT_ERRORS! issue^(s^)^). Fix them and re-run.
    pause
    exit /b 1
)

if "%DRY_RUN%"=="true" (
    echo.
    echo [OK] Dry run complete -- no changes made.
    pause
    exit /b 0
)

:: ── Step 2: Ollama install prompt ─────────────────────────────────────────────
echo.
echo -- Ollama --
where ollama >nul 2>&1
if errorlevel 1 (
    echo [WARN] Ollama is not installed.
    echo.
    echo   Download and install from: https://ollama.com/download
    echo   Run the .exe installer, check "Add to PATH", then press any key.
    echo.
    pause

    where ollama >nul 2>&1
    if errorlevel 1 (
        echo [ERROR] Ollama still not found. Install it and re-run setup.bat.
        pause
        exit /b 1
    )
)
echo [OK]   Ollama ready

:: ── Step 3: Virtual environment ──────────────────────────────────��────────────
echo.
echo -- Python virtual environment --
if "%UPDATE_MODE%"=="true" (
    echo [INFO] Update mode -- skipping venv recreation
) else (
    if exist ".venv\" (
        echo [INFO] Virtual environment already exists -- skipping
    ) else (
        echo [INFO] Creating virtual environment...
        %PYTHON_CMD% -m venv .venv
        echo [OK]   Virtual environment created
    )
)

call ".venv\Scripts\activate.bat"

:: ── Step 4: Python dependencies ───────────────────────────────��───────────────
echo.
echo -- Python dependencies --
echo [INFO] Installing Python dependencies (this may take 2-5 minutes)...
python -m pip install --upgrade pip --quiet
pip install -r "%PROJECT_ROOT%\requirements.txt" --quiet
echo [OK]   Python dependencies installed

:: ── Step 5: Detect RAM ────────────────────────────────────────────────────────
set "TOTAL_RAM_GB=8"
set "MAIN_MODEL=qwen3:4b"
set "FAST_MODEL=qwen2.5:0.5b"
set "TIER=balanced"

for /f "skip=1 tokens=*" %%m in ('wmic computersystem get TotalPhysicalMemory 2^>nul ^| findstr /r "[0-9]"') do (
    set "RAM_BYTES=%%m"
    :: Strip spaces
    set "RAM_BYTES=!RAM_BYTES: =!"
)
if defined RAM_BYTES (
    if not "!RAM_BYTES:~9,1!"=="" (
        set "RAM_BYTES_TRUNC=!RAM_BYTES:~0,-9!"
        set /a TOTAL_RAM_GB=!RAM_BYTES_TRUNC!+0 2>nul || set "TOTAL_RAM_GB=8"
        if !TOTAL_RAM_GB! LSS 1 set "TOTAL_RAM_GB=8"
    )
)

if !TOTAL_RAM_GB! GEQ 16 (
    set "TIER=quality (16 GB+ RAM)"
) else if !TOTAL_RAM_GB! GEQ 8 (
    set "TIER=balanced (8 GB+ RAM)"
) else (
    set "MAIN_MODEL=qwen2.5:1.5b"
    set "TIER=ultra-light (< 8 GB RAM)"
)

:: ── Step 6: Pull Ollama models ────────────────────────────────────────────────
echo.
echo -- AI models --
echo   Detected ~%TOTAL_RAM_GB% GB RAM -- using %TIER% tier
echo   Main model : %MAIN_MODEL%
echo   Fast model : %FAST_MODEL%
echo.

:: Start Ollama service (idempotent -- /b means background)
start /b ollama serve >nul 2>&1
timeout /t 4 /nobreak >nul

if "%OFFLINE%"=="true" (
    :: Check if models are cached
    ollama list 2>nul | findstr /i "%MAIN_MODEL%" >nul 2>&1
    if not errorlevel 1 (
        echo [OK]   Models already cached -- skipping download ^(offline^)
    ) else (
        echo [WARN] Offline and models not cached.
        echo        Connect to internet and re-run, or pull manually:
        echo          ollama pull %MAIN_MODEL%
        echo          ollama pull %FAST_MODEL%
    )
) else (
    :: Pull only if not already present
    ollama list 2>nul | findstr /i "%MAIN_MODEL%" >nul 2>&1
    if not errorlevel 1 (
        echo [OK]   %MAIN_MODEL% already cached
    ) else (
        echo [INFO] Pulling %MAIN_MODEL% ^(~2-3 GB^)...
        ollama pull %MAIN_MODEL%
        echo [OK]   %MAIN_MODEL% ready
    )

    ollama list 2>nul | findstr /i "%FAST_MODEL%" >nul 2>&1
    if not errorlevel 1 (
        echo [OK]   %FAST_MODEL% already cached
    ) else (
        echo [INFO] Pulling %FAST_MODEL% ^(~400 MB^)...
        ollama pull %FAST_MODEL%
        echo [OK]   %FAST_MODEL% ready
    )
)

:: ── Step 7: Data directories ─────────────────────────────────��────────────────
echo.
echo -- Data directories --
if not exist "data\"               mkdir data
if not exist "db\"                 mkdir db
if not exist "downloaded_files\"   mkdir downloaded_files
if not exist "tenants\default\"    mkdir tenants\default
echo [OK]   Directories ready

:: ── Step 8: .env ─────────────────────────────────────────────────────────────���
echo.
echo -- Configuration --
if exist ".env" (
    echo [INFO] .env already exists -- not overwritten
) else (
    (
        echo # HealthyPartner v2 -- Auto-generated by installer
        echo HP_MODEL_TIER=balanced
        echo HP_MAIN_MODEL=%MAIN_MODEL%
        echo HP_FAST_MODEL=%FAST_MODEL%
    ) > .env
    echo [OK]   .env written
)

:: ── Step 9: start.bat ─────────────────────────────────────────────────────────
echo.
echo -- Launch script --
(
echo @echo off
echo :: HealthyPartner v2 -- Launch Script
echo :: Double-click to start HealthyPartner
echo setlocal
echo cd /d "%%~dp0"
echo.
echo echo.
echo echo Starting HealthyPartner v2...
echo echo.
echo.
echo :: Start Ollama
echo echo [1/3] Starting Ollama...
echo start /b ollama serve
echo timeout /t 3 /nobreak ^>nul
echo.
echo :: Activate venv and start backend
echo call ".venv\Scripts\activate.bat"
echo echo [2/3] Starting backend on http://localhost:8000 ...
echo start /b uvicorn app.main:app --host 0.0.0.0 --port 8000
echo timeout /t 4 /nobreak ^>nul
echo.
echo :: Start frontend
echo echo [3/3] Starting frontend...
echo echo.
echo echo HealthyPartner is running^^!
echo echo Open your browser at: http://localhost:8501
echo echo Close this window to stop.
echo echo.
echo streamlit run frontend.py --server.port 8501 --server.address 0.0.0.0
) > start.bat

echo [OK]   start.bat created

:: ── Done ──────────────────────────────────────────────────���──────────────────
echo.
echo ============================================
echo   Installation Complete^^!
echo ============================================
echo.
if "%UPDATE_MODE%"=="true" (
    echo   HealthyPartner has been updated.
) else (
    echo   To start HealthyPartner:
    echo   Double-click  start.bat
    echo   or run from Command Prompt.
)
echo.
pause
