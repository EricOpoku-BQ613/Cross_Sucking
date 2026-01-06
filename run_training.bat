@echo off
REM Quick training launcher for Windows
REM Usage: run_training.bat [config_name]
REM Example: run_training.bat train_binary_v3_intravideo

setlocal

if "%~1"=="" (
    echo Usage: run_training.bat [config_name]
    echo Example: run_training.bat train_binary_v3_intravideo
    echo.
    echo Available configs:
    echo   - train_binary_v3_intravideo          [RECOMMENDED]
    echo   - train_binary_v3_intravideo_boosted  [30%% tail]
    echo   - train_binary_baseline_old_split     [For comparison]
    exit /b 1
)

set CONFIG_NAME=%~1
set CONFIG_PATH=configs\%CONFIG_NAME%.yaml

if not exist "%CONFIG_PATH%" (
    echo Error: Config not found: %CONFIG_PATH%
    exit /b 1
)

echo ======================================================================
echo STARTING TRAINING
echo ======================================================================
echo Config: %CONFIG_PATH%
echo.

venv\Scripts\python.exe scripts\train_supervised.py --config %CONFIG_PATH%

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ======================================================================
    echo TRAINING COMPLETED SUCCESSFULLY
    echo ======================================================================
) else (
    echo.
    echo ======================================================================
    echo TRAINING FAILED WITH ERROR CODE: %ERRORLEVEL%
    echo ======================================================================
)

endlocal
