@echo off
REM Setup Virtual Environment for Cross-Sucking Project
REM Windows batch script

echo Creating virtual environment...
py -3.11 -m venv venv

echo.
echo Activating virtual environment...
call venv\Scripts\activate.bat

echo.
echo Upgrading pip...
python -m pip install --upgrade pip

echo.
echo Installing project in editable mode...
pip install -e .

echo.
echo Setup complete!
echo.
echo To activate the environment in the future, run:
echo     venv\Scripts\activate.bat
echo.
pause
