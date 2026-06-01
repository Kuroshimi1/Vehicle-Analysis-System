@off
echo =======================================================
echo   Zapusk Real-Time Vehicle Analysis System...
echo =======================================================

:: Указываем путь к портативному Python
set PYTHON_PATH=%~dp0.python-portable\python.exe

:: Проверяем наличие интерпретатора
if not exist "%PYTHON_PATH%" (
    color 0C
    echo ERR: Portable Python environment not found in .python-portable/
    echo Please make sure you packed the environment correctly.
    pause
    exit /b
)

:: Запуск главного скрипта
"%PYTHON_PATH%" "%~dp0main.py"

if %errorlevel% neq 0 (
    color 0C
    echo.
    echo Application crashed with exit code %errorlevel%
    pause
)