@echo off
setlocal

REM 检查 python3 是否存在
where python3 >nul 2>nul
if %errorlevel%==0 (
    set PYTHON=python3
) else (
    set PYTHON=python
)

REM 检查 pip3 是否存在
where pip3 >nul 2>nul
if %errorlevel%==0 (
    set PIP=pip3
) else (
    set PIP=pip
)

%PIP% install -r requirements.txt
if %errorlevel% neq 0 (
    echo "Failed to install dependencies."
    exit /b %errorlevel%
)

%PYTHON% setup.py build_ext --inplace
if %errorlevel% neq 0 (
    echo "Failed to compile"
    exit /b %errorlevel%
)

echo "Done."
pause
endlocal
