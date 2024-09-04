@echo off
setlocal

:confirm
set /p "yn=Are you sure you want to delete the 'runs' and 'params' directories? [y/n]: "

if /i "%yn%"=="y" (
    echo Deleting directories...
    rmdir /s /q runs
    rmdir /s /q params
    mkdir params
    mkdir runs
    echo Operation completed.
) else if /i "%yn%"=="n" (
    echo Operation cancelled.
) else (
    echo Please answer y or n.
    goto confirm
)

endlocal