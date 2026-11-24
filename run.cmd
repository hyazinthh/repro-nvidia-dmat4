@echo off
rem Configures, builds and runs the repro. Extra arguments are passed to the program.
setlocal
cd /d "%~dp0"
cmake -B build || exit /b 1
cmake --build build --config Release || exit /b 1
if exist build\Release\dmat4_repro.exe (
    build\Release\dmat4_repro.exe %*
) else (
    build\dmat4_repro.exe %*
)
