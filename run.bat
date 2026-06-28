@echo off
if exist "build\MemristorSim.exe" (
    echo Launching MemristorSim 3D...
    start "" "build\MemristorSim.exe"
) else (
    echo Error: build\MemristorSim.exe not found. Build the project first using cmake.
    pause
)
