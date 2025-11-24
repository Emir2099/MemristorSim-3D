# MemristorSim 3D

Desktop simulator for voltage‑controlled memristors with 3D visualization. Built with C++20, CMake, GLFW, OpenGL 4.5, Dear ImGui (docking), ImPlot, and GLM.

## Build

Requirements:
- CMake 3.20+
- Ninja or Make
- C++20 compiler (MSYS2/MinGW, MSVC, or Clang)
- GPU/driver with OpenGL 4.5

Commands:
```sh
cmake -S . -B build -G "Ninja"
cmake --build build -j 8
./build/MemristorSim.exe
```

## Features
- VTEAM memristor physics with RK4 integration and thermal reset
- 3D lab scene with electrodes, oxide, and dynamic filament
- Filament heat glow tied to device power
- ImGui docking UI with Controls, Viewport, and Oscilloscope (ImPlot)

## Project Layout
- `src/physics/` VTEAM model and solver
- `src/render/` framebuffer, geometry, shaders loader
- `src/gui/` ImGui + ImPlot UI
- `shaders/` GLSL programs
- `CMakeLists.txt` Fetches dependencies with FetchContent
