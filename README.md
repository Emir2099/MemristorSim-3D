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

## Creating Your GitHub Repo
1. Initialize in project root (not inside `build/`):
   ```sh
   git init
   git add .
   git commit -m "Initial scaffolding"
   git branch -M main
   git remote add origin https://github.com/<your-user>/<your-repo>.git
   git push -u origin main
   ```
2. Keep third‑party sources out of your repo:
   - `build/` is ignored in `.gitignore` so `_deps/*-src` clones from CMake won’t be committed.
   - Do not run `git init` inside `build/`.

## Why VS Code Shows Many Repos
- CMake `FetchContent` pulls dependencies (GLFW, GLAD, ImGui, ImPlot, GLM) into `build/_deps/*-src`.
- Those folders are independent upstream repos and VS Code detects them for convenience; they are not part of your project repository.
- To hide them in VS Code:
  - Settings → search “Git: Auto Repository Detection” → set to `openEditors`.
  - Optionally set “Git: Repository Scan Max Depth” to a small value.

## Optional: Avoid nested repos entirely
Switch `FetchContent_Declare` entries to release archives instead of Git:
```cmake
FetchContent_Declare(glfw
  URL https://github.com/glfw/glfw/archive/refs/tags/3.3.9.zip)
# Repeat similarly for glad, imgui, implot, glm
```
This downloads sources without `.git` metadata, so VS Code won’t list them as repos.

## Notes
- Runtime config like `imgui.ini` is ignored by `.gitignore`.
- If you change build generator or compiler, you may need to delete `build/` and reconfigure.
