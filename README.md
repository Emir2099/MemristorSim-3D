# High-Fidelity 3D Memristor Simulator & Hardware-Aware Training (HAT) Framework

[![Language](https://img.shields.io/badge/Language-C%2B%2B20-blue.svg)](https://en.cppreference.com/w/cpp/20)
[![Library](https://img.shields.io/badge/Python-pybind11%20%7C%20PyTorch-orange.svg)](https://github.com/pybind/pybind11)
[![Graphics](https://img.shields.io/badge/Graphics-OpenGL%204.5-red.svg)](https://www.opengl.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A cross-disciplinary simulation framework bridging nanoscale device physics, circuit-level parasitics, and deep-learning compiler interfaces for **Compute-in-Memory (CIM)** and **Neuromorphic Computing** research. 

This framework provides a desktop GUI application (OpenGL/Dear ImGui) for real-time visualization of conductive filaments and oscilloscope telemetry, a compiled Python scripting module (`pybind11`), and a custom **PyTorch Autograd Operator** for Hardware-Aware Training (HAT) of neural networks under physical non-idealities.

---

## 🛰️ Core Features

1. **High-Fidelity Device Physics**: 
   * Integrates the Voltage Threshold Adaptive Memristor (VTEAM) state equations using highly stable **Runge-Kutta 4 (RK4)** numerical integration.
   * Models temperature-driven filament dissolution and thermal reset under critical local power thresholds.
   * Supports Sinh, **Poole-Frenkel Emission**, and **Schottky Barrier Tunneling** transport modes.
2. **True Nanoscale Noise & Stochasticity**:
   * **Spatial (Device-to-Device, D2D)**: Normal and log-normal distributions of initial states ($w_{init}$) and switching rates ($k_{on}, k_{off}$).
   * **Temporal (Cycle-to-Cycle, C2C)**: Write-noise modeled as a Stochastic Differential Equation (SDE) using Euler-Maruyama integration.
   * **Random Telegraph Noise (RTN)**: Trapping/detrapping events modeled as a two-state Markov chain producing discrete current jumps.
3. **CIM Crossbar Parasitics & Nodal Drop Solver**:
   * Simulates metal wire segment resistance ($r_{wire}$) using an iterative **Modified Nodal Analysis (MNA)** solver via Gauss-Seidel relaxation.
   * Solves sneak-path currents through unselected cells by implementing volatile threshold switches (**1S1R**) or transistor gates (**1T1R**) in series.
   * Models finite-precision data converter noise using uniform **1-to-8 bit DAC and ADC** quantization models.
4. **Research Software Bridge**:
   * Native C++ bindings compiled as a `.pyd` module for Python scripting sweeps.
   * Custom PyTorch layer wrapper utilizing **Straight-Through Estimator (STE)** backpropagation for training CNNs under physical array constraints.
5. **Nelder-Mead Parameter Extraction**:
   * Built-in simplex optimizer to automatically fit physical VTEAM parameters to experimental CSV current-voltage (I-V) curves.

---

## 📊 Mathematical Formulations

### 1. Conduction & Transport Models

The cell current is evaluated dynamically based on the selected conduction model:

* **Sinh Conduction**:

$$ I(V_m) = w \cdot a_1 \sinh(b_1 V_m) + (1-w) \cdot a_2 \sinh(b_2 V_m) $$

* **Poole-Frenkel Emission**:

$$ I_{PF}(V_m) = c_1 \cdot V_m \cdot \exp\left(d_1 \sqrt{|V_m|} - e_1\right) $$

* **Schottky Barrier Tunneling**:

$$ I_{Schottky}(V_m) = c_2 \cdot \exp\left(d_2 \sqrt{|V_m|} - e_2\right) \cdot \operatorname{sgn}(V_m) $$

### 2. Series Selector Solver (1S1R / 1T1R)

For a cell under total voltage bias $V_{total}$, the loop solves for the intermediate voltage $V_m$ across the memristor where:

$$ V_{total} = V_s + V_m \quad \text{and} \quad I_{cell} = I_{sel}(V_s) = I_{mem}(V_m) $$

We use a fast numerical binary bisection search to resolve $V_m$ at each time step.

* **1S1R (Volatile Threshold Switch)**:

$$ I_{sel}(V_s) = V_s \cdot \left(G_{off} + \frac{G_{on} - G_{off}}{1 + \exp\left(-(|V_s| - V_{th})/0.05\right)}\right) $$

### 3. Modified Nodal Analysis (MNA)

At each junction $(i,j)$ in the $8\times8$ crossbar, KCL row node voltage $V^r_{i,j}$ and column node voltage $V^c_{i,j}$ are updated iteratively using Gauss-Seidel relaxation to solve the wire resistance drops:

$$ V^r_{i,j} = \frac{V^r_{i,j-1} + V^r_{i,j+1} - r_{wire} \cdot I_{cell,i,j}}{2} $$

$$ V^c_{i,j} = \frac{V^c_{i-1,j} + V^c_{i+1,j} + r_{wire} \cdot I_{cell,i,j}}{2} $$

---

## 🛠️ Building & Compilation

### Prerequisites
* **C++ Compiler**: GCC 14+ or MSVC (supporting C++20)
* **CMake**: Version 3.20 or newer
* **Python**: Python 3.10 (with header libraries)

### Build Steps (Windows / Linux)
Open a terminal in the project root:
```bash
# Configure the build system
cmake -B build -DCMAKE_BUILD_TYPE=Release

# Compile both the C++ 3D Desktop Application and the Python bindings module
cmake --build build --config Release
```
This builds:
1. `build/MemristorSim.exe` (or `MemristorSim` on Linux): Desktop application.
2. `build/memristorsim.cp310-win_amd64.pyd` (or `.so`): Compiled Python module.

---

## 🐍 Python & PyTorch Usage

### 1. Path Configuration
To run scripts in any folder on your PC, configure a `local_settings.py` file in the project root to map your compiler binary DLL path (needed on Windows for importing C++ compiled modules):
```python
# local_settings.py
MINGW_BIN = r"C:\path\to\your\mingw64\bin"
```

### 2. High-Throughput Parameter Sweeps
Researchers can write Python sweeps to characterize device behavior (e.g. studying programming write-verify pulse overheads under C2C write noise):
```python
import sys
import os
sys.path.append("./build")

# Resolve DLL dependencies automatically
try:
    import local_settings
    os.add_dll_directory(local_settings.MINGW_BIN)
except ImportError:
    pass

import memristorsim

params = memristorsim.MemristorParams()
params.enable_variability = True
params.sigma_c2c = 0.05  # 5% cycle-to-cycle write noise

device = memristorsim.PhysicsEngine(params)
device.set_w(0.0)  # Reset state

# Run closed-loop feedback write-verify
pulses, energy = device.program_write_verify(w_target=0.70, tolerance=0.01)
print(f"Target reached in {pulses} pulses. Total write energy: {energy*1e6:.1f} uJ")
```

### 3. Hardware-Aware Training (HAT) in PyTorch
Use the custom PyTorch layer to inject crossbar line losses and ADC quantization directly into the forward pass of your neural networks. Gradients backpropagate using the Straight-Through Estimator (STE) approximation:

```python
import torch
import crossbar_pytorch

# Create a hardware-aware linear layer (8 inputs, 8 outputs)
model = crossbar_pytorch.CrossbarLinear(
    enable_ir_drop=True, 
    r_wire=1.5,          # 1.5 Ohm wire segment parasitics
    enable_dac=True,     # 8-bit Input digital-to-analog converter
    enable_adc=True      # 8-bit Output analog-to-digital converter
)

# Scale output current to normalize gradient ranges in backpropagation
bn = torch.nn.BatchNorm1d(8)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

# Dummy training loop
for epoch in range(10):
    optimizer.zero_grad()
    inputs = torch.rand((32, 8))
    targets = torch.randint(0, 10, (32,))
    
    # Forward pass executes physical solver; backward pass backpropagates using STE
    outputs = bn(model(inputs))
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()
    
    # Clip logical weights to physical range bounds
    with torch.no_grad():
        model.weight.clamp_(-1.0, 1.0)
```

---

## 🎨 Interactive GUI Visualization

Run the desktop executable (`.\build\MemristorSim.exe`) to launch the interactive workspace:
* **Bipolar Device Tab**: Run AC/DC waveforms and watch the 3D filament expand (LOW resistance) or dissolve (HIGH resistance). View real-time hysteresis loops in the ImPlot oscilloscope.
* **8x8 CIM Crossbar Tab**:
  * Hover over any synaptic junction to read row and column node voltages, net voltage drops, power dissipation, and temperature.
  * Toggle **Show Sneak-Path Leakage Currents** to activate a logarithmic neon-green heatmap highlighting parasitic leakage flows. Turn on the **Series Selector** to observe leakages dissolve immediately.
  * Run a **Sobel Filter edge detection** convolution on contrast step matrices to verify the analog vector-matrix computing accuracy.
* **Auto-Fitting Tab**: Upload experimental CSV data sheets (Voltage, Current columns) and run the simplex fitting optimizer to extract physical device parameters in real time.
