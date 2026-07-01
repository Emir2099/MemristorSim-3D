import sys
import os

# Append build directories to path
sys.path.append(os.path.abspath("./build"))
sys.path.append(os.path.abspath("./build/Release"))

# Resolve local MinGW compiler paths on Windows if configured
try:
    import local_settings
    if hasattr(local_settings, 'MINGW_BIN') and os.path.exists(local_settings.MINGW_BIN):
        if hasattr(os, 'add_dll_directory'):
            os.add_dll_directory(local_settings.MINGW_BIN)
except ImportError:
    pass

try:
    import memristorsim
    print("SUCCESS: Successfully imported memristorsim python bindings module!")
except ImportError as e:
    print(f"FAILED: Could not import memristorsim. Error: {e}")
    sys.exit(1)

# Initialize parameters
params = memristorsim.MemristorParams()
params.R_on = 100.0
params.R_off = 10000.0
params.v_on = -0.5
params.v_off = 0.5
params.k_on = 50.0
params.k_off = -50.0

# Test single device physics
print("\n--- 1. Testing Single Device Physics ---")
device = memristorsim.PhysicsEngine(params)
print(f"Initial state w: {device.w()}")

# Apply a positive voltage pulse to turn it ON (SET)
print("Applying -1.0V (SET pulse) for 50 steps...")
dt = 0.001
for step in range(50):
    device.update(dt, -1.0)
print(f"Final state w after SET: {device.w():.4f}")
print(f"Device resistance: {device.r():.1f} Ohms")
print(f"Device current: {device.i():.6f} A")

# Test crossbar array
print("\n--- 2. Testing 8x8 Crossbar Array ---")
crossbar = memristorsim.CrossbarArray()

# Program diagonal to ON state
print("Programming diagonal to ON state...")
for i in range(8):
    crossbar.program_cell(i, i, 1.0)

# Apply inputs [1.0V, 0.0V, 1.0V, ...]
inputs = [1.0 if i % 2 == 0 else 0.0 for i in range(8)]
crossbar.set_inputs(inputs)

# Run VMM update without IR drop
crossbar.set_enable_ir_drop(False)
crossbar.update(0.001)
print("Outputs without IR drop:")
print([f"{out:.4f} A" for out in crossbar.outputs()])

# Run VMM update with IR drop (r_wire = 2.0 Ohm)
crossbar.set_enable_ir_drop(True)
crossbar.set_r_wire(2.0)
crossbar.update(0.001)
print("\nOutputs with IR drop (r_wire = 2.0 Ohm):")
print([f"{out:.4f} A" for out in crossbar.outputs()])

# Sweep wire resistance
print("\nSweeping wire resistance and plotting readout current on channel 0:")
for r_w in [0.0, 1.0, 2.0, 5.0, 10.0]:
    if r_w == 0.0:
        crossbar.set_enable_ir_drop(False)
    else:
        crossbar.set_enable_ir_drop(True)
        crossbar.set_r_wire(r_w)
    crossbar.update(0.001)
    print(f"  r_wire = {r_w:4.1f} Ohm -> I_BL[0] = {crossbar.outputs()[0]:.6f} A")

print("\nAll python binding checks completed successfully!")
