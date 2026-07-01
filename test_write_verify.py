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

import memristorsim

# 1. Initialize parameters with write noise (C2C)
params = memristorsim.MemristorParams()
params.enable_variability = True
params.sigma_w_init = 0.0
params.sigma_k_on = 0.0

# 2. Test closed-loop write-verify on a single device
print("--- 1. Single Cell Program-and-Verify under Stochastic Write Noise ---")
params.sigma_c2c = 0.02 # C2C noise level (standard deviation)

device = memristorsim.PhysicsEngine(params)
device.set_w(0.0) # Start from fully RESET OFF state
target_w = 0.70

# Run closed-loop write-verify
pulses, energy = device.program_write_verify(target_w, tolerance=0.01, max_pulses=50)

print(f"Target w: {target_w:.2f}")
print(f"Final w:  {device.w():.4f}")
print(f"Programming cost: {pulses} write pulses applied")
print(f"Total write energy consumed: {energy*1e6:.3f} uJ")

# 3. Sweep C2C write noise level to show scaling of programming cost
print("\n--- 2. Sweeping C2C Write Noise vs. Programming Overhead ---")
for noise in [0.02, 0.10, 0.25, 0.50]:
    params.sigma_c2c = noise
    
    # Run 50 trials per noise level to compute average cost
    total_pulses = 0
    total_energy = 0.0
    for trial in range(50):
        device = memristorsim.PhysicsEngine(params)
        device.set_w(0.0)
        pulses, energy = device.program_write_verify(target_w, tolerance=0.01, max_pulses=100)
        total_pulses += pulses
        total_energy += energy
        
    avg_pulses = total_pulses / 50.0
    avg_energy = total_energy / 50.0
    print(f"C2C Noise sigma_c2c = {noise:.3f} -> Avg Pulses = {avg_pulses:5.1f} | Avg Energy = {avg_energy*1e6:7.3f} uJ")

print("\nWrite-verify sweep completed successfully!")
