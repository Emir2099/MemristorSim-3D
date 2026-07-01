import sys
import os

try:
    import torch
except ImportError:
    print("PyTorch is not installed in the current environment.")
    print("Please install PyTorch using: pip install torch")
    sys.exit(0)

import crossbar_pytorch

print("PyTorch successfully detected!")
print("Initializing Hardware-Aware Training (HAT) simulation using CrossbarLinear layer...")

# Initialize our hardware-aware layer
# Let's enable IR drop (1.5 Ohm) and 8-bit converters to make it a realistic HAT setup
model = crossbar_pytorch.CrossbarLinear(
    enable_ir_drop=True,
    r_wire=1.5,
    enable_dac=True,
    dac_bits=8,
    enable_adc=True,
    adc_bits=8
)

# Set optimizer
optimizer = torch.optim.SGD([model.weight], lr=0.1)
criterion = torch.nn.MSELoss()

# Create dummy training data with signed target outputs
weight_target = torch.randn(8, 8) * 0.5
x_train = torch.rand((16, 8)) # 16 samples of size 8
y_target = torch.matmul(x_train, weight_target) * 0.02

print("\n--- Starting Hardware-Aware Training (HAT) Loop ---")
for epoch in range(1, 21):
    optimizer.zero_grad()
    
    # Forward pass runs through C++ crossbar physical simulation (with G+/G- differential mapping)
    outputs = model(x_train)
    
    # Calculate loss
    loss = criterion(outputs, y_target)
    
    # Backward pass calculates gradients using Straight-Through Estimator (STE)
    loss.backward()
    
    # Optimizer step updates weight parameter (conductance w)
    optimizer.step()
    
    # Clip logical weights to physical boundaries [-1.0, 1.0]
    with torch.no_grad():
        model.weight.clamp_(-1.0, 1.0)
        
    print(f"Epoch {epoch:2d}/20 -> Loss: {loss.item():.8f}")

print("\nHAT loop completed successfully!")
print("Trained signed weight matrix (clamped to [-1.0, 1.0]):")
print(model.weight.detach().cpu().numpy())
