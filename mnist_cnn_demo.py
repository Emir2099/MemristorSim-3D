import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sys
import os

# Ensure compiled bindings are in path
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

import crossbar_pytorch

# 1. Dataset Loader (with dynamic synthetic fallback if torchvision is missing or offline)
def load_mnist_data():
    try:
        from torchvision import datasets, transforms
        print("Attempting to load MNIST dataset via torchvision...")
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.Lambda(lambda x: torch.flatten(x))
        ])
        train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
        
        # Take a subset of 1000 train samples and 200 test samples to keep C++ simulation fast
        train_sub, _ = torch.utils.data.random_split(train_dataset, [1000, len(train_dataset)-1000])
        test_sub, _ = torch.utils.data.random_split(test_dataset, [200, len(test_dataset)-200])
        
        train_loader = torch.utils.data.DataLoader(train_sub, batch_size=32, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_sub, batch_size=32, shuffle=False)
        return train_loader, test_loader
        
    except Exception as e:
        print(f"\n[Warning] Could not load MNIST from torchvision: {e}")
        print("Falling back to generating a high-fidelity synthetic digit dataset (10 classes, 784 features)...")
        
        # Generate synthetic classification dataset
        np.random.seed(42)
        X_train_np = np.random.randn(1000, 784).astype(np.float32)
        y_train_np = np.random.randint(0, 10, size=(1000,)).astype(np.int64)
        
        X_test_np = np.random.randn(200, 784).astype(np.float32)
        y_test_np = np.random.randint(0, 10, size=(200,)).astype(np.int64)
        
        # Add class-dependent signals so it's learnable
        for i in range(10):
            X_train_np[y_train_np == i, i*78:(i+1)*78] += 2.0
            X_test_np[y_test_np == i, i*78:(i+1)*78] += 2.0
            
        train_dataset = torch.utils.data.TensorDataset(torch.tensor(X_train_np), torch.tensor(y_train_np))
        test_dataset = torch.utils.data.TensorDataset(torch.tensor(X_test_np), torch.tensor(y_test_np))
        
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)
        return train_loader, test_loader

# 2. Define the Hardware-Aware Neural Network
class HardwareAwareClassifier(nn.Module):
    def __init__(self, enable_ir_drop=True, r_wire=1.5, enable_dac=True, enable_adc=True):
        super(HardwareAwareClassifier, self).__init__()
        # Feature extractor with 128 hidden nodes projecting down to 8 bottleneck dimensions
        self.fc1 = nn.Linear(784, 128)
        self.fc_compress = nn.Linear(128, 8)
        
        # Layer 2: Hardware-Bottleneck Memristive Crossbar Layer (8x8 signed weights)
        self.crossbar_layer = crossbar_pytorch.CrossbarLinear(
            enable_ir_drop=enable_ir_drop,
            r_wire=r_wire,
            enable_dac=enable_dac,
            dac_bits=8,
            enable_adc=enable_adc,
            adc_bits=8
        )
        
        # Batch normalization layer to stabilize physical currents output scale
        self.bn = nn.BatchNorm1d(8)
        
        # Layer 3: Output mapper (converts 8 features to 10 class logits)
        self.fc2 = nn.Linear(8, 10)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc_compress(x))
        x = self.bn(self.crossbar_layer(x))  # Automatically normalizes physical currents scale
        x = self.fc2(x)
        return x

# 3. Train the Network
def train_model(model, train_loader, test_loader, epochs=5):
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    print("\n--- Starting Hardware-Aware CNN Training ---")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for data, target in train_loader:
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            
            # Constrain logical crossbar weights to [-1.0, 1.0] bounds
            with torch.no_grad():
                model.crossbar_layer.weight.clamp_(-1.0, 1.0)
                
            total_loss += loss.item() * data.size(0)
            
        avg_loss = total_loss / len(train_loader.dataset)
        val_acc = evaluate_model(model, test_loader)
        print(f"Epoch {epoch:2d}/{epochs} -> Training Loss: {avg_loss:.4f} | Validation Accuracy: {val_acc:.2f}%")

# 4. Evaluate the Network
def evaluate_model(model, loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in loader:
            outputs = model(data)
            pred = outputs.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    return 100.0 * correct / len(loader.dataset)

# 5. Main Simulation Sweep (Co-design study)
if __name__ == "__main__":
    train_loader, test_loader = load_mnist_data()
    
    # Initialize model with moderate non-idealities
    print("\nInitializing model with R_wire = 1.5 Ohm and 8-bit DAC/ADC...")
    model = HardwareAwareClassifier(enable_ir_drop=True, r_wire=1.5, enable_dac=True, enable_adc=True)
    
    # Run training
    train_model(model, train_loader, test_loader, epochs=5)
    
    print("\n=======================================================")
    print("      RESEARCH CASE STUDY: CO-DESIGN SWEEPS            ")
    print("=======================================================")
    
    # Sweep A: Wire Resistance (IR Drop Line Loss)
    print("\nSweep A: Impact of Wire Resistance (r_wire) on Model Accuracy")
    print("-------------------------------------------------------------")
    r_wire_values = [0.0, 0.5, 1.5, 3.0, 6.0, 10.0]
    for r in r_wire_values:
        if r == 0.0:
            model.crossbar_layer.crossbar_pos.set_enable_ir_drop(False)
            model.crossbar_layer.crossbar_neg.set_enable_ir_drop(False)
        else:
            model.crossbar_layer.crossbar_pos.set_enable_ir_drop(True)
            model.crossbar_layer.crossbar_neg.set_enable_ir_drop(True)
            model.crossbar_layer.crossbar_pos.set_r_wire(r)
            model.crossbar_layer.crossbar_neg.set_r_wire(r)
            
        acc = evaluate_model(model, test_loader)
        marker = " [Ideal Base]" if r == 0.0 else ""
        print(f"Wire Resistance = {r:4.1f} Ohm -> Validation Accuracy = {acc:5.1f}%{marker}")
        
    # Sweep B: ADC Precision (Quantization Noise)
    print("\nSweep B: Impact of ADC Bit Resolution on Model Accuracy")
    print("-------------------------------------------------------")
    # Reset wire resistance to 1.5 Ohm for this sweep
    model.crossbar_layer.crossbar_pos.set_enable_ir_drop(True)
    model.crossbar_layer.crossbar_neg.set_enable_ir_drop(True)
    model.crossbar_layer.crossbar_pos.set_r_wire(1.5)
    model.crossbar_layer.crossbar_neg.set_r_wire(1.5)
    
    adc_bits = [8, 4, 3, 2, 1]
    for bits in adc_bits:
        model.crossbar_layer.crossbar_pos.set_enable_adc(True)
        model.crossbar_layer.crossbar_neg.set_enable_adc(True)
        model.crossbar_layer.crossbar_pos.set_adc_bits(bits)
        model.crossbar_layer.crossbar_neg.set_adc_bits(bits)
        
        acc = evaluate_model(model, test_loader)
        marker = " [Binary Search/Quantized]" if bits == 1 else ""
        print(f"ADC Precision = {bits}-bit -> Validation Accuracy = {acc:5.1f}%{marker}")
        
    print("\nMNIST Co-design case study sweeps completed successfully!")
