import torch
import numpy as np
import os
import sys

# Ensure compiled bindings are in path
sys.path.append(os.path.abspath("./build"))
sys.path.append(os.path.abspath("./build/Release"))

# MinGW runtime DLL path
mingw_bin = r"C:\Users\capta\Downloads\winlibs-x86_64-posix-seh-gcc-14.2.0-llvm-19.1.1-mingw-w64ucrt-12.0.0-r2\mingw64\bin"
if os.path.exists(mingw_bin) and hasattr(os, 'add_dll_directory'):
    os.add_dll_directory(mingw_bin)

import memristorsim

class CrossbarFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, crossbar):
        """
        Forward VMM pass using the physical C++ Crossbar solver.
        - x: [batch_size, 8] input activations mapped to row voltages
        - weight: [8, 8] synaptic weights mapped to conductances w (0.0 to 1.0)
        - crossbar: The CrossbarArray C++ instance
        """
        ctx.save_for_backward(x, weight)
        
        batch_size = x.shape[0]
        device = x.device
        
        # Output tensor of currents: [batch_size, 8]
        out_currents = torch.zeros((batch_size, 8), device=device)
        
        # Program weights into C++ crossbar array cells
        w_np = weight.detach().cpu().numpy()
        for r in range(8):
            for c in range(8):
                crossbar.program_cell(r, c, float(np.clip(w_np[r, c], 0.0, 1.0)))
        
        # Run C++ solver for each vector in the batch
        x_np = x.detach().cpu().numpy()
        for b in range(batch_size):
            row_voltages = [float(val) for val in x_np[b, :]]
            crossbar.set_inputs(row_voltages)
            crossbar.update(0.001)  # step forward dt=1ms
            
            # Read output currents from bitlines
            out_currents[b, :] = torch.tensor(crossbar.outputs(), device=device)
            
        return out_currents

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass using Straight-Through Estimator (STE) gradient approximation.
        Since hardware non-idealities (noise, drop, quantization) are non-differentiable,
        we use the ideal mathematical VMM gradient.
        """
        x, weight = ctx.saved_tensors
        
        grad_input = None
        grad_weight = None
        
        # Y = X * W  => dL/dX = dL/dY * W_transpose
        if ctx.needs_input_grad[0]:
            grad_input = torch.matmul(grad_output, weight.t())
            
        # Y = X * W  => dL/dW = X_transpose * dL/dY
        if ctx.needs_input_grad[1]:
            grad_weight = torch.matmul(x.t(), grad_output)
            
        return grad_input, grad_weight, None
        
class CrossbarLinear(torch.nn.Module):
    def __init__(self, enable_ir_drop=False, r_wire=1.5, enable_dac=False, dac_bits=8, enable_adc=False, adc_bits=8):
        super(CrossbarLinear, self).__init__()
        # Trainable weights: initialized randomly as conductances [0.0, 1.0]
        self.weight = torch.nn.Parameter(torch.rand(8, 8))
        
        # Initialize C++ crossbar backend
        self.crossbar = memristorsim.CrossbarArray()
        self.crossbar.set_enable_ir_drop(enable_ir_drop)
        self.crossbar.set_r_wire(r_wire)
        self.crossbar.set_enable_dac(enable_dac)
        self.crossbar.set_dac_bits(dac_bits)
        self.crossbar.set_enable_adc(enable_adc)
        self.crossbar.set_adc_bits(adc_bits)
        
    def forward(self, x):
        # Enforce batch layout checks
        if x.shape[-1] != 8:
            raise ValueError("CrossbarLinear layer inputs must have shape [..., 8]")
            
        orig_shape = x.shape
        x_flat = x.view(-1, 8)
        
        # Run autograd function
        y_flat = CrossbarFunction.apply(
            x_flat, self.weight, self.crossbar
        )
        
        # Reshape to original batch dimensions
        new_shape = list(orig_shape[:-1]) + [8]
        return y_flat.view(*new_shape)
