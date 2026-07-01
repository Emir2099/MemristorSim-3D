import torch
import numpy as np
import os
import sys

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
        # Trainable weights: initialized in range [-1.0, 1.0] for signed weights representation
        self.weight = torch.nn.Parameter(torch.rand(8, 8) * 2.0 - 1.0)
        
        # Positive and Negative crossbar arrays representing G+ and G- columns
        self.crossbar_pos = memristorsim.CrossbarArray()
        self.crossbar_neg = memristorsim.CrossbarArray()
        
        for cb in [self.crossbar_pos, self.crossbar_neg]:
            cb.set_enable_ir_drop(enable_ir_drop)
            cb.set_r_wire(r_wire)
            cb.set_enable_dac(enable_dac)
            cb.set_dac_bits(dac_bits)
            cb.set_enable_adc(enable_adc)
            cb.set_adc_bits(adc_bits)
        
    def forward(self, x):
        # Enforce batch layout checks
        if x.shape[-1] != 8:
            raise ValueError("CrossbarLinear layer inputs must have shape [..., 8]")
            
        orig_shape = x.shape
        x_flat = x.view(-1, 8)
        
        # Map logical weights to positive and negative conductances (G+ / G- differential pair)
        weight_pos = torch.clamp(self.weight, min=0.0)
        weight_neg = torch.clamp(-self.weight, min=0.0)
        
        # Run forward evaluations on both positive and negative physical arrays
        out_pos = CrossbarFunction.apply(x_flat, weight_pos, self.crossbar_pos)
        out_neg = CrossbarFunction.apply(x_flat, weight_neg, self.crossbar_neg)
        
        # Net output current is the differential: I_net = I+ - I-
        y_flat = out_pos - out_neg
        
        # Reshape to original batch dimensions
        new_shape = list(orig_shape[:-1]) + [8]
        return y_flat.view(*new_shape)
