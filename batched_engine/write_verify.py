"""
Batched current-margin write-verify — port of error_dist_validation.common.write_verify_current.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

if TYPE_CHECKING:
    from .device import BatchedMemristorArray, WriteVerifyResult


def write_verify_batched(
    engine: "BatchedMemristorArray",
    I_target: torch.Tensor,
    *,
    max_pulses: Optional[int] = None,
    noisy_read: bool = True,
    margin: Optional[float] = None,
    V_SET: Optional[float] = None,
    V_RESET: Optional[float] = None,
    pulse_width: Optional[float] = None,
) -> "WriteVerifyResult":
    """
    Closed-loop write-verify targeting read current (Yao / Component #4).

    Correctness constraints (real bug classes elsewhere in this project):
    - Converged cells receive no further pulses (active masking on V_applied).
    - Idle cells accrue no C2C noise.
    - Margins / voltages / pulse width come from calibrated_params, not invented.
    """
    from .device import WriteVerifyResult

    if I_target.shape != (engine.E, engine.N):
        raise ValueError(
            f"I_target shape {tuple(I_target.shape)} != ({engine.E}, {engine.N})"
        )

    I_target = I_target.to(device=engine.device, dtype=engine.dtype)
    max_pulses = engine.max_pulses_default if max_pulses is None else int(max_pulses)
    margin = engine.margin if margin is None else float(margin)
    V_SET = engine.V_SET if V_SET is None else float(V_SET)
    V_RESET = engine.V_RESET if V_RESET is None else float(V_RESET)
    pulse_width = engine.pulse_width if pulse_width is None else float(pulse_width)

    active = torch.ones(engine.E, engine.N, dtype=torch.bool, device=engine.device)
    set_pulses = torch.zeros(engine.E, engine.N, dtype=engine.dtype, device=engine.device)
    reset_pulses = torch.zeros(
        engine.E, engine.N, dtype=engine.dtype, device=engine.device
    )
    energy = torch.zeros(engine.E, engine.N, dtype=engine.dtype, device=engine.device)

    for _ in range(max_pulses):
        if not bool(active.any()):
            break

        I_read = engine.read(engine.V_READ, noisy=noisy_read)
        error = I_read - I_target
        converged = error.abs() <= margin
        active = active & ~converged

        if not bool(active.any()):
            break

        need_set = active & (error < -margin)
        need_reset = active & (error > margin)

        V_applied = torch.zeros(
            engine.E, engine.N, dtype=engine.dtype, device=engine.device
        )
        V_applied = torch.where(
            need_set, torch.as_tensor(V_SET, dtype=engine.dtype, device=engine.device), V_applied
        )
        V_applied = torch.where(
            need_reset,
            torch.as_tensor(V_RESET, dtype=engine.dtype, device=engine.device),
            V_applied,
        )

        # Pulse only cells that still need SET/RESET this iteration.
        pulse_mask = need_set | need_reset
        x_before = engine.x.clone()
        engine.pulse(V_applied, dt=pulse_width, active_mask=pulse_mask, apply_c2c=True)

        set_pulses = set_pulses + need_set.to(engine.dtype)
        reset_pulses = reset_pulses + need_reset.to(engine.dtype)

        # Midpoint-current energy (noiseless Sinh; RTN applied if enable_rtn).
        x_mid = (x_before + engine.x) / 2.0
        I_mid = engine._sinh(x_mid, V_applied)
        energy = energy + (V_applied.abs() * I_mid.abs() * pulse_width)

    # Final read + stuck classification (matches C4 write_verify_current)
    final_I = engine.read(engine.V_READ, noisy=noisy_read)
    total = set_pulses + reset_pulses
    within = (final_I - I_target).abs() <= margin
    stuck = (~within) & (total >= max_pulses)

    engine.set_pulses = set_pulses
    engine.reset_pulses = reset_pulses
    engine.energy = energy
    engine.stuck_flag = stuck

    return WriteVerifyResult(
        final_current=final_I,
        final_x=engine.x.clone(),
        set_pulses=set_pulses,
        reset_pulses=reset_pulses,
        energy=energy,
        stuck_flag=stuck,
    )
