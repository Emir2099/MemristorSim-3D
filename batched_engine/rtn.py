"""
Optional RTN (random telegraph noise) — off by default.

Literal port of Memristor.cpp / Memristor.h two-state Markov RTN.
Component #4 omitted RTN; Gate 1/2 must keep enable_rtn=False.
"""

from __future__ import annotations

from typing import Optional

import torch

# Memristor.h defaults
RTN_AMPLITUDE_DEFAULT = 0.03
RTN_TAU_C_DEFAULT = 0.05  # mean capture time (s)
RTN_TAU_E_DEFAULT = 0.05  # mean emission time (s)


def step_rtn_state(
    rtn_state: torch.Tensor,
    dt: float,
    *,
    tau_c: float = RTN_TAU_C_DEFAULT,
    tau_e: float = RTN_TAU_E_DEFAULT,
    active_mask: Optional[torch.Tensor] = None,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """
    Two-state Markov update — PhysicsEngine::update RTN block.

    State 0 (empty) → 1 with p = 1 - exp(-dt / tau_c)
    State 1 (occupied) → 0 with p = 1 - exp(-dt / tau_e)
    """
    if dt <= 0.0:
        return rtn_state

    u = torch.rand(
        rtn_state.shape,
        dtype=torch.float64,
        device=rtn_state.device,
        generator=generator,
    )
    # Promote u to rtn dtype for comparisons only via float values
    p_capture = 1.0 - torch.exp(torch.tensor(-dt / tau_c, dtype=u.dtype, device=u.device))
    p_emit = 1.0 - torch.exp(torch.tensor(-dt / tau_e, dtype=u.dtype, device=u.device))

    empty = rtn_state == 0
    occupied = ~empty
    flip_to_1 = empty & (u < p_capture)
    flip_to_0 = occupied & (u < p_emit)

    new_state = rtn_state.clone()
    new_state = torch.where(flip_to_1, torch.ones_like(new_state), new_state)
    new_state = torch.where(flip_to_0, torch.zeros_like(new_state), new_state)

    if active_mask is not None:
        new_state = torch.where(active_mask, new_state, rtn_state)
    return new_state


def apply_rtn_factor(
    raw_i: torch.Tensor,
    rtn_state: torch.Tensor,
    amplitude: float = RTN_AMPLITUDE_DEFAULT,
) -> torch.Tensor:
    """
    Relative current fluctuation from calculate_memristor_current:

      factor = 1 + (±0.5) * rtn_amplitude
      raw_i *= factor
    """
    # state 1 → +0.5 * amp; state 0 → -0.5 * amp
    half = torch.tensor(0.5, dtype=raw_i.dtype, device=raw_i.device)
    amp = torch.tensor(amplitude, dtype=raw_i.dtype, device=raw_i.device)
    sign = torch.where(rtn_state > 0, half, -half)
    factor = 1.0 + sign * amp
    return raw_i * factor
