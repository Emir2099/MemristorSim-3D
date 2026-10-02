"""
Per-cell D2D parameter sampling for the batched engine.

(default): log-normal k_on/k_off, Normal x_init; global v_on/v_off.
stretch: threshold D2D with uncalibrated σ_v — off by default.
"""

from __future__ import annotations

import warnings
from typing import Dict, Optional

import torch

# Placeholder only — This never calibrated threshold D2D.
# Do not treat results using this as published / Gate-passing claims.
UNCALIBRATED_SIGMA_V_PLACEHOLDER = 0.05


def warn_uncalibrated_threshold_d2d(sigma_v: float) -> None:
    warnings.warn(
        f"threshold_d2d enabled with σ_v={sigma_v} — UNCALIBRATED placeholder. "
        "Component #4 never fit switching-threshold D2D. Do not use for Gate 1/2 "
        "or published results until a Stage-2-style calibration is run.",
        UserWarning,
        stacklevel=2,
    )


def sample_d2d_params(
    E: int,
    N: int,
    *,
    k_on0: float,
    k_off0: float,
    v_on0: float,
    v_off0: float,
    sigma_d2d: float,
    enable_variability: bool = True,
    threshold_d2d: bool = False,
    sigma_v: float = UNCALIBRATED_SIGMA_V_PLACEHOLDER,
    w_init_mean: float = 1.0,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
    generator: Optional[torch.Generator] = None,
    x_init: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """
    Match PhysicsEngine::apply_d2d_variability:

      w_init += N(0,1) * sigma_w_init
      k_on  *= 10 ** (N(0,1) * sigma_k_on)
      k_off *= 10 ** (N(0,1) * sigma_k_on)
    """
    device = torch.device(device)
    shape = (E, N)

    def _randn() -> torch.Tensor:
        return torch.randn(shape, dtype=dtype, device=device, generator=generator)

    if enable_variability and sigma_d2d != 0.0:
        k_on = torch.full(shape, k_on0, dtype=dtype, device=device) * torch.pow(
            10.0, _randn() * sigma_d2d
        )
        k_off = torch.full(shape, k_off0, dtype=dtype, device=device) * torch.pow(
            10.0, _randn() * sigma_d2d
        )
        if x_init is None:
            x = (torch.full(shape, w_init_mean, dtype=dtype, device=device) + _randn() * sigma_d2d).clamp(
                0.0, 1.0
            )
        else:
            x = x_init.to(device=device, dtype=dtype).expand(E, N).clone()
    else:
        k_on = torch.full(shape, k_on0, dtype=dtype, device=device)
        k_off = torch.full(shape, k_off0, dtype=dtype, device=device)
        if x_init is None:
            x = torch.full(shape, w_init_mean, dtype=dtype, device=device)
        else:
            x = x_init.to(device=device, dtype=dtype).expand(E, N).clone()

    if threshold_d2d:
        warn_uncalibrated_threshold_d2d(sigma_v)
        v_on = torch.full(shape, v_on0, dtype=dtype, device=device) + _randn() * sigma_v
        v_off = torch.full(shape, v_off0, dtype=dtype, device=device) + _randn() * sigma_v
    else:
        v_on = torch.full(shape, v_on0, dtype=dtype, device=device)
        v_off = torch.full(shape, v_off0, dtype=dtype, device=device)

    return {"k_on": k_on, "k_off": k_off, "v_on": v_on, "v_off": v_off, "x": x}
