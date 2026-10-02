"""
Batched VTEAM device physics — literal port of Memristor.cpp.

See PORTED_CONSTANTS.md. Do not re-derive Biolek / power-law / read noise.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent
CALIBRATED_PARAMS_PATH = PROJECT_ROOT / "error_dist_validation" / "calibrated_params.json"

# Engine defaults matching Memristor.h
GAMMA_SINH_DEFAULT = 2.0
READ_NOISE_FRAC = 0.05
T_CRITICAL_DEFAULT = 5.0
THETA_THERMAL_DEFAULT = 0.01
TAU_THERMAL = 0.01
I_COMPLIANCE_DEFAULT = 1.0  # C4 make_base_params uses 1.0


def load_calibrated_params(path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    import json
    p = Path(path) if path is not None else CALIBRATED_PARAMS_PATH
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def clamp01(x: torch.Tensor) -> torch.Tensor:
    return x.clamp(0.0, 1.0)


def dw_dt(
    w: torch.Tensor,
    v: torch.Tensor,
    k_on: torch.Tensor,
    k_off: torch.Tensor,
    v_on: torch.Tensor,
    v_off: torch.Tensor,
    alpha_on: float,
    alpha_off: float,
    dT: Optional[torch.Tensor] = None,
    T_critical: float = T_CRITICAL_DEFAULT,
) -> torch.Tensor:
    """
    Vectorized PhysicsEngine::get_dw_dt.

    Strict voltage branches; raw pow; Biolek keyed by SET/RESET branch.
    """
    w_c = clamp01(w)
    dw = torch.zeros_like(w_c)

    reset_mask = v > v_off
    set_mask = (~reset_mask) & (v < v_on)

    if reset_mask.any():
        base = (v / v_off) - 1.0
        # Raw pow matches std::pow. Safe under v1 fixed global thresholds (Yao
        # |V_pulse| > |v_th| keeps bases > 0). NOT safe if v2 threshold-D2D is
        # on: a per-cell random v_on/v_off can make (v/v_th - 1) negative under
        # fractional alpha — then abs/sign (or a branch guard) is required.
        term = k_off * torch.pow(base, alpha_off) * (1.0 - torch.pow(w_c - 1.0, 8.0))
        dw = torch.where(reset_mask, term, dw)

    if set_mask.any():
        base = (v / v_on) - 1.0
        # Same v1-only raw-pow caveat as RESET branch above.
        term = k_on * torch.pow(base, alpha_on) * (1.0 - torch.pow(w_c, 8.0))
        dw = torch.where(set_mask, term, dw)

    if dT is not None:
        hot = dT > T_critical
        if hot.any():
            decay = (
                -torch.abs(k_off)
                * ((dT - T_critical) / T_critical)
                * w_c
            )
            dw = torch.where(hot, dw + decay, dw)

    return dw


def rk4_step(
    w: torch.Tensor,
    v: torch.Tensor,
    dt: float,
    k_on: torch.Tensor,
    k_off: torch.Tensor,
    v_on: torch.Tensor,
    v_off: torch.Tensor,
    alpha_on: float,
    alpha_off: float,
    dT: Optional[torch.Tensor] = None,
    T_critical: float = T_CRITICAL_DEFAULT,
) -> torch.Tensor:
    """One classic RK4 step — matches PhysicsEngine::rk4 (no sub-stepping)."""
    k1 = dw_dt(w, v, k_on, k_off, v_on, v_off, alpha_on, alpha_off, dT, T_critical)
    k2 = dw_dt(
        w + 0.5 * dt * k1, v, k_on, k_off, v_on, v_off, alpha_on, alpha_off, dT, T_critical
    )
    k3 = dw_dt(
        w + 0.5 * dt * k2, v, k_on, k_off, v_on, v_off, alpha_on, alpha_off, dT, T_critical
    )
    k4 = dw_dt(
        w + dt * k3, v, k_on, k_off, v_on, v_off, alpha_on, alpha_off, dT, T_critical
    )
    return w + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def apply_c2c_noise(
    w: torch.Tensor,
    sigma_c2c: float,
    dt: float,
    active_mask: torch.Tensor,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Euler–Maruyama C2C, gated to cells that received a pulse this step."""
    if sigma_c2c == 0.0 or not active_mask.any():
        return w
    noise = sigma_c2c * (dt ** 0.5) * torch.randn(
        w.shape, dtype=w.dtype, device=w.device, generator=generator
    )
    return torch.where(active_mask, w + noise, w)


def sinh_current(
    w: torch.Tensor,
    V: torch.Tensor,
    R_on: float,
    R_off: float,
    gamma: float = GAMMA_SINH_DEFAULT,
    I_compliance: float = I_COMPLIANCE_DEFAULT,
    rtn_state: Optional[torch.Tensor] = None,
    rtn_amplitude: float = 0.03,
    enable_rtn: bool = False,
) -> torch.Tensor:
    """
    Deterministic Sinh I–V — calculate_memristor_current with selector off.

    RTN factor (if enabled) applied before I_compliance clamp, matching C++.
    """
    i_on = V / R_on
    sinh_v = torch.sinh(gamma * V)
    sinh_1 = torch.sinh(torch.tensor(gamma, dtype=w.dtype, device=w.device))
    i_off = sinh_v / (R_off * sinh_1)
    raw = w * i_on + (1.0 - w) * i_off
    if enable_rtn and rtn_state is not None:
        from .rtn import apply_rtn_factor

        raw = apply_rtn_factor(raw, rtn_state, rtn_amplitude)
    return raw.clamp(-I_compliance, I_compliance)


def apply_read_noise(
    I: torch.Tensor,
    frac: float = READ_NOISE_FRAC,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """m_i = raw_i + N(0,1) * (0.05 * raw_i)."""
    noise = torch.randn(I.shape, dtype=I.dtype, device=I.device, generator=generator) * (
        frac * I
    )
    return I + noise


def update_thermal(
    dT: torch.Tensor,
    power: torch.Tensor,
    dt: float,
    theta_thermal: float = THETA_THERMAL_DEFAULT,
) -> torch.Tensor:
    """
    Per-cell self-heating (GUI 'Temp Rise (dT)') from PhysicsEngine::update.

    Not inter-cell thermal coupling — that is Component #5 and out of scope.
    """
    dT_target = power * theta_thermal
    return dT + (dt / (dt + TAU_THERMAL)) * (dT_target - dT)


@dataclass
class WriteVerifyResult:
    final_current: torch.Tensor
    final_x: torch.Tensor
    set_pulses: torch.Tensor
    reset_pulses: torch.Tensor
    energy: torch.Tensor
    stuck_flag: torch.Tensor


class BatchedMemristorArray:
    """
    Vectorized VTEAM array over shape [E, N].

    v1: global v_on/v_off; per-cell k_on/k_off/x from D2D sampling.
    """

    def __init__(
        self,
        n_ensemble: int,
        n_cells: int,
        calibrated_params: Optional[Dict[str, Any]] = None,
        device: str = "cpu",
        dtype: torch.dtype = torch.float64,
        seed: Optional[int] = None,
        enable_variability: bool = True,
        threshold_d2d: bool = False,
        sigma_v: float = 0.05,
        enable_rtn: bool = False,
        rtn_amplitude: float = 0.03,
        rtn_tau_c: float = 0.05,
        rtn_tau_e: float = 0.05,
    ):
        from .d2d import warn_uncalibrated_threshold_d2d
        from .rtn import (
            RTN_AMPLITUDE_DEFAULT,
            RTN_TAU_C_DEFAULT,
            RTN_TAU_E_DEFAULT,
        )

        if calibrated_params is None:
            calibrated_params = load_calibrated_params()
        self.calibrated_params = calibrated_params
        self.E = int(n_ensemble)
        self.N = int(n_cells)
        self.device = torch.device(device)
        self.dtype = dtype
        self.enable_variability = bool(enable_variability)
        self.threshold_d2d = bool(threshold_d2d)
        # RTN off by default — do not enable for Gate 1/2.
        self.enable_rtn = bool(enable_rtn)
        self.rtn_amplitude = float(
            rtn_amplitude if rtn_amplitude is not None else RTN_AMPLITUDE_DEFAULT
        )
        self.rtn_tau_c = float(rtn_tau_c if rtn_tau_c is not None else RTN_TAU_C_DEFAULT)
        self.rtn_tau_e = float(rtn_tau_e if rtn_tau_e is not None else RTN_TAU_E_DEFAULT)

        shape = calibrated_params["shape"]
        noise = calibrated_params.get("noise", {})
        pulse = calibrated_params.get("pulse_conditions", {})

        self.k_on0 = float(shape["k_on"])
        self.k_off0 = float(shape["k_off"])
        self.v_on0 = float(shape["v_on"])
        self.v_off0 = float(shape["v_off"])
        self.alpha_on = float(shape["alpha_on"])
        self.alpha_off = float(shape["alpha_off"])
        self.R_on = float(shape["R_on"])
        self.R_off = float(shape["R_off"])
        self.gamma_sinh = GAMMA_SINH_DEFAULT
        self.I_compliance = I_COMPLIANCE_DEFAULT
        self.T_critical = T_CRITICAL_DEFAULT
        self.theta_thermal = THETA_THERMAL_DEFAULT

        self.sigma_d2d = float(noise.get("sigma_d2d", 0.0)) if enable_variability else 0.0
        self.sigma_c2c = float(noise.get("sigma_c2c", 0.0)) if enable_variability else 0.0

        self.V_SET = float(pulse.get("V_SET_eff", -1.5))
        self.V_RESET = float(pulse.get("V_RESET_eff", 1.2))
        self.V_READ = float(pulse.get("V_READ", 0.2))
        self.pulse_width = float(pulse.get("pulse_width_s", 50e-9))
        self.margin = float(pulse.get("margin_A", 100e-9))
        self.max_pulses_default = int(pulse.get("max_pulses", 500))

        self._generator: Optional[torch.Generator] = None
        if seed is not None:
            self._generator = torch.Generator(device=self.device)
            self._generator.manual_seed(int(seed))

        if threshold_d2d:
            warn_uncalibrated_threshold_d2d(sigma_v)

        self.sigma_v = float(sigma_v)
        self.k_on: torch.Tensor
        self.k_off: torch.Tensor
        self.v_on: torch.Tensor
        self.v_off: torch.Tensor
        self.x: torch.Tensor
        self.dT: torch.Tensor
        self.set_pulses: torch.Tensor
        self.reset_pulses: torch.Tensor
        self.energy: torch.Tensor
        self.stuck_flag: torch.Tensor
        self.rtn_state: torch.Tensor

        self.reset()

    def _empty(self) -> torch.Tensor:
        return torch.zeros(self.E, self.N, dtype=self.dtype, device=self.device)

    def _sinh(
        self,
        w: torch.Tensor,
        V: torch.Tensor,
    ) -> torch.Tensor:
        return sinh_current(
            w,
            V,
            self.R_on,
            self.R_off,
            self.gamma_sinh,
            self.I_compliance,
            rtn_state=self.rtn_state if self.enable_rtn else None,
            rtn_amplitude=self.rtn_amplitude,
            enable_rtn=self.enable_rtn,
        )

    def reset(self, x_init: Optional[torch.Tensor] = None) -> None:
        from .d2d import sample_d2d_params

        sampled = sample_d2d_params(
            self.E,
            self.N,
            k_on0=self.k_on0,
            k_off0=self.k_off0,
            v_on0=self.v_on0,
            v_off0=self.v_off0,
            sigma_d2d=self.sigma_d2d,
            enable_variability=self.enable_variability,
            threshold_d2d=self.threshold_d2d,
            sigma_v=self.sigma_v,
            device=self.device,
            dtype=self.dtype,
            generator=self._generator,
            x_init=x_init,
        )
        self.k_on = sampled["k_on"]
        self.k_off = sampled["k_off"]
        self.v_on = sampled["v_on"]
        self.v_off = sampled["v_off"]
        self.x = sampled["x"]
        self.dT = self._empty()
        self.set_pulses = self._empty()
        self.reset_pulses = self._empty()
        self.energy = self._empty()
        self.stuck_flag = torch.zeros(
            self.E, self.N, dtype=torch.bool, device=self.device
        )
        # RTN trap empty at reset (matches PhysicsEngine constructor / reset)
        self.rtn_state = torch.zeros(
            self.E, self.N, dtype=torch.int64, device=self.device
        )

    def read(self, V_read: Optional[float] = None, noisy: bool = True) -> torch.Tensor:
        from .rtn import step_rtn_state

        V = self.V_READ if V_read is None else float(V_read)
        V_t = torch.full(
            (self.E, self.N), V, dtype=self.dtype, device=self.device
        )
        # Noisy verify mirrors C4: subthreshold update advances RTN with dt=1e-12
        if self.enable_rtn and noisy:
            self.rtn_state = step_rtn_state(
                self.rtn_state,
                1e-12,
                tau_c=self.rtn_tau_c,
                tau_e=self.rtn_tau_e,
                generator=self._generator,
            )
        I = self._sinh(self.x, V_t)
        if noisy:
            I = apply_read_noise(I, generator=self._generator)
        return I

    def pulse(
        self,
        V_applied: torch.Tensor,
        dt: Optional[float] = None,
        active_mask: Optional[torch.Tensor] = None,
        apply_c2c: bool = True,
    ) -> None:
        """
        Apply one voltage pulse to cells where |V| > 0 (or active_mask).

        Matches PhysicsEngine::update with enable_selector=false:
        RK4 → optional C2C → clamp → RTN step → thermal.
        """
        from .rtn import step_rtn_state

        dt = self.pulse_width if dt is None else float(dt)
        if active_mask is None:
            active_mask = V_applied != 0

        x_before = self.x
        x_after = rk4_step(
            x_before,
            V_applied,
            dt,
            self.k_on,
            self.k_off,
            self.v_on,
            self.v_off,
            self.alpha_on,
            self.alpha_off,
            self.dT,
            self.T_critical,
        )
        if apply_c2c and self.sigma_c2c != 0.0:
            x_after = apply_c2c_noise(
                x_after, self.sigma_c2c, dt, active_mask, self._generator
            )
        x_after = clamp01(x_after)
        self.x = torch.where(active_mask, x_after, x_before)

        if self.enable_rtn:
            self.rtn_state = step_rtn_state(
                self.rtn_state,
                dt,
                tau_c=self.rtn_tau_c,
                tau_e=self.rtn_tau_e,
                active_mask=active_mask,
                generator=self._generator,
            )

        I_now = self._sinh(self.x, V_applied)
        power = (I_now * V_applied).abs()
        dT_new = update_thermal(self.dT, power, dt, self.theta_thermal)
        self.dT = torch.where(active_mask, dT_new, self.dT)

    def write_verify(
        self,
        I_target: torch.Tensor,
        max_pulses: Optional[int] = None,
        noisy_read: bool = True,
        margin: Optional[float] = None,
        V_SET: Optional[float] = None,
        V_RESET: Optional[float] = None,
        pulse_width: Optional[float] = None,
    ) -> WriteVerifyResult:
        from .write_verify import write_verify_batched

        return write_verify_batched(
            self,
            I_target,
            max_pulses=max_pulses,
            noisy_read=noisy_read,
            margin=margin,
            V_SET=V_SET,
            V_RESET=V_RESET,
            pulse_width=pulse_width,
        )
