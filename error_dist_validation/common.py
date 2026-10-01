"""
Shared constants and Yao-style current-margin write-verify for the
Error Distribution Validation pilot.

Yao et al., Nature 577, 2020 — keep these THREE experiments distinct:
  (A) 8-group pulse-count table  → Stage-1 calibration target
  (B) 32-state characterization  → pulse cap = 500 only
  (C) ResNet/CIFAR weight-transfer → assumed Gaussian N(0, 108 nA) only
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent
PLOTS_DIR = PACKAGE_DIR / "plots"
CALIBRATED_PARAMS_PATH = PACKAGE_DIR / "calibrated_params.json"
ERROR_SAMPLES_PATH = PACKAGE_DIR / "error_samples.parquet"
ERROR_SAMPLES_CSV_PATH = PACKAGE_DIR / "error_samples.csv"
FITTED_DISTRIBUTION_PATH = PACKAGE_DIR / "fitted_distribution.json"


def ensure_plots_dir() -> Path:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    return PLOTS_DIR


# ---------------------------------------------------------------------------
# Experiment (A) — 8-group pulse-count table (Extended Data Fig. 8a,b)
# ---------------------------------------------------------------------------

# Targets in amperes (0.5–4.0 µA)
TARGETS_UA = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0], dtype=float)
TARGETS_A = TARGETS_UA * 1e-6

# Average pulse counts per group
YAO_SET_AVG = np.array([0.2, 0.9, 3.6, 7.0, 9.5, 11.9, 7.9, 14.0], dtype=float)
YAO_RESET_AVG = np.array([72.2, 28.3, 24.6, 14.0, 10.4, 6.6, 4.3, 2.0], dtype=float)
YAO_TOTAL_AVG = np.array([72.4, 29.2, 28.2, 21.0, 19.9, 18.5, 12.2, 16.0], dtype=float)

# Pulse / read conditions (Fig. 1 caption) — terminal differentials.
V_SET_TERMINAL = -2.0   # V_BL=2.0, V_SL=0
V_RESET_TERMINAL = 1.8  # V_SL=1.8, V_BL=0
V_READ = 0.2
PULSE_WIDTH_S = 50e-9

# Yao DC-vs-pulse voltage gap (literature-fixed, NOT fitted):
# effective |V_mem| is lower than the applied terminal pulse by ~0.5 V (SET)
# and ~0.6 V (RESET). Reduces |V| so the memristor sees a weaker drive —
# the hypothesized cause of under-counting deep-RESET pulses at 0.5 uA.
V_SET_GAP = 0.5
V_RESET_GAP = 0.6
V_SET = V_SET_TERMINAL + V_SET_GAP      # -2.0 + 0.5 = -1.5 V
V_RESET = V_RESET_TERMINAL - V_RESET_GAP  # 1.8 - 0.6 = 1.2 V

# Experiment (A) programming margin
MARGIN_A = 100e-9     # ±100 nA

# Experiment (B) — pulse cap only (do NOT mix with (A)'s ±50 nA margin)
MAX_PULSES = 500

# Experiment (C) — assumed Gaussian std used ONLY as Stage-2 noise anchor
GAUSSIAN_STD_NA = 108.0  # nA; NOT a measured programming-error shape from (A)

# Fixed resistance bounds for Stage-1 device dynamics (sanity band).
# Deployment mapping (MNIST) uses the characterized window separately — see
# DEPLOY_R_ON / DEPLOY_R_OFF after scoping.
R_ON = 15e3
R_OFF = 1e6

# Characterized conductance window at V_read=0.2 V for surviving groups
# (1.0–4.0 uA → ~50–200 kOhm). Used for MNIST weight→G mapping at scale=1.0.
# Updated by apply_group_scope() if 0.5 uA is excluded.
DEPLOY_R_ON = 50e3    # ~4.0 uA @ 0.2 V
DEPLOY_R_OFF = 200e3  # ~1.0 uA @ 0.2 V

# Active calibration / MC target set (may drop 0.5 uA under hard-stop scoping)
ACTIVE_TARGETS_UA = TARGETS_UA.copy()
ACTIVE_TARGETS_A = TARGETS_A.copy()
ACTIVE_YAO_SET = YAO_SET_AVG.copy()
ACTIVE_YAO_RESET = YAO_RESET_AVG.copy()
ACTIVE_YAO_TOTAL = YAO_TOTAL_AVG.copy()
SCOPED_EXCLUDE_05UA = False


def apply_group_scope(*, exclude_05ua: bool) -> None:
    """
    Hard-stop scoping: if 0.5 uA cannot be fit after the Yao V_mem correction,
    exclude it and restrict validation to 1.0–4.0 uA. Call once; do not retune.
    """
    global ACTIVE_TARGETS_UA, ACTIVE_TARGETS_A
    global ACTIVE_YAO_SET, ACTIVE_YAO_RESET, ACTIVE_YAO_TOTAL
    global SCOPED_EXCLUDE_05UA, DEPLOY_R_ON, DEPLOY_R_OFF
    SCOPED_EXCLUDE_05UA = bool(exclude_05ua)
    if exclude_05ua:
        mask = TARGETS_UA >= 1.0 - 1e-12
        ACTIVE_TARGETS_UA = TARGETS_UA[mask].copy()
        ACTIVE_TARGETS_A = TARGETS_A[mask].copy()
        ACTIVE_YAO_SET = YAO_SET_AVG[mask].copy()
        ACTIVE_YAO_RESET = YAO_RESET_AVG[mask].copy()
        ACTIVE_YAO_TOTAL = YAO_TOTAL_AVG[mask].copy()
        # 1.0 uA → 200 kOhm, 4.0 uA → 50 kOhm @ 0.2 V
        DEPLOY_R_ON = V_READ / 4.0e-6
        DEPLOY_R_OFF = V_READ / 1.0e-6
    else:
        ACTIVE_TARGETS_UA = TARGETS_UA.copy()
        ACTIVE_TARGETS_A = TARGETS_A.copy()
        ACTIVE_YAO_SET = YAO_SET_AVG.copy()
        ACTIVE_YAO_RESET = YAO_RESET_AVG.copy()
        ACTIVE_YAO_TOTAL = YAO_TOTAL_AVG.copy()
        DEPLOY_R_ON = V_READ / 4.0e-6
        DEPLOY_R_OFF = V_READ / 0.5e-6  # full 0.5–4.0 uA window → 50–400 kOhm


# Stage-1 acceptance gates
STAGE1_MAX_MEAN_REL_ERROR = 0.20
# Per-group gate: catches a single bad group (esp. 0.5 uA HRS-boundary) hiding in the mean
STAGE1_MAX_PER_GROUP_REL_ERROR = 0.40

# Engine always-on read noise (Memristor.cpp): i += N(0, 0.05*|raw_i|)
READ_NOISE_FRAC = 0.05
READ_DT_S = 1e-12  # subthreshold probe; negligible C2C / state drift

# Initial state: LRS (w=1) — matches RESET-heavy low-target pulse counts
W_INIT_LRS = 1.0


# ---------------------------------------------------------------------------
# Engine bootstrap
# ---------------------------------------------------------------------------

def bootstrap_memristorsim():
    """Import the compiled pybind11 module, resolving MinGW DLL path on Windows."""
    for p in (PROJECT_ROOT / "build", PROJECT_ROOT / "build" / "Release"):
        sp = str(p)
        if sp not in sys.path:
            sys.path.insert(0, sp)

    # Also allow importing from project root (local_settings)
    root = str(PROJECT_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)

    try:
        import local_settings  # noqa: F401
        mingw = getattr(local_settings, "MINGW_BIN", None)
        if mingw and os.path.exists(mingw) and hasattr(os, "add_dll_directory"):
            os.add_dll_directory(mingw)
    except ImportError:
        pass

    import memristorsim
    return memristorsim


def make_base_params(ms, *, enable_variability: bool = False,
                     sigma_d2d: float = 0.0, sigma_c2c: float = 0.0,
                     k_on: float = 2e4, k_off: float = -2e4,
                     v_on: float = -0.5, v_off: float = 0.5,
                     alpha_on: float = 3.0, alpha_off: float = 3.0,
                     enable_selector: bool = False,
                     selector_type: int = 1,
                     selector_v_gate: float = 1.8):
    """
    Build MemristorParams with Yao-appropriate fixed R range and Sinh conduction.
    Default |k| ~ 2e4 is a starting guess for 50 ns pulses (fit will refine).

    enable_selector=True with selector_type=1 routes terminal voltages through the
    engine's 1T1R bisection solver so the memristor sees a reduced V_mem (IR drop
    across the access transistor) — the hypothesized fix for the 0.5 uA under-count.
    """
    p = ms.MemristorParams()
    p.R_on = R_ON
    p.R_off = R_OFF
    p.k_on = float(k_on)
    p.k_off = float(k_off)
    p.v_on = float(v_on)
    p.v_off = float(v_off)
    p.alpha_on = float(alpha_on)
    p.alpha_off = float(alpha_off)
    p.w_init = W_INIT_LRS
    p.conduction_model = ms.ConductionModel.Sinh
    p.enable_selector = bool(enable_selector)
    p.selector_type = int(selector_type)  # 1 = 1T1R
    p.selector_v_gate = float(selector_v_gate)
    p.enable_rtn = False
    p.enable_variability = bool(enable_variability)
    p.sigma_k_on = float(sigma_d2d)
    p.sigma_w_init = float(sigma_d2d)
    p.sigma_c2c = float(sigma_c2c)
    p.I_compliance = 1.0
    return p


def params_from_dict(ms, d: Dict[str, Any], *, enable_variability: Optional[bool] = None):
    """Reconstruct MemristorParams from calibrated_params.json content."""
    shape = d["shape"]
    noise = d.get("noise", {})
    ev = enable_variability
    if ev is None:
        ev = bool(noise.get("enable_variability", False))
    return make_base_params(
        ms,
        enable_variability=ev,
        sigma_d2d=float(noise.get("sigma_d2d", 0.0)),
        sigma_c2c=float(noise.get("sigma_c2c", 0.0)),
        k_on=float(shape["k_on"]),
        k_off=float(shape["k_off"]),
        v_on=float(shape["v_on"]),
        v_off=float(shape["v_off"]),
        alpha_on=float(shape["alpha_on"]),
        alpha_off=float(shape["alpha_off"]),
        enable_selector=bool(shape.get("enable_selector", False)),
        selector_type=int(shape.get("selector_type", 1)),
        selector_v_gate=float(shape.get("selector_v_gate", 1.8)),
    )


def load_calibrated_params() -> Dict[str, Any]:
    if not CALIBRATED_PARAMS_PATH.exists():
        raise FileNotFoundError(
            f"Missing {CALIBRATED_PARAMS_PATH}. Run calibration.py first."
        )
    with open(CALIBRATED_PARAMS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Write-verify result
# ---------------------------------------------------------------------------

@dataclass
class WriteVerifyResult:
    group_ua: float
    target_current: float
    final_current: float
    set_pulses: int
    reset_pulses: int
    total_pulses: int
    stuck_flag: bool
    final_error: float

    def as_row(self) -> Dict[str, Any]:
        return {
            "group": self.group_ua,
            "final_current": self.final_current,
            "set_pulses": self.set_pulses,
            "reset_pulses": self.reset_pulses,
            "stuck_flag": self.stuck_flag,
            "final_error": self.final_error,
        }


def read_current(device, v_read: float = V_READ, *, noisy: bool = False) -> float:
    """
    Verify read at v_read.

    noisy=False (Stage 1): calculate_current — deterministic, no 5% read noise.
    noisy=True  (Phase 2+): apply a subthreshold update so engine i() includes
              the always-on 5% read noise (real verify imprecision).
    """
    if not noisy:
        return float(device.calculate_current(v_read))
    device.update(READ_DT_S, v_read)
    return float(device.i())


def write_verify_current(
    device,
    target_current: float,
    *,
    margin: float = MARGIN_A,
    max_pulses: int = MAX_PULSES,
    v_set: float = V_SET,
    v_reset: float = V_RESET,
    v_read: float = V_READ,
    pulse_width: float = PULSE_WIDTH_S,
    group_ua: float = 0.0,
    w_init: float = W_INIT_LRS,
    noisy_read: bool = False,
) -> WriteVerifyResult:
    """
    Closed-loop write-verify targeting read current (Yao-style).

    Loop: read @ v_read → compare to target ± margin → SET or RESET pulse
    until converged or capped at max_pulses (experiment B).

    noisy_read: Stage 1 must be False; Monte Carlo / Stage 2 should be True so
    verify-step imprecision contributes to early/late stop and final error.
    """
    device.set_w(w_init)
    set_pulses = 0
    reset_pulses = 0
    stuck = False

    for _ in range(max_pulses):
        i_now = read_current(device, v_read, noisy=noisy_read)
        err = i_now - target_current
        if abs(err) <= margin:
            break
        if i_now < target_current - margin:
            device.update(pulse_width, v_set)
            set_pulses += 1
        else:
            device.update(pulse_width, v_reset)
            reset_pulses += 1
    else:
        # Loop exhausted without break → hit pulse cap
        stuck = True

    final_i = read_current(device, v_read, noisy=noisy_read)
    total = set_pulses + reset_pulses
    if abs(final_i - target_current) <= margin:
        stuck = False
    elif total >= max_pulses:
        stuck = True

    return WriteVerifyResult(
        group_ua=group_ua,
        target_current=target_current,
        final_current=final_i,
        set_pulses=set_pulses,
        reset_pulses=reset_pulses,
        total_pulses=total,
        stuck_flag=stuck,
        final_error=final_i - target_current,
    )


def simulate_pulse_counts(
    ms,
    params,
    *,
    w_init: float = W_INIT_LRS,
    noisy_read: bool = False,
    targets_a=None,
    targets_ua=None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Deterministic (or single-shot) pulse counts for active target groups.
    Returns (set_counts, reset_counts, total_counts).
    """
    t_a_arr = ACTIVE_TARGETS_A if targets_a is None else targets_a
    t_ua_arr = ACTIVE_TARGETS_UA if targets_ua is None else targets_ua
    set_c = np.zeros(len(t_a_arr))
    reset_c = np.zeros(len(t_a_arr))
    total_c = np.zeros(len(t_a_arr))
    for i, (t_a, t_ua) in enumerate(zip(t_a_arr, t_ua_arr)):
        device = ms.PhysicsEngine(params)
        res = write_verify_current(
            device, float(t_a),
            group_ua=float(t_ua), w_init=w_init, noisy_read=noisy_read,
            v_set=V_SET, v_reset=V_RESET,
        )
        set_c[i] = res.set_pulses
        reset_c[i] = res.reset_pulses
        total_c[i] = res.total_pulses
    return set_c, reset_c, total_c


def per_group_relative_errors(
    sim_totals: np.ndarray, ref_totals: np.ndarray = None
) -> np.ndarray:
    if ref_totals is None:
        ref_totals = ACTIVE_YAO_TOTAL
    return np.abs(sim_totals - ref_totals) / np.maximum(ref_totals, 1e-12)


def mean_relative_pulse_error(sim_totals: np.ndarray, ref_totals: np.ndarray = None) -> float:
    return float(np.mean(per_group_relative_errors(sim_totals, ref_totals)))


def max_relative_pulse_error(sim_totals: np.ndarray, ref_totals: np.ndarray = None) -> float:
    return float(np.max(per_group_relative_errors(sim_totals, ref_totals)))


def pulse_count_loss(sim_totals: np.ndarray, ref_totals: np.ndarray = None) -> float:
    """Stage-1 loss: sum of squared relative errors on total pulse counts."""
    if ref_totals is None:
        ref_totals = ACTIVE_YAO_TOTAL
    rel = (sim_totals - ref_totals) / np.maximum(ref_totals, 1e-12)
    return float(np.sum(rel ** 2))


def save_json(path: Path, obj: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
