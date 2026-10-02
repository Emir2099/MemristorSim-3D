"""
RTN hook smoke tests.

Default off must not change Gate-1 physics. When enabled, current gets the
C++ multiplicative RTN factor.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from batched_engine.device import BatchedMemristorArray, load_calibrated_params
from batched_engine.rtn import apply_rtn_factor, step_rtn_state


def test_rtn_default_off_matches_no_rtn():
    calib = load_calibrated_params()
    eng = BatchedMemristorArray(
        1, 1, calibrated_params=calib, enable_variability=False, enable_rtn=False, seed=0
    )
    eng.reset(x_init=torch.tensor([[0.5]], dtype=torch.float64))
    assert eng.enable_rtn is False
    I = eng.read(0.2, noisy=False)
    # Force rtn_state occupied — must still be ignored when enable_rtn=False
    eng.rtn_state.fill_(1)
    I2 = eng.read(0.2, noisy=False)
    assert torch.allclose(I, I2), "RTN state must not affect current when enable_rtn=False"
    print("RTN default-off: OK")


def test_rtn_factor_matches_cpp_formula():
    raw = torch.tensor([[1e-6]], dtype=torch.float64)
    empty = torch.tensor([[0]], dtype=torch.int64)
    occupied = torch.tensor([[1]], dtype=torch.int64)
    amp = 0.03
    i0 = apply_rtn_factor(raw, empty, amp)
    i1 = apply_rtn_factor(raw, occupied, amp)
    assert abs(float(i0) - 1e-6 * (1.0 - 0.5 * amp)) / 1e-6 < 1e-12
    assert abs(float(i1) - 1e-6 * (1.0 + 0.5 * amp)) / 1e-6 < 1e-12
    print("RTN factor formula: OK")


def test_rtn_enabled_changes_read_current():
    calib = load_calibrated_params()
    eng = BatchedMemristorArray(
        1,
        1,
        calibrated_params=calib,
        enable_variability=False,
        enable_rtn=True,
        rtn_amplitude=0.03,
        seed=0,
    )
    eng.reset(x_init=torch.tensor([[0.5]], dtype=torch.float64))
    eng.rtn_state.fill_(0)
    I_empty = eng.read(0.2, noisy=False)
    eng.rtn_state.fill_(1)
    I_occ = eng.read(0.2, noisy=False)
    ratio = float(I_occ / I_empty)
    expected = (1.0 + 0.5 * 0.03) / (1.0 - 0.5 * 0.03)
    assert abs(ratio - expected) < 1e-9, f"ratio={ratio} expected={expected}"
    print("RTN enabled read: OK")


def test_rtn_markov_can_flip_with_large_dt():
    state = torch.zeros(100, 10, dtype=torch.int64)
    # With tau=0.05 and dt=1.0, p ≈ 1 - exp(-20) ≈ 1 → almost all flip to 1
    new_state = step_rtn_state(state, dt=1.0, tau_c=0.05, tau_e=0.05)
    assert int(new_state.sum()) > 900, "expected nearly all traps to capture"
    print("RTN Markov step: OK")


if __name__ == "__main__":
    test_rtn_default_off_matches_no_rtn()
    test_rtn_factor_matches_cpp_formula()
    test_rtn_enabled_changes_read_current()
    test_rtn_markov_can_flip_with_large_dt()
    print("\nRTN hook tests passed.")
