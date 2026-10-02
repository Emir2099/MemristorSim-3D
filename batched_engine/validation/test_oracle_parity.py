"""
Gate 1 — deterministic parity vs C++ PhysicsEngine oracle.

E=1, N=1, variability off, noisy_read=False.
x after each pulse must match within 1e-6 relative.

If this fails, do NOT proceed to Gate 2.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from batched_engine.device import BatchedMemristorArray, load_calibrated_params, rk4_step, sinh_current
from error_dist_validation.common import (
    V_READ,
    V_RESET,
    V_SET,
    PULSE_WIDTH_S,
    bootstrap_memristorsim,
    params_from_dict,
    write_verify_current,
)


REL_TOL = 1e-6
ABS_TOL = 1e-12


def _rel_err(a: float, b: float) -> float:
    denom = max(abs(b), ABS_TOL)
    return abs(a - b) / denom


def _assert_close(name: str, got: float, ref: float, tol: float = REL_TOL) -> None:
    err = _rel_err(got, ref)
    assert err <= tol, f"{name}: got={got:.12e} ref={ref:.12e} rel_err={err:.3e} > {tol}"


def test_single_pulse_trajectory():
    """Fixed SET then RESET pulses — compare w after each update."""
    ms = bootstrap_memristorsim()
    calib = load_calibrated_params()
    params = params_from_dict(ms, calib, enable_variability=False)
    params.enable_variability = False
    params.sigma_c2c = 0.0
    params.sigma_k_on = 0.0
    params.sigma_w_init = 0.0

    cpp = ms.PhysicsEngine(params)
    cpp.set_w(1.0)

    eng = BatchedMemristorArray(
        1, 1, calibrated_params=calib, enable_variability=False, seed=0
    )
    eng.reset(x_init=torch.tensor([[1.0]], dtype=torch.float64))

    pulses = [
        (V_SET, PULSE_WIDTH_S),
        (V_SET, PULSE_WIDTH_S),
        (V_RESET, PULSE_WIDTH_S),
        (V_RESET, PULSE_WIDTH_S),
        (V_RESET, PULSE_WIDTH_S),
        (V_SET, PULSE_WIDTH_S),
    ]

    print("Gate 1a — fixed pulse trajectory")
    for i, (v, dt) in enumerate(pulses):
        cpp.update(dt, v)
        V_t = torch.tensor([[v]], dtype=torch.float64)
        eng.pulse(V_t, dt=dt, active_mask=torch.tensor([[True]]), apply_c2c=False)
        w_cpp = float(cpp.w())
        w_py = float(eng.x.item())
        print(f"  pulse {i}: V={v:+.2f}  w_cpp={w_cpp:.12e}  w_py={w_py:.12e}")
        _assert_close(f"pulse[{i}].w", w_py, w_cpp)

        # Also compare noiseless current at V_READ
        i_cpp = float(cpp.calculate_current(V_READ))
        i_py = float(eng.read(V_READ, noisy=False).item())
        _assert_close(f"pulse[{i}].I_read", i_py, i_cpp)


def test_rk4_matches_cpp_get_dw_path():
    """Direct RK4 parity from identical (w, v) — isolates ODE from write-verify."""
    ms = bootstrap_memristorsim()
    calib = load_calibrated_params()
    params = params_from_dict(ms, calib, enable_variability=False)
    cpp = ms.PhysicsEngine(params)

    shape = calib["shape"]
    k_on = torch.tensor([[float(shape["k_on"])]], dtype=torch.float64)
    k_off = torch.tensor([[float(shape["k_off"])]], dtype=torch.float64)
    v_on = torch.tensor([[float(shape["v_on"])]], dtype=torch.float64)
    v_off = torch.tensor([[float(shape["v_off"])]], dtype=torch.float64)
    alpha_on = float(shape["alpha_on"])
    alpha_off = float(shape["alpha_off"])

    print("Gate 1b — RK4 step from shared w0")
    for w0 in (1.0, 0.7, 0.3, 0.0):
        for v in (V_SET, V_RESET, 0.0, -0.1, 0.1):
            cpp.set_w(w0)
            # Force dT=0 path: reset then set_w (reset re-applies D2D but variability off)
            cpp.reset()
            cpp.set_w(w0)
            cpp.update(PULSE_WIDTH_S, v)
            w_cpp = float(cpp.w())

            w = torch.tensor([[w0]], dtype=torch.float64)
            V = torch.tensor([[v]], dtype=torch.float64)
            dT = torch.zeros(1, 1, dtype=torch.float64)
            w_new = rk4_step(
                w, V, PULSE_WIDTH_S, k_on, k_off, v_on, v_off, alpha_on, alpha_off, dT
            ).clamp(0, 1)
            w_py = float(w_new.item())
            print(f"  w0={w0:.1f} V={v:+.2f}  w_cpp={w_cpp:.12e}  w_py={w_py:.12e}")
            _assert_close(f"rk4(w0={w0},V={v})", w_py, w_cpp)


def test_write_verify_parity():
    """Full noiseless write-verify to several targets — pulse counts + final w/I."""
    ms = bootstrap_memristorsim()
    calib = load_calibrated_params()
    params = params_from_dict(ms, calib, enable_variability=False)

    targets_ua = [1.0, 2.0, 3.0, 4.0]
    print("Gate 1c — noiseless write-verify vs C4 protocol")
    for t_ua in targets_ua:
        t_a = t_ua * 1e-6
        cpp = ms.PhysicsEngine(params)
        res_cpp = write_verify_current(
            cpp, t_a, group_ua=t_ua, w_init=1.0, noisy_read=False,
            v_set=V_SET, v_reset=V_RESET,
        )

        eng = BatchedMemristorArray(
            1, 1, calibrated_params=calib, enable_variability=False, seed=0
        )
        eng.reset(x_init=torch.tensor([[1.0]], dtype=torch.float64))
        I_target = torch.tensor([[t_a]], dtype=torch.float64)
        res_py = eng.write_verify(I_target, noisy_read=False)

        print(
            f"  {t_ua:.1f} uA  cpp: set={res_cpp.set_pulses} reset={res_cpp.reset_pulses} "
            f"I={res_cpp.final_current:.6e} w={cpp.w():.6e}"
        )
        print(
            f"         py:  set={int(res_py.set_pulses.item())} "
            f"reset={int(res_py.reset_pulses.item())} "
            f"I={res_py.final_current.item():.6e} w={res_py.final_x.item():.6e}"
        )

        assert int(res_py.set_pulses.item()) == res_cpp.set_pulses, (
            f"{t_ua}uA set pulse mismatch"
        )
        assert int(res_py.reset_pulses.item()) == res_cpp.reset_pulses, (
            f"{t_ua}uA reset pulse mismatch"
        )
        _assert_close(f"{t_ua}uA.w", float(res_py.final_x.item()), float(cpp.w()))
        _assert_close(
            f"{t_ua}uA.I",
            float(res_py.final_current.item()),
            float(res_cpp.final_current),
        )


def test_sinh_current_parity():
    ms = bootstrap_memristorsim()
    calib = load_calibrated_params()
    params = params_from_dict(ms, calib, enable_variability=False)
    cpp = ms.PhysicsEngine(params)
    shape = calib["shape"]
    R_on = float(shape["R_on"])
    R_off = float(shape["R_off"])

    print("Gate 1d — Sinh I–V parity")
    for w0 in (0.0, 0.25, 0.5, 0.75, 1.0):
        cpp.set_w(w0)
        for v in (0.2, -0.2, 1.2, -1.5):
            i_cpp = float(cpp.calculate_current(v))
            w = torch.tensor([[w0]], dtype=torch.float64)
            V = torch.tensor([[v]], dtype=torch.float64)
            i_py = float(sinh_current(w, V, R_on, R_off).item())
            _assert_close(f"I(w={w0},V={v})", i_py, i_cpp)


if __name__ == "__main__":
    test_sinh_current_parity()
    test_rk4_matches_cpp_get_dw_path()
    test_single_pulse_trajectory()
    test_write_verify_parity()
    print("\nGATE 1 PASSED — deterministic parity within 1e-6 relative.")
