"""
Two-stage Nelder-Mead calibration against Yao et al. experiment (A).

Stage 1: deterministic VTEAM shape fit (noise OFF). Hard gate: mean relative
         pulse-count error < 20%. If this fails, abort — do NOT tune noise.
Stage 2: fit σ_D2D, σ_C2C so programming-error std ≈ 108 nA from experiment (C)
         (cross-experiment approximation — flagged in output JSON).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

# Allow running as script from repo root or package dir
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import error_dist_validation.common as C
from error_dist_validation.common import (
    CALIBRATED_PARAMS_PATH,
    GAUSSIAN_STD_NA,
    MARGIN_A,
    MAX_PULSES,
    R_OFF,
    R_ON,
    STAGE1_MAX_MEAN_REL_ERROR,
    STAGE1_MAX_PER_GROUP_REL_ERROR,
    V_READ,
    V_RESET,
    V_RESET_GAP,
    V_RESET_TERMINAL,
    V_SET,
    V_SET_GAP,
    V_SET_TERMINAL,
    W_INIT_LRS,
    apply_group_scope,
    bootstrap_memristorsim,
    ensure_plots_dir,
    make_base_params,
    max_relative_pulse_error,
    mean_relative_pulse_error,
    per_group_relative_errors,
    pulse_count_loss,
    save_json,
    simulate_pulse_counts,
    write_verify_current,
)


# ---------------------------------------------------------------------------
# Stage 1 encoding: x = [log10(|k_on|), log10(|k_off|), v_on, v_off, α_on, α_off]
# ---------------------------------------------------------------------------

def _decode_stage1(x: np.ndarray):
    k_on = 10.0 ** x[0]
    k_off = -(10.0 ** x[1])
    v_on = float(x[2])
    v_off = float(x[3])
    alpha_on = float(max(x[4], 0.1))
    alpha_off = float(max(x[5], 0.1))
    return k_on, k_off, v_on, v_off, alpha_on, alpha_off


def _stage1_init() -> np.ndarray:
    # Seed near a workable basin under Yao V_mem-corrected amplitudes
    return np.array([4.55, 4.55, -0.35, 0.55, 3.0, 4.5], dtype=float)


def _stage1_multistart_seeds(rng: np.random.Generator, n_extra: int = 15) -> list:
    """
    Diverse simplex starts. v_off must stay below V_RESET (effective 1.2 V after
    Yao gap correction).
    """
    v_reset_cap = V_RESET - 0.05
    seeds = [
        _stage1_init(),
        np.array([4.3, 4.3, -0.4, 0.8, 3.0, 5.0]),
        np.array([4.0, 4.0, -0.5, 0.9, 2.5, 6.0]),
        np.array([3.8, 3.8, -0.3, 1.0, 2.0, 7.0]),
        np.array([4.7, 4.2, -0.6, 0.7, 4.0, 5.0]),
        np.array([5.0, 3.7, -0.4, 0.85, 3.0, 8.0]),
        np.array([4.2, 4.8, -0.8, 0.45, 2.0, 3.0]),
        np.array([3.9, 4.5, -0.35, 0.95, 3.5, 4.5]),
    ]
    for _ in range(n_extra):
        seeds.append(np.array([
            rng.uniform(3.6, 5.5),
            rng.uniform(3.6, 5.5),
            rng.uniform(-1.2, -0.15),
            rng.uniform(0.25, v_reset_cap),
            rng.uniform(1.0, 8.0),
            rng.uniform(1.0, 10.0),
        ], dtype=float))
    return seeds


def stage1_fit(ms, maxiter: int = 100, n_starts: int = 12,
               enable_selector: bool = False) -> dict:
    print("=" * 70)
    print("STAGE 1 - Deterministic VTEAM shape fit (noise OFF)")
    print(f"  Active groups (uA): {C.ACTIVE_TARGETS_UA.tolist()}")
    print(f"  V_SET_eff={V_SET:.2f} V (terminal {V_SET_TERMINAL} + gap {V_SET_GAP})")
    print(f"  V_RESET_eff={V_RESET:.2f} V (terminal {V_RESET_TERMINAL} - gap {V_RESET_GAP})")
    print(f"  Gates:  mean rel. err < {STAGE1_MAX_MEAN_REL_ERROR*100:.0f}%")
    print(f"          AND every group < {STAGE1_MAX_PER_GROUP_REL_ERROR*100:.0f}%")
    print(f"  Multi-start Nelder-Mead: {n_starts} starts")
    print(f"  1T1R selector: {'ON' if enable_selector else 'OFF'}")
    print("=" * 70)

    eval_count = [0]
    global_best = {"loss": np.inf, "x": None, "totals": None, "start_idx": -1}

    def cost(x):
        if x[2] >= -0.05 or x[3] <= 0.05:
            return 1e6
        if x[2] <= V_SET or x[3] >= V_RESET - 0.02:
            return 1e6
        if x[4] < 0.1 or x[5] < 0.1 or x[0] < 3.5 or x[1] < 3.5:
            return 1e6
        if x[0] > 12.0 or x[1] > 12.0 or x[4] > 12.0 or x[5] > 12.0:
            return 1e6

        k_on, k_off, v_on, v_off, alpha_on, alpha_off = _decode_stage1(x)
        params = make_base_params(
            ms,
            enable_variability=False,
            k_on=k_on, k_off=k_off,
            v_on=v_on, v_off=v_off,
            alpha_on=alpha_on, alpha_off=alpha_off,
            enable_selector=enable_selector,
        )
        try:
            _, _, totals = simulate_pulse_counts(ms, params, noisy_read=False)
        except Exception as e:
            print(f"  [eval error] {e}")
            return 1e6

        loss = pulse_count_loss(totals)
        eval_count[0] += 1
        if loss < global_best["loss"]:
            global_best["loss"] = loss
            global_best["x"] = x.copy()
            global_best["totals"] = totals.copy()
            mre = mean_relative_pulse_error(totals)
            mx = max_relative_pulse_error(totals)
            print(
                f"  eval {eval_count[0]:4d}  loss={loss:.4f}  "
                f"mean={mre*100:.1f}%  max={mx*100:.1f}%  "
                f"totals={np.round(totals, 1)}"
            )
        return loss

    rng = np.random.default_rng(42)
    seeds = _stage1_multistart_seeds(rng, n_extra=max(0, n_starts - 8))[:n_starts]

    t0 = time.time()
    for si, x0 in enumerate(seeds):
        print(f"\n--- start {si+1}/{len(seeds)}  x0={np.round(x0, 3)} ---")
        local_best_before = global_best["loss"]
        minimize(
            cost,
            x0,
            method="Nelder-Mead",
            options={"maxiter": maxiter, "xatol": 1e-3, "fatol": 1e-4, "disp": False},
        )
        if global_best["loss"] < local_best_before:
            global_best["start_idx"] = si
            print(f"  * new global best from start {si+1}")
    elapsed = time.time() - t0

    x_best = global_best["x"]
    if x_best is None:
        raise RuntimeError("Stage 1 multi-start produced no valid parameter set.")

    k_on, k_off, v_on, v_off, alpha_on, alpha_off = _decode_stage1(x_best)
    params = make_base_params(
        ms,
        enable_variability=False,
        k_on=k_on, k_off=k_off,
        v_on=v_on, v_off=v_off,
        alpha_on=alpha_on, alpha_off=alpha_off,
        enable_selector=enable_selector,
    )
    set_c, reset_c, totals = simulate_pulse_counts(ms, params, noisy_read=False)
    mre = mean_relative_pulse_error(totals)
    mx = max_relative_pulse_error(totals)
    per_g = per_group_relative_errors(totals)
    loss = pulse_count_loss(totals)

    print(f"\nStage 1 finished in {elapsed:.1f}s  ({eval_count[0]} evals, "
          f"best from start {global_best['start_idx']+1})")
    print(f"  k_on={k_on:.4e}  k_off={k_off:.4e}")
    print(f"  v_on={v_on:.4f}  v_off={v_off:.4f}")
    print(f"  alpha_on={alpha_on:.3f}  alpha_off={alpha_off:.3f}")
    print(f"  loss={loss:.4f}  mean_rel_err={mre*100:.2f}%  max_rel_err={mx*100:.2f}%")
    print(f"  sim totals : {np.round(totals, 1)}")
    print(f"  Yao totals : {C.ACTIVE_YAO_TOTAL}")
    print("  per-group relative error:")
    for t_ua, e, sim, ref in zip(C.ACTIVE_TARGETS_UA, per_g, totals, C.ACTIVE_YAO_TOTAL):
        flag = " <-- FAIL" if e >= STAGE1_MAX_PER_GROUP_REL_ERROR else ""
        print(f"    {t_ua:.1f} uA: sim={sim:.1f}  Yao={ref:.1f}  rel={e*100:.1f}%{flag}")

    return {
        "k_on": k_on,
        "k_off": k_off,
        "v_on": v_on,
        "v_off": v_off,
        "alpha_on": alpha_on,
        "alpha_off": alpha_off,
        "enable_selector": enable_selector,
        "selector_type": 1,
        "selector_v_gate": 1.8,
        "set_counts": set_c,
        "reset_counts": reset_c,
        "total_counts": totals,
        "mean_rel_error": mre,
        "max_rel_error": mx,
        "per_group_rel_error": per_g,
        "loss": loss,
        "n_evals": eval_count[0],
        "elapsed_s": elapsed,
        "n_starts": len(seeds),
        "best_start_idx": int(global_best["start_idx"]),
    }


def stage1_gate(stage1: dict) -> bool:
    mre = stage1["mean_rel_error"]
    mx = stage1.get("max_rel_error", max_relative_pulse_error(stage1["total_counts"]))
    per_g = stage1.get("per_group_rel_error", per_group_relative_errors(stage1["total_counts"]))
    mean_ok = mre < STAGE1_MAX_MEAN_REL_ERROR
    per_ok = bool(np.all(per_g < STAGE1_MAX_PER_GROUP_REL_ERROR))
    ok = mean_ok and per_ok

    print("\n" + "=" * 70)
    print("STAGE 1 GATE")
    print(f"  mean rel. err  = {mre*100:.2f}%  "
          f"({'PASS' if mean_ok else 'FAIL'} < {STAGE1_MAX_MEAN_REL_ERROR*100:.0f}%)")
    print(f"  max  rel. err  = {mx*100:.2f}%  "
          f"({'PASS' if per_ok else 'FAIL'} every group < {STAGE1_MAX_PER_GROUP_REL_ERROR*100:.0f}%)")
    if not per_ok:
        worst_i = int(np.argmax(per_g))
        print(f"  worst group: {C.ACTIVE_TARGETS_UA[worst_i]:.1f} uA  "
              f"({per_g[worst_i]*100:.1f}%) — HRS-boundary / deep-RESET regime")
    if ok:
        print("  RESULT: PASSED (mean + per-group)")
    else:
        print("  RESULT: FAILED")
        print("  Do NOT proceed to noise fitting to paper over a shape miss.")
        if mean_ok and not per_ok:
            print("  Mean gate alone would have passed — per-group gate caught the hide.")
    print("=" * 70)
    return ok


# ---------------------------------------------------------------------------
# Stage 2 — noise magnitude fit → 108 nA (experiment C anchor)
# ---------------------------------------------------------------------------

def _collect_errors(ms, shape: dict, sigma_d2d: float, sigma_c2c: float,
                    n_per_group: int, rng: np.random.Generator) -> np.ndarray:
    errors = []
    for t_a, t_ua in zip(C.ACTIVE_TARGETS_A, C.ACTIVE_TARGETS_UA):
        for _ in range(n_per_group):
            params = make_base_params(
                ms,
                enable_variability=True,
                sigma_d2d=sigma_d2d,
                sigma_c2c=sigma_c2c,
                k_on=shape["k_on"],
                k_off=shape["k_off"],
                v_on=shape["v_on"],
                v_off=shape["v_off"],
                alpha_on=shape["alpha_on"],
                alpha_off=shape["alpha_off"],
                enable_selector=bool(shape.get("enable_selector", False)),
                selector_type=int(shape.get("selector_type", 1)),
                selector_v_gate=float(shape.get("selector_v_gate", 1.8)),
            )
            device = ms.PhysicsEngine(params)
            # D2D already applied in constructor; start from sampled w_init if variability on
            w0 = float(device.params().w_init) if hasattr(device.params(), "w_init") else W_INIT_LRS
            # After construction, active w_init may differ; use device.w() after reset
            device.reset()
            w0 = float(device.w())
            res = write_verify_current(
                device, float(t_a), group_ua=float(t_ua), w_init=w0, noisy_read=True
            )
            errors.append(res.final_error)
    return np.asarray(errors, dtype=float)


def stage2_fit(ms, shape: dict, n_per_group: int = 200, maxiter: int = 40) -> dict:
    print("\n" + "=" * 70)
    print("STAGE 2 - Noise magnitude fit (shape frozen)")
    print(f"  Anchor: experiment (C) assumed Gaussian std = {GAUSSIAN_STD_NA} nA")
    print(f"  CAUTION: 108 nA is from array-scale weight-transfer, NOT experiment (A).")
    print(f"  N = {n_per_group} devices/group")
    print("  NOTE: engine C2C uses sigma_c2c*sqrt(dt); at dt=50ns the fitted")
    print("        sigma_c2c magnitude is larger than for ms-scale pulses.")
    print("=" * 70)

    # Optimize in log-space:
    #   x[0] = log10(sigma_d2d)
    #   x[1] = log10(sigma_c2c)
    eval_count = [0]
    best = {"loss": np.inf, "x": None, "std_na": None}

    def cost(x):
        sigma_d2d = float(10.0 ** np.clip(x[0], -4.0, -0.7))  # ~1e-4 .. 0.2
        sigma_c2c = float(10.0 ** np.clip(x[1], -2.0, 3.0))    # ~0.01 .. 1000
        errors = _collect_errors(ms, shape, sigma_d2d, sigma_c2c, n_per_group, None)
        std_na = float(np.std(errors) * 1e9)  # A -> nA
        loss = (std_na - GAUSSIAN_STD_NA) ** 2
        eval_count[0] += 1
        if loss < best["loss"]:
            best["loss"] = loss
            best["x"] = np.array([sigma_d2d, sigma_c2c])
            best["std_na"] = std_na
            print(
                f"  eval {eval_count[0]:3d}  sigma_D2D={sigma_d2d:.5f}  "
                f"sigma_C2C={sigma_c2c:.3f}  std={std_na:.1f} nA  loss={loss:.1f}"
            )
        return loss

    # Seed near basin where residual + rare outliers ≈ 108 nA
    x0 = np.array([-2.3, 0.7], dtype=float)  # sigma_d2d≈0.005, sigma_c2c≈5
    t0 = time.time()
    minimize(
        cost,
        x0,
        method="Nelder-Mead",
        options={"maxiter": maxiter, "xatol": 1e-2, "fatol": 20.0, "disp": True},
    )
    elapsed = time.time() - t0

    sigma_d2d, sigma_c2c = best["x"] if best["x"] is not None else (0.005, 5.0)
    # Average several final draws — write-verify noise makes single-shot std noisy
    finals = [
        float(np.std(_collect_errors(ms, shape, float(sigma_d2d), float(sigma_c2c), n_per_group, None)) * 1e9)
        for _ in range(3)
    ]
    std_na = float(np.mean(finals))

    print(f"\nStage 2 finished in {elapsed:.1f}s")
    print(f"  sigma_D2D={sigma_d2d:.5f}  sigma_C2C={sigma_c2c:.4f}")
    print(f"  simulated error std = {std_na:.1f} nA  (3-run mean; target {GAUSSIAN_STD_NA} nA)")
    print(f"  per-run stds (nA): {np.round(finals, 1)}")
    if best.get("std_na") is not None:
        print(f"  best-eval std during search = {best['std_na']:.1f} nA")

    return {
        "sigma_d2d": float(sigma_d2d),
        "sigma_c2c": float(sigma_c2c),
        "simulated_error_std_nA": std_na,
        "target_std_nA": GAUSSIAN_STD_NA,
        "target_source": "Yao et al. experiment (C) ResNet/CIFAR-10 weight-transfer "
                         "assumed Gaussian N(0, 108 nA) - NOT measured from experiment (A)",
        "n_per_group": n_per_group,
        "n_evals": eval_count[0],
        "elapsed_s": elapsed,
        "c2c_timescale_note": (
            "sigma_c2c is the engine SDE coefficient (w += sigma*sqrt(dt)*xi). "
            "At pulse_width=50ns its numeric value is much larger than for ms pulses."
        ),
    }


def plot_pulse_count_fit(stage1: dict) -> Path:
    import matplotlib.pyplot as plt

    plots = ensure_plots_dir()
    path = plots / "pulse_count_fit.png"

    x = C.ACTIVE_TARGETS_UA
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharex=True)

    axes[0].plot(x, C.ACTIVE_YAO_SET, "o-", label="Yao (A)", color="C0")
    axes[0].plot(x, stage1["set_counts"], "s--", label="Sim", color="C1")
    axes[0].set_title("SET pulses")
    axes[0].set_ylabel("Avg pulses")

    axes[1].plot(x, C.ACTIVE_YAO_RESET, "o-", label="Yao (A)", color="C0")
    axes[1].plot(x, stage1["reset_counts"], "s--", label="Sim", color="C1")
    axes[1].set_title("RESET pulses")

    axes[2].plot(x, C.ACTIVE_YAO_TOTAL, "o-", label="Yao (A)", color="C0")
    axes[2].plot(x, stage1["total_counts"], "s--", label="Sim", color="C1")
    axes[2].set_title("Total pulses")

    for ax in axes:
        ax.set_xlabel("Target current (uA)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"Stage-1 pulse-count fit  (mean rel. err = {stage1['mean_rel_error']*100:.1f}%)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Wrote {path}")
    return path


def _stage1_blocks(stage1: dict) -> tuple:
    per_g = np.asarray(stage1["per_group_rel_error"]).tolist()
    shape_block = {
        "k_on": stage1["k_on"],
        "k_off": stage1["k_off"],
        "v_on": stage1["v_on"],
        "v_off": stage1["v_off"],
        "alpha_on": stage1["alpha_on"],
        "alpha_off": stage1["alpha_off"],
        "R_on": R_ON,
        "R_off": R_OFF,
        "enable_selector": stage1["enable_selector"],
        "selector_type": stage1["selector_type"],
        "selector_v_gate": stage1["selector_v_gate"],
    }
    stage1_block = {
        "mean_rel_error": stage1["mean_rel_error"],
        "max_rel_error": stage1["max_rel_error"],
        "per_group_rel_error": per_g,
        "per_group_targets_uA": C.ACTIVE_TARGETS_UA.tolist(),
        "loss": stage1["loss"],
        "sim_set": stage1["set_counts"].tolist(),
        "sim_reset": stage1["reset_counts"].tolist(),
        "sim_totals": stage1["total_counts"].tolist(),
        "yao_set": C.ACTIVE_YAO_SET.tolist(),
        "yao_reset": C.ACTIVE_YAO_RESET.tolist(),
        "yao_totals": C.ACTIVE_YAO_TOTAL.tolist(),
        "gate_mean_threshold": STAGE1_MAX_MEAN_REL_ERROR,
        "gate_per_group_threshold": STAGE1_MAX_PER_GROUP_REL_ERROR,
        "n_starts": stage1["n_starts"],
        "best_start_idx": stage1["best_start_idx"],
        "n_evals": stage1["n_evals"],
        "enable_selector": stage1["enable_selector"],
        "scoped_exclude_05ua": C.SCOPED_EXCLUDE_05UA,
    }
    return shape_block, stage1_block


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--starts", type=int, default=12)
    parser.add_argument("--maxiter", type=int, default=60)
    parser.add_argument(
        "--selector", action="store_true",
        help="Enable engine 1T1R (legacy test; prefer Yao V_mem gap which is always on)",
    )
    args = parser.parse_args()

    ms = bootstrap_memristorsim()
    apply_group_scope(exclude_05ua=False)  # full 8-group attempt first

    print("\n*** STEP 1a: Yao fixed V_mem gap (SET -0.5 / RESET -0.6 from |V_term|) ***\n")
    stage1 = stage1_fit(
        ms, maxiter=args.maxiter, n_starts=args.starts, enable_selector=args.selector
    )
    plot_pulse_count_fit(stage1)
    shape_block, stage1_block = _stage1_blocks(stage1)

    pulse_block = {
        "V_SET_terminal": V_SET_TERMINAL,
        "V_RESET_terminal": V_RESET_TERMINAL,
        "V_SET_eff": V_SET,
        "V_RESET_eff": V_RESET,
        "V_SET_gap": V_SET_GAP,
        "V_RESET_gap": V_RESET_GAP,
        "V_READ": V_READ,
        "pulse_width_s": 50e-9,
        "margin_A": MARGIN_A,
        "max_pulses": MAX_PULSES,
        "voltage_model": (
            "Literature-fixed Yao DC-vs-pulse gap: "
            "V_SET_eff = V_term + 0.5, V_RESET_eff = V_term - 0.6 (magnitude reduced). "
            "NOT a fitted transistor."
        ),
    }

    if not stage1_gate(stage1):
        # Hard stop: one scoped refit excluding 0.5 uA — then no more parameter hunting
        print("\n*** HARD STOP: full 8-group fit failed after Yao V_mem correction. ***")
        print("*** Scoping call: exclude 0.5 uA; validate 1.0–4.0 uA only. One refit. ***\n")
        apply_group_scope(exclude_05ua=True)
        stage1 = stage1_fit(
            ms, maxiter=args.maxiter, n_starts=args.starts, enable_selector=args.selector
        )
        plot_pulse_count_fit(stage1)
        shape_block, stage1_block = _stage1_blocks(stage1)
        stage1_block["scoping_note"] = (
            "0.5 uA excluded after Yao V_mem correction failed per-group gate. "
            "Validated model covers 1.0–4.0 uA only. No further parameter search."
        )

        if not stage1_gate(stage1):
            stage1_block["gate_passed"] = False
            save_json(
                CALIBRATED_PARAMS_PATH,
                {
                    "status": "stage1_failed_even_scoped",
                    "shape": shape_block,
                    "stage1": stage1_block,
                    "deploy": {
                        "R_on": C.DEPLOY_R_ON,
                        "R_off": C.DEPLOY_R_OFF,
                        "groups_uA": C.ACTIVE_TARGETS_UA.tolist(),
                    },
                    "pulse_conditions": pulse_block,
                },
            )
            print("Scoped 1.0–4.0 uA fit ALSO failed gates. Stop. See calibrated_params.json.")
            sys.exit(1)

        print("\nScoped Stage 1 PASSED (1.0–4.0 uA). Proceeding to Stage 2.\n")
    else:
        apply_group_scope(exclude_05ua=False)
        print("\nFull 8-group Stage 1 PASSED with Yao V_mem correction.\n")

    noise = stage2_fit(ms, stage1, n_per_group=200, maxiter=30)
    stage1_block["gate_passed"] = True
    shape_block["note"] = (
        "Shape parameters FIT to Yao experiment (A) pulse counts "
        f"({'1.0–4.0 uA scoped' if C.SCOPED_EXCLUDE_05UA else 'all 8 groups'}); "
        "not independently measured for the Yao device."
    )

    out = {
        "status": "ok",
        "shape": shape_block,
        "noise": {
            "enable_variability": True,
            "sigma_d2d": noise["sigma_d2d"],
            "sigma_c2c": noise["sigma_c2c"],
            "sigma_d2d_maps_to": ["sigma_k_on", "sigma_w_init"],
            "simulated_error_std_nA": noise["simulated_error_std_nA"],
            "target_std_nA": noise["target_std_nA"],
            "target_source": noise["target_source"],
            "rtn": "omitted for this pilot",
            "verify_read": "noisy (engine 5% via update+i) for Stage 2 / Monte Carlo",
        },
        "stage1": stage1_block,
        "stage2": {
            "n_per_group": noise["n_per_group"],
            "elapsed_s": noise["elapsed_s"],
        },
        "deploy": {
            "R_on": C.DEPLOY_R_ON,
            "R_off": C.DEPLOY_R_OFF,
            "groups_uA": C.ACTIVE_TARGETS_UA.tolist(),
            "scoped_exclude_05ua": C.SCOPED_EXCLUDE_05UA,
            "note": (
                "MNIST weight→G mapping MUST use these deploy R_on/R_off "
                "(characterized window), not Stage-1 sanity R_on/R_off."
            ),
        },
        "pulse_conditions": pulse_block,
        "experiments_disclaimer": {
            "A": "8-group pulse-count table - Stage-1 calibration target",
            "B": "32-state - source of pulse cap 500 only",
            "C": "ResNet/CIFAR weight-transfer - source of 108 nA Gaussian assumption only",
        },
    }
    save_json(CALIBRATED_PARAMS_PATH, out)
    print(f"\nWrote {CALIBRATED_PARAMS_PATH}")
    print(f"Deploy window: R_on={C.DEPLOY_R_ON:.0f}  R_off={C.DEPLOY_R_OFF:.0f}  "
          f"groups={C.ACTIVE_TARGETS_UA.tolist()}")
    print("Calibration complete.")


if __name__ == "__main__":
    main()
