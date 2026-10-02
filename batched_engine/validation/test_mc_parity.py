"""
Gate 2 — statistical parity vs Component #4 Phase-2 Monte Carlo.

Rerun N=2000/group × 7 groups through the batched engine with the same
calibrated_params.json and noisy-read protocol. Pass criteria:

1. Mean pulse totals vs C4 Phase-2 MC: <20% mean, <40% per-group relative
2. Pooled skew / excess kurtosis within C4 bootstrap CIs
3. AIC-winning model still gmm2

If Gate 1 has not passed, do not trust this result.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Import torch before sklearn-heavy modules (Windows DLL init can fail otherwise).
import torch  # noqa: E402

from batched_engine.device import BatchedMemristorArray, load_calibrated_params  # noqa: E402
from error_dist_validation import common as c4_common  # noqa: E402
from error_dist_validation.common import (  # noqa: E402
    STAGE1_MAX_MEAN_REL_ERROR,
    STAGE1_MAX_PER_GROUP_REL_ERROR,
    apply_group_scope,
)

# Gate-2 CIs loaded from Component #4's rescoped characterize artifact on disk
# (error_dist_validation/fitted_distribution.json → reports[name=pooled]).
# Do NOT hardcode alternate pre-scope numbers.
def _load_c4_pooled_cis():
    import json
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / "error_dist_validation" / "fitted_distribution.json"
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    pooled = next(r for r in data["reports"] if r["name"] == "pooled")
    skew = pooled["shape"]["skewness"]
    kurt = pooled["shape"]["excess_kurtosis"]
    groups = data.get("groups_uA", [])
    return {
        "skew_point": float(skew["value"]),
        "skew_ci": (float(skew["ci95"][0]), float(skew["ci95"][1])),
        "kurt_point": float(kurt["value"]),
        "kurt_ci": (float(kurt["ci95"][0]), float(kurt["ci95"][1])),
        "n": int(pooled["n"]),
        "groups_uA": groups,
        "source": str(path),
    }


_C4 = _load_c4_pooled_cis()
C4_SKEW_CI = _C4["skew_ci"]
C4_KURT_CI = _C4["kurt_ci"]
C4_SKEW_POINT = _C4["skew_point"]
C4_KURT_POINT = _C4["kurt_point"]

# Phase-2 MC mean total pulses per group from error_dist_validation/error_samples.parquet
# (N=2000/group). Stage-1 noiseless sim_totals differ; Gate 2 compares to this MC oracle.
# Note: these do NOT pass Yao Stage-1 gates (noise changes programming paths) — C4's own
# Phase-2 would also fail a Yao pulse-count gate; parity is vs C4 MC, not vs Yao (A).
C4_MC_PULSE_MEANS = {
    1.0: 22.9210,
    1.5: 23.5300,
    2.0: 22.2875,
    2.5: 21.4200,
    3.0: 21.2315,
    3.5: 20.7790,
    4.0: 20.1255,
}


def run_batched_monte_carlo(
    n_per_group: int = 2000,
    seed: int = 0,
    device: str = "cpu",
) -> pd.DataFrame:
    calib = load_calibrated_params()
    if calib.get("status") != "ok":
        raise RuntimeError(f"Calibration status is '{calib.get('status')}'")

    deploy = calib.get("deploy", {})
    groups_ua = np.asarray(
        deploy.get("groups_uA", [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]), dtype=float
    )
    apply_group_scope(exclude_05ua=bool(deploy.get("scoped_exclude_05ua", True)))
    groups_a = groups_ua * 1e-6
    n_groups = len(groups_ua)

    print("=" * 70)
    print(f"GATE 2 — Batched Monte Carlo  N={n_per_group}/group  ({n_per_group * n_groups} devices)")
    print(f"  Groups (uA): {groups_ua.tolist()}")
    print(f"  Noise: D2D + C2C ON; RTN off; noisy verify reads")
    print("=" * 70)

    # Shape [E, N] = [n_per_group, n_groups] — one column per current group
    eng = BatchedMemristorArray(
        n_ensemble=n_per_group,
        n_cells=n_groups,
        calibrated_params=calib,
        device=device,
        enable_variability=True,
        threshold_d2d=False,
        seed=seed,
    )
    I_target = torch.tensor(
        groups_a, dtype=torch.float64, device=eng.device
    ).unsqueeze(0).expand(n_per_group, n_groups)

    t0 = time.time()
    res = eng.write_verify(I_target, noisy_read=True)
    elapsed = time.time() - t0
    print(f"  write_verify wall time: {elapsed:.1f}s")

    final_i = res.final_current.detach().cpu().numpy()
    set_p = res.set_pulses.detach().cpu().numpy()
    reset_p = res.reset_pulses.detach().cpu().numpy()
    stuck = res.stuck_flag.detach().cpu().numpy()
    targets = np.broadcast_to(groups_a.reshape(1, -1), (n_per_group, n_groups))
    group_col = np.broadcast_to(groups_ua.reshape(1, -1), (n_per_group, n_groups))

    for g in range(n_groups):
        totals = set_p[:, g] + reset_p[:, g]
        err_std_na = float(np.std(final_i[:, g] - groups_a[g]) * 1e9)
        print(
            f"  group {groups_ua[g]:.1f} uA  mean_pulses={totals.mean():.2f}  "
            f"stuck={stuck[:, g].mean()*100:.1f}%  err_std={err_std_na:.1f} nA"
        )

    return pd.DataFrame(
        {
            "group": group_col.ravel(),
            "final_current": final_i.ravel(),
            "set_pulses": set_p.ravel(),
            "reset_pulses": reset_p.ravel(),
            "stuck_flag": stuck.ravel(),
            "final_error": (final_i - targets).ravel(),
        }
    )


def check_pulse_gates(df: pd.DataFrame) -> Dict[str, Any]:
    """Compare mean pulse totals to Component #4 Phase-2 MC (not Yao Stage-1)."""
    apply_group_scope(exclude_05ua=True)
    groups = sorted(df["group"].unique())

    sim_totals = []
    ref_totals = []
    rel_errs = []
    for g in groups:
        sub = df[df["group"] == g]
        total = float((sub["set_pulses"] + sub["reset_pulses"]).mean())
        ref = float(C4_MC_PULSE_MEANS[float(g)])
        sim_totals.append(total)
        ref_totals.append(ref)
        rel_errs.append(abs(total - ref) / max(ref, 1e-12))

    mean_rel = float(np.mean(rel_errs))
    max_rel = float(np.max(rel_errs))
    # Same relative tolerances C4 used for Stage-1, applied to C4 MC baselines.
    passed = (
        mean_rel < STAGE1_MAX_MEAN_REL_ERROR
        and max_rel < STAGE1_MAX_PER_GROUP_REL_ERROR
    )
    print("\nPulse-count gate vs Component #4 Phase-2 MC means:")
    for g, s, r, e in zip(groups, sim_totals, ref_totals, rel_errs):
        print(f"  {g:.1f} uA  batched={s:.2f}  C4_MC={r:.2f}  rel_err={e*100:.1f}%")
    print(f"  mean_rel={mean_rel*100:.1f}%  max_rel={max_rel*100:.1f}%  pass={passed}")
    # Also report Yao comparison for context (not a pass/fail gate).
    yao = np.asarray(c4_common.ACTIVE_YAO_TOTAL, dtype=float)
    yao_rels = [
        abs(s - y) / max(y, 1e-12) for s, y in zip(sim_totals, yao)
    ]
    print(
        f"  (context) vs Yao mean_rel={np.mean(yao_rels)*100:.1f}% — "
        "C4 Phase-2 itself also fails Yao Stage-1 gates under noisy verify"
    )
    return {
        "sim_totals": sim_totals,
        "c4_mc_totals": ref_totals,
        "per_group_rel_error": rel_errs,
        "mean_rel_error": mean_rel,
        "max_rel_error": max_rel,
        "passed": passed,
    }


def check_distribution(df: pd.DataFrame) -> Dict[str, Any]:
    from error_dist_validation.characterize import fit_candidates, shape_stats

    errors = df["final_error"].to_numpy(dtype=float)
    shape = shape_stats(errors)
    skew = shape["skewness"]["value"]
    kurt = shape["excess_kurtosis"]["value"]

    skew_ok = C4_SKEW_CI[0] <= skew <= C4_SKEW_CI[1]
    kurt_ok = C4_KURT_CI[0] <= kurt <= C4_KURT_CI[1]

    print("\nPooled error-distribution moments:")
    print(
        f"  skew={skew:.4f}  (C4 point={C4_SKEW_POINT:.2f}, CI={list(C4_SKEW_CI)})  "
        f"in_CI={skew_ok}"
    )
    print(
        f"  excess_kurtosis={kurt:.4f}  (C4 point={C4_KURT_POINT:.2f}, "
        f"CI={list(C4_KURT_CI)})  in_CI={kurt_ok}"
    )
    print(f"  std={shape['std_nA']:.1f} nA")

    candidates = fit_candidates(errors)
    # Prefer lowest AIC among KS-pass; else lowest AIC
    ks_ok = [c for c in candidates if c.get("ks_pvalue", 0.0) >= 0.05]
    pool = ks_ok if ks_ok else candidates
    winner = min(pool, key=lambda c: c["aic"])
    winner_name = winner["name"]
    gmm_ok = winner_name == "gmm2"

    print("\nModel selection (pooled):")
    for c in candidates:
        print(
            f"  {c['name']:10s}  AIC={c['aic']:.1f}  KS_p={c.get('ks_pvalue', float('nan')):.4g}"
        )
    print(f"  winner={winner_name}  (expect gmm2)  pass={gmm_ok}")

    return {
        "skew": skew,
        "kurtosis": kurt,
        "skew_in_ci": skew_ok,
        "kurtosis_in_ci": kurt_ok,
        "winner": winner_name,
        "gmm2_wins": gmm_ok,
        "shape": shape,
        "passed": skew_ok and kurt_ok and gmm_ok,
    }


def main(n_per_group: int = 2000, seed: int = 0, device: str = "cpu") -> None:
    df = run_batched_monte_carlo(n_per_group=n_per_group, seed=seed, device=device)
    pulse = check_pulse_gates(df)
    dist = check_distribution(df)

    stuck = float(df["stuck_flag"].mean())
    print(f"\nStuck fraction: {stuck*100:.2f}%")

    all_pass = pulse["passed"] and dist["passed"]
    if not all_pass:
        reasons = []
        if not pulse["passed"]:
            reasons.append("pulse-count gate")
        if not dist["skew_in_ci"]:
            reasons.append("skew outside C4 CI")
        if not dist["kurtosis_in_ci"]:
            reasons.append("kurtosis outside C4 CI")
        if not dist["gmm2_wins"]:
            reasons.append(f"winner={dist['winner']} (not gmm2)")
        raise SystemExit(f"GATE 2 FAILED: {', '.join(reasons)}")

    out = Path(__file__).resolve().parent.parent / "gate2_mc_results.parquet"
    try:
        df.to_parquet(out, index=False)
        print(f"Wrote {out}")
    except Exception as e:
        csv = out.with_suffix(".csv")
        df.to_csv(csv, index=False)
        print(f"Parquet failed ({e}); wrote {csv}")

    print("\nGATE 2 PASSED — statistical parity with Component #4.")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-group", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args()
    main(n_per_group=args.n_per_group, seed=args.seed, device=args.device)
