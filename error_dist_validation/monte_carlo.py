"""
Large-N Monte Carlo programming-error distribution.

Uses calibrated VTEAM + noise parameters. Simulates N=2000 devices per
*active* current group (may exclude 0.5 uA under hard-stop scoping).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from error_dist_validation.common import (
    ERROR_SAMPLES_CSV_PATH,
    ERROR_SAMPLES_PATH,
    V_RESET,
    V_SET,
    apply_group_scope,
    bootstrap_memristorsim,
    load_calibrated_params,
    params_from_dict,
    write_verify_current,
)


def run_monte_carlo(n_per_group: int = 2000, seed: int = 0) -> pd.DataFrame:
    ms = bootstrap_memristorsim()
    calib = load_calibrated_params()
    if calib.get("status") != "ok":
        raise RuntimeError(
            f"Calibration status is '{calib.get('status')}'. "
            "Stage-1 gate must pass before Monte Carlo."
        )

    deploy = calib.get("deploy", {})
    groups_ua = np.asarray(deploy.get("groups_uA", [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]), dtype=float)
    apply_group_scope(exclude_05ua=bool(deploy.get("scoped_exclude_05ua", True)))
    groups_a = groups_ua * 1e-6

    rows = []
    t0 = time.time()
    total_devices = n_per_group * len(groups_ua)
    done = 0

    print("=" * 70)
    print(f"PHASE 2 - Monte Carlo  N={n_per_group}/group  ({total_devices} devices)")
    print(f"  Groups (uA): {groups_ua.tolist()}")
    print(f"  V_SET_eff={V_SET:.2f}  V_RESET_eff={V_RESET:.2f}")
    print("  Noise: D2D + C2C ON; RTN omitted; verify uses NOISY reads (engine 5%)")
    print("=" * 70)

    for t_a, t_ua in zip(groups_a, groups_ua):
        group_t0 = time.time()
        for i in range(n_per_group):
            params = params_from_dict(ms, calib, enable_variability=True)
            device = ms.PhysicsEngine(params)
            device.reset()
            w0 = float(device.w())
            res = write_verify_current(
                device, float(t_a), group_ua=float(t_ua), w_init=w0,
                noisy_read=True, v_set=V_SET, v_reset=V_RESET,
            )
            rows.append(res.as_row())
            done += 1
            if (i + 1) % 500 == 0:
                print(
                    f"  group {t_ua:.1f} uA  {i+1}/{n_per_group}  "
                    f"({100*done/total_devices:.0f}% overall)"
                )
        elapsed_g = time.time() - group_t0
        stuck_rate = np.mean([r["stuck_flag"] for r in rows[-n_per_group:]])
        err_std_na = np.std([r["final_error"] for r in rows[-n_per_group:]]) * 1e9
        print(
            f"  group {t_ua:.1f} uA done in {elapsed_g:.1f}s  "
            f"stuck={stuck_rate*100:.1f}%  err_std={err_std_na:.1f} nA"
        )

    df = pd.DataFrame(rows)
    print(f"\nTotal wall time: {time.time()-t0:.1f}s  rows={len(df)}")
    return df


def save_samples(df: pd.DataFrame) -> Path:
    try:
        df.to_parquet(ERROR_SAMPLES_PATH, index=False)
        print(f"Wrote {ERROR_SAMPLES_PATH}")
        return ERROR_SAMPLES_PATH
    except Exception as e:
        print(f"Parquet write failed ({e}); falling back to CSV")
        df.to_csv(ERROR_SAMPLES_CSV_PATH, index=False)
        print(f"Wrote {ERROR_SAMPLES_CSV_PATH}")
        return ERROR_SAMPLES_CSV_PATH


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Phase 2 Monte Carlo error samples")
    parser.add_argument("--n", type=int, default=2000, help="Devices per group (default 2000)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    df = run_monte_carlo(n_per_group=args.n, seed=args.seed)
    save_samples(df)
    print("Monte Carlo complete.")


if __name__ == "__main__":
    main()
