"""
performance benchmarking for the batched engine.

Run only after Gates 1 and 2 pass. Reports cells/sec and E×N scaling;
does not claim success without those gates.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from batched_engine.device import BatchedMemristorArray, load_calibrated_params


def _bench_once(
    E: int,
    N: int,
    *,
    max_pulses: int,
    device: str,
    enable_variability: bool,
    target_ua: float = 2.0,
    force_stuck: bool = False,
) -> dict:
    calib = load_calibrated_params()
    eng = BatchedMemristorArray(
        E,
        N,
        calibrated_params=calib,
        device=device,
        enable_variability=enable_variability,
        seed=0,
    )
    # Unreachable target forces full max_pulses loop (worst-case training-step bound).
    t_ua = 100.0 if force_stuck else target_ua
    I_target = torch.full(
        (E, N), t_ua * 1e-6, dtype=torch.float64, device=eng.device
    )

    if device.startswith("cuda"):
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    res = eng.write_verify(I_target, max_pulses=max_pulses, noisy_read=False)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    cells = E * N
    return {
        "E": E,
        "N": N,
        "cells": cells,
        "device": device,
        "max_pulses": max_pulses,
        "force_stuck": force_stuck,
        "enable_variability": enable_variability,
        "elapsed_s": elapsed,
        "cells_per_sec": cells / elapsed if elapsed > 0 else float("inf"),
        "mean_set": float(res.set_pulses.mean().item()),
        "mean_reset": float(res.reset_pulses.mean().item()),
        "stuck_frac": float(res.stuck_flag.float().mean().item()),
    }


def main() -> None:
    calib = load_calibrated_params()
    print("=" * 70)
    print("Batched engine benchmark (Section 6)")
    print("  Prerequisite: Gate 1 + Gate 2 must already pass.")
    print(f"  calibrated status={calib.get('status')}")
    print("=" * 70)

    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")

    rows = []

    # Worst-case bound: unreachable target → all cells hit max_pulses=500
    for device in devices:
        for N in (10_000, 50_000):
            r = _bench_once(
                1,
                N,
                max_pulses=500,
                device=device,
                enable_variability=False,
                force_stuck=True,
            )
            rows.append(r)
            print(
                f"  {device:4s} WORST E=1 N={N:6d}  {r['elapsed_s']:.3f}s  "
                f"{r['cells_per_sec']:.0f} cells/s  "
                f"stuck={r['stuck_frac']*100:.1f}%  (max_pulses=500)"
            )

        # E=1 vs E=10 scaling at fixed N (converging targets, shorter cap)
        N = 5_000
        r1 = _bench_once(1, N, max_pulses=100, device=device, enable_variability=False)
        r10 = _bench_once(10, N, max_pulses=100, device=device, enable_variability=False)
        rows.extend([r1, r10])
        ratio = r10["elapsed_s"] / max(r1["elapsed_s"], 1e-12)
        print(
            f"  {device:4s} scaling N={N}: E=1 {r1['elapsed_s']:.3f}s -> "
            f"E=10 {r10['elapsed_s']:.3f}s  (time ratio={ratio:.2f}, expect ~10x on E*N)"
        )

        # Typical programming depth (reachable target, variability ON)
        r_typ = _bench_once(
            1, 20_000, max_pulses=500, device=device, enable_variability=True
        )
        rows.append(r_typ)
        print(
            f"  {device:4s} typical E=1 N=20000 variability ON  "
            f"{r_typ['elapsed_s']:.3f}s  {r_typ['cells_per_sec']:.0f} cells/s  "
            f"mean pulses~={r_typ['mean_set']+r_typ['mean_reset']:.1f}"
        )

    print("\nDone. Numbers are throughput reports only — Gate 2 is the correctness proof.")


if __name__ == "__main__":
    main()
