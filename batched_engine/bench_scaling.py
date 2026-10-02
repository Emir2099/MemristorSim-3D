"""E×N scaling table for handoff confirmation"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from batched_engine.device import BatchedMemristorArray, load_calibrated_params


def main() -> None:
    calib = load_calibrated_params()
    cuda = torch.cuda.is_available()
    devices = ["cpu"] + (["cuda"] if cuda else [])

    combos = [
        (1, 1_000),
        (1, 5_000),
        (1, 10_000),
        (1, 50_000),
        (10, 1_000),
        (10, 5_000),
        (10, 10_000),
        (50, 1_000),
        (50, 5_000),
    ]
    max_pulses = 100  # fixed budget so E*N is the only variable

    for device in devices:
        print("=" * 78)
        print(
            f"device={device}  force_stuck=True  max_pulses={max_pulses}  "
            f"(unreachable I_target — fixed pulse budget)"
        )
        if device == "cpu" and not cuda:
            print("NOTE: torch.cuda.is_available()=False on this machine — no GPU datum.")
        print(
            f"{'E':>4} {'N':>6} {'ExN':>8} {'time_s':>10} "
            f"{'cells/s':>12} {'us/cell':>10} {'rel_tpc':>8}"
        )
        rows = []
        for E, N in combos:
            eng = BatchedMemristorArray(
                E,
                N,
                calibrated_params=calib,
                device=device,
                enable_variability=False,
                seed=0,
            )
            I = torch.full(
                (E, N), 100e-6, dtype=torch.float64, device=eng.device
            )
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            eng.write_verify(I, max_pulses=max_pulses, noisy_read=False)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
            cells = E * N
            rows.append((E, N, cells, elapsed))

        base_tpc = rows[0][3] / rows[0][2]
        for E, N, cells, elapsed in rows:
            tpc = elapsed / cells
            print(
                f"{E:4d} {N:6d} {cells:8d} {elapsed:10.4f} "
                f"{cells/elapsed:12.1f} {1e6*tpc:10.2f} {tpc/base_tpc:8.2f}"
            )
        print(
            "rel_tpc = per-cell time / baseline(E=1,N=1000); "
            "ideal linear scaling keeps rel_tpc ~ 1; >1 would be superlinear."
        )


if __name__ == "__main__":
    main()
