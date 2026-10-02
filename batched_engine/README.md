# Batched Device + Write-Verify Engine (Component #2)

Standalone PyTorch module that vectorizes the existing C++ VTEAM engine over
`(n_ensemble × n_cells)` and ports Component #4’s current-margin write-verify
protocol. **No new physics claim** — sole deliverable is *the same validated
results, faster*. Gate 2 is the proof; benchmarks are throughput only.

**No C++ changes. No IR-drop. No STE. No inter-cell thermal coupling.**

Physics lock (Step 0): [`PORTED_CONSTANTS.md`](PORTED_CONSTANTS.md)  
Calibrated params (do not re-fit): [`../error_dist_validation/calibrated_params.json`](../error_dist_validation/calibrated_params.json)  
Gate-2 CI source (rescoped 7-group C4 characterize): [`../error_dist_validation/fitted_distribution.json`](../error_dist_validation/fitted_distribution.json)

---

## Layout

```
batched_engine/
├── PORTED_CONSTANTS.md      # exact C++ formulas — read before changing physics
├── device.py                # RK4, Biolek, C2C, Sinh read, per-cell dT
├── write_verify.py          # batched current-margin loop
├── d2d.py                   # v1 D2D + v2 threshold-D2D stub (default off)
├── rtn.py                   # optional RTN Markov hook (default off)
├── benchmark.py             # §6 worst-case / typical throughput
├── bench_scaling.py         # E×N wall-clock table
├── requirements.txt
└── validation/
    ├── test_oracle_parity.py   # Gate 1
    ├── test_mc_parity.py       # Gate 2
    └── test_rtn_hook.py        # RTN default-off + formula smoke
```

---

## Quick start

```bash
# from repo root
pip install -r batched_engine/requirements.txt

# Gate 1 — deterministic parity vs C++ (must pass first)
python batched_engine/validation/test_oracle_parity.py

# Gate 2 — statistical parity vs Component #4 Phase-2 MC
python -c "from batched_engine.validation.test_mc_parity import main; main(n_per_group=2000, seed=0)"

# Optional smoke
python batched_engine/validation/test_rtn_hook.py

# Throughput (after both gates)
python batched_engine/benchmark.py
python batched_engine/bench_scaling.py
```

Requires the compiled `memristorsim` pybind module (same bootstrap as Component #4)
for Gate 1 only. Gate 2 is pure PyTorch against C4 artifacts.

---

## API

```python
from batched_engine import BatchedMemristorArray, load_calibrated_params

eng = BatchedMemristorArray(
    n_ensemble=E,
    n_cells=N,
    calibrated_params=load_calibrated_params(),
    device="cpu",              # "cuda" when available — see GPU note below
    enable_variability=True,   # D2D + C2C from calibrated JSON
    threshold_d2d=False,       # v2 stretch — uncalibrated if True
    enable_rtn=False,          # keep False for Gates / C4 parity
)
eng.reset()
result = eng.write_verify(I_target, max_pulses=500, noisy_read=True)
I = eng.read(V_read=0.2, noisy=True)
```

`WriteVerifyResult`: `final_current`, `final_x`, `set_pulses`, `reset_pulses`,
`energy`, `stuck_flag` — all `[E, N]`.

---

## Validation outcomes (frozen)

### Gate 1 — deterministic oracle parity

`E=N=1`, variability off, `noisy_read=False`. Trajectory / pulse counts / Sinh
current match `PhysicsEngine` within **1e-6 relative**.

### Gate 2 — statistical parity vs C4 Phase-2

Protocol: N=2000 × **7 groups** (1.0–4.0 µA; 0.5 µA excluded), noisy verify,
same `calibrated_params.json`. Comparison CIs loaded from
`fitted_distribution.json` → `reports[name=pooled]` (n=14000):

| Check | Batched result | Bound / reference |
| --- | --- | --- |
| Pulse means vs C4 MC | mean rel **0.7%**, max **2.0%** | &lt;20% mean, &lt;40% per-group |
| Pooled skew | **0.2976** | C4 **0.2967** CI **[0.2247, 0.3691]** |
| Excess kurtosis | **1.4911** | C4 **1.5822** CI **[1.3803, 1.7877]** |
| AIC winner | **gmm2** | must remain gmm2 |
| Stuck | **0%** | — |
| Wall time (write_verify) | **~0.4 s** for 14 000 cells (CPU) | — |

Do **not** compare Gate 2 to pre-scope 8-group skew/kurtosis numbers.

---

## Performance (CPU; GPU untested on build machine)

`torch.cuda.is_available() = False` when these numbers were recorded. The API
exposes `device="cuda"`; **log at least one GPU row before treating performance
work as done for Component #1.**

### Worst-case write-verify (`max_pulses=500`, unreachable target)

| Device | E | N | time (s) | cells/s |
| --- | ---: | ---: | ---: | ---: |
| cpu | 1 | 10 000 | ~1.3 | ~8k |
| cpu | 1 | 50 000 | ~4.8 | ~10k |

### E×N scaling (`max_pulses=100`, stuck budget — from `bench_scaling.py`)

| E | N | E×N | time (s) | cells/s | µs/cell | rel_tpc |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1 000 | 1 000 | 0.119 | 8 383 | 119.3 | 1.00 |
| 1 | 5 000 | 5 000 | 0.138 | 36 179 | 27.6 | 0.23 |
| 1 | 10 000 | 10 000 | 0.225 | 44 402 | 22.5 | 0.19 |
| 1 | 50 000 | 50 000 | 0.754 | 66 339 | 15.1 | 0.13 |
| 10 | 1 000 | 10 000 | 0.301 | 33 204 | 30.1 | 0.25 |
| 10 | 5 000 | 50 000 | 0.734 | 68 104 | 14.7 | 0.12 |
| 10 | 10 000 | 100 000 | 0.946 | 105 664 | 9.5 | 0.08 |
| 50 | 1 000 | 50 000 | 0.766 | 65 296 | 15.3 | 0.13 |
| 50 | 5 000 | 250 000 | 1.872 | 133 530 | 7.5 | 0.06 |

`rel_tpc` = per-cell time / baseline(`E=1,N=1000`). Values &lt; 1 ⇒ **sublinear**
(amortized overhead), not superlinear.

---

## Intentional Step-0 corrections vs the C2 spec placeholders

| Spec placeholder | Actual engine (ported) |
| --- | --- |
| `a1*sinh(b1*V)` two-branch | Single-γ Sinh, `γ=2.0` |
| Additive Normal D2D on `k` | Log-normal `k · 10^(σ·N)` |
| Classic Biolek `stp(−i)` | Voltage-branch windows `1−w^8` / `1−(w−1)^8` |
| `.abs().pow().sign()` | Raw `pow` (safe under **v1** global thresholds only) |

Raw `pow` stops being safe if **v2 threshold-D2D** is enabled (per-cell `v_on`/`v_off`
can flip `(v/v_th−1)` sign under fractional α). See comment in `device.py`.

---

## Caveats

1. **v1:** `v_on` / `v_off` are global scalars (matches C4). Threshold D2D is an
   uncalibrated stretch behind `threshold_d2d=True`.
2. **Energy** uses midpoint-current approximation — order-of-magnitude, not
   bit-matched to C++ `m_power`.
3. **RTN** defaults off (`enable_rtn=False`), consistent with C4.
4. **`dT`** is per-cell self-heating (GUI “Temp Rise”), not Component #5 coupling.
5. Do not report a benchmark as success if Gate 2 has not passed.

---

## Out of scope

- Crossbar IR-drop / electrical coupling  
- STE / autograd (Component #1)  
- Inter-cell thermal coupling (Component #5)  
- Any change to the C++ engine  
- Custom CUDA kernels  
