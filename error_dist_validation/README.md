# Error Distribution Validation Pilot

Standalone Python pilot that calibrates the existing `memristorsim` VTEAM engine against Yao et al. (*Nature* 577, 2020) pulse-count data, generates a programming-error distribution, tests non-Gaussianity, and checks whether the true error shape changes MNIST deployment accuracy versus a moment-matched Gaussian.

**No C++ changes. No batching. No STE pipeline.**

**Definitive write-up of Steps 1–4:** [`DEFINITIVE_RESULTS.md`](DEFINITIVE_RESULTS.md)

## Do not conflate these three Yao experiments

| Label | What it is | How we use it |
| --- | --- | --- |
| **(A)** | 8-group pulse-count table (Ext. Data Fig. 8) | Stage-1 calibration target only |
| **(B)** | 32-state characterization | Pulse cap = 500 only |
| **(C)** | ResNet/CIFAR-10 weight-transfer | Assumed `N(0, 108 nA)` — Stage-2 noise-magnitude anchor only |

## Stage-1 hard gates (both required)

Stage 1 fits deterministic VTEAM shape parameters with **noise off**:

1. Mean relative pulse-count error **&lt; 20%** across active groups
2. **Every** active group relative error **&lt; 40%**

Multi-start Nelder-Mead. If either gate fails: **do not** tune noise.

### Final status (hard-stop protocol)

| Attempt | Result |
| --- | --- |
| Full 8 groups + Yao V_mem gap (SET −1.5 V / RESET +1.2 V) | **FAIL** — 0.5 µA ~63% |
| Stock 1T1R selector | **FAIL** — 0.5 µA ~65% |
| **Scoped 1.0–4.0 µA** (exclude 0.5 µA) | **PASS** — mean 14.3%, max 31.1% |

**True Claim:** VTEAM+Biolek is validated for **1.0–4.0 µA** only. Deploy R window = **50–200 kΩ**.

## Verify-read noise

| Phase | Read | Why |
| --- | --- | --- |
| Stage 1 | `calculate_current(0.2)` noiseless | Deterministic shape fit |
| Stage 2 / Monte Carlo | `update` + `i()` (engine **5%** read noise) | Hardware verify imprecision |

## Characterization (scoped MC)

- N = 14 000 devices (2000 × 7 groups); pooled skew **0.30**, excess kurtosis **1.58**
- **Winner: 2-GMM** (KS p ≈ 0.10)

## Phase 4 — definitive MNIST (transition scale)

Sweep (plain FP32, no HA-FT): scales 0.03→0.7. Only **0.2** sits in the 40–90% band
(mean ~68%). Full Gaussian vs gmm2 at that scale:

| Metric | Value |
| --- | ---: |
| Clean / zero-noise | 0.972 |
| Gaussian mean acc | 0.687 ± 0.086 |
| True-shape mean acc | 0.710 ± 0.067 |
| Mean Δ (true − gauss) | **+0.022** · CI [−0.017, +0.067] |
| Wilcoxon p | **0.465** — no significant difference |
| Collapse (&lt;80%) | Gauss 93% / true 90% · McNemar p=1 |

**Null shape effect** in the first non-degenerate regime. Does not justify STE/batching.
Sanity: scale 0.015 → ~98% (pipeline OK); scale 1.0 → chance (magnitude saturation).

## Diagnostics

```bash
python error_dist_validation/dump_diagnostics.py
# → error_dist_validation/diagnostics_report.md
```

## Setup

```bash
cmake --build build --config Release
pip install -r error_dist_validation/requirements.txt
```

Requires `local_settings.py` with `MINGW_BIN` on Windows.

## Run order

```bash
python error_dist_validation/calibration.py
python error_dist_validation/monte_carlo.py
python error_dist_validation/characterize.py
python error_dist_validation/mnist_validation.py --sweep-then-transition
# log-spaced sweep → full Gaussian vs gmm2 at transition scale
python error_dist_validation/dump_diagnostics.py
```

## Caveats for any write-up

1. Shape parameters are **fit** to pulse counts (A), not independently measured for Yao’s device.
2. The 108 nA Stage-2 target is from experiment **(C)** — cross-experiment limitation.
3. RTN omitted.
4. 0.5 µA is **out of scope** — disclose the 1.0–4.0 µA restriction.
5. Phase-4 null is under absolute ΔI→ΔG on 50–200 kΩ; a different weight↔G scheme could change the downstream result without changing the error distribution.
