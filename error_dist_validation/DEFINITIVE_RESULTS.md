# Component 4 — Definitive results (Steps 1–4 hard-stop protocol)

Generated after the Yao V_mem-gap correction, honest 0.5 µA scoping, fresh MC/characterize,
and MNIST at `noise_scale=1.0` with the **characterized** deploy R window.

## Step 1 — Stage 1 calibration

**Voltage model (fixed, not fitted):**
- `V_SET_eff = −1.5 V` (terminal −2.0 + 0.5 V Yao DC–pulse gap)
- `V_RESET_eff = +1.2 V` (terminal +1.8 − 0.6 V)

**Full 8-group attempt:** FAILED per-group gate (0.5 µA still ~63% relative error; 3.5 µA also >40%).

**Hard-stop scoping:** exclude 0.5 µA. One refit on **1.0–4.0 µA only:**
- mean rel. err = **14.3%** PASS
- max rel. err = **31.1%** PASS

**Honest claim:** VTEAM+Biolek+Yao voltage gap is validated for **1.0–4.0 µA**, not below ~1 µA.

Deploy mapping window: `R_on = 50 kΩ`, `R_off = 200 kΩ` (4.0–1.0 µA @ 0.2 V).

## Step 2 — Monte Carlo + characterization (scoped)

- N = 2000 × 7 groups = 14 000 devices; noisy verify reads
- Pooled: skewness **0.30** [0.22, 0.37], excess kurtosis **1.58** [1.38, 1.79]
- Normality rejected; stuck = 0%
- **Winner: 2-GMM** (lowest AIC, KS p = **0.104** — passes α=0.05)

Do **not** reuse pre-scope skew/kurtosis numbers.

## Step 3–4 — MNIST (definitive = transition-scale comparison)

Pipeline sanity: `noise_scale=0.015` recovers ~98% (mapping OK). `noise_scale=1.0`
(+ HA-FT) collapses both arms to chance — magnitude saturation, not a bug.

### Noise-scale sweep (plain FP32, no HA-FT, 8 Gaussian probe seeds)

| scale | mean acc | collapse &lt;80% |
| ---: | ---: | ---: |
| 0.03 | 0.970 | 0% |
| 0.06 | 0.966 | 0% |
| 0.10 | 0.948 | 0% |
| **0.20** | **0.676** | 88% ← transition band |
| 0.35 | 0.214 | 100% |
| 0.50 | 0.142 | 100% |
| 0.70 | 0.131 | 100% |

### Phase 4 @ transition `noise_scale=0.2` (M=30, Gaussian vs gmm2)

| Item | Value |
| --- | ---: |
| Clean / zero-noise | 0.9715 / 0.9715 |
| Gaussian mean acc | 0.6872 ± 0.0862 |
| True-shape mean acc | 0.7095 ± 0.0669 |
| Mean Δ (true − gauss) | **+0.0223** |
| 95% CI Δ | [−0.0167, +0.0674] |
| Wilcoxon p | **0.465** |
| Collapse (&lt;80%) Gaussian | **93.3%** |
| Collapse (&lt;80%) true-shape | **90.0%** |
| McNemar (collapse) p | 1.0 |

### Interpretation (closed)

This is the first comparison run in a non-degenerate mid-accuracy regime. CI for Δ
covers zero; Wilcoxon and collapse McNemar are null. **No detectable shape effect**
of gmm2 vs moment-matched Gaussian on this CNN under the characterized mapping.
Step-2 non-Gaussianity of the programming-error distribution still stands.
**STE / batched pipeline is not justified by a positive Phase-4 effect.**
