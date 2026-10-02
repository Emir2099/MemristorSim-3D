# PORTED_CONSTANTS — Step 0 ground truth

**Do not write batched physics until this file exists and matches the C++ oracle.**
This document is extracted from `src/physics/Memristor.cpp` / `Memristor.h` and
Component (`error_dist_validation/`). Spec placeholders that disagree with
these formulas must be ignored.

Source commit context: calibrated params already validated against
this engine. This component ports them; it does not re-fit them.

---

## 1. VTEAM `dw/dt` + Biolek window (voltage-branch, not current-`stp`)

Canonical Biolek paper form `f = 1 − (x − stp(−i))^(2p)` is **NOT** what the
engine implements. VTEAM is voltage-driven; the C++ keys the window off the
SET/RESET **voltage branch**, with hardcoded exponent `8` (= `2p`, `p = 4`).

From `Memristor.cpp` `PhysicsEngine::get_dw_dt`:

```cpp
// RESET: v > v_off  (strict)
dw = k_off * pow((v / v_off) - 1.0, alpha_off);
dw *= (1.0 - pow(w - 1.0, 8.0));   // protects w → 0

// SET: v < v_on  (strict)
dw = k_on * pow((v / v_on) - 1.0, alpha_on);
dw *= (1.0 - pow(w, 8.0));         // protects w → 1

// else: dw = 0
```

Notes that burned earlier ports:

- Inequalities are **strict** (`>` / `<`), not `≥` / `≤`.
- Power law is raw `std::pow(base, alpha)` — **no** `.abs().pow().sign()`.
  For Yao pulses with `|V| > |threshold|`, bases are positive; do not “fix”
  undefined fractional powers of negative bases with abs/sign.
- Intermediate `w` is clamped to `[0, 1]` at the start of `get_dw_dt`.

### Thermal dissolution (included for fidelity)

```cpp
if (dT > T_critical) {
    dw += -abs(k_off) * ((dT - T_critical) / T_critical) * w;
}
```

Defaults: `T_critical = 5.0`, `theta_thermal = 0.01`, thermal time constant
`tau_thermal = 0.01` inside `update`. At Component #4 pulse width `dt = 50 ns`,
`dt/(dt+tau) ≈ 5e-6`, so `dT` barely moves and the thermal term is effectively
inert for Gate 1 / Gate 2 protocols. Still ported so trajectories match `update`.

### PyTorch equivalent (literal)

```python
# RESET mask: v > v_off
dw = k_off * ((v / v_off) - 1.0).pow(alpha_off) * (1.0 - (w - 1.0).pow(8.0))
# SET mask: v < v_on
dw = k_on  * ((v / v_on)  - 1.0).pow(alpha_on)  * (1.0 - w.pow(8.0))
```

---

## 2. RK4 stepping

One classic RK4 call per `update(dt, voltage)` — **no** sub-stepping of the
50 ns pulse.

```cpp
k1 = get_dw_dt(v, w0, dT);
k2 = get_dw_dt(v, w0 + 0.5*dt*k1, dT);
k3 = get_dw_dt(v, w0 + 0.5*dt*k2, dT);
k4 = get_dw_dt(v, w0 + dt*k3, dT);
return w0 + (dt/6)*(k1 + 2*k2 + 2*k3 + k4);
```

After RK4 (+ optional C2C): `w = clamp(w_new, 0, 1)`.

Selector: when `enable_selector=false` (calibrated scope), `v_mem = voltage`
directly — no 1T1R bisection.

---

## 3. C2C write noise (Euler–Maruyama)

```cpp
if (enable_variability) {
    w_new += sigma_c2c * sqrt(dt) * N(0, 1);
}
```

Batched port: apply **only where `active_mask` is true** (cell received a pulse
this step). Idle / converged cells must not randomly walk.

---

## 4. Read current (Sinh conduction) + 5% read noise

### Deterministic Sinh (`calculate_memristor_current`)

```cpp
i_on  = V / R_on;
i_off = sinh(gamma * V) / (R_off * sinh(gamma));  // gamma_sinh = 2.0 default
raw_i = w * i_on + (1 - w) * i_off;
// then I_compliance clamp
```

With `enable_selector=false`, `calculate_current(V) == calculate_memristor_current(V)`.

**Do not** use the spec placeholder `a1*sinh(b1*V) + a2*sinh(b2*V)`.

### Always-on read noise inside `update`

```cpp
noise = N(0,1) * (0.05 * raw_i);   // σ = 5% of raw_i
m_i   = raw_i + noise;
```

Component #4 verify protocol (`error_dist_validation/common.py`):

- `noisy=False` (Stage 1 / Gate 1): `device.calculate_current(v_read)` — no noise.
- `noisy=True` (MC / Gate 2): `device.update(1e-12, v_read)` then `device.i()` so
  the 5% noise attaches without meaningful state drift (`READ_DT_S = 1e-12`).

`READ_NOISE_FRAC = 0.05`.

---

## 5. D2D variability (v1)

From `apply_d2d_variability`:

```cpp
w_init += N(0,1) * sigma_w_init;          // Normal, then clamp01
k_on   *= pow(10.0, N(0,1) * sigma_k_on); // Log-normal
k_off  *= pow(10.0, N(0,1) * sigma_k_on); // same sigma
```

Component #4 maps `sigma_d2d` → both `sigma_k_on` and `sigma_w_init`.

**v1:** `v_on`, `v_off` stay **global scalars** from calibrated JSON.
**v2 stretch (off by default):** per-cell `v_on_i`, `v_off_i ~ N(v, σ_v)` with
**uncalibrated** placeholder `σ_v` — never used for Gate 1/2 published claims.

---

## 5b. RTN (optional, default off — Spec §4.2)

From `Memristor.cpp` / `Memristor.h`. **Not used by Component #4 or Gates 1/2.**

Two-state Markov chain per cell:

```cpp
// empty (0) → occupied (1)
p = 1 - exp(-dt / rtn_tau_c);
// occupied (1) → empty (0)
p = 1 - exp(-dt / rtn_tau_e);
```

Current factor (multiplicative, applied before I_compliance):

```cpp
factor = 1.0 + (rtn_state == 1 ? +0.5 : -0.5) * rtn_amplitude;
raw_i *= factor;
```

Defaults: `enable_rtn=false`, `rtn_amplitude=0.03`, `rtn_tau_c=rtn_tau_e=0.05 s`.

Batched API: `BatchedMemristorArray(..., enable_rtn=False)`. When enabled, state
advances on pulses (`dt=pulse_width`) and on noisy reads (`dt=1e-12`, matching
C4's subthreshold verify update). Keep off for published / Gate results.

---

## 6. Write-verify protocol (Component #4, not C++ `program_write_verify`)

Port `write_verify_current` in `error_dist_validation/common.py`:

1. Start from LRS `w_init = 1.0` (plus D2D on init when variability on).
2. Loop up to `max_pulses=500`:
   - Read at `V_READ=0.2`.
   - If `|I - I_target| ≤ margin` → stop (converged).
   - If `I < I_target - margin` → SET pulse `V_SET_eff=-1.5`, `dt=50 ns`.
   - Else → RESET pulse `V_RESET_eff=+1.2`, `dt=50 ns`.
3. Margin `±100 nA`.
4. Do **not** use C++ `program_write_verify` (w-target, 1 ms pulses, adaptive V).

Energy in the batched engine uses midpoint-current approximation
`Σ |V| · |I_mid(noiseless)| · dt` — order-of-magnitude, not bit-matched to
`update`'s noisy `m_power`.

---

## 7. Calibrated parameters (load from JSON — do not hardcode in physics)

File: `error_dist_validation/calibrated_params.json`

| Quantity | Value | Notes |
| --- | ---: | --- |
| `k_on` | 4.9173551603980275e4 | shape fit |
| `k_off` | −8.646573422065927e4 | shape fit |
| `v_on` | −0.3671035919638096 | global scalar (v1) |
| `v_off` | 0.45453731191773283 | global scalar (v1) |
| `alpha_on` | 2.9417779925201675 | |
| `alpha_off` | 4.801772094341022 | |
| `R_on` / `R_off` (Stage-1 dynamics) | 15e3 / 1e6 | used for write-verify physics |
| Deploy `R_on` / `R_off` | 50e3 / 200e3 | MNIST mapping only — **not** this engine’s dynamics |
| `sigma_d2d` | 0.0031551261173307903 | → `sigma_k_on`, `sigma_w_init` |
| `sigma_c2c` | 5.325875270032986 | large: C2C ∝ √dt at 50 ns |
| `enable_selector` | false | |
| `enable_rtn` | false (default) | Optional Markov RTN hook; off for Gates |
| `gamma_sinh` | 2.0 | header default |
| `V_SET_eff` / `V_RESET_eff` | −1.5 / +1.2 | Yao DC–pulse gap, literature-fixed |
| `V_READ` | 0.2 | |
| `pulse_width_s` | 5e-8 | |
| `margin_A` | 1e-7 | |
| `max_pulses` | 500 | |
| Active groups | 1.0–4.0 µA | 0.5 µA excluded |

---

## 8. Gate baselines (Component #4 published)

- Stage-1 pulse gates: mean rel error &lt; 20%, per-group &lt; 40% vs Yao totals.
- Pooled programming-error moments (N=14000): skew ≈ 0.30 CI [0.22, 0.37],
  excess kurtosis ≈ 1.58 CI [1.38, 1.79]; AIC winner **gmm2**.

---

## 9. Caveats

1. v1: no threshold D2D (`v_on`/`v_off` global).
2. Energy = midpoint approximation.
3. RTN available behind `enable_rtn=False` by default (C++ Markov port); omitted
   for Gates / Component #4 parity.
4. Sole deliverable claim: same validated results, faster — Gate 2 proves it.
5. **Never** substitute literature Biolek-on-current or abs/sign power laws for
   the formulas above.
