"""
Downstream MNIST training validation.

Train a small CNN in FP32, then optionally noise-aware fine-tune (Gaussian
programming noise on the forward pass only; clean backward/update — standard
hardware-aware training, not STE). Deploy the resulting weights and for M=30
seeds inject:
  - Baseline: moment-matched Gaussian noise (empirical std from Phase 2)
  - True-shape: samples from Phase-3 winning fitted distribution

Always runs a zero-noise mapping control first. Reports collapse rates and
per-seed histograms in addition to mean accuracy / Wilcoxon — mean+/-std can
hide bimodal catastrophic-failure structure.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from error_dist_validation.common import (
    ERROR_SAMPLES_CSV_PATH,
    ERROR_SAMPLES_PATH,
    FITTED_DISTRIBUTION_PATH,
    R_OFF,
    R_ON,
    ensure_plots_dir,
    load_calibrated_params,
    load_json,
)


def load_mnist(batch_size: int = 64, train_n: int = 8000, test_n: int = 2000):
    try:
        from torchvision import datasets, transforms
        transform = transforms.Compose([transforms.ToTensor()])
        train_full = datasets.MNIST("./data", train=True, download=True, transform=transform)
        test_full = datasets.MNIST("./data", train=False, download=True, transform=transform)
        train_sub, _ = torch.utils.data.random_split(
            train_full, [train_n, len(train_full) - train_n],
            generator=torch.Generator().manual_seed(0),
        )
        test_sub, _ = torch.utils.data.random_split(
            test_full, [test_n, len(test_full) - test_n],
            generator=torch.Generator().manual_seed(1),
        )
        train_loader = torch.utils.data.DataLoader(train_sub, batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_sub, batch_size=batch_size, shuffle=False)
        return train_loader, test_loader
    except Exception as e:
        print(f"[Warning] MNIST unavailable ({e}); using synthetic digits.")
        rng = np.random.default_rng(0)
        Xtr = rng.standard_normal((train_n, 1, 28, 28)).astype(np.float32)
        ytr = rng.integers(0, 10, size=(train_n,))
        Xte = rng.standard_normal((test_n, 1, 28, 28)).astype(np.float32)
        yte = rng.integers(0, 10, size=(test_n,))
        for c in range(10):
            Xtr[ytr == c, :, c, :] += 1.5
            Xte[yte == c, :, c, :] += 1.5
        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(torch.tensor(Xtr), torch.tensor(ytr)),
            batch_size=batch_size, shuffle=True,
        )
        test_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(torch.tensor(Xte), torch.tensor(yte)),
            batch_size=batch_size, shuffle=False,
        )
        return train_loader, test_loader


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def train_fp32(model, train_loader, epochs: int = 5, lr: float = 1e-3, device=None):
    device = device or torch.device("cpu")
    model.to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    model.train()
    for ep in range(epochs):
        total, correct, loss_sum = 0, 0, 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = crit(logits, y)
            loss.backward()
            opt.step()
            loss_sum += loss.item() * y.size(0)
            pred = logits.argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
        print(f"  epoch {ep+1}/{epochs}  loss={loss_sum/total:.4f}  acc={correct/total:.4f}")
    return model


def finetune_noise_aware(
    model: nn.Module,
    train_loader,
    *,
    emp_mean: float,
    emp_std: float,
    g_on: float,
    g_off: float,
    noise_scale: float = 1.0,
    include_bias: bool = False,
    epochs: int = 3,
    lr: float = 1e-4,
    device=None,
    seed: int = 42,
):
    """
    Hardware-aware fine-tune: inject moment-matched Gaussian programming noise
    on the forward pass only; restore clean weights before the optimizer step
    so backward/update stay on the clean parameters (no STE / no true-shape).
    """
    device = device or torch.device("cpu")
    model.to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    rng = np.random.default_rng(seed)
    model.train()
    print(
        f"\nNoise-aware fine-tune ({epochs} epochs, Gaussian forward noise, "
        f"clean update, lr={lr}, noise_scale={noise_scale})..."
    )
    for ep in range(epochs):
        total, correct, loss_sum = 0, 0, 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            full_clean = {n: p.detach().cpu().clone() for n, p in model.named_parameters()}
            prog = collect_weights(model, include_bias=include_bias)
            n_params = sum(w.numel() for w in prog.values())
            noise = rng.normal(emp_mean, emp_std, size=n_params)
            inject_noise_into_model(model, prog, noise, g_on, g_off, noise_scale)

            opt.zero_grad()
            logits = model(x)
            loss = crit(logits, y)
            loss.backward()
            # Grads stay on .grad; restore clean values so the step updates clean weights.
            restore_weights(model, full_clean, device)
            opt.step()

            loss_sum += loss.item() * y.size(0)
            pred = logits.argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
        print(
            f"  HA-FT epoch {ep+1}/{epochs}  "
            f"noisy-fwd loss={loss_sum/total:.4f}  acc={correct/total:.4f}"
        )
    return model


@torch.no_grad()
def evaluate(model, test_loader, device=None) -> float:
    device = device or torch.device("cpu")
    model.eval()
    correct, total = 0, 0
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return correct / max(total, 1)


def collect_weights(model: nn.Module, include_bias: bool = False) -> Dict[str, torch.Tensor]:
    """
    By default skip biases — they are not memristor-programmed conductances in a
    typical CIM crossbar; injecting Yao-scale programming noise into them is
    physically mismatched and can dominate collapse.
    """
    out = {}
    for n, p in model.named_parameters():
        if p.ndim < 1:
            continue
        if (not include_bias) and n.endswith(".bias"):
            continue
        out[n] = p.detach().cpu().clone()
    return out


def weights_to_conductance(w: torch.Tensor, g_on: float, g_off: float) -> torch.Tensor:
    s = torch.sigmoid(w)
    return (g_on - g_off) * s + g_off


def conductance_to_weights(g: torch.Tensor, g_on: float, g_off: float,
                           eps: float = 1e-6) -> torch.Tensor:
    frac = (g - g_off) / max(g_on - g_off, eps)
    frac = frac.clamp(eps, 1.0 - eps)
    return torch.logit(frac)


def load_empirical_error_std() -> float:
    if ERROR_SAMPLES_PATH.exists():
        df = __import__("pandas").read_parquet(ERROR_SAMPLES_PATH)
    elif ERROR_SAMPLES_CSV_PATH.exists():
        df = __import__("pandas").read_csv(ERROR_SAMPLES_CSV_PATH)
    else:
        raise FileNotFoundError("Run monte_carlo.py first.")
    return float(df["final_error"].std())


def sample_true_shape(n: int, fitted: Dict[str, Any], rng: np.random.Generator) -> np.ndarray:
    winner = fitted["pooled_winner"]
    name = winner["name"]
    p = winner["params"]
    emp = fitted["empirical_moments"]
    emp_mean, emp_std = emp["mean"], emp["std"]

    if name == "gaussian":
        samples = rng.normal(p["loc"], p["scale"], size=n)
    elif name == "skewnorm":
        samples = stats.skewnorm.rvs(p["a"], loc=p["loc"], scale=p["scale"], size=n, random_state=rng)
    elif name == "truncnorm":
        samples = stats.truncnorm.rvs(
            p["a"], p["b"], loc=p["loc"], scale=p["scale"], size=n, random_state=rng
        )
    elif name == "gmm2":
        weights = np.asarray(p["weights"])
        means = np.asarray(p["means"])
        vars_ = np.asarray(p["variances"])
        comp = rng.choice(len(weights), size=n, p=weights)
        samples = rng.normal(means[comp], np.sqrt(vars_[comp]))
    else:
        samples = rng.normal(emp_mean, emp_std, size=n)

    s_mean, s_std = float(np.mean(samples)), float(np.std(samples))
    if s_std < 1e-18:
        return np.full(n, emp_mean)
    samples = (samples - s_mean) / s_std * emp_std + emp_mean
    return samples.astype(np.float64)


def inject_noise_into_model(
    model: nn.Module,
    clean_weights: Dict[str, torch.Tensor],
    noise_amps: np.ndarray,
    g_on: float,
    g_off: float,
    noise_scale: float = 1.0,
) -> None:
    """
    Map weight → conductance, add ΔG = noise_scale * ΔI / V_read, map back.

    noise_scale=1.0 is the literal Yao absolute programming-error conversion.
    That is often catastrophic for sigmoid or logit on small trained weights
    (logit amplifies full-range G noise). Use noise_scale<1 for SNR-matched
    shape comparisons; always report which scale was used.
    """
    v_read = 0.2
    idx = 0
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name not in clean_weights:
                continue
            w = clean_weights[name]
            g = weights_to_conductance(w, g_on, g_off)
            n_el = g.numel()
            if idx + n_el > len(noise_amps):
                chunk = np.resize(noise_amps, idx + n_el)[idx: idx + n_el]
            else:
                chunk = noise_amps[idx: idx + n_el]
            idx += n_el
            dg = torch.from_numpy((noise_scale * chunk).astype(np.float32)).reshape(g.shape) / v_read
            g_noisy = (g + dg).clamp(min=g_off * 0.5, max=g_on * 1.5)
            p.copy_(conductance_to_weights(g_noisy, g_on, g_off).to(p.device))


def restore_weights(model, clean_weights, device):
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name in clean_weights:
                p.copy_(clean_weights[name].to(device))


def collapse_stats(accs: np.ndarray, thresholds=(0.90, 0.80, 0.50)) -> Dict[str, float]:
    out = {}
    for t in thresholds:
        out[f"frac_below_{int(t*100)}"] = float(np.mean(accs < t))
    out["min"] = float(accs.min())
    out["max"] = float(accs.max())
    out["median"] = float(np.median(accs))
    return out


def plot_seed_histograms(acc_gauss, acc_true, zero_acc, clean_acc, path):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    bins = np.linspace(0, 1, 21)
    axes[0].hist(acc_gauss, bins=bins, color="C0", alpha=0.75, edgecolor="k", linewidth=0.3)
    axes[0].axvline(clean_acc, color="g", ls="--", label=f"clean {clean_acc:.3f}")
    axes[0].axvline(zero_acc, color="k", ls=":", label=f"zero-noise {zero_acc:.3f}")
    axes[0].axvline(0.8, color="r", ls="--", alpha=0.5, label="collapse thr 0.8")
    axes[0].set_title("Gaussian seeds")
    axes[0].set_xlabel("test accuracy")
    axes[0].legend(fontsize=7)
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(acc_true, bins=bins, color="C1", alpha=0.75, edgecolor="k", linewidth=0.3)
    axes[1].axvline(clean_acc, color="g", ls="--", label=f"clean {clean_acc:.3f}")
    axes[1].axvline(zero_acc, color="k", ls=":", label=f"zero-noise {zero_acc:.3f}")
    axes[1].axvline(0.8, color="r", ls="--", alpha=0.5, label="collapse thr 0.8")
    axes[1].set_title("True-shape seeds")
    axes[1].set_xlabel("test accuracy")
    axes[1].legend(fontsize=7)
    axes[1].grid(True, alpha=0.3)
    fig.suptitle("Per-seed accuracy distribution (look for bimodality / collapse)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_validation(
    m_seeds: int = 30,
    epochs: int = 5,
    noise_scale: float = 1.0,
    include_bias: bool = False,
    collapse_threshold: float = 0.80,
    ha_finetune_epochs: int = 3,
    ha_finetune_lr: float = 1e-4,
    skip_ha_finetune: bool = False,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    fitted = load_json(FITTED_DISTRIBUTION_PATH)
    emp_std = load_empirical_error_std()
    emp_mean = float(fitted["empirical_moments"]["mean"])

    # Step 3: use characterized deploy R window from calibration — NOT Stage-1 sanity band
    try:
        calib = load_calibrated_params()
        deploy = calib.get("deploy", {})
        r_on_dep = float(deploy.get("R_on", 50e3))
        r_off_dep = float(deploy.get("R_off", 200e3))
        print(f"Deploy R_on={r_on_dep:.0f}  R_off={r_off_dep:.0f}  "
              f"groups={deploy.get('groups_uA')}  "
              f"scoped_exclude_05ua={deploy.get('scoped_exclude_05ua')}")
    except Exception as e:
        print(f"WARNING: deploy R fallback ({e})")
        r_on_dep, r_off_dep = 50e3, 200e3
    g_on = 1.0 / r_on_dep
    g_off = 1.0 / r_off_dep
    i_fs = 0.2 * (g_on - g_off)
    print(f"Empirical error: mean={emp_mean*1e9:.2f} nA  std={emp_std*1e9:.2f} nA")
    print(f"Winning distribution: {fitted['pooled_winner']['name']}")
    print(f"G range={g_on-g_off:.3e} S  I_fs@0.2V={i_fs*1e6:.2f} uA")
    print(f"dg_std/range = {(emp_std/0.2)/(g_on-g_off)*100:.2f}% of full-scale "
          f"(noise_scale={noise_scale})")
    print(f"Noise on biases: {include_bias}")
    print(f"Noise-aware fine-tune: "
          f"{'OFF' if skip_ha_finetune else f'{ha_finetune_epochs} epochs @ lr={ha_finetune_lr}'}")

    train_loader, test_loader = load_mnist()
    model = SmallCNN()
    print("\nTraining FP32 baseline...")
    train_fp32(model, train_loader, epochs=epochs, device=device)
    clean_acc_pre = evaluate(model, test_loader, device=device)
    print(f"Clean FP32 test accuracy (pre fine-tune): {clean_acc_pre:.4f}")

    if not skip_ha_finetune and ha_finetune_epochs > 0:
        finetune_noise_aware(
            model,
            train_loader,
            emp_mean=emp_mean,
            emp_std=emp_std,
            g_on=g_on,
            g_off=g_off,
            noise_scale=noise_scale,
            include_bias=include_bias,
            epochs=ha_finetune_epochs,
            lr=ha_finetune_lr,
            device=device,
        )
        clean_acc = evaluate(model, test_loader, device=device)
        print(f"Clean FP32 test accuracy (post noise-aware FT): {clean_acc:.4f}")
    else:
        clean_acc = clean_acc_pre

    # Snapshot programmable weights AFTER fine-tune (deployed network)
    full_snapshot = {n: p.detach().cpu().clone() for n, p in model.named_parameters()}
    clean_weights = collect_weights(model, include_bias=include_bias)
    n_params = sum(w.numel() for w in clean_weights.values())
    print(f"Programmable parameters: {n_params}")

    # --- Zero-noise control ---
    print("\n--- ZERO-NOISE CONTROL (mapping only) ---")
    inject_noise_into_model(
        model, clean_weights, np.zeros(n_params), g_on, g_off, noise_scale=1.0
    )
    zero_acc = evaluate(model, test_loader, device=device)
    # restore everything including biases
    restore_weights(model, full_snapshot, device)
    drop = clean_acc - zero_acc
    print(f"Zero-noise mapped accuracy: {zero_acc:.4f}  (drop vs clean = {drop:.4f})")
    if drop > 0.01:
        print("BUG: zero-noise mapping loses >1pp accuracy — fix mapping before interpreting noise results.")
        raise SystemExit(2)
    print("Zero-noise control PASSED (mapping roundtrip preserves accuracy).")

    acc_gauss: List[float] = []
    acc_true: List[float] = []

    print(f"\nDeploying M={m_seeds} noisy copies "
          f"(post HA fine-tune={not skip_ha_finetune}, noise_scale={noise_scale})...")
    for seed in range(m_seeds):
        rng = np.random.default_rng(1000 + seed)

        noise_g = rng.normal(emp_mean, emp_std, size=n_params)
        inject_noise_into_model(model, clean_weights, noise_g, g_on, g_off, noise_scale)
        a_g = evaluate(model, test_loader, device=device)
        acc_gauss.append(a_g)
        restore_weights(model, full_snapshot, device)

        noise_t = sample_true_shape(n_params, fitted, rng)
        inject_noise_into_model(model, clean_weights, noise_t, g_on, g_off, noise_scale)
        a_t = evaluate(model, test_loader, device=device)
        acc_true.append(a_t)
        restore_weights(model, full_snapshot, device)

        print(f"  seed {seed:02d}: Gaussian={a_g:.4f}  true-shape={a_t:.4f}  Δ={a_t-a_g:+.4f}")

    acc_gauss = np.asarray(acc_gauss)
    acc_true = np.asarray(acc_true)
    deltas = acc_true - acc_gauss

    if np.allclose(deltas, 0):
        stat, p_value = 0.0, 1.0
    else:
        stat, p_value = stats.wilcoxon(acc_true, acc_gauss, zero_method="wilcox")

    mean_delta = float(np.mean(deltas))
    rng = np.random.default_rng(0)
    boot = np.array([np.mean(rng.choice(deltas, size=len(deltas), replace=True)) for _ in range(2000)])
    ci_lo, ci_hi = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))
    significant = bool(p_value < 0.05)

    cg = collapse_stats(acc_gauss)
    ct = collapse_stats(acc_true)
    # Paired collapse: McNemar-style counts
    coll_g = acc_gauss < collapse_threshold
    coll_t = acc_true < collapse_threshold
    # Fisher / proportions
    from scipy.stats import fisher_exact
    # contingency for unpaired rate comparison is weak; report rates + paired discordant
    n_g_only = int(np.sum(coll_g & ~coll_t))
    n_t_only = int(np.sum(~coll_g & coll_t))
    n_both = int(np.sum(coll_g & coll_t))
    n_neither = int(np.sum(~coll_g & ~coll_t))
    # McNemar mid-p on discordant pairs
    if n_g_only + n_t_only > 0:
        mcnemar_p = float(stats.binomtest(n_t_only, n_g_only + n_t_only, 0.5).pvalue)
    else:
        mcnemar_p = 1.0

    print("\n" + "=" * 70)
    print("PHASE 4 RESULTS (noise-aware fine-tune → Gaussian vs true-shape)")
    print(f"  clean FP32 (pre HA-FT)  = {clean_acc_pre:.4f}")
    print(f"  clean FP32 (deployed)   = {clean_acc:.4f}"
          f"  (HA-FT={'on' if not skip_ha_finetune else 'off'})")
    print(f"  zero-noise mapped       = {zero_acc:.4f}")
    print(f"  Gaussian mean acc       = {acc_gauss.mean():.4f} ± {acc_gauss.std():.4f}  "
          f"median={cg['median']:.4f}  range=[{cg['min']:.4f},{cg['max']:.4f}]")
    print(f"  True-shape mean acc     = {acc_true.mean():.4f} ± {acc_true.std():.4f}  "
          f"median={ct['median']:.4f}  range=[{ct['min']:.4f},{ct['max']:.4f}]")
    print(f"  Mean Δ (true−gauss)     = {mean_delta:+.4f}  95% CI [{ci_lo:+.4f}, {ci_hi:+.4f}]")
    print(f"  Wilcoxon                = {stat:.4f}  p = {p_value:.6g}  "
          f"({'PASS' if significant else 'FAIL'} diff)")
    print(f"  Collapse rate (<{collapse_threshold:.0%}):")
    print(f"    Gaussian   = {cg['frac_below_80']*100:.1f}%")
    print(f"    True-shape = {ct['frac_below_80']*100:.1f}%")
    print(f"    paired discordant: gauss-only={n_g_only}  true-only={n_t_only}  "
          f"both={n_both}  neither={n_neither}")
    print(f"    McNemar p (collapse)  = {mcnemar_p:.6g}")
    if cg["frac_below_80"] > 0.3 or ct["frac_below_80"] > 0.3:
        print("  NOTE: high collapse rate after HA fine-tune — network still lacks "
              "tolerance margin; shape comparison remains in a degenerate regime.")
    print("=" * 70)

    plots = ensure_plots_dir()
    hist_path = plots / "accuracy_seed_histograms.png"
    plot_seed_histograms(acc_gauss, acc_true, zero_acc, clean_acc, hist_path)
    print(f"Wrote {hist_path}")

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(acc_gauss, acc_true, c="C0", alpha=0.8, edgecolors="k", linewidths=0.4)
    lims = [0.0, 1.0]
    ax.plot(lims, lims, "k--", lw=1, label="y = x")
    ax.axhline(collapse_threshold, color="r", ls=":", alpha=0.5)
    ax.axvline(collapse_threshold, color="r", ls=":", alpha=0.5)
    ax.set_xlabel("Accuracy (moment-matched Gaussian)")
    ax.set_ylabel("Accuracy (true-shape)")
    ax.set_title(f"MNIST  p={p_value:.4g}  Δ={mean_delta:+.4f}  scale={noise_scale}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)
    path = plots / "accuracy_comparison.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Wrote {path}")

    result = {
        "clean_fp32_accuracy_pre_haft": clean_acc_pre,
        "clean_fp32_accuracy": clean_acc,
        "noise_aware_finetune": {
            "enabled": not skip_ha_finetune and ha_finetune_epochs > 0,
            "epochs": 0 if skip_ha_finetune else ha_finetune_epochs,
            "lr": ha_finetune_lr,
            "forward_noise": "gaussian_matched_moments",
            "backward_update": "clean",
        },
        "zero_noise_mapped_accuracy": zero_acc,
        "zero_noise_control_passed": drop <= 0.01,
        "noise_scale": noise_scale,
        "include_bias": include_bias,
        "n_programmable_params": n_params,
        "gaussian_accuracies": acc_gauss.tolist(),
        "true_shape_accuracies": acc_true.tolist(),
        "mean_delta": mean_delta,
        "ci95_delta": [ci_lo, ci_hi],
        "wilcoxon_statistic": float(stat),
        "wilcoxon_pvalue": float(p_value),
        "pass_significant_difference": significant,
        "collapse_threshold": collapse_threshold,
        "collapse_gaussian": cg,
        "collapse_true_shape": ct,
        "collapse_paired": {
            "gauss_only": n_g_only,
            "true_only": n_t_only,
            "both": n_both,
            "neither": n_neither,
            "mcnemar_pvalue": mcnemar_p,
        },
        "winning_distribution": fitted["pooled_winner"]["name"],
        "empirical_std_nA": emp_std * 1e9,
        "dg_std_over_G_range": float((emp_std / 0.2) / (g_on - g_off)),
    }
    out_path = Path(__file__).resolve().parent / "mnist_validation_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"Wrote {out_path}")
    return result


def _deploy_g_range():
    try:
        calib = load_calibrated_params()
        deploy = calib.get("deploy", {})
        r_on = float(deploy.get("R_on", 50e3))
        r_off = float(deploy.get("R_off", 200e3))
    except Exception:
        r_on, r_off = 50e3, 200e3
    return 1.0 / r_on, 1.0 / r_off


def _paired_comparison(acc_gauss, acc_true, collapse_threshold=0.80):
    deltas = acc_true - acc_gauss
    if np.allclose(deltas, 0):
        stat, p_value = 0.0, 1.0
    else:
        stat, p_value = stats.wilcoxon(acc_true, acc_gauss, zero_method="wilcox")
    mean_delta = float(np.mean(deltas))
    rng = np.random.default_rng(0)
    boot = np.array([
        np.mean(rng.choice(deltas, size=len(deltas), replace=True)) for _ in range(2000)
    ])
    ci_lo, ci_hi = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))
    cg = collapse_stats(acc_gauss)
    ct = collapse_stats(acc_true)
    coll_g = acc_gauss < collapse_threshold
    coll_t = acc_true < collapse_threshold
    n_g_only = int(np.sum(coll_g & ~coll_t))
    n_t_only = int(np.sum(~coll_g & coll_t))
    n_both = int(np.sum(coll_g & coll_t))
    n_neither = int(np.sum(~coll_g & ~coll_t))
    if n_g_only + n_t_only > 0:
        mcnemar_p = float(stats.binomtest(n_t_only, n_g_only + n_t_only, 0.5).pvalue)
    else:
        mcnemar_p = 1.0
    return {
        "mean_delta": mean_delta,
        "ci95_delta": [ci_lo, ci_hi],
        "wilcoxon_statistic": float(stat),
        "wilcoxon_pvalue": float(p_value),
        "pass_significant_difference": bool(p_value < 0.05),
        "collapse_gaussian": cg,
        "collapse_true_shape": ct,
        "collapse_paired": {
            "gauss_only": n_g_only,
            "true_only": n_t_only,
            "both": n_both,
            "neither": n_neither,
            "mcnemar_pvalue": mcnemar_p,
        },
    }


def pick_transition_scale(
    sweep_rows: List[Dict[str, Any]],
    lo: float = 0.40,
    hi: float = 0.90,
) -> float:
    """Prefer scale whose Gaussian mean acc sits in (lo, hi); else closest to mid."""
    in_band = [r for r in sweep_rows if lo < r["gaussian_mean"] < hi]
    if in_band:
        # Prefer closest to midpoint of band (most informative)
        mid = 0.5 * (lo + hi)
        best = min(in_band, key=lambda r: abs(r["gaussian_mean"] - mid))
        return float(best["noise_scale"])
    # No scale in band — pick closest to mid anyway (report as best available)
    mid = 0.5 * (lo + hi)
    best = min(sweep_rows, key=lambda r: abs(r["gaussian_mean"] - mid))
    return float(best["noise_scale"])


def run_sweep_then_transition(
    scales: Optional[List[float]] = None,
    sweep_seeds: int = 8,
    compare_seeds: int = 30,
    epochs: int = 5,
    include_bias: bool = False,
    collapse_threshold: float = 0.80,
    acc_lo: float = 0.40,
    acc_hi: float = 0.90,
):
    """
    Train FP32 once (no HA-FT). Probe Gaussian accuracy across noise scales,
    pick the transition scale, then run full Gaussian-vs-true-shape comparison
    at that scale only.
    """
    if scales is None:
        scales = [0.03, 0.06, 0.1, 0.2, 0.35, 0.5, 0.7]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fitted = load_json(FITTED_DISTRIBUTION_PATH)
    emp_std = load_empirical_error_std()
    emp_mean = float(fitted["empirical_moments"]["mean"])
    g_on, g_off = _deploy_g_range()

    print(f"Device: {device}")
    print(f"Deploy G_on={g_on:.3e}  G_off={g_off:.3e}")
    print(f"Empirical std={emp_std*1e9:.2f} nA  winner={fitted['pooled_winner']['name']}")
    print(f"Sweep scales={scales}  probe_seeds={sweep_seeds}  (HA-FT OFF)")

    train_loader, test_loader = load_mnist()
    model = SmallCNN()
    print("\nTraining FP32 baseline (once)...")
    train_fp32(model, train_loader, epochs=epochs, device=device)
    clean_acc = evaluate(model, test_loader, device=device)
    print(f"Clean FP32 test accuracy: {clean_acc:.4f}")

    full_snapshot = {n: p.detach().cpu().clone() for n, p in model.named_parameters()}
    clean_weights = collect_weights(model, include_bias=include_bias)
    n_params = sum(w.numel() for w in clean_weights.values())

    inject_noise_into_model(model, clean_weights, np.zeros(n_params), g_on, g_off, 1.0)
    zero_acc = evaluate(model, test_loader, device=device)
    restore_weights(model, full_snapshot, device)
    print(f"Zero-noise mapped: {zero_acc:.4f}")
    if clean_acc - zero_acc > 0.01:
        raise SystemExit(2)

    # --- Sweep (Gaussian probe only) ---
    print("\n" + "=" * 70)
    print("NOISE-SCALE SWEEP (Gaussian probe, plain FP32)")
    print("=" * 70)
    sweep_rows: List[Dict[str, Any]] = []
    for scale in scales:
        accs = []
        for s in range(sweep_seeds):
            rng = np.random.default_rng(2000 + s)
            noise = rng.normal(emp_mean, emp_std, size=n_params)
            inject_noise_into_model(model, clean_weights, noise, g_on, g_off, scale)
            accs.append(evaluate(model, test_loader, device=device))
            restore_weights(model, full_snapshot, device)
        accs_a = np.asarray(accs)
        row = {
            "noise_scale": float(scale),
            "gaussian_mean": float(accs_a.mean()),
            "gaussian_std": float(accs_a.std()),
            "gaussian_min": float(accs_a.min()),
            "gaussian_max": float(accs_a.max()),
            "collapse_frac_below_80": float(np.mean(accs_a < collapse_threshold)),
            "in_transition_band": bool(acc_lo < accs_a.mean() < acc_hi),
        }
        sweep_rows.append(row)
        flag = " <-- TRANSITION" if row["in_transition_band"] else ""
        print(
            f"  scale={scale:.3f}  mean={row['gaussian_mean']:.4f} ± {row['gaussian_std']:.4f}  "
            f"range=[{row['gaussian_min']:.3f},{row['gaussian_max']:.3f}]  "
            f"collapse={row['collapse_frac_below_80']*100:.0f}%{flag}"
        )

    transition = pick_transition_scale(sweep_rows, lo=acc_lo, hi=acc_hi)
    in_band = any(r["in_transition_band"] for r in sweep_rows)
    print(f"\nSelected transition scale = {transition}"
          f"  (band [{acc_lo},{acc_hi}] hit={in_band})")

    # --- Full comparison at transition ---
    print("\n" + "=" * 70)
    print(f"PHASE 4 @ transition noise_scale={transition}  M={compare_seeds}")
    print("=" * 70)
    acc_gauss: List[float] = []
    acc_true: List[float] = []
    for seed in range(compare_seeds):
        rng = np.random.default_rng(1000 + seed)
        noise_g = rng.normal(emp_mean, emp_std, size=n_params)
        inject_noise_into_model(model, clean_weights, noise_g, g_on, g_off, transition)
        a_g = evaluate(model, test_loader, device=device)
        acc_gauss.append(a_g)
        restore_weights(model, full_snapshot, device)

        noise_t = sample_true_shape(n_params, fitted, rng)
        inject_noise_into_model(model, clean_weights, noise_t, g_on, g_off, transition)
        a_t = evaluate(model, test_loader, device=device)
        acc_true.append(a_t)
        restore_weights(model, full_snapshot, device)
        print(f"  seed {seed:02d}: Gaussian={a_g:.4f}  true-shape={a_t:.4f}  Δ={a_t-a_g:+.4f}")

    acc_gauss_a = np.asarray(acc_gauss)
    acc_true_a = np.asarray(acc_true)
    stats_out = _paired_comparison(acc_gauss_a, acc_true_a, collapse_threshold)
    cg, ct = stats_out["collapse_gaussian"], stats_out["collapse_true_shape"]
    cp = stats_out["collapse_paired"]

    print("\n" + "=" * 70)
    print(f"PHASE 4 RESULTS @ transition scale={transition}")
    print(f"  clean FP32              = {clean_acc:.4f}")
    print(f"  zero-noise mapped       = {zero_acc:.4f}")
    print(f"  Gaussian mean acc       = {acc_gauss_a.mean():.4f} ± {acc_gauss_a.std():.4f}  "
          f"median={cg['median']:.4f}  range=[{cg['min']:.4f},{cg['max']:.4f}]")
    print(f"  True-shape mean acc     = {acc_true_a.mean():.4f} ± {acc_true_a.std():.4f}  "
          f"median={ct['median']:.4f}  range=[{ct['min']:.4f},{ct['max']:.4f}]")
    print(f"  Mean Δ (true−gauss)     = {stats_out['mean_delta']:+.4f}  "
          f"95% CI [{stats_out['ci95_delta'][0]:+.4f}, {stats_out['ci95_delta'][1]:+.4f}]")
    print(f"  Wilcoxon                = {stats_out['wilcoxon_statistic']:.4f}  "
          f"p = {stats_out['wilcoxon_pvalue']:.6g}  "
          f"({'PASS' if stats_out['pass_significant_difference'] else 'FAIL'} diff)")
    print(f"  Collapse rate (<{collapse_threshold:.0%}):")
    print(f"    Gaussian   = {cg['frac_below_80']*100:.1f}%")
    print(f"    True-shape = {ct['frac_below_80']*100:.1f}%")
    print(f"    paired discordant: gauss-only={cp['gauss_only']}  true-only={cp['true_only']}  "
          f"both={cp['both']}  neither={cp['neither']}")
    print(f"    McNemar p (collapse)  = {cp['mcnemar_pvalue']:.6g}")
    print("=" * 70)

    plots = ensure_plots_dir()
    # Sweep plot
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    xs = [r["noise_scale"] for r in sweep_rows]
    ys = [r["gaussian_mean"] for r in sweep_rows]
    yerr = [r["gaussian_std"] for r in sweep_rows]
    ax.errorbar(xs, ys, yerr=yerr, fmt="o-", color="C0", capsize=3, label="Gaussian probe")
    ax.axhline(clean_acc, color="g", ls="--", label=f"clean {clean_acc:.3f}")
    ax.axhline(0.1, color="gray", ls=":", label="chance ~0.1")
    ax.axhspan(acc_lo, acc_hi, color="C1", alpha=0.12, label=f"transition band [{acc_lo},{acc_hi}]")
    ax.axvline(transition, color="r", ls="--", label=f"selected {transition}")
    ax.set_xscale("log")
    ax.set_xlabel("noise_scale")
    ax.set_ylabel("test accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("Noise-scale sweep (plain FP32, no HA-FT)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, which="both")
    sweep_plot = plots / "noise_scale_sweep.png"
    fig.tight_layout()
    fig.savefig(sweep_plot, dpi=150)
    plt.close(fig)
    print(f"Wrote {sweep_plot}")

    hist_path = plots / "accuracy_seed_histograms.png"
    plot_seed_histograms(acc_gauss_a, acc_true_a, zero_acc, clean_acc, hist_path)
    print(f"Wrote {hist_path}")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(acc_gauss_a, acc_true_a, c="C0", alpha=0.8, edgecolors="k", linewidths=0.4)
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.axhline(collapse_threshold, color="r", ls=":", alpha=0.5)
    ax.axvline(collapse_threshold, color="r", ls=":", alpha=0.5)
    ax.set_xlabel("Accuracy (moment-matched Gaussian)")
    ax.set_ylabel("Accuracy (true-shape)")
    ax.set_title(
        f"MNIST @ scale={transition}  p={stats_out['wilcoxon_pvalue']:.4g}  "
        f"Δ={stats_out['mean_delta']:+.4f}"
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    cmp_path = plots / "accuracy_comparison.png"
    fig.tight_layout()
    fig.savefig(cmp_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {cmp_path}")

    result = {
        "protocol": "sweep_then_transition",
        "ha_finetune": False,
        "clean_fp32_accuracy": clean_acc,
        "zero_noise_mapped_accuracy": zero_acc,
        "sweep": sweep_rows,
        "transition_scale": transition,
        "transition_band": [acc_lo, acc_hi],
        "transition_band_hit": in_band,
        "noise_scale": transition,
        "n_programmable_params": n_params,
        "gaussian_accuracies": acc_gauss_a.tolist(),
        "true_shape_accuracies": acc_true_a.tolist(),
        "gaussian_mean": float(acc_gauss_a.mean()),
        "gaussian_std": float(acc_gauss_a.std()),
        "true_shape_mean": float(acc_true_a.mean()),
        "true_shape_std": float(acc_true_a.std()),
        **stats_out,
        "winning_distribution": fitted["pooled_winner"]["name"],
        "empirical_std_nA": emp_std * 1e9,
        "dg_std_over_G_range_at_unit_scale": float((emp_std / 0.2) / (g_on - g_off)),
    }
    out_path = Path(__file__).resolve().parent / "mnist_validation_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    sweep_path = Path(__file__).resolve().parent / "mnist_noise_scale_sweep.json"
    with open(sweep_path, "w", encoding="utf-8") as f:
        json.dump({"sweep": sweep_rows, "transition_scale": transition}, f, indent=2)
    print(f"Wrote {out_path}")
    print(f"Wrote {sweep_path}")
    return result


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument(
        "--noise-scale", type=float, default=1.0,
        help="Multiplier on ΔI→ΔG (1.0=literal Yao absolute error)",
    )
    parser.add_argument("--include-bias", action="store_true")
    parser.add_argument("--collapse-threshold", type=float, default=0.80)
    parser.add_argument(
        "--ha-epochs", type=int, default=3,
        help="Noise-aware fine-tune epochs (Gaussian forward, clean update)",
    )
    parser.add_argument("--ha-lr", type=float, default=1e-4)
    parser.add_argument(
        "--skip-ha-finetune", action="store_true",
        help="Skip hardware-aware fine-tune (legacy direct-deploy comparison)",
    )
    parser.add_argument(
        "--sweep-then-transition", action="store_true",
        help="Log-spaced noise-scale sweep (no HA-FT), then compare at transition",
    )
    parser.add_argument("--sweep-seeds", type=int, default=8)
    args = parser.parse_args()
    if args.sweep_then_transition:
        run_sweep_then_transition(
            sweep_seeds=args.sweep_seeds,
            compare_seeds=args.seeds,
            epochs=args.epochs,
            include_bias=args.include_bias,
            collapse_threshold=args.collapse_threshold,
        )
    else:
        run_validation(
            m_seeds=args.seeds,
            epochs=args.epochs,
            noise_scale=args.noise_scale,
            include_bias=args.include_bias,
            collapse_threshold=args.collapse_threshold,
            ha_finetune_epochs=args.ha_epochs,
            ha_finetune_lr=args.ha_lr,
            skip_ha_finetune=args.skip_ha_finetune,
        )


if __name__ == "__main__":
    main()
