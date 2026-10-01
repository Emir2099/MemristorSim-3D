"""
Statistical characterization of the programming-error distribution.

Normality tests, shape statistics, truncation/stuck rates, candidate distribution
fits (Gaussian, skew-normal, truncated normal, 2-GMM), AIC/BIC/KS model selection.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.mixture import GaussianMixture

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from error_dist_validation.common import (
    ERROR_SAMPLES_CSV_PATH,
    ERROR_SAMPLES_PATH,
    FITTED_DISTRIBUTION_PATH,
    MARGIN_A,
    R_OFF,
    R_ON,
    V_READ,
    ensure_plots_dir,
    load_calibrated_params,
    save_json,
)


def load_samples() -> pd.DataFrame:
    if ERROR_SAMPLES_PATH.exists():
        return pd.read_parquet(ERROR_SAMPLES_PATH)
    if ERROR_SAMPLES_CSV_PATH.exists():
        return pd.read_csv(ERROR_SAMPLES_CSV_PATH)
    raise FileNotFoundError(
        f"Missing {ERROR_SAMPLES_PATH} (or CSV fallback). Run monte_carlo.py first."
    )


def bootstrap_ci(data: np.ndarray, stat_fn, n_boot: int = 1000,
                 alpha: float = 0.05, seed: int = 0) -> Tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    point = float(stat_fn(data))
    boots = np.empty(n_boot)
    n = len(data)
    for i in range(n_boot):
        sample = rng.choice(data, size=n, replace=True)
        boots[i] = stat_fn(sample)
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point, float(lo), float(hi)


def normality_tests(errors: np.ndarray) -> Dict[str, Any]:
    # Shapiro-Wilk is limited to n<=5000; subsample if needed
    rng = np.random.default_rng(0)
    sw_data = errors if len(errors) <= 5000 else rng.choice(errors, 5000, replace=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sw_stat, sw_p = stats.shapiro(sw_data)
        ad = stats.anderson(errors, dist="norm")
        # Anderson-Darling: compare statistic to 1% critical value
        ad_crit_1pct = float(ad.critical_values[ad.significance_level.tolist().index(1.0)]) \
            if 1.0 in list(ad.significance_level) else float(ad.critical_values[-1])
        ad_reject = bool(ad.statistic > ad_crit_1pct)
        dp_stat, dp_p = stats.normaltest(errors)

    reject_any = bool(sw_p < 0.01 or ad_reject or dp_p < 0.01)
    return {
        "shapiro_wilk": {"statistic": float(sw_stat), "p_value": float(sw_p)},
        "anderson_darling": {
            "statistic": float(ad.statistic),
            "critical_1pct": ad_crit_1pct,
            "reject_at_0.01": ad_reject,
        },
        "dagostino_pearson": {"statistic": float(dp_stat), "p_value": float(dp_p)},
        "reject_normality_any_p_lt_0.01": reject_any,
    }


def shape_stats(errors: np.ndarray) -> Dict[str, Any]:
    skew, skew_lo, skew_hi = bootstrap_ci(errors, stats.skew)
    kurt, kurt_lo, kurt_hi = bootstrap_ci(errors, lambda x: stats.kurtosis(x, fisher=True))
    return {
        "skewness": {"value": skew, "ci95": [skew_lo, skew_hi]},
        "excess_kurtosis": {"value": kurt, "ci95": [kurt_lo, kurt_hi]},
        "mean": float(np.mean(errors)),
        "std": float(np.std(errors)),
        "std_nA": float(np.std(errors) * 1e9),
    }


def truncation_and_stuck(df: pd.DataFrame, errors: np.ndarray) -> Dict[str, Any]:
    # Conductance bounds from R_on / R_off at V_read
    i_max = V_READ / R_ON   # ~13.3 µA
    i_min = V_READ / R_OFF  # ~0.2 µA
    finals = df["final_current"].to_numpy()
    near_hi = np.abs(finals - i_max) <= MARGIN_A
    near_lo = np.abs(finals - i_min) <= MARGIN_A
    stuck = df["stuck_flag"].to_numpy().astype(bool)
    return {
        "fraction_near_Gmax_bound": float(np.mean(near_hi)),
        "fraction_near_Gmin_bound": float(np.mean(near_lo)),
        "fraction_near_either_bound": float(np.mean(near_hi | near_lo)),
        "stuck_fraction": float(np.mean(stuck)),
        "i_min_A": float(i_min),
        "i_max_A": float(i_max),
        "note": "Gaussian predicts negligible boundary fraction; "
                "stuck_flag qualitatively compared to Yao Ext. Data Fig. 1 outliers.",
    }


def _aic_bic(nll: float, k: int, n: int) -> Tuple[float, float]:
    aic = 2 * k + 2 * nll
    bic = k * np.log(n) + 2 * nll
    return float(aic), float(bic)


def fit_candidates(errors: np.ndarray) -> List[Dict[str, Any]]:
    """
    Fit candidate distributions. Work in nanoamperes for numerical stability,
    then convert fitted location/scale params back to amperes for serialization.
    """
    scale_to_nA = 1e9
    e = errors * scale_to_nA
    n = len(e)
    results = []

    def to_amps_params(params: Dict[str, Any], keys=("loc", "scale", "means", "variances",
                                                      "bound_lo", "bound_hi")) -> Dict[str, Any]:
        out = dict(params)
        for k in keys:
            if k not in out:
                continue
            if k == "means":
                out[k] = [float(v) / scale_to_nA for v in out[k]]
            elif k == "variances":
                out[k] = [float(v) / (scale_to_nA ** 2) for v in out[k]]
            else:
                out[k] = float(out[k]) / scale_to_nA
        return out

    # --- Gaussian ---
    mu, sigma = stats.norm.fit(e)
    nll = float(-np.sum(stats.norm.logpdf(e, mu, sigma)))
    aic, bic = _aic_bic(nll, 2, n)
    ks_stat, ks_p = stats.kstest(e, "norm", args=(mu, sigma))
    results.append({
        "name": "gaussian",
        "params": to_amps_params({"loc": mu, "scale": sigma}),
        "n_params": 2,
        "nll": nll,
        "aic": aic,
        "bic": bic,
        "ks_statistic": float(ks_stat),
        "ks_pvalue": float(ks_p),
    })

    # --- Skew-normal ---
    a, loc, scale = stats.skewnorm.fit(e)
    nll = float(-np.sum(stats.skewnorm.logpdf(e, a, loc, scale)))
    aic, bic = _aic_bic(nll, 3, n)
    ks_stat, ks_p = stats.kstest(e, "skewnorm", args=(a, loc, scale))
    results.append({
        "name": "skewnorm",
        "params": to_amps_params({"a": a, "loc": loc, "scale": scale}, keys=("loc", "scale")),
        "n_params": 3,
        "nll": nll,
        "aic": aic,
        "bic": bic,
        "ks_statistic": float(ks_stat),
        "ks_pvalue": float(ks_p),
    })
    # keep shape param a as-is (dimensionless)
    results[-1]["params"]["a"] = float(a)

    # --- Truncated normal ---
    lo = float(np.min(e) - 1e-9)
    hi = float(np.max(e) + 1e-9)
    mu0, sig0 = float(np.mean(e)), float(np.std(e) + 1e-12)
    a_t = (lo - mu0) / sig0
    b_t = (hi - mu0) / sig0
    try:
        _a, _b, loc_t, scale_t = stats.truncnorm.fit(e, fa=a_t, fb=b_t)
        a_t = (lo - loc_t) / max(scale_t, 1e-12)
        b_t = (hi - loc_t) / max(scale_t, 1e-12)
        logpdf = stats.truncnorm.logpdf(e, a_t, b_t, loc=loc_t, scale=scale_t)
        nll = float(-np.sum(logpdf))
        aic, bic = _aic_bic(nll, 2, n)
        cdf = lambda x, a=a_t, b=b_t, loc=loc_t, sc=scale_t: stats.truncnorm.cdf(
            x, a, b, loc=loc, scale=sc
        )
        ks_stat, ks_p = stats.kstest(e, cdf)
        results.append({
            "name": "truncnorm",
            "params": to_amps_params({
                "a": a_t, "b": b_t, "loc": loc_t, "scale": scale_t,
                "bound_lo": lo, "bound_hi": hi,
            }, keys=("loc", "scale", "bound_lo", "bound_hi")),
            "n_params": 2,
            "nll": nll,
            "aic": aic,
            "bic": bic,
            "ks_statistic": float(ks_stat),
            "ks_pvalue": float(ks_p),
        })
        results[-1]["params"]["a"] = float(a_t)
        results[-1]["params"]["b"] = float(b_t)
    except Exception as ex:
        results.append({
            "name": "truncnorm",
            "params": {},
            "error": str(ex),
            "aic": 1e30,
            "bic": 1e30,
            "ks_pvalue": 0.0,
        })

    # --- 2-component Gaussian mixture ---
    X = e.reshape(-1, 1)
    gmm = GaussianMixture(n_components=2, covariance_type="full", random_state=0, n_init=5)
    gmm.fit(X)
    nll = float(-gmm.score(X) * n)
    aic, bic = _aic_bic(nll, 5, n)
    mix_samples = gmm.sample(n)[0].ravel()
    ks_stat, ks_p = stats.ks_2samp(e, mix_samples)
    results.append({
        "name": "gmm2",
        "params": to_amps_params({
            "weights": gmm.weights_.tolist(),
            "means": gmm.means_.ravel().tolist(),
            "variances": [float(v[0][0]) for v in gmm.covariances_],
        }, keys=("means", "variances")),
        "n_params": 5,
        "nll": nll,
        "aic": aic,
        "bic": bic,
        "ks_statistic": float(ks_stat),
        "ks_pvalue": float(ks_p),
    })
    results[-1]["params"]["weights"] = gmm.weights_.tolist()

    return results


def select_winner(candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    valid = [c for c in candidates if "aic" in c and np.isfinite(c["aic"])]
    valid_sorted = sorted(valid, key=lambda c: c["aic"])
    # Prefer lowest AIC that also passes KS at α=0.05
    for c in valid_sorted:
        if c.get("ks_pvalue", 0.0) >= 0.05:
            return {**c, "selection": "lowest_aic_with_ks_pass", "ks_pass": True}
    # Fallback: best AIC, flag KS failure
    best = valid_sorted[0]
    return {**best, "selection": "lowest_aic_ks_failed", "ks_pass": False}


def characterize_subset(name: str, df: pd.DataFrame) -> Dict[str, Any]:
    errors = df["final_error"].to_numpy(dtype=float)
    print(f"\n--- {name}  (n={len(errors)}) ---")
    norm = normality_tests(errors)
    shape = shape_stats(errors)
    trunc = truncation_and_stuck(df, errors)
    cands = fit_candidates(errors)
    winner = select_winner(cands)

    print(f"  normality reject (any p<0.01): {norm['reject_normality_any_p_lt_0.01']}")
    print(f"  skew={shape['skewness']['value']:.3f}  "
          f"ex_kurt={shape['excess_kurtosis']['value']:.3f}  "
          f"std={shape['std_nA']:.1f} nA")
    print(f"  stuck={trunc['stuck_fraction']*100:.2f}%  "
          f"near_bound={trunc['fraction_near_either_bound']*100:.2f}%")
    print("  candidates:")
    for c in cands:
        if "error" in c:
            print(f"    {c['name']}: FAILED ({c['error']})")
        else:
            print(
                f"    {c['name']}: AIC={c['aic']:.1f}  BIC={c['bic']:.1f}  "
                f"KS_p={c['ks_pvalue']:.4g}"
            )
    print(f"  WINNER: {winner['name']}  ({winner['selection']})")

    return {
        "name": name,
        "n": int(len(errors)),
        "normality": norm,
        "shape": shape,
        "truncation_stuck": trunc,
        "candidates": cands,
        "winner": winner,
    }


def plot_histograms(df: pd.DataFrame, pooled_winner: Dict[str, Any], groups) -> None:
    import matplotlib.pyplot as plt

    plots = ensure_plots_dir()
    errors_na = df["final_error"].to_numpy() * 1e9
    n_g = len(groups)
    ncols = 3
    nrows = int(np.ceil((n_g + 1) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3.2 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for i, g in enumerate(groups):
        ax = axes[i]
        e = df.loc[df["group"] == g, "final_error"].to_numpy() * 1e9
        ax.hist(e, bins=40, density=True, alpha=0.6, color="C0", label="emp.")
        mu, sig = np.mean(e), np.std(e)
        xs = np.linspace(e.min(), e.max(), 200) if len(e) > 1 else np.linspace(-1, 1, 200)
        ax.plot(xs, stats.norm.pdf(xs, mu, sig), "r-", lw=1.5, label="Gaussian")
        ax.set_title(f"{g:.1f} uA")
        ax.set_xlabel("error (nA)")
        if i == 0:
            ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    ax = axes[n_g]
    ax.hist(errors_na, bins=60, density=True, alpha=0.6, color="C0", label="emp.")
    mu, sig = np.mean(errors_na), np.std(errors_na)
    xs = np.linspace(np.percentile(errors_na, 0.5), np.percentile(errors_na, 99.5), 300)
    ax.plot(xs, stats.norm.pdf(xs, mu, sig), "r-", lw=1.5, label="Gaussian")
    wname = pooled_winner["name"]
    ax.set_title(f"Pooled (winner={wname})")
    ax.set_xlabel("error (nA)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    for j in range(n_g + 1, len(axes)):
        axes[j].axis("off")

    fig.suptitle("Programming-error histograms (nA)", fontsize=12)
    fig.tight_layout()
    path = plots / "error_histograms_per_group.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Wrote {path}")


def plot_qq(errors: np.ndarray) -> None:
    import matplotlib.pyplot as plt

    plots = ensure_plots_dir()
    fig, ax = plt.subplots(figsize=(5, 5))
    stats.probplot(errors * 1e9, dist="norm", plot=ax)
    ax.set_title("Q-Q plot vs Gaussian (error in nA)")
    ax.grid(True, alpha=0.3)
    path = plots / "qq_plot.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Wrote {path}")


def main():
    df = load_samples()
    print(f"Loaded {len(df)} samples from Monte Carlo")
    groups = sorted(df["group"].unique().tolist())
    print(f"Groups (uA): {groups}")

    reports = []
    for g in groups:
        sub = df[df["group"] == g]
        reports.append(characterize_subset(f"group_{g:.1f}uA", sub))

    pooled = characterize_subset("pooled", df)
    reports.append(pooled)

    plot_histograms(df, pooled["winner"], groups)
    plot_qq(df["final_error"].to_numpy())

    table = []
    for c in pooled["candidates"]:
        if "error" in c:
            continue
        table.append({
            "model": c["name"],
            "aic": c["aic"],
            "bic": c["bic"],
            "ks_statistic": c["ks_statistic"],
            "ks_pvalue": c["ks_pvalue"],
        })

    out = {
        "pooled_winner": pooled["winner"],
        "empirical_moments": {
            "mean": pooled["shape"]["mean"],
            "std": pooled["shape"]["std"],
            "std_nA": pooled["shape"]["std_nA"],
        },
        "groups_uA": groups,
        "model_comparison_pooled": table,
        "reports": reports,
        "notes": [
            "RTN omitted in simulation.",
            "108 nA Stage-2 anchor is from Yao experiment (C), not (A).",
            "Winner selected by lowest AIC among models with KS p>=0.05; "
            "fallback to lowest AIC with ks_pass=False.",
            "Stats are for active/scoped groups only — do not reuse pre-scope numbers.",
        ],
    }
    save_json(FITTED_DISTRIBUTION_PATH, out)
    print(f"\nWrote {FITTED_DISTRIBUTION_PATH}")
    print("Characterization complete.")


if __name__ == "__main__":
    main()
