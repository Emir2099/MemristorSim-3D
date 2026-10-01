"""Dump Phase-3 diagnostics table + Phase-4 effect size to stdout/markdown."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from error_dist_validation.common import (
    FITTED_DISTRIBUTION_PATH,
    PACKAGE_DIR,
)


def main():
    fitted = json.loads(FITTED_DISTRIBUTION_PATH.read_text(encoding="utf-8"))
    mnist_path = PACKAGE_DIR / "mnist_validation_results.json"
    mnist = json.loads(mnist_path.read_text(encoding="utf-8")) if mnist_path.exists() else None

    lines = []
    lines.append("# Error-distribution diagnostics\n")

    for r in fitted["reports"]:
        sh = r["shape"]
        tr = r["truncation_stuck"]
        lines.append(f"## {r['name']}  (n={r['n']})\n")
        lines.append(
            f"- skewness = {sh['skewness']['value']:.4f}  "
            f"95% CI [{sh['skewness']['ci95'][0]:.4f}, {sh['skewness']['ci95'][1]:.4f}]"
        )
        lines.append(
            f"- excess kurtosis = {sh['excess_kurtosis']['value']:.4f}  "
            f"95% CI [{sh['excess_kurtosis']['ci95'][0]:.4f}, {sh['excess_kurtosis']['ci95'][1]:.4f}]"
        )
        lines.append(f"- stuck fraction = {tr['stuck_fraction']*100:.2f}%")
        lines.append(
            f"- near Gmin/Gmax bound = {tr['fraction_near_either_bound']*100:.2f}% "
            f"(Gmin {tr['fraction_near_Gmin_bound']*100:.2f}%, "
            f"Gmax {tr['fraction_near_Gmax_bound']*100:.2f}%)"
        )
        lines.append(
            f"- normality rejected (any p<0.01) = "
            f"{r['normality']['reject_normality_any_p_lt_0.01']}"
        )
        lines.append("")
        lines.append("| model | AIC | BIC | KS | KS p |")
        lines.append("| --- | ---: | ---: | ---: | ---: |")
        for c in r["candidates"]:
            if "error" in c:
                lines.append(f"| {c['name']} | FAILED | | | |")
                continue
            lines.append(
                f"| {c['name']} | {c['aic']:.1f} | {c['bic']:.1f} | "
                f"{c['ks_statistic']:.4f} | {c['ks_pvalue']:.4g} |"
            )
        w = r["winner"]
        lines.append(
            f"\n**Winner:** `{w['name']}`  ({w['selection']}, "
            f"ks_pass={w.get('ks_pass')})\n"
        )

    if mnist:
        import numpy as np
        g = np.asarray(mnist["gaussian_accuracies"])
        t = np.asarray(mnist["true_shape_accuracies"])
        lines.append("## Phase 4 effect size (MNIST)\n")
        lines.append(f"- clean FP32 accuracy = {mnist['clean_fp32_accuracy']:.4f}")
        lines.append(
            f"- moment-matched Gaussian mean acc = {g.mean():.4f} ± {g.std():.4f}"
        )
        lines.append(
            f"- true-shape mean acc = {t.mean():.4f} ± {t.std():.4f}"
        )
        lines.append(
            f"- mean Δ (true − gaussian) = {mnist['mean_delta']:+.4f}  "
            f"95% CI [{mnist['ci95_delta'][0]:+.4f}, {mnist['ci95_delta'][1]:+.4f}]"
        )
        lines.append(
            f"- Wilcoxon p = {mnist['wilcoxon_pvalue']:.6g}  "
            f"(pass={mnist['pass_significant_difference']})"
        )
        lines.append(f"- winning distribution used = {mnist['winning_distribution']}")
        lines.append("")

    text = "\n".join(lines)
    out = PACKAGE_DIR / "diagnostics_report.md"
    out.write_text(text, encoding="utf-8")
    print(text)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
