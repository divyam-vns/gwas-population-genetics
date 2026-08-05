"""
06_manhattan_qq_plots.py

Generates Manhattan and QQ plots comparing the naive vs PC-corrected GWAS
models, visually demonstrating the effect of population-stratification
correction (genomic inflation) and confirming recovery of the known
simulated causal SNPs.

Output: results/plots/manhattan_naive_vs_corrected.png
        results/plots/qq_naive_vs_corrected.png
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

naive = pd.read_csv("results/gwas_results_naive.csv")
corr = pd.read_csv("results/gwas_results_corrected.csv")

GENOME_WIDE_SIG = -np.log10(0.05 / len(naive))  # Bonferroni line for this SNP set

# --- Manhattan plots ----------------------------------------------------------
fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

for ax, df, title in zip(
    axes, [naive, corr],
    ["Naive model: trait ~ genotype  (population stratification NOT corrected)",
     "Corrected model: trait ~ genotype + PC1 + PC2  (stratification corrected)"]
):
    colors = np.where(df["is_true_causal"], "#C0392B", "#2E5B8A")
    sizes = np.where(df["is_true_causal"], 40, 6)
    ax.scatter(df["position_bp"] / 1e6, -np.log10(df["p_value"]), c=colors, s=sizes, alpha=0.7)
    ax.axhline(GENOME_WIDE_SIG, color="grey", linestyle="--", linewidth=1,
               label=f"Bonferroni threshold (-log10 p = {GENOME_WIDE_SIG:.1f})")
    ax.set_ylabel("-log10(p-value)")
    ax.set_title(title, fontsize=11)
    ax.legend(loc="upper right", fontsize=8)

axes[-1].set_xlabel("Position (Mb)")
handles = [
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#C0392B', markersize=8, label='True causal SNP'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#2E5B8A', markersize=6, label='Non-causal SNP'),
]
fig.legend(handles=handles, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.02), fontsize=9)
plt.tight_layout()
plt.savefig("results/plots/manhattan_naive_vs_corrected.png", dpi=150, bbox_inches="tight")
plt.close()

# --- QQ plots -------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 5))

for ax, df, title in zip(
    axes, [naive, corr],
    ["Naive model\n(inflated -- population stratification)",
     "PC-corrected model\n(inflation largely resolved)"]
):
    p = np.sort(df["p_value"].values)
    n = len(p)
    expected = -np.log10(np.arange(1, n + 1) / (n + 1))
    observed = -np.log10(p)
    ax.scatter(expected, observed, s=4, color="#2E5B8A", alpha=0.5)
    max_val = max(expected.max(), observed.max())
    ax.plot([0, max_val], [0, max_val], color="red", linestyle="--", linewidth=1)
    ax.set_xlabel("Expected -log10(p)")
    ax.set_ylabel("Observed -log10(p)")
    ax.set_title(title, fontsize=10)

plt.tight_layout()
plt.savefig("results/plots/qq_naive_vs_corrected.png", dpi=150, bbox_inches="tight")
plt.close()

print("Saved: results/plots/manhattan_naive_vs_corrected.png, results/plots/qq_naive_vs_corrected.png")
