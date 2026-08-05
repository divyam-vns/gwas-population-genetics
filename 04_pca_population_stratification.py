"""
04_pca_population_stratification.py

Runs LD pruning followed by PCA on the QC'd genotype matrix to detect
population structure. LD pruning is performed first because unpruned PCA can
be dominated by a small number of high-LD genomic blocks rather than
reflecting genome-wide ancestry signal -- pruning is standard practice
before any ancestry/stratification PCA.

Validates that PCA correctly recovers the known simulated population split
(POP_A vs POP_B) using genotype data alone.

Output: data/genotypes_ld_pruned.npy, data/variant_positions_ld_pruned.csv
        results/pca_components.csv (sample_id, PC1..PC10, population)
        results/plots/pca_population_structure.png
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

WINDOW_SNPS = 50     # number of already-kept SNPs considered when testing a candidate
R2_THRESHOLD = 0.2   # standard PLINK-style pairwise r^2 pruning threshold

dosage = np.load("data/genotypes_qc.npy")
meta = pd.read_csv("data/sample_metadata_qc.csv")
positions = pd.read_csv("data/variant_positions_qc.csv")

# ---------------------------------------------------------------------------
# LD pruning: greedy sequential pairwise-r^2 pruning in genomic order.
# For each candidate SNP (in position order), compute r^2 against the most
# recently kept SNPs; drop the candidate if any exceeds the threshold.
# ---------------------------------------------------------------------------
order = np.argsort(positions["position_bp"].values)
dosage_ordered = dosage[:, order]
n_ind, n_snps = dosage_ordered.shape

geno_centered = dosage_ordered - dosage_ordered.mean(axis=0)
geno_norm = np.linalg.norm(geno_centered, axis=0)
geno_norm[geno_norm == 0] = 1e-9  # guard against monomorphic columns post-QC (should not occur)

kept_mask = np.zeros(n_snps, dtype=bool)
kept_recent = []  # indices (into dosage_ordered) of most recently kept SNPs

for j in range(n_snps):
    candidate = geno_centered[:, j]
    cand_norm = geno_norm[j]
    correlated = False
    for k in kept_recent[-WINDOW_SNPS:]:
        r = np.dot(candidate, geno_centered[:, k]) / (cand_norm * geno_norm[k])
        if r * r > R2_THRESHOLD:
            correlated = True
            break
    if not correlated:
        kept_mask[j] = True
        kept_recent.append(j)

dosage_pruned = dosage_ordered[:, kept_mask]
positions_pruned = positions.iloc[order].iloc[kept_mask].reset_index(drop=True)
positions_pruned["snp_index_pruned"] = np.arange(len(positions_pruned))

print(f"LD pruning: {n_snps} SNPs -> {dosage_pruned.shape[1]} SNPs retained "
      f"(pairwise r^2 < {R2_THRESHOLD} within a {WINDOW_SNPS}-SNP window)")

np.save("data/genotypes_ld_pruned.npy", dosage_pruned.astype(np.float32))
positions_pruned.to_csv("data/variant_positions_ld_pruned.csv", index=False)

# ---------------------------------------------------------------------------
# PCA on the pruned, standardized genotype matrix (Patterson et al. 2006
# EIGENSTRAT-style standardization: center by allele frequency, scale by
# sqrt(p(1-p)))
# ---------------------------------------------------------------------------
af = dosage_pruned.mean(axis=0) / 2
af = np.clip(af, 1e-6, 1 - 1e-6)
scale = np.sqrt(af * (1 - af))
geno_std = (dosage_pruned - dosage_pruned.mean(axis=0)) / scale

n_components = 10
pca = PCA(n_components=n_components, random_state=42)
pcs = pca.fit_transform(geno_std)

pc_cols = [f"PC{i+1}" for i in range(n_components)]
pc_df = pd.DataFrame(pcs, columns=pc_cols)
pc_df.insert(0, "sample_id", meta["sample_id"])
pc_df["population"] = meta["population"]
pc_df.to_csv("results/pca_components.csv", index=False)

var_explained = pca.explained_variance_ratio_ * 100
print("Variance explained by top 5 PCs (%):", np.round(var_explained[:5], 2))

popA_pc1 = pc_df.loc[pc_df["population"] == "POP_A", "PC1"]
popB_pc1 = pc_df.loc[pc_df["population"] == "POP_B", "PC1"]
separation = abs(popA_pc1.mean() - popB_pc1.mean()) / pc_df["PC1"].std()
print(f"PC1 mean separation between POP_A and POP_B: {separation:.2f} standard deviations")

# --- Plot ---------------------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(11, 5))

colors = {"POP_A": "#1F3864", "POP_B": "#C0392B"}
for pop, group in pc_df.groupby("population"):
    ax[0].scatter(group["PC1"], group["PC2"], s=18, alpha=0.75, label=pop, color=colors[pop])
ax[0].set_xlabel(f"PC1 ({var_explained[0]:.1f}% variance explained)")
ax[0].set_ylabel(f"PC2 ({var_explained[1]:.1f}% variance explained)")
ax[0].set_title("PCA recovers known population structure\nfrom genotype data alone (LD-pruned)")
ax[0].legend(title="True simulated population")

ax[1].bar(range(1, 11), var_explained, color="#2E5B8A")
ax[1].set_xlabel("Principal Component")
ax[1].set_ylabel("% Variance Explained")
ax[1].set_title("Scree plot")

plt.tight_layout()
plt.savefig("results/plots/pca_population_structure.png", dpi=150)
print("Saved: data/genotypes_ld_pruned.npy, results/pca_components.csv, "
      "results/plots/pca_population_structure.png")
