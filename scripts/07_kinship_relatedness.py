"""
07_kinship_relatedness.py

Simulates individuals with KNOWN family relationships via literal Mendelian
transmission, then tests whether a kinship estimator can recover those known
relationships from genotype data alone -- the same core principle behind
IBD-based relatedness and pedigree inference.

Steps:
  1. Take founder individuals from the QC'd genotype matrix (unrelated,
     drawn from the simulated population).
  2. Simulate offspring via explicit Mendelian transmission: each offspring
     allele at each SNP is drawn from one randomly-chosen parental allele
     per parent (true biological inheritance, not a shortcut).
  3. Build known relationship pairs: parent-offspring, full-siblings,
     and unrelated pairs.
  4. Implement the KING-robust kinship coefficient estimator (Manichaikul
     et al. 2010) directly from genotype dosages -- no external library.
  5. Validate that estimated kinship coefficients correctly separate the
     three known relationship classes.

Expected theoretical kinship coefficients:
  parent-offspring   -> 0.25
  full siblings       -> 0.25 (expected value; IBD0/IBD1/IBD2 sharing pattern
                          differs from parent-offspring, but mean kinship
                          coefficient is the same for outbred populations)
  unrelated            -> 0.0

Output: results/kinship_estimates.csv
        results/kinship_validation_summary.csv
        results/plots/kinship_validation.png
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 123
N_TRIOS = 40
N_SIBSHIPS = 40
N_UNRELATED_PAIRS = 200

rng = np.random.default_rng(SEED)

dosage = np.load("data/genotypes_qc.npy").astype(int)
n_ind, n_snps = dosage.shape


def transmit_allele(parent_dosage):
    """Simulate transmission of one allele from a parent given their dosage.
    For a parent with dosage g in {0,1,2}, the allele transmitted to a
    child is ALT with probability g/2 -- equivalent in expectation to
    transmitting one randomly chosen parental allele at each locus."""
    p_alt = parent_dosage / 2.0
    return rng.random(parent_dosage.shape) < p_alt


def make_offspring(mother_dosage, father_dosage):
    allele_from_mother = transmit_allele(mother_dosage).astype(int)
    allele_from_father = transmit_allele(father_dosage).astype(int)
    return allele_from_mother + allele_from_father


# --- Build parent-offspring trios ---------------------------------------------
founder_idx = rng.choice(n_ind, size=2 * N_TRIOS, replace=False)
mothers_idx = founder_idx[:N_TRIOS]
fathers_idx = founder_idx[N_TRIOS:]

offspring_trio = np.array([
    make_offspring(dosage[m], dosage[f]) for m, f in zip(mothers_idx, fathers_idx)
])

# --- Build full-sibling pairs (two offspring from the same parent pair) ------
remaining_idx = np.setdiff1d(np.arange(n_ind), founder_idx)
sib_founder_idx = rng.choice(remaining_idx, size=2 * N_SIBSHIPS, replace=False)
sib_mothers_idx = sib_founder_idx[:N_SIBSHIPS]
sib_fathers_idx = sib_founder_idx[N_SIBSHIPS:]

sib1 = np.array([make_offspring(dosage[m], dosage[f]) for m, f in zip(sib_mothers_idx, sib_fathers_idx)])
sib2 = np.array([make_offspring(dosage[m], dosage[f]) for m, f in zip(sib_mothers_idx, sib_fathers_idx)])

# --- Unrelated pairs (drawn independently from the original founder pool) ----
unrel_a = rng.choice(n_ind, size=N_UNRELATED_PAIRS, replace=True)
unrel_b = rng.choice(n_ind, size=N_UNRELATED_PAIRS, replace=True)
keep_unrel = unrel_a != unrel_b
unrel_a, unrel_b = unrel_a[keep_unrel], unrel_b[keep_unrel]


def king_robust_kinship(g1, g2):
    """
    KING-robust kinship coefficient estimator (Manichaikul et al. 2010,
    Bioinformatics), which does not require allele frequencies and is
    robust to population structure -- implemented directly from genotype
    dosage vectors in {0,1,2}.

    phi_hat = ( N_het_shared - 2*N_IBS0 ) / ( 2 * min(N_het1, N_het2) )
    """
    het1 = g1 == 1
    het2 = g2 == 1
    n_het1 = het1.sum()
    n_het2 = het2.sum()
    n_het_shared = (het1 & het2).sum()
    n_ibs0 = (((g1 == 0) & (g2 == 2)) | ((g1 == 2) & (g2 == 0))).sum()
    denom = 2 * min(n_het1, n_het2)
    if denom == 0:
        return 0.0
    return (n_het_shared - 2 * n_ibs0) / denom


records = []
for i in range(N_TRIOS):
    phi_mother = king_robust_kinship(offspring_trio[i], dosage[mothers_idx[i]])
    phi_father = king_robust_kinship(offspring_trio[i], dosage[fathers_idx[i]])
    records.append({"pair_type": "parent-offspring", "kinship_estimate": phi_mother})
    records.append({"pair_type": "parent-offspring", "kinship_estimate": phi_father})

for i in range(N_SIBSHIPS):
    phi_sib = king_robust_kinship(sib1[i], sib2[i])
    records.append({"pair_type": "full-sibling", "kinship_estimate": phi_sib})

for a, b in zip(unrel_a, unrel_b):
    phi_unrel = king_robust_kinship(dosage[a], dosage[b])
    records.append({"pair_type": "unrelated", "kinship_estimate": phi_unrel})

kinship_df = pd.DataFrame(records)
kinship_df.to_csv("results/kinship_estimates.csv", index=False)

summary = kinship_df.groupby("pair_type")["kinship_estimate"].agg(["mean", "std", "count"])
summary["theoretical_expected"] = [0.25, 0.25, 0.0]
print(summary)
summary.to_csv("results/kinship_validation_summary.csv")

fig, ax = plt.subplots(figsize=(8, 5))
order = ["unrelated", "full-sibling", "parent-offspring"]
colors = {"unrelated": "#7F8C8D", "full-sibling": "#2E5B8A", "parent-offspring": "#C0392B"}
for pt in order:
    vals = kinship_df.loc[kinship_df["pair_type"] == pt, "kinship_estimate"]
    ax.hist(vals, bins=25, alpha=0.6, label=f"{pt} (n={len(vals)})", color=colors[pt])
ax.axvline(0.25, color="black", linestyle="--", linewidth=1, label="Theoretical 1st-degree kinship (0.25)")
ax.axvline(0.0, color="black", linestyle=":", linewidth=1, label="Theoretical unrelated (0.0)")
ax.set_xlabel("KING-robust kinship coefficient estimate")
ax.set_ylabel("Pair count")
ax.set_title("Recovering known family relationships from genotype data alone\n(KING-robust kinship estimator, implemented from scratch)")
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig("results/plots/kinship_validation.png", dpi=150)
print("\nSaved: results/kinship_estimates.csv, results/kinship_validation_summary.csv, "
      "results/plots/kinship_validation.png")
