"""
02_simulate_phenotype.py

Simulates a quantitative trait with:
  - A small number of true causal SNPs with known effect sizes (ground truth
    for validating the association test later)
  - A population-stratification confound (POP_B has a higher baseline trait
    value, independent of genotype)
  - Gaussian noise

Output: data/phenotype.csv (sample_id, trait, population)
        data/causal_snps.csv (ground truth: which SNPs are truly causal, effect size)
"""

import numpy as np
import pandas as pd

SEED = 42
N_CAUSAL = 5
STRATIFICATION_EFFECT = 1.5   # mean trait shift for POP_B vs POP_A (the confound)
NOISE_SD = 1.0

rng = np.random.default_rng(SEED)

dosage = np.load("data/genotypes.npy")
meta = pd.read_csv("data/sample_metadata.csv")
n_ind, n_snps = dosage.shape

# Pick causal SNPs at random from a middle-frequency range (MAF 10-45%) so
# they are realistically detectable
af = dosage.mean(axis=0) / 2
mid_freq_snps = np.where((af > 0.10) & (af < 0.45))[0]
causal_idx = rng.choice(mid_freq_snps, size=N_CAUSAL, replace=False)
effect_sizes = rng.uniform(0.6, 1.2, size=N_CAUSAL) * rng.choice([-1, 1], size=N_CAUSAL)

genetic_effect = dosage[:, causal_idx] @ effect_sizes

# Population-stratification confound: POP_B individuals get a mean shift
# unrelated to genotype -- this simulates real-world confounding by ancestry
strat_effect = np.where(meta["population"].values == "POP_B", STRATIFICATION_EFFECT, 0.0)

noise = rng.normal(0, NOISE_SD, size=n_ind)

trait = genetic_effect + strat_effect + noise

pheno = pd.DataFrame({
    "sample_id": meta["sample_id"],
    "trait": trait,
    "population": meta["population"],
})
pheno.to_csv("data/phenotype.csv", index=False)

causal_df = pd.DataFrame({
    "snp_index": causal_idx,
    "true_effect_size": effect_sizes,
    "allele_frequency": af[causal_idx],
})
causal_df.to_csv("data/causal_snps.csv", index=False)

print(f"Simulated trait for {n_ind} individuals.")
print(f"True causal SNPs (indices): {list(causal_idx)}")
print(f"True effect sizes: {np.round(effect_sizes, 3)}")
print(f"Stratification confound: +{STRATIFICATION_EFFECT} mean trait shift in POP_B")
print("Saved: data/phenotype.csv, data/causal_snps.csv")
