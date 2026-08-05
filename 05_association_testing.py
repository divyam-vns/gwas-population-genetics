"""
05_association_testing.py

Runs genome-wide association testing two ways, to demonstrate *why*
population-structure correction matters (not just that it exists):

  Model A (naive):      trait ~ genotype
  Model B (corrected):  trait ~ genotype + PC1 + PC2

The association scan itself is run on the full QC'd SNP set (not the
LD-pruned set) -- LD pruning is only appropriate for computing ancestry
PCs, not for the association scan, which should retain all QC-passing
SNPs as candidates. The PCs used as covariates here are the ones computed
from the LD-pruned set in script 04, which is the correct standard practice.

Because the simulated phenotype has a population-stratification confound
(POP_B has a higher baseline trait, unrelated to genotype), Model A is
expected to show inflated/spurious association signal at SNPs whose allele
frequency happens to differ between POP_A and POP_B, while Model B should
suppress that spurious signal and recover the true causal SNPs more cleanly.

Output: results/gwas_results_naive.csv
        results/gwas_results_corrected.csv
        results/association_validation_summary.csv
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

dosage = np.load("data/genotypes_qc.npy")
positions = pd.read_csv("data/variant_positions_qc.csv")
meta = pd.read_csv("data/sample_metadata_qc.csv")
pheno = pd.read_csv("data/phenotype.csv").set_index("sample_id").loc[meta["sample_id"]].reset_index()
pcs = pd.read_csv("results/pca_components.csv")
causal = pd.read_csv("data/causal_snps.csv")

y = pheno["trait"].values
n_snps = dosage.shape[1]

# Map original causal SNP indices -> post-QC column indices via genomic position
orig_positions = pd.read_csv("data/variant_positions.csv")
causal_positions = orig_positions.set_index("snp_index").loc[causal["snp_index"]]["position_bp"].values
qc_pos_to_idx = dict(zip(positions["position_bp"], positions["snp_index_qc"]))
causal_qc_idx = [qc_pos_to_idx[p] for p in causal_positions if p in qc_pos_to_idx]
print(f"{len(causal_qc_idx)} of {len(causal)} true causal SNPs survived QC and are trackable.")


def run_gwas(y, dosage, covariates=None):
    n_snps = dosage.shape[1]
    betas = np.zeros(n_snps)
    pvals = np.ones(n_snps)
    base_X = np.ones((len(y), 1)) if covariates is None else np.column_stack([np.ones(len(y)), covariates])
    for j in range(n_snps):
        X = np.column_stack([base_X, dosage[:, j]])
        try:
            model = sm.OLS(y, X).fit()
            betas[j] = model.params[-1]
            pvals[j] = model.pvalues[-1]
        except Exception:
            betas[j] = 0.0
            pvals[j] = 1.0
    return betas, pvals


print("Running naive model (trait ~ genotype)...")
beta_naive, p_naive = run_gwas(y, dosage, covariates=None)

print("Running PC-corrected model (trait ~ genotype + PC1 + PC2)...")
pc_cov = pcs[["PC1", "PC2"]].values
beta_corr, p_corr = run_gwas(y, dosage, covariates=pc_cov)

fdr_naive = multipletests(p_naive, method="fdr_bh")[1]
fdr_corr = multipletests(p_corr, method="fdr_bh")[1]

results_naive = pd.DataFrame({
    "snp_index_qc": np.arange(n_snps),
    "position_bp": positions["position_bp"],
    "beta": beta_naive,
    "p_value": p_naive,
    "p_fdr": fdr_naive,
    "is_true_causal": np.isin(np.arange(n_snps), causal_qc_idx),
})
results_corr = pd.DataFrame({
    "snp_index_qc": np.arange(n_snps),
    "position_bp": positions["position_bp"],
    "beta": beta_corr,
    "p_value": p_corr,
    "p_fdr": fdr_corr,
    "is_true_causal": np.isin(np.arange(n_snps), causal_qc_idx),
})

results_naive.to_csv("results/gwas_results_naive.csv", index=False)
results_corr.to_csv("results/gwas_results_corrected.csv", index=False)


def genomic_inflation(pvals):
    from scipy.stats import chi2
    chisq = chi2.isf(pvals, df=1)
    return np.median(chisq) / chi2.ppf(0.5, df=1)


lambda_naive = genomic_inflation(p_naive)
lambda_corr = genomic_inflation(p_corr)

sig_thresh = 0.05
n_causal_sig_naive = int((results_naive.loc[results_naive["is_true_causal"], "p_fdr"] < sig_thresh).sum())
n_causal_sig_corr = int((results_corr.loc[results_corr["is_true_causal"], "p_fdr"] < sig_thresh).sum())
n_false_pos_naive = int(((results_naive["p_fdr"] < sig_thresh) & (~results_naive["is_true_causal"])).sum())
n_false_pos_corr = int(((results_corr["p_fdr"] < sig_thresh) & (~results_corr["is_true_causal"])).sum())

summary = pd.DataFrame([
    {"model": "naive (trait ~ genotype)", "genomic_inflation_lambda_GC": round(lambda_naive, 3),
     "true_causal_snps_recovered": n_causal_sig_naive, "total_true_causal": len(causal_qc_idx),
     "significant_non_causal_hits_FDR<0.05": n_false_pos_naive},
    {"model": "PC-corrected (trait ~ genotype + PC1 + PC2)", "genomic_inflation_lambda_GC": round(lambda_corr, 3),
     "true_causal_snps_recovered": n_causal_sig_corr, "total_true_causal": len(causal_qc_idx),
     "significant_non_causal_hits_FDR<0.05": n_false_pos_corr},
])
summary.to_csv("results/association_validation_summary.csv", index=False)
print(summary.to_string(index=False))
print("\nSaved: results/gwas_results_naive.csv, results/gwas_results_corrected.csv, "
      "results/association_validation_summary.csv")
