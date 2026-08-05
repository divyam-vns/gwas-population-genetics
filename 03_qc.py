"""
03_qc.py

Standard GWAS QC steps, applied to the simulated genotype matrix:
  1. Introduce realistic missingness (simulated genotyping call-rate dropout)
  2. Filter SNPs with high missingness (>5%)
  3. Filter individuals with high missingness (>5%)
  4. Filter SNPs by minor allele frequency (MAF < 1%)
  5. Filter SNPs failing Hardy-Weinberg equilibrium (p < 1e-6) within POP_A
     (HWE is only expected to hold within a randomly-mating subpopulation,
     not across a stratified sample -- testing within a single subpopulation
     is the methodologically correct approach and is documented as such)

Output: data/genotypes_qc.npy, data/sample_metadata_qc.csv,
        data/variant_positions_qc.csv, results/qc_summary.csv
"""

import numpy as np
import pandas as pd
from scipy.stats import chi2

SEED = 42
MISSINGNESS_RATE = 0.02
SNP_CALL_RATE_THRESH = 0.05
IND_CALL_RATE_THRESH = 0.05
MAF_THRESH = 0.01
HWE_P_THRESH = 1e-6

rng = np.random.default_rng(SEED)

dosage = np.load("data/genotypes.npy").astype(float)
meta = pd.read_csv("data/sample_metadata.csv")
positions = pd.read_csv("data/variant_positions.csv")
n_ind, n_snps = dosage.shape

# --- Step 1: introduce missingness (marked as NaN) ---------------------------
missing_mask = rng.random(dosage.shape) < MISSINGNESS_RATE
dosage_missing = dosage.copy()
dosage_missing[missing_mask] = np.nan

# --- Step 2: SNP call-rate filter --------------------------------------------
snp_missing_rate = np.isnan(dosage_missing).mean(axis=0)
snp_pass_callrate = snp_missing_rate <= SNP_CALL_RATE_THRESH

# --- Step 3: individual call-rate filter -------------------------------------
ind_missing_rate = np.isnan(dosage_missing[:, snp_pass_callrate]).mean(axis=1)
ind_pass_callrate = ind_missing_rate <= IND_CALL_RATE_THRESH

# --- Step 4: MAF filter (computed on non-missing calls) ----------------------
d1 = dosage_missing[np.ix_(ind_pass_callrate, snp_pass_callrate)]
af = np.nanmean(d1, axis=0) / 2
maf = np.minimum(af, 1 - af)
snp_pass_maf = maf >= MAF_THRESH

# --- Step 5: HWE filter, tested within POP_A only -----------------------------
meta_pass = meta.loc[ind_pass_callrate].reset_index(drop=True)
d2 = d1[:, snp_pass_maf]
popA_rows = (meta_pass["population"] == "POP_A").values
d_popA = d2[popA_rows, :]


def hwe_chisq_pvalue(genotype_col):
    """HWE chi-square goodness-of-fit test for a single SNP column of
    {0,1,2} dosages (NaNs excluded). df=1 (3 genotype classes - 1 estimated
    allele-frequency parameter - 1)."""
    g = genotype_col[~np.isnan(genotype_col)]
    n = len(g)
    if n < 10:
        return 1.0
    obs_hom_ref = np.sum(g == 0)
    obs_het = np.sum(g == 1)
    obs_hom_alt = np.sum(g == 2)
    p = (2 * obs_hom_ref + obs_het) / (2 * n)
    q = 1 - p
    exp_hom_ref = (p ** 2) * n
    exp_het = 2 * p * q * n
    exp_hom_alt = (q ** 2) * n
    exp = np.array([exp_hom_ref, exp_het, exp_hom_alt])
    obs = np.array([obs_hom_ref, obs_het, obs_hom_alt])
    exp = np.where(exp == 0, 1e-6, exp)
    stat = np.sum((obs - exp) ** 2 / exp)
    return 1 - chi2.cdf(stat, df=1)


hwe_pvals = np.array([hwe_chisq_pvalue(d_popA[:, j]) for j in range(d_popA.shape[1])])
snp_pass_hwe = hwe_pvals >= HWE_P_THRESH

final_snp_mask = snp_pass_hwe
d_final = d2[:, final_snp_mask]
col_means = np.nanmean(d_final, axis=0)
nan_inds = np.where(np.isnan(d_final))
d_final[nan_inds] = np.take(col_means, nan_inds[1])

orig_idx_after_callrate = np.where(snp_pass_callrate)[0]
orig_idx_after_maf = orig_idx_after_callrate[snp_pass_maf]
orig_idx_final = orig_idx_after_maf[final_snp_mask]

positions_final = positions.set_index("snp_index").loc[orig_idx_final].reset_index()
positions_final["snp_index_qc"] = np.arange(len(positions_final))

meta_final = meta_pass.reset_index(drop=True)

np.save("data/genotypes_qc.npy", d_final.astype(np.float32))
meta_final.to_csv("data/sample_metadata_qc.csv", index=False)
positions_final.to_csv("data/variant_positions_qc.csv", index=False)

summary = pd.DataFrame([{
    "individuals_before_qc": n_ind,
    "individuals_after_qc": int(ind_pass_callrate.sum()),
    "snps_before_qc": n_snps,
    "snps_after_callrate_filter": int(snp_pass_callrate.sum()),
    "snps_after_maf_filter": int(snp_pass_maf.sum()),
    "snps_after_hwe_filter": int(final_snp_mask.sum()),
    "missingness_rate_simulated": MISSINGNESS_RATE,
    "maf_threshold": MAF_THRESH,
    "hwe_p_threshold": HWE_P_THRESH,
}])
summary.to_csv("results/qc_summary.csv", index=False)

print(summary.to_string(index=False))
print("Saved: data/genotypes_qc.npy, data/sample_metadata_qc.csv, "
      "data/variant_positions_qc.csv, results/qc_summary.csv")
