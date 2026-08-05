"""
01_simulate_population.py

Simulates genotype data for a structured human-like population using msprime
(coalescent simulation), producing realistic linkage disequilibrium and
population stratification.

Demographic model:
  - Ancestral population splits into POP_A and POP_B `split_time_gens` generations ago
  - Low symmetric migration between them after the split
  - Effective population size Ne is constant within each population (simplification)

Output: data/genotypes.npy (individuals x SNPs, values in {0,1,2} = ALT allele dosage)
        data/sample_metadata.csv (sample_id, population)
        data/variant_positions.csv (SNP index, genomic position)
"""

import msprime
import numpy as np
import pandas as pd

SEED = 42
N_PER_POP = 150          # diploid individuals per population -> 300 total founders
NE = 10_000              # effective population size per subpopulation
SPLIT_TIME_GENS = 2000   # generations since population split
MIGRATION_RATE = 1e-5    # symmetric migration rate per generation
SEQ_LENGTH = 5_000_000   # 5 Mb region simulated
RECOMB_RATE = 1e-8       # per bp per generation, human-like
MUT_RATE = 1e-8          # per bp per generation, human-like

demography = msprime.Demography()
demography.add_population(name="POP_A", initial_size=NE)
demography.add_population(name="POP_B", initial_size=NE)
demography.add_population(name="ANC", initial_size=NE)
demography.set_symmetric_migration_rate(["POP_A", "POP_B"], MIGRATION_RATE)
demography.add_population_split(time=SPLIT_TIME_GENS, derived=["POP_A", "POP_B"], ancestral="ANC")

ts = msprime.sim_ancestry(
    samples={"POP_A": N_PER_POP, "POP_B": N_PER_POP},
    demography=demography,
    sequence_length=SEQ_LENGTH,
    recombination_rate=RECOMB_RATE,
    random_seed=SEED,
)
ts = msprime.sim_mutations(ts, rate=MUT_RATE, random_seed=SEED)

print(f"Simulated tree sequence: {ts.num_samples} haploid samples, {ts.num_sites} variant sites")

# Diploid individuals correspond to consecutive haploid sample-node pairs
# (0,1), (2,3), ... under msprime's default individual/node table layout,
# and individuals are ordered POP_A block followed by POP_B block for a
# samples={"POP_A": n, "POP_B": n} specification. Both properties are relied
# on below and are standard msprime tree-sequence conventions.
geno_haploid = ts.genotype_matrix()  # shape: (num_sites, num_haploid_samples)
dosage = geno_haploid[:, 0::2] + geno_haploid[:, 1::2]  # (num_sites, n_diploid)
dosage = dosage.T.astype(np.int8)  # (n_diploid, num_sites) = (individuals, SNPs)

# Filter to biallelic, non-monomorphic sites with MAF > 0.5% for a workable SNP set
maf = np.minimum(dosage.mean(axis=0) / 2, 1 - dosage.mean(axis=0) / 2)
keep = maf > 0.005
dosage = dosage[:, keep]
positions = np.array([s.position for s in ts.sites()])[keep]

print(f"After MAF>0.5% filter: {dosage.shape[1]} SNPs retained across {dosage.shape[0]} individuals")

n_diploid = dosage.shape[0]
pop_labels = (["POP_A"] * N_PER_POP) + (["POP_B"] * N_PER_POP)
sample_ids = [f"IND{i:04d}" for i in range(n_diploid)]

meta = pd.DataFrame({"sample_id": sample_ids, "population": pop_labels})

np.save("data/genotypes.npy", dosage)
meta.to_csv("data/sample_metadata.csv", index=False)
pd.DataFrame({"snp_index": np.arange(dosage.shape[1]), "position_bp": positions}).to_csv(
    "data/variant_positions.csv", index=False
)

print("Saved: data/genotypes.npy, data/sample_metadata.csv, data/variant_positions.csv")
