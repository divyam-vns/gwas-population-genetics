# Population-Structured GWAS Simulation & Kinship Recovery Pipeline

## What this is

An end-to-end, executed pipeline that:

1. Simulates a realistic, population-structured human genotype dataset using coalescent theory (`msprime`)
2. Simulates a quantitative trait with known causal SNPs and a population-stratification confound
3. Runs standard GWAS QC (call-rate, MAF, Hardy-Weinberg equilibrium filtering)
4. LD-prunes the genotype matrix and runs PCA, showing it recovers the known population split from genotype data alone
5. Runs association testing two ways (naive vs. PC-corrected) to demonstrate why stratification correction matters, not just that it exists
6. Simulates individuals with known family relationships via literal Mendelian transmission, then recovers those relationships from genotype data alone using a from-scratch implementation of the KING-robust kinship estimator (Manichaikul et al. 2010)

## Why simulated data, not a public dataset

Built in an environment without network access to genomics data repositories (1000 Genomes/EBI/NCBI). `msprime` is genuine, widely-used population genetics simulation software — the same category of tool used in real population genetics research for method validation. Every result below is from code that actually ran.

## Scope and limitations

- This is a methods demonstration on simulated data, not a discovery analysis on real human genetics data.
- The demographic model (two-population split with migration) is deliberately simple.
- The kinship simulation uses an unphased, dosage-based transmission approximation (documented in `scripts/07_kinship_relatedness.py`): each parental allele is transmitted independently per locus with probability equal to half the parent's dosage, which is unbiased at each locus but does not model haplotype-block/linkage structure in transmission.
- A 5 Mb simulated region is a small fraction of a real genome-wide panel; absolute values (e.g. λGC magnitude) will not numerically match a real genome-wide GWAS.

## Pipeline design notes

- **LD pruning precedes PCA, but not the association scan.** PCA is computed on an LD-pruned SNP set (pairwise r² < 0.2 in a 50-SNP sliding window) so that ancestry components reflect genome-wide structure rather than being dominated by a handful of highly-correlated genomic blocks. The association scan itself is run on the full QC-passing SNP set, using the PCs computed from the pruned set as covariates — this is standard practice, not a shortcut.
- **Hardy-Weinberg equilibrium is tested within a single subpopulation (POP_A)**, not across the full stratified sample. HWE is only expected to hold under random mating within a subpopulation; testing it across a structured sample would incorrectly flag real ancestry-informative SNPs as QC failures.

## Results (from actual pipeline output)

### Population structure (PCA)
After LD pruning (10,855 → 2,214 SNPs), PC1 alone separates the two simulated populations by **1.97 standard deviations**, recovering the known population split from genotype data with no population labels provided to the PCA itself. See `results/plots/pca_population_structure.png`.

### Why stratification correction matters
| Model | Genomic inflation (λGC) | True causal SNPs recovered | Significant non-causal hits (FDR<0.05) |
|---|---|---|---|
| Naive: `trait ~ genotype` | **3.12** (inflated) | 4 / 5 | 1,040 |
| Corrected: `trait ~ genotype + PC1 + PC2` | **1.55** | 4 / 5 | 48 |

Both models recover the same 4 of 5 known causal SNPs at genome-wide significance, but the naive model produces far more spurious signal from unmodeled population structure — the failure mode PCA-based correction exists to prevent in real GWAS. See `results/plots/manhattan_naive_vs_corrected.png` and `qq_naive_vs_corrected.png`.

### Recovering known family relationships from genotype data alone
Using a from-scratch implementation of the KING-robust kinship estimator (no external library):

| Relationship (known by simulation) | Mean estimated kinship | Theoretical expected | n pairs |
|---|---|---|---|
| Parent–offspring | 0.255 | 0.25 | 80 |
| Full-sibling | 0.263 | 0.25 | 40 |
| Unrelated | −0.055 | 0.00 | 198 |

Related and unrelated pairs separate cleanly with no overlap between the unrelated distribution and the first-degree-relative cluster. See `results/plots/kinship_validation.png`.

## Repository structure

```
.
├── README.md
├── requirements.txt
├── run_all.sh                          # runs the full pipeline in order
├── scripts/
│   ├── 01_simulate_population.py       # msprime coalescent simulation, 2-population split
│   ├── 02_simulate_phenotype.py        # quantitative trait, known causal SNPs + confound
│   ├── 03_qc.py                        # call-rate, MAF, HWE filtering
│   ├── 04_pca_population_stratification.py   # LD pruning + PCA
│   ├── 05_association_testing.py       # naive vs. PC-corrected GWAS
│   ├── 06_manhattan_qq_plots.py
│   └── 07_kinship_relatedness.py       # Mendelian simulation + KING-robust estimator
├── data/                                # generated by running the pipeline
└── results/
    ├── plots/
    └── *.csv                            # all summary tables referenced above
```

## Running it

```bash
pip install -r requirements.txt
./run_all.sh
```

Runs in under a minute on a standard machine (300 individuals, ~10,000–12,000 SNPs).

## Methods referenced

- Kelleher J, Etheridge AM, McVean G. "Efficient coalescent simulation and genealogical analysis for large sample sizes." *PLoS Comput Biol.* 2016. (msprime)
- Patterson N, Price AL, Reich D. "Population structure and eigenanalysis." *PLoS Genet.* 2006. (PCA for population structure)
- Manichaikul A, Mychaleckyj JC, Rich SS, et al. "Robust relationship inference in genome-wide association studies." *Bioinformatics.* 2010. (KING-robust kinship estimator)
