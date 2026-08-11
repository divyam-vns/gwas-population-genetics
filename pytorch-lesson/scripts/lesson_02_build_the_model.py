"""
LESSON 2 — Building the real model: predict the simulated trait from genotype.

Run this with: python3 scripts/lesson_02_build_the_model.py

We reuse the EXACT same simulated data from the GWAS project
(../gwas-population-genetics/data/). This is real, honest reuse -- not a toy
dataset invented for this lesson.

Task: given a person's genotype (10,855 SNPs), predict their trait value.
This is a REGRESSION problem (predicting a continuous number), same as the
linear regression used in the GWAS association test -- except a neural
network can, in principle, learn nonlinear and interaction effects between
SNPs that a plain linear model cannot.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

GWAS_DATA_DIR = "/home/claude/gwas-population-genetics/data"

# ---------------------------------------------------------------------------
# Load the real simulated data
# ---------------------------------------------------------------------------
dosage = np.load(f"{GWAS_DATA_DIR}/genotypes_qc.npy").astype(np.float32)
meta = pd.read_csv(f"{GWAS_DATA_DIR}/sample_metadata_qc.csv")
pheno = pd.read_csv(f"{GWAS_DATA_DIR}/phenotype.csv").set_index("sample_id").loc[meta["sample_id"]].reset_index()
trait = pheno["trait"].values.astype(np.float32)

n_individuals, n_snps = dosage.shape
print(f"Loaded {n_individuals} individuals x {n_snps} SNPs (the same QC'd data from the GWAS project)")

# ---------------------------------------------------------------------------
# Why we design the network the way we do
# ---------------------------------------------------------------------------
print("""
Design decisions, and why:

  Input layer size = 10,855 (one input per SNP -- the genotype dosage 0/1/2)
  Hidden layer 1: 64 neurons, ReLU activation
  Hidden layer 2: 16 neurons, ReLU activation
  Output layer: 1 neuron, no activation (we want a raw number, the trait value)

  Why so few hidden neurons for so many inputs? With only 300 individuals
  total, a large network would simply memorize the training data instead of
  learning a generalizable pattern (this is called OVERFITTING -- it's the
  single biggest practical risk in deep learning on small datasets, and
  genomics datasets are very often small relative to the number of SNPs).
  Keeping the network small is a deliberate, defensible choice here, not
  a limitation we're hiding.
""")


class GenotypeTraitNet(nn.Module):
    """A small feedforward neural network: SNPs in, trait prediction out."""

    def __init__(self, n_snps):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(n_snps, 64),
            nn.ReLU(),
            nn.Dropout(0.3),     # randomly "turn off" 30% of neurons each training step
                                  # -- another overfitting safeguard, explained below
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.layers(x).squeeze(-1)


model = GenotypeTraitNet(n_snps)
print(model)

n_params = sum(p.numel() for p in model.parameters())
print(f"\nTotal trainable parameters: {n_params:,}")
print(f"Compare to: only {n_individuals} training examples available.")
print("This gap is exactly why Dropout and a held-out test set (lesson_03) matter.")

print("""
What Dropout does, in plain terms: during training, at each step, it randomly
"switches off" a fraction of neurons in that layer, forcing the remaining
ones to not over-rely on any single neuron. It's a standard, simple, and
effective regularization technique for small datasets. It's automatically
turned OFF during evaluation (model.eval()), so predictions are stable.
""")
