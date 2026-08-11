# PyTorch Deep Learning — Genotype-to-Trait Prediction

A hands-on PyTorch project, built by reusing the real simulated genotype/phenotype
data from the `gwas-population-genetics` project. Trains a neural network to predict
the simulated trait from genotype, and honestly compares it to XGBoost on an
identical held-out test set.

## What's here

```
scripts/
  lesson_01_pytorch_fundamentals.py   # tensors, autograd, layers, training loop -- from scratch
  lesson_02_build_the_model.py        # the real network architecture, and why it's designed this way
  lesson_03_train_and_compare.py      # real training + honest PyTorch vs. XGBoost comparison
results/
  training_curves.png                  # training/test loss curves + prediction scatter plots
  model_comparison.csv                 # final test R^2 / RMSE for both models
```

## Running it

```bash
pip install -r requirements.txt
./run_all.sh
```

Requires the `gwas-population-genetics` project's `data/` folder to exist one directory up
(reuses `genotypes_qc.npy` and `phenotype.csv` — no new data generation needed).

## Actual results from this run

| Model | Test R² | Test RMSE |
|---|---|---|
| PyTorch neural network | 0.193 | 1.430 |
| XGBoost | 0.308 | 1.325 |

## The takeaway

With only 300 individuals and 10,855 SNPs, XGBoost outperformed the neural network —
a real, expected result for small tabular genomics data, not a failure of the deep
learning implementation. The training curves show textbook overfitting (training
loss keeps falling while test loss plateaus), which is itself the point: on datasets
this size relative to feature count, that gap is very hard to fully close, and
knowing *when* deep learning is and isn't the right tool is a stronger signal than
defaulting to it. Deep learning's real advantages emerge with larger sample sizes or
data with spatial/sequential structure (imaging, raw sequence, text) that a plain
feedforward network on tabular SNP dosages doesn't exploit.

## Honest limitations

- Small dataset (n=300) — results will have high variance across different random
  train/test splits; this is a learning exercise, not a claim of state-of-the-art
  genomic prediction.
- No hyperparameter tuning was done for either model — this is a first-pass,
  reasonably-configured comparison, not an optimized benchmark.
- CPU-only training (no GPU used or needed at this data scale).
