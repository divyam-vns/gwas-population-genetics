"""
LESSON 3 — Train the PyTorch network for real, train XGBoost on the same
data, and compare them HONESTLY on data neither model has seen.

Run with: python3 scripts/lesson_03_train_and_compare.py

This is the actual deliverable: real training, real held-out evaluation,
real numbers -- not a narrated example.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import xgboost as xgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

GWAS_DATA_DIR = "/home/claude/gwas-population-genetics/data"
RESULTS_DIR = "/home/claude/pytorch-lesson/results"
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# ---------------------------------------------------------------------------
# Load data (same as lesson_02)
# ---------------------------------------------------------------------------
dosage = np.load(f"{GWAS_DATA_DIR}/genotypes_qc.npy").astype(np.float32)
meta = pd.read_csv(f"{GWAS_DATA_DIR}/sample_metadata_qc.csv")
pheno = pd.read_csv(f"{GWAS_DATA_DIR}/phenotype.csv").set_index("sample_id").loc[meta["sample_id"]].reset_index()
trait = pheno["trait"].values.astype(np.float32)
n_individuals, n_snps = dosage.shape

# ---------------------------------------------------------------------------
# The single most important step for an honest comparison: split the data
# BEFORE training anything, and never let either model see the test set
# until final evaluation. This mirrors exactly what a real interviewer will
# expect you to explain and defend.
# ---------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    dosage, trait, test_size=0.2, random_state=SEED
)
print(f"Train set: {X_train.shape[0]} individuals | Test set: {X_test.shape[0]} individuals")
print("Both models are trained ONLY on the train set, and scored ONLY on the")
print("never-before-seen test set, using the identical split.\n")

# ---------------------------------------------------------------------------
# PART A: Train the PyTorch neural network
# ---------------------------------------------------------------------------
print("=" * 70)
print("PART A: Training the PyTorch neural network")
print("=" * 70)


class GenotypeTraitNet(nn.Module):
    def __init__(self, n_snps):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(n_snps, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 16), nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.layers(x).squeeze(-1)


X_train_t = torch.tensor(X_train)
y_train_t = torch.tensor(y_train)
X_test_t = torch.tensor(X_test)
y_test_t = torch.tensor(y_test)

model = GenotypeTraitNet(n_snps)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-3)
loss_fn = nn.MSELoss()

n_epochs = 150
train_losses, test_losses = [], []

for epoch in range(n_epochs):
    model.train()                      # turns Dropout ON
    optimizer.zero_grad()
    pred = model(X_train_t)
    loss = loss_fn(pred, y_train_t)
    loss.backward()
    optimizer.step()

    model.eval()                       # turns Dropout OFF for evaluation
    with torch.no_grad():
        test_pred = model(X_test_t)
        test_loss = loss_fn(test_pred, y_test_t)

    train_losses.append(loss.item())
    test_losses.append(test_loss.item())

    if epoch % 30 == 0 or epoch == n_epochs - 1:
        print(f"epoch {epoch:3d}  train_loss={loss.item():.3f}  test_loss={test_loss.item():.3f}")

model.eval()
with torch.no_grad():
    nn_test_pred = model(X_test_t).numpy()

nn_r2 = r2_score(y_test, nn_test_pred)
nn_rmse = np.sqrt(mean_squared_error(y_test, nn_test_pred))
print(f"\nNeural network -- test R^2: {nn_r2:.3f}   test RMSE: {nn_rmse:.3f}")

print("""
What to watch in the epoch printout above: if train_loss keeps dropping
while test_loss stops improving or gets WORSE, that's overfitting happening
live -- the network is memorizing the training examples rather than
learning a generalizable pattern. This is the single most important
diagnostic plot in practical deep learning; results/training_curves.png
shows it visually for this run.
""")

# ---------------------------------------------------------------------------
# PART B: Train XGBoost on the identical split, for an honest comparison
# ---------------------------------------------------------------------------
print("=" * 70)
print("PART B: Training XGBoost on the identical train/test split")
print("=" * 70)

xgb_model = xgb.XGBRegressor(
    n_estimators=200, max_depth=3, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.3, random_state=SEED,
)
xgb_model.fit(X_train, y_train)
xgb_test_pred = xgb_model.predict(X_test)

xgb_r2 = r2_score(y_test, xgb_test_pred)
xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_test_pred))
print(f"XGBoost -- test R^2: {xgb_r2:.3f}   test RMSE: {xgb_rmse:.3f}")

# ---------------------------------------------------------------------------
# PART C: Honest comparison
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("PART C: Honest comparison")
print("=" * 70)

summary = pd.DataFrame([
    {"model": "PyTorch neural network", "test_R2": round(nn_r2, 3), "test_RMSE": round(nn_rmse, 3)},
    {"model": "XGBoost", "test_R2": round(xgb_r2, 3), "test_RMSE": round(xgb_rmse, 3)},
])
print(summary.to_string(index=False))
summary.to_csv(f"{RESULTS_DIR}/model_comparison.csv", index=False)

# --- Plots -------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

axes[0].plot(train_losses, label="train loss")
axes[0].plot(test_losses, label="test loss")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("MSE loss")
axes[0].set_title("Neural network training curves")
axes[0].legend()

axes[1].scatter(y_test, nn_test_pred, alpha=0.6, color="#2E5B8A")
lims = [min(y_test.min(), nn_test_pred.min()), max(y_test.max(), nn_test_pred.max())]
axes[1].plot(lims, lims, "r--", linewidth=1)
axes[1].set_xlabel("True trait value")
axes[1].set_ylabel("Predicted trait value")
axes[1].set_title(f"Neural network (test R\u00b2={nn_r2:.3f})")

axes[2].scatter(y_test, xgb_test_pred, alpha=0.6, color="#C0392B")
axes[2].plot(lims, lims, "r--", linewidth=1)
axes[2].set_xlabel("True trait value")
axes[2].set_ylabel("Predicted trait value")
axes[2].set_title(f"XGBoost (test R\u00b2={xgb_r2:.3f})")

plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/training_curves.png", dpi=150)
print(f"\nSaved: {RESULTS_DIR}/model_comparison.csv, {RESULTS_DIR}/training_curves.png")

print("""
HONEST TAKEAWAY (write this in your own words for an interview):
With only 300 individuals and 10,855 SNPs, this is exactly the kind of
"wide, shallow" tabular dataset where tree-based methods like XGBoost
typically match or beat a from-scratch neural network -- deep learning's
real advantage shows up with much larger sample sizes, or with data that
has spatial/sequential structure (images, raw DNA sequence, text), which
a plain feedforward network on tabular SNP dosages does not exploit.
Knowing WHEN to reach for deep learning vs. gradient boosting -- and being
able to defend that choice with a real head-to-head result -- is a
stronger interview answer than only knowing how to build a neural network.
""")
