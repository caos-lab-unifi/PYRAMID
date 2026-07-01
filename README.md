# PYRAMID
**PYthon Radiomics And Machine learnIng Data analysis**

PYRAMID is a modular command-line toolkit for radiomics-based machine learning workflows, covering the full path from raw feature matrices to trained and validated predictive models.

It is designed to work on quantitative imaging features extracted with [PyRadiomics](https://pyradiomics.readthedocs.io), and supports both binary classification and survival analysis pipelines.

---

## Modules

| Module | Description |
|--------|-------------|
| **DaTrax** | Feature-space transformation (Z-score, robust, Yeo-Johnson, Box-Cox, quantile) |
| **SLIC** | Class imbalance correction (SMOTE, ADASYN, SMOTETomek, SMOTEEN, AllKNN, CNN, RENN) |
| **FS** | Feature selection with per-algorithm ranked lists (ANOVA, MI, RFECV, Lasso, ElasticNet, Agglomerative) |
| **FeatX** | Consensus feature selection across algorithms → single reproducible feature set |
| **HyPerTune** | Hyperparameter tuning via GridSearchCV or RandomizedSearchCV |
| **PREDICT** | Validation on held-out test set with optional permutation testing |
| **ENSEMBLE** | Aggregation of top-performing pipelines by posterior probability averaging |
| **SURV** | Survival analysis (CoxPH, penalised Cox, RSF, GBS, CWGB, SSVM, archetypal analysis) |

---

## Installation

PYRAMID requires **Python ≥ 3.10**.

```bash
conda create -y -n pyramidenv "python>3.10"
conda activate pyramidenv
git clone https://github.com/caos-lab-unifi/PYRAMID.git
cd PYRAMID
pip install .
```

Verify the installation:

```bash
PYRAMID --help
```

---

## Documentation

Full documentation is available at: [https://pyramid-caos.readthedocs.io](https://pyramid-caos.readthedocs.io)

---
