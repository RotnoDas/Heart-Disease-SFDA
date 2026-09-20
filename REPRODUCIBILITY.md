# Reproducibility

## Study design

Four UCI Heart Disease clinical centers are treated as separate domains:

- Cleveland
- Hungary
- Switzerland
- VA Long Beach

All 12 directional source-to-target transfers are evaluated.

## Outcome

Binary heart-disease outcome:

- 0: num == 0
- 1: num > 0

## Primary feature set

CORE8:

- age
- sex
- cp
- trestbps
- restecg
- thalach
- exang
- oldpeak

## Source-only preprocessing

All preprocessing is fitted using source-domain data only.

Numerical variables:
- median imputation
- standardization

Categorical variables:
- most-frequent imputation
- one-hot encoding

Target labels are never used during preprocessing or adaptation.

## Single-head source training epochs

- Cleveland: 34
- Hungary: 32
- Switzerland: 44
- VA Long Beach: 33

These epochs were selected using source-only cross-validation.

## Evaluation

- 10 source-model initialization seeds
- 5-fold unlabeled target cross-fitting
- threshold = 0.5
- hierarchical paired bootstrap = 5000 iterations

## Final single-head methods

- source-only
- pseudo-label SFDA
- entropy SFDA
- reliability-gated v1
- conservative candidate

## DLAR-LCL

DLAR-LCL uses a separate dual-head architecture.

Its adaptation effect is evaluated only relative to its own
architecture-matched dual-head source-only reference.

## Statistical interpretation

Positive delta AUROC:
adapted AUROC - source-only AUROC

Positive Brier improvement:
source-only Brier - adapted Brier

95% hierarchical bootstrap intervals are exploratory.
No multiplicity correction was applied.

## Analysis freeze

No adaptation hyperparameters or methods are modified after the
final multi-source evaluation.