# Comparing Classifiers — Practical Application III

## Notebook
[prompt_III.ipynb](prompt_III.ipynb)

## Summary
This notebook applies CRISP-DM to the UCI Bank Marketing problem: predict whether a customer will subscribe to a term deposit (target: `y`, positive class: `yes`). The modeling objective is aligned to a high cost of false negatives (missed likely subscribers), so model selection and hyperparameter tuning prioritize recall via the $F_2$ score (β=2) while still checking for operationally-degenerate behavior (e.g., “predict everyone as `yes`”).

Key takeaways reflected in the notebook:
- A tuned Logistic Regression model (elastic net via `solver='saga'`) provides strong $F_2$ performance with an interpretable and operationally-feasible confusion matrix.
- Some high-recall configurations (notably certain SVC settings) can collapse into near-trivial “predict all positives” behavior; these are flagged as unsuitable for deployment despite high recall.
- Deployment is framed as a capacity-constrained targeting problem (threshold/top‑k outreach) and should be evaluated with an experiment (A/B test or clustered RCT) and an incremental-profit calculation.

## Dataset
- Source: UCI Bank Marketing dataset (see notebook for citations)
- Local file used: [data/bank-additional/bank-additional-full.csv](data/bank-additional/bank-additional-full.csv)

## Approach (CRISP-DM)
- **Business Understanding:** Define success in terms of prioritizing recall of `yes` outcomes (minimize costly false negatives), using $F_2$ as the primary tuning metric.
- **Data Understanding:** Review distributions, class balance, and feature availability constraints.
- **Data Preparation:** Train/test split with stratification; preprocessing via `ColumnTransformer` (imputation + scaling for numeric, imputation + one‑hot encoding for categorical with `handle_unknown='ignore'`).
- **Modeling:** Baseline plus multiple classifiers (Logistic Regression, KNN, Decision Tree, SVC).
- **Evaluation:** Accuracy/precision/recall/F1 and $F_2$ (positive class), plus confusion matrices and timing (fit/predict) to support operational feasibility.
- **Deployment:** Use a threshold/top‑k policy to match outreach capacity; monitor performance/drift; quantify value via incremental lift × margin − outreach cost.

## Results
The notebook produces:
- A baseline comparison table across candidate models.
- Separate `GridSearchCV` runs per model, all tuned with an $F_2$ scorer on the positive class (`yes`).
- A tuned-model comparison table (including timing) and a grid of tuned confusion matrices.

The selected recommendation in the notebook is the tuned Logistic Regression model because it balances high recall (as emphasized by $F_2$) with a non-degenerate prediction profile (see confusion matrix) and clear operational interpretability.

Note: Exact metric values depend on the notebook run (random state is fixed where applicable), so this README intentionally summarizes outcomes qualitatively; the notebook contains the tables/plots.

## Reproducibility
From the repository root:
1. Install dependencies: `pip install -r requirements.txt`
2. Open and run: `prompt_III.ipynb` (Run All).

### Data Processing Notes
To regenerate `CRISP-DM-BANK.txt` from a source `CRISP-DM-BANK.pdf` using OCR:
```bash
# Install dependencies (Ubuntu/Debian)
sudo apt-get install ocrmypdf poppler-utils

# Add OCR layer to PDF and extract text
ocrmypdf CRISP-DM-BANK.pdf - | pdftotext - CRISP-DM-BANK.txt
```

## Notes
- Selected features: `age`, `job`, `marital`, `education`, `default`, `housing`, `loan`.
- Leakage feature `duration` should always be excluded for realistic modeling.
- Supporting CRISP-DM reference material:
  - [CRISP-DM-BANK.txt](docs/CRISP-DM-BANK.pdf)
  - [crisp-dm-manual.pdf](docs/CRISP-DM_overview/crisp-dm-manual.pdf)
  - [crisp-dm-overview.pdf](docs/CRISP-DM_overview/crisp-dm-overview.pdf)
