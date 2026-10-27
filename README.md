# Data pitfalls in explainable ML for UPJ obstruction in horseshoe kidney (synthetic cohort)

Code, data and results accompanying the manuscript
"Data Pitfalls in Explainable ML for UPJ Obstruction in Horseshoe Kidney" (IJASIT, submission 67, revised version).

**Important:** the HSK–UPJO files are *synthetic*. They contain no real patient information and must not be used for clinical decisions. The script that originally generated the data was not retained; the label-generating structure is characterized a posteriori in the manuscript (Sections III-B and IV-D). What this repository makes reproducible is the **analysis**: every number, table and data-driven figure in the paper.

## Contents
- `data/HSK_UPJO_synthetic_1000.csv` – synthetic cohort as originally distributed (1,000 records, 18 columns).
- `HSK_UPJO_synthetic_1000_v2.csv` – corrected release: `Surgical_modality` and `Postoperative_DRF_percent` are blank when `UPJO = 0`. Neither field is used by any model, so results are identical for both files.
- `data/urine.csv` – urine-analysis data (Andrews & Herzberg, 1985; boot R package, via Rdatasets).
- `scripts/reproduce_all.py` – **single entry point** for the synthetic cohort and the urine data: audit, univariate tests, holdout benchmark with bootstrap CIs (Table IV, Figs. 5–6), baselines and repeated CV (Table V), 10-draw label- and measurement-noise experiments (Tables VI–VII), gain importance and TreeSHAP (Figs. 7–11), urine-analysis benchmark (Table IX), plus Figs. 3–4.
- `scripts/make_extra_figures.py` – renogram schematic (Fig. 2) and the noise-sensitivity figure (Fig. 12).
- `scripts/ckd_pipeline.py` – UCI Chronic Kidney Disease benchmark (Table VIII).
- `scripts/hsk_noise_experiments.py` – earlier single-draw version of the noise experiments (kept for transparency; superseded by `reproduce_all.py`).
- `scripts/UPJO_original_analysis_notebook.ipynb` – original exploratory Colab notebook (earlier library versions, no standardization for LR/SVM; holdout figures differ by ≤ 3 test cases per model).
- `results/` – output tables (CSV), `summary.json` with every scalar result, and 600-dpi figures.

## Reproduce
```bash
pip install -r requirements.txt
python scripts/reproduce_all.py data/HSK_UPJO_synthetic_1000.csv data/urine.csv results/
python scripts/make_extra_figures.py results/
# UCI CKD: download chronic_kidney_disease.arff from https://archive.ics.uci.edu/dataset/336/chronic+kidney+disease
python scripts/ckd_pipeline.py chronic_kidney_disease.arff results/
```
Seed 42 throughout. Runtime of `reproduce_all.py` is about 15 min on two CPU cores (mostly the 10-draw noise experiments).

## Model configuration (all synthetic-cohort and urine experiments)
| Model | Configuration |
|---|---|
| Logistic Regression | L2, C = 1.0, lbfgs, max_iter = 1000; standardized inputs |
| SVM | RBF, C = 1.0, gamma = "scale", Platt probabilities; standardized inputs |
| Random Forest | 300 trees, class_weight = "balanced" |
| XGBoost | 300 trees, max_depth = 4, learning_rate = 0.05, subsample = 0.8, colsample_bytree = 0.8 |

Categorical variables: one-hot, drop = "first", handle_unknown = "ignore" (11 indicators); 5 numeric inputs; M = 16. Holdout: stratified 80:20. CV: stratified 5-fold × 3 repeats (seeds 42–44). Noise experiments: 10 independent draws, each assessed with stratified 5-fold CV.

## License
[Choose before publishing, e.g. MIT for code and CC BY 4.0 for data.]

## Citation
[Add manuscript citation / DOI after publication.]
