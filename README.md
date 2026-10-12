# Reproducible explainable-ML benchmark for UPJO in horseshoe kidney (synthetic cohort)

Code, data and results accompanying the manuscript
"A Reproducible Explainable Machine-Learning Benchmark for Ureteropelvic Junction Obstruction in Horseshoe Kidney Using a Synthetic Cohort".

**Important:** `data/HSK_UPJO_synthetic_1000.csv` is a *synthetic* dataset. It contains no real patient information and must not be used for clinical decisions. The script that originally generated it was not retained; the label structure was characterized post hoc (see manuscript, Sections III-D and V).

## Contents
- `data/HSK_UPJO_synthetic_1000.csv` – synthetic cohort (1,000 records, 18 columns).
- `scripts/hsk_noise_experiments.py` – baselines, repeated CV, label-noise and measurement-noise experiments (Tables III–V, Fig. 11).
- `scripts/ckd_pipeline.py` – pipeline on the public UCI Chronic Kidney Disease dataset (Table VI).
- `scripts/UPJO_original_analysis_notebook.ipynb` – original notebook (holdout split, Table II, SHAP figures).
- `results/` – output tables and figures.

## Reproduce
```bash
pip install -r requirements.txt
python scripts/hsk_noise_experiments.py data/HSK_UPJO_synthetic_1000.csv results/
# UCI CKD (download chronic_kidney_disease.arff from https://archive.ics.uci.edu/dataset/336/chronic+kidney+disease)
python scripts/ckd_pipeline.py chronic_kidney_disease.arff results/
```
Random seed 42 throughout. The UCI CKD dataset is not redistributed here; cite: L. J. Rubini, P. Soundarapandian, P. Eswaran, "Chronic Kidney Disease," UCI Machine Learning Repository, 2015.

## License
[Choose before publishing, e.g. MIT for code and CC BY 4.0 for data.]

## Citation
[Add manuscript citation / DOI after publication.]
