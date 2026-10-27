"""Single-entry reproduction script for the revised manuscript
"Pitfalls of Synthetic Data in Explainable Machine Learning for Rare Congenital
Urology: A Reproducible Case Study of UPJO in Horseshoe Kidney".

Regenerates every number, table and data-driven figure reported for the synthetic
cohort and the urine-analysis dataset (the UCI CKD analysis is in ckd_pipeline.py).

Usage
    python scripts/reproduce_all.py data/HSK_UPJO_synthetic_1000.csv data/urine.csv results/

Environment used for the revised manuscript (requirements.txt):
    Python 3.11, scikit-learn 1.9.1, xgboost 3.4.1, shap 0.54.1, pandas 3.0.5,
    numpy 2.5.3, scipy, matplotlib 3.11.2. Seed 42 throughout.
"""
import sys, os, json, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, chi2_contingency
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, roc_curve, confusion_matrix, brier_score_loss)
from xgboost import XGBClassifier
import shap

SEED = 42
N_BOOT = 2000
N_DRAWS = 10
csv, urine_csv, out = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(out, exist_ok=True)
R = {}  # all scalar results -> results/summary.json

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 8.5,
                     "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
                     "legend.fontsize": 7, "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.dpi": 600, "savefig.bbox": "tight", "axes.linewidth": 0.7})
COL = {"Logistic Regression": "#1f77b4", "Random Forest": "#2ca02c", "SVM (RBF)": "#9467bd", "XGBoost": "#d62728"}

# ---------------------------------------------------------------- data + audit
df_raw = pd.read_csv(csv)
R["audit"] = dict(rows=int(len(df_raw)), cols=int(df_raw.shape[1]),
                  missing=int(df_raw.isna().sum().sum()), duplicated_rows=int(df_raw.duplicated().sum()),
                  unique_ids=int(df_raw.Patient_ID.nunique()),
                  class_counts={int(k): int(v) for k, v in df_raw.UPJO.value_counts().items()},
                  nonobstructed_with_surgery=int(((df_raw.UPJO == 0) & df_raw.Surgical_modality.notna()).sum()),
                  nonobstructed_with_postop_drf=int(((df_raw.UPJO == 0) & df_raw.Postoperative_DRF_percent.notna()).sum()))
y = df_raw.UPJO.values
X = df_raw.drop(columns=["UPJO", "Patient_ID", "Postoperative_DRF_percent", "Surgical_modality"])
CAT = ["Sex", "Laterality", "Prenatal_hydronephrosis", "Initial_presentation", "Flank_pain",
       "Recurrent_UTI", "High_ureteral_insertion", "Crossing_vessel", "Renal_stone"]
NUM = ["Age_years", "Hydronephrosis_SFU_grade", "APD_mm", "Preoperative_DRF_percent", "Diuretic_renogram_T1_2_min"]
assert sorted(CAT + NUM) == sorted(X.columns)

# univariate audit
uni = {}
for c in ["APD_mm", "Diuretic_renogram_T1_2_min", "Preoperative_DRF_percent", "Age_years"]:
    a, b = X.loc[y == 0, c], X.loc[y == 1, c]
    u = mannwhitneyu(a, b)
    uni[c] = dict(med0=float(a.median()), med1=float(b.median()), U=float(u.statistic), p=float(u.pvalue))
for c in CAT + ["Hydronephrosis_SFU_grade"]:
    chi = chi2_contingency(pd.crosstab(X[c], y))
    uni[c] = dict(chi2=float(chi[0]), p=float(chi[1]))
R["univariate"] = uni
R["sfu_by_class"] = pd.crosstab(X.Hydronephrosis_SFU_grade, y).to_dict()


# ---------------------------------------------------------------- model factory
def make(name, num=NUM, cat=CAT, impute=False):
    """Canonical configuration used for EVERY experiment (holdout, CV, noise)."""
    scale = name in ("Logistic Regression", "SVM (RBF)")
    num_steps = ([("imp", SimpleImputer(strategy="median"))] if impute else []) + \
                ([("sc", StandardScaler())] if scale else [])
    num_tf = Pipeline(num_steps) if num_steps else "passthrough"
    tfs = [("num", num_tf, num)]
    if cat:
        tfs.append(("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat))
    model = {"Logistic Regression": LogisticRegression(max_iter=1000, random_state=SEED),
             "Random Forest": RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=SEED, n_jobs=-1),
             "SVM (RBF)": SVC(kernel="rbf", probability=True, random_state=SEED),
             "XGBoost": XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
                                      colsample_bytree=0.8, eval_metric="logloss", random_state=SEED, n_jobs=4),
             "Decision tree (depth 3)": DecisionTreeClassifier(max_depth=3, random_state=SEED)}[name]
    return Pipeline([("prep", ColumnTransformer(tfs)), ("model", model)])


MODELS = ["XGBoost", "Logistic Regression", "SVM (RBF)", "Random Forest"]


def boot_ci(yt, p, fn, n=N_BOOT, seed=SEED):
    rng = np.random.RandomState(seed); v = []
    for _ in range(n):
        i = rng.randint(0, len(yt), len(yt))
        if len(np.unique(yt[i])) < 2: continue
        v.append(fn(yt[i], p[i]))
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


# ---------------------------------------------------------------- A. holdout
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
R["split"] = dict(train=int(len(ytr)), train_pos=int(ytr.sum()), test=int(len(yte)), test_pos=int(yte.sum()))
hold, probs, fitted = {}, {}, {}
for n in MODELS:
    m = make(n).fit(Xtr, ytr); p = m.predict_proba(Xte)[:, 1]; yh = (p >= 0.5).astype(int)
    fitted[n], probs[n] = m, p
    hold[n] = dict(acc=accuracy_score(yte, yh), prec=precision_score(yte, yh), rec=recall_score(yte, yh),
                   f1=f1_score(yte, yh), auc=roc_auc_score(yte, p), brier=brier_score_loss(yte, p),
                   cm=confusion_matrix(yte, yh).tolist(),
                   auc_ci=boot_ci(yte, p, roc_auc_score),
                   acc_ci=boot_ci(yte, (p >= .5).astype(float), lambda a, b: accuracy_score(a, b)))
# ablation without DRF
num_nd = [c for c in NUM if c != "Preoperative_DRF_percent"]
m = make("XGBoost", num=num_nd).fit(Xtr.drop(columns="Preoperative_DRF_percent"), ytr)
p = m.predict_proba(Xte.drop(columns="Preoperative_DRF_percent"))[:, 1]; yh = (p >= .5).astype(int)
hold["XGBoost (without DRF)"] = dict(acc=accuracy_score(yte, yh), prec=precision_score(yte, yh), rec=recall_score(yte, yh),
                                     f1=f1_score(yte, yh), auc=roc_auc_score(yte, p), brier=brier_score_loss(yte, p),
                                     cm=confusion_matrix(yte, yh).tolist(), auc_ci=boot_ci(yte, p, roc_auc_score),
                                     acc_ci=boot_ci(yte, yh.astype(float), lambda a, b: accuracy_score(a, b)))
R["holdout"] = hold
pd.DataFrame(hold).T.to_csv(f"{out}/table_holdout.csv")

# ---------------------------------------------------------------- B. repeated CV + baselines
def cv(Xd, ytrain, yeval, name, num, cat, reps=3):
    aucs, accs, oof_brier = [], [], []
    for r in range(reps):
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=SEED + r).split(Xd, y):
            p = make(name, num, cat).fit(Xd.iloc[tr], ytrain[tr]).predict_proba(Xd.iloc[te])[:, 1]
            aucs.append(roc_auc_score(yeval[te], p)); accs.append(accuracy_score(yeval[te], p >= .5))
            oof_brier.append(brier_score_loss(yeval[te], p))
    return dict(auc=float(np.mean(aucs)), auc_sd=float(np.std(aucs)), acc=float(np.mean(accs)),
                acc_sd=float(np.std(accs)), brier=float(np.mean(oof_brier)))

cvres = {}
for lab, cols in {"APD_mm only (LR)": ["APD_mm"], "T1/2 only (LR)": ["Diuretic_renogram_T1_2_min"],
                  "APD_mm + T1/2 (LR)": ["APD_mm", "Diuretic_renogram_T1_2_min"]}.items():
    cvres[lab] = cv(X[cols], y, y, "Logistic Regression", cols, [])
cvres["APD_mm + T1/2 (depth-3 tree)"] = cv(X[["APD_mm", "Diuretic_renogram_T1_2_min"]], y, y,
                                           "Decision tree (depth 3)", ["APD_mm", "Diuretic_renogram_T1_2_min"], [])
for n in ["Logistic Regression", "Random Forest", "SVM (RBF)", "XGBoost"]:
    cvres["All 14: " + n] = cv(X, y, y, n, NUM, CAT); print("cv", n, cvres["All 14: " + n], flush=True)
R["cv"] = cvres
pd.DataFrame(cvres).T.to_csv(f"{out}/table_cv_baselines.csv")
# the rule the depth-3 tree learns (post-hoc characterisation of the label-generating structure)
tree = make("Decision tree (depth 3)", ["APD_mm", "Diuretic_renogram_T1_2_min"], []).fit(X[["APD_mm", "Diuretic_renogram_T1_2_min"]], y)
from sklearn.tree import export_text
R["tree_rule"] = export_text(tree.named_steps["model"], feature_names=["APD_mm", "T1_2"])

# ---------------------------------------------------------------- C. noise experiments (10 independent draws)
def cv5(Xd, ytrain, yeval, name, seed):
    a = []
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(Xd, y):
        p = make(name).fit(Xd.iloc[tr], ytrain[tr]).predict_proba(Xd.iloc[te])[:, 1]
        a.append(roc_auc_score(yeval[te], p))
    return float(np.mean(a))

noise = []
NM = ["Logistic Regression", "Random Forest", "XGBoost"]
for f in [0, .05, .10, .20, .30]:
    for d in range(N_DRAWS):
        rng = np.random.RandomState(1000 + d)
        yn = y.copy(); idx = rng.rand(len(y)) < f; yn[idx] = 1 - yn[idx]
        for n in NM:
            noise.append(dict(exp="label_observed", level=f, draw=d, model=n, auc=cv5(X, yn, yn, n, SEED + d)))
            noise.append(dict(exp="label_train_only", level=f, draw=d, model=n, auc=cv5(X, yn, y, n, SEED + d)))
    print("label noise", f, flush=True)
for k in [0, .25, .5, 1.0]:
    for d in range(N_DRAWS):
        rng = np.random.RandomState(2000 + d); Xn = X.copy()
        for c in ["APD_mm", "Diuretic_renogram_T1_2_min"]:
            Xn[c] = Xn[c] + rng.normal(0, k * X[c].std(), len(X))
        for n in NM:
            noise.append(dict(exp="measurement", level=k, draw=d, model=n, auc=cv5(Xn, y, y, n, SEED + d)))
    print("meas noise", k, flush=True)
N = pd.DataFrame(noise); N.to_csv(f"{out}/noise_draws.csv", index=False)
NS = N.groupby(["exp", "model", "level"]).auc.agg(["mean", "std"]).reset_index()
NS.to_csv(f"{out}/table_noise_summary.csv", index=False)

# ---------------------------------------------------------------- D. explainability
xgb = fitted["XGBoost"]; prep = xgb.named_steps["prep"]; booster = xgb.named_steps["model"]
fn = [s.replace("num__", "").replace("cat__", "") for s in prep.get_feature_names_out()]
R["n_model_features"] = len(fn)
Xte_t = prep.transform(Xte); Xte_t = Xte_t.toarray() if hasattr(Xte_t, "toarray") else Xte_t
gain = pd.Series(booster.feature_importances_, index=fn).sort_values(ascending=False)
R["gain_importance"] = gain.round(4).to_dict()
explainer = shap.TreeExplainer(booster)
sv = explainer.shap_values(Xte_t)
mabs = pd.Series(np.abs(sv).mean(0), index=fn).sort_values(ascending=False)
R["shap_mean_abs"] = mabs.round(4).to_dict()
R["shap_base"] = float(np.ravel(explainer.expected_value)[0])

# ---------------------------------------------------------------- E. urine-analysis dataset
U = pd.read_csv(urine_csv)
U = U.drop(columns=[c for c in U.columns if c.lower().startswith("rownames") or c.startswith("Unnamed")])
yu = U.pop("r").values.astype(int); UNUM = list(U.columns)
R["urine_audit"] = dict(n=int(len(yu)), pos=int(yu.sum()), missing_cells=int(U.isna().sum().sum()),
                        missing_frac=float(U.isna().sum().sum() / U.size))
ures = {}
for n in ["Logistic Regression", "Random Forest", "SVM (RBF)", "XGBoost"]:
    aucs, accs, oof = [], [], np.zeros((5, len(yu)))
    for r in range(5):
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=SEED + r).split(U, yu):
            p = make(n, UNUM, [], impute=True).fit(U.iloc[tr], yu[tr]).predict_proba(U.iloc[te])[:, 1]
            oof[r, te] = p; aucs.append(roc_auc_score(yu[te], p)); accs.append(accuracy_score(yu[te], p >= .5))
    ures[n] = dict(acc=float(np.mean(accs)), acc_sd=float(np.std(accs)), auc=float(np.mean(aucs)),
                   auc_sd=float(np.std(aucs)), auc_ci=boot_ci(yu, oof.mean(0), roc_auc_score))
for c in UNUM:
    aucs = []
    for r in range(5):
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=SEED + r).split(U, yu):
            p = make("Logistic Regression", [c], [], impute=True).fit(U.iloc[tr][[c]], yu[tr]).predict_proba(U.iloc[te][[c]])[:, 1]
            aucs.append(roc_auc_score(yu[te], p))
    ures["single: " + c] = dict(auc=float(np.mean(aucs)))
mu = make("XGBoost", UNUM, [], impute=True).fit(U, yu)
Ut = mu.named_steps["prep"].transform(U)
svu = shap.TreeExplainer(mu.named_steps["model"]).shap_values(Ut)
ures["shap_mean_abs"] = dict(zip(UNUM, np.abs(svu).mean(0).round(3).tolist()))
R["urine"] = ures

json.dump(R, open(f"{out}/summary.json", "w"), indent=1, default=float)
print(json.dumps({k: R[k] for k in ["audit", "split", "holdout", "cv"]}, indent=1, default=float))

# ================================================================ FIGURES (600 dpi)
W1 = 3.5  # single IEEE column, inches

# Fig: class distribution
fig, ax = plt.subplots(figsize=(W1, 2.3))
cnt = df_raw.UPJO.value_counts().sort_index()
b = ax.bar(["Class 0\n(no UPJO)", "Class 1\n(UPJO)"], cnt.values, color=["#7f9fbf", "#c0504d"], width=.55)
for r_, v in zip(b, cnt.values):
    ax.text(r_.get_x() + r_.get_width() / 2, v + 10, f"{v} ({v / 10:.1f}%)", ha="center", fontsize=7.5)
ax.set_ylabel("Number of synthetic records"); ax.set_ylim(0, 760)
fig.savefig(f"{out}/fig_class_distribution.png"); plt.close(fig)

# Fig: SFU grade vs class
ct = pd.crosstab(X.Hydronephrosis_SFU_grade, y)
fig, ax = plt.subplots(figsize=(W1, 2.3)); w = .38; xs = np.arange(len(ct))
for j, (lab, c) in enumerate([("Class 0 (no UPJO)", "#7f9fbf"), ("Class 1 (UPJO)", "#c0504d")]):
    bb = ax.bar(xs + (j - .5) * w, ct[j].values, w, label=lab, color=c)
    for r_, v in zip(bb, ct[j].values): ax.text(r_.get_x() + w / 2, v + 6, str(v), ha="center", fontsize=7)
ax.set_xticks(xs); ax.set_xticklabels([f"Grade {g}" for g in ct.index]); ax.set_ylabel("Number of records")
ax.set_xlabel("SFU hydronephrosis grade"); ax.legend(frameon=False); ax.set_ylim(0, 430)
fig.savefig(f"{out}/fig_sfu_by_class.png"); plt.close(fig)

# Fig: confusion matrices 2x2
fig, axs = plt.subplots(2, 2, figsize=(W1, 3.3))
for ax, n in zip(axs.ravel(), MODELS):
    cm = np.array(hold[n]["cm"]); ax.imshow(cm, cmap="Blues", vmin=0, vmax=140)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{['TN','FP','FN','TP'][2*i+j]}\n{cm[i, j]}", ha="center", va="center",
                    fontsize=8, color="white" if cm[i, j] > 70 else "black")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1]); ax.set_xticklabels(["0", "1"]); ax.set_yticklabels(["0", "1"])
    ax.set_title(n, fontsize=8); ax.set_xlabel("Predicted", fontsize=7); ax.set_ylabel("True", fontsize=7)
    for s in ax.spines.values(): s.set_visible(False)
fig.tight_layout(); fig.savefig(f"{out}/fig_confusion_matrices.png"); plt.close(fig)

# Fig: ROC curves (all four models; inset zoom)
fig, ax = plt.subplots(figsize=(W1, 3.0))
for n in MODELS:
    f_, t_, _ = roc_curve(yte, probs[n]); ax.plot(f_, t_, lw=1.2, color=COL[n], label=f"{n} (AUC = {hold[n]['auc']:.3f})")
ax.plot([0, 1], [0, 1], ls="--", lw=.8, color="grey", label="Chance")
ax.set_xlabel("False positive rate (1 − specificity)"); ax.set_ylabel("True positive rate (sensitivity)")
ax.legend(loc="lower right", frameon=False, fontsize=6.5); ax.set_xlim(-.01, 1); ax.set_ylim(0, 1.01)
ins = ax.inset_axes([.40, .50, .52, .30])
for n in MODELS:
    f_, t_, _ = roc_curve(yte, probs[n]); ins.plot(f_, t_, lw=1.1, color=COL[n])
ins.set_xlim(0, .15); ins.set_ylim(.8, 1.005); ins.tick_params(labelsize=6); ins.set_title("Zoom: FPR 0–0.15", fontsize=6.5)
fig.savefig(f"{out}/fig_roc.png"); plt.close(fig)

# Fig: gain importance (top 12, readable names)
def pretty(s):
    return (s.replace("Diuretic_renogram_T1_2_min", "Renogram T½ (min)").replace("Preoperative_DRF_percent", "Preop. DRF (%)")
             .replace("Hydronephrosis_SFU_grade", "SFU grade").replace("APD_mm", "APD (mm)").replace("Age_years", "Age (years)")
             .replace("_", " "))
g = gain.head(12)[::-1]
fig, ax = plt.subplots(figsize=(W1, 2.9))
ax.barh([pretty(s) for s in g.index], g.values, color="#4c72b0")
for i, v in enumerate(g.values): ax.text(v + .003, i, f"{v:.3f}", va="center", fontsize=6.5)
ax.set_xlabel("XGBoost gain-based importance (normalised)"); ax.set_xlim(0, g.max() * 1.18)
fig.savefig(f"{out}/fig_gain_importance.png"); plt.close(fig)

# Fig: SHAP beeswarm (drawn directly with matplotlib so labels stay legible at column width)
order = np.argsort(np.abs(sv).mean(0))[::-1][:10][::-1]
fig, ax = plt.subplots(figsize=(W1, 3.3)); rs = np.random.RandomState(SEED)
for row, k in enumerate(order):
    v = Xte_t[:, k].astype(float); lo, hi = np.nanpercentile(v, 5), np.nanpercentile(v, 95)
    cv_ = np.clip((v - lo) / (hi - lo + 1e-12), 0, 1)
    # density-aware vertical jitter
    xs = sv[:, k]; bins = np.round(xs / (np.ptp(sv) / 80)); jit = np.zeros(len(xs))
    for b in np.unique(bins):
        idx = np.where(bins == b)[0]; n_ = len(idx)
        jit[idx] = (np.arange(n_) - (n_ - 1) / 2) * min(.06, .38 / max(n_, 1)) * rs.permutation(n_) / max(n_ - 1, 1) * 2 if n_ > 1 else 0
    sc = ax.scatter(xs, row + jit, c=cv_, cmap="coolwarm", s=6, lw=0, vmin=0, vmax=1)
ax.axvline(0, color="grey", lw=.6)
ax.set_yticks(range(len(order))); ax.set_yticklabels([pretty(fn[k]) for k in order], fontsize=7)
ax.set_xlabel("SHAP value (impact on log-odds of UPJO)"); ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
cb = fig.colorbar(sc, ax=ax, pad=.02, ticks=[0, 1], aspect=30); cb.ax.set_yticklabels(["Low", "High"], fontsize=6.5)
cb.set_label("Feature value", fontsize=7)
fig.savefig(f"{out}/fig_shap_beeswarm.png"); plt.close(fig)

# Fig: dependence plots
def dep(feat, inter, fname, xlabel, vline=None):
    i, j = fn.index(feat), fn.index(inter)
    fig, ax = plt.subplots(figsize=(W1, 2.4))
    s = ax.scatter(Xte_t[:, i], sv[:, i], c=Xte_t[:, j], cmap="coolwarm", s=9, lw=0)
    cb = fig.colorbar(s, ax=ax, pad=.02); cb.set_label(pretty(inter), fontsize=7); cb.ax.tick_params(labelsize=6.5)
    ax.axhline(0, color="grey", lw=.6, ls=":")
    if vline is not None:
        ax.axvline(vline, color="k", lw=.7, ls="--"); ax.text(vline, ax.get_ylim()[1] * .92, f" {vline}", fontsize=6.5)
    ax.set_xlabel(xlabel); ax.set_ylabel("SHAP value (log-odds)")
    fig.savefig(f"{out}/{fname}"); plt.close(fig)
dep("APD_mm", "Diuretic_renogram_T1_2_min", "fig_dep_apd.png", "Anteroposterior pelvic diameter, APD (mm)")
dep("Diuretic_renogram_T1_2_min", "APD_mm", "fig_dep_t12.png", "Diuretic renogram drainage half-time, T½ (min)", vline=20)
dep("Preoperative_DRF_percent", "APD_mm", "fig_dep_drf.png", "Preoperative differential renal function (%)")

# Fig: noise sensitivity (two panels)
fig, axs = plt.subplots(1, 2, figsize=(7.16, 2.6), sharey=True)
for ax, e, xl, title in [(axs[0], "label_observed", "Fraction of labels flipped", "(a) Label noise (train + evaluation)"),
                         (axs[1], "measurement", "Noise SD (× feature SD) on APD and T½", "(b) Measurement noise")]:
    for n in NM:
        g = NS[(NS.exp == e) & (NS.model == n)]
        ax.errorbar(g.level, g["mean"], g["std"], marker="o", ms=3.5, lw=1.1, capsize=2.5, color=COL[n], label=n)
    if e == "label_observed":
        g = NS[(NS.exp == "label_train_only") & (NS.model == "XGBoost")]
        ax.plot(g.level, g["mean"], ls="--", lw=1, color=COL["XGBoost"], marker="s", ms=3,
                label="XGBoost, noisy training only\n(clean evaluation labels)")
    ax.set_xlabel(xl); ax.set_title(title, loc="left"); ax.grid(alpha=.25, lw=.5)
axs[0].set_ylabel("Cross-validated ROC-AUC"); axs[0].set_ylim(.6, 1.0); axs[0].legend(frameon=False, loc="lower left")
axs[0].set_xticks([0, .05, .1, .2, .3]); axs[0].set_xticklabels(["0", "5%", "10%", "20%", "30%"])
axs[1].set_xticks([0, .25, .5, 1.0])
fig.tight_layout(); fig.savefig(f"{out}/fig_noise_sensitivity.png"); plt.close(fig)
print("done")
