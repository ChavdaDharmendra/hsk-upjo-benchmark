"""Reproducible CKD (UCI id 336, Apollo Hospitals) benchmark: repeated CV + bootstrap CI + SHAP.
Usage: python ckd_pipeline.py path/to/chronic_kidney_disease.arff outdir"""
import sys, re, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, brier_score_loss
from sklearn.calibration import calibration_curve
from xgboost import XGBClassifier
import shap
SEED=42
arff, out = sys.argv[1], sys.argv[2]
# ---- parse ARFF
cols, rows, data = [], [], False
for ln in open(arff, encoding="latin-1"):
    s = ln.strip()
    if not s or s.startswith("%"): continue
    if s.lower().startswith("@attribute"): cols.append(re.match(r"@attribute\s+'?(\w+)'?", s, re.I).group(1))
    elif s.lower().startswith("@data"): data = True
    elif data:
        t=[v.strip() for v in s.split(",")]
        if len(t)>len(cols): t=[v for v in t if v!=""]   # source file has stray empty fields in 4 rows
        rows.append(t)
df = pd.DataFrame(rows, columns=cols).replace("?", np.nan)
df = df.apply(lambda c: c.str.strip() if c.dtype==object else c)
y = (df.pop("class")=="ckd").astype(int).values
num = ["age","bp","bgr","bu","sc","sod","pot","hemo","pcv","wbcc","rbcc"]
for c in num: df[c]=pd.to_numeric(df[c])
# sg, al, su are ordinal -> numeric; remaining nominal
for c in ["sg","al","su"]: df[c]=pd.to_numeric(df[c])
num += ["sg","al","su"]; cat=[c for c in df.columns if c not in num]
print("n=",len(df),"CKD=",y.sum(),"notCKD=",(1-y).sum(),"missing cells %=",round(df.isna().mean().mean()*100,1))
df.isna().sum().sort_values(ascending=False).to_csv(f"{out}/missing_counts.csv")

def prep(nc, cc, scale):
    n = [("imp",SimpleImputer(strategy="median"))] + ([("sc",StandardScaler())] if scale else [])
    return ColumnTransformer([("n",Pipeline(n),nc),("c",Pipeline([("imp",SimpleImputer(strategy="most_frequent")),
        ("oh",OneHotEncoder(drop="if_binary",handle_unknown="ignore"))]),cc)])
def models(nc, cc):
    mk = {"Logistic Regression":(LogisticRegression(max_iter=2000),True),
          "Random Forest":(RandomForestClassifier(300,random_state=SEED),False),
          "SVM (RBF)":(SVC(probability=True,random_state=SEED),True),
          "XGBoost":(XGBClassifier(n_estimators=200,max_depth=3,learning_rate=0.05,subsample=0.9,colsample_bytree=0.9,eval_metric="logloss",random_state=SEED),False)}
    return {k:Pipeline([("p",prep(nc,cc,sc)),("m",m)]) for k,(m,sc) in mk.items()}

def boot_ci(y,p,f,B=2000):
    r=np.random.RandomState(SEED); v=[]
    for _ in range(B):
        i=r.randint(0,len(y),len(y))
        if len(set(y[i]))<2: continue
        v.append(f(y[i],p[i]))
    return np.percentile(v,[2.5,97.5])
def evaluate(X, nc, cc, label):
    rows=[]; oof_store={}
    cv=RepeatedStratifiedKFold(n_splits=5,n_repeats=5,random_state=SEED)
    for name,pipe in models(nc,cc).items():
        a,f,u=[],[],[]
        for tr,te in cv.split(X,y):
            pipe.fit(X.iloc[tr],y[tr]); p=pipe.predict_proba(X.iloc[te])[:,1]
            a.append(accuracy_score(y[te],p>.5)); f.append(f1_score(y[te],p>.5)); u.append(roc_auc_score(y[te],p))
        oof=cross_val_predict(pipe,X,y,cv=StratifiedKFold(5,shuffle=True,random_state=SEED),method="predict_proba")[:,1]
        oof_store[name]=oof; lo,hi=boot_ci(y,oof,roc_auc_score)
        rows.append(dict(setting=label,model=name,acc=f"{np.mean(a):.4f} ± {np.std(a):.4f}",f1=f"{np.mean(f):.4f} ± {np.std(f):.4f}",
            auc=f"{np.mean(u):.4f} ± {np.std(u):.4f}",auc_boot95=f"[{lo:.4f}, {hi:.4f}]",brier=round(brier_score_loss(y,oof),4)))
        print(rows[-1],flush=True)
    return rows,oof_store
allc=list(df.columns)
r1,oof=evaluate(df,num,cat,"All 24 features")
# baselines: single-variable (hemo, sc, sg) via LR
rb=[]
for v in ["sc","hemo","sg"]:
    p=cross_val_predict(Pipeline([("i",SimpleImputer(strategy="median")),("s",StandardScaler()),("m",LogisticRegression())]),df[[v]],y,cv=StratifiedKFold(5,shuffle=True,random_state=SEED),method="predict_proba")[:,1]
    lo,hi=boot_ci(y,p,roc_auc_score); rb.append(dict(setting="Single-feature baseline",model=f"LR on {v} only",acc=round(accuracy_score(y,p>.5),4),f1=round(f1_score(y,p>.5),4),auc=round(roc_auc_score(y,p),4),auc_boot95=f"[{lo:.4f}, {hi:.4f}]",brier=round(brier_score_loss(y,p),4)))
# ablation: drop strongest renal/lab proxies
drop=["sc","bu","hemo","pcv","rbcc","sg","al"]
r2,_=evaluate(df.drop(columns=drop),[c for c in num if c not in drop],cat,"Ablation (no sc,bu,hemo,pcv,rbcc,sg,al)")
pd.DataFrame(r1+rb+r2).to_csv(f"{out}/results.csv",index=False)
# calibration + ROC-ish plot
fig,ax=plt.subplots(figsize=(5,4.5)); ax.plot([0,1],[0,1],"k--",lw=1)
for n,p in oof.items():
    fr,mp=calibration_curve(y,p,n_bins=8,strategy="quantile"); ax.plot(mp,fr,marker="o",label=n)
ax.set_xlabel("Predicted probability"); ax.set_ylabel("Observed fraction CKD"); ax.legend(fontsize=8); ax.set_title("Calibration (5-fold out-of-fold)")
fig.tight_layout(); fig.savefig(f"{out}/calibration.png",dpi=200)
# SHAP on XGBoost fit on all data (explanatory only)
pipe=models(num,cat)["XGBoost"]; pipe.fit(df,y)
Xt=pipe["p"].transform(df); names=[n.split("__")[1] for n in pipe["p"].get_feature_names_out()]
sv=shap.TreeExplainer(pipe["m"]).shap_values(Xt)
plt.figure(); shap.summary_plot(sv,Xt,feature_names=names,show=False); plt.tight_layout(); plt.savefig(f"{out}/shap_beeswarm.png",dpi=200,bbox_inches="tight")
imp=pd.Series(np.abs(sv).mean(0),index=names).sort_values(ascending=False); imp.to_csv(f"{out}/shap_mean_abs.csv"); print(imp.head(8))
