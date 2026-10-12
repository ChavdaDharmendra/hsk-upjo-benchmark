"""Baselines + label-noise + measurement-noise sensitivity on HSK_UPJO_synthetic_1000.csv
Usage: python hsk_noise_experiments.py HSK_UPJO_synthetic_1000.csv outdir"""
import sys, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, accuracy_score
from xgboost import XGBClassifier
SEED=42; csv,out=sys.argv[1],sys.argv[2]
df=pd.read_csv(csv); y0=df.pop("UPJO").values
df=df.drop(columns=["Patient_ID","Postoperative_DRF_percent","Surgical_modality"])
cat=[c for c in df.columns if df[c].dtype==object or str(df[c].dtype).startswith("str")]
num=[c for c in df.columns if c not in cat]
print("categorical:",cat)
def mk(name,nc,cc):
    ct=lambda sc: ColumnTransformer([("n",StandardScaler() if sc else "passthrough",nc),("c",OneHotEncoder(drop="if_binary",handle_unknown="ignore"),cc)])
    m={"Logistic Regression":(LogisticRegression(max_iter=2000),1),"Random Forest":(RandomForestClassifier(300,random_state=SEED),0),
       "SVM (RBF)":(SVC(probability=True,random_state=SEED),1),
       "XGBoost":(XGBClassifier(n_estimators=200,max_depth=4,learning_rate=0.05,subsample=0.9,colsample_bytree=0.9,eval_metric="logloss",random_state=SEED),0)}[name]
    return Pipeline([("p",ct(m[1])),("m",m[0])])
def cv_eval(X,ytr,yte,nc,cc,name,reps=3):
    """train on ytr labels, score against yte labels; returns mean AUC/acc over reps x 5 folds"""
    a=[];c=[]
    for r in range(reps):
        for tr,te in StratifiedKFold(5,shuffle=True,random_state=SEED+r).split(X,y0):
            p=mk(name,nc,cc).fit(X.iloc[tr],ytr[tr]).predict_proba(X.iloc[te])[:,1]
            a.append(roc_auc_score(yte[te],p)); c.append(accuracy_score(yte[te],p>.5))
    return np.mean(a),np.std(a),np.mean(c)
rows=[]
# A. baselines (clean labels)
for lab,cols in {"APD only":["APD_mm"],"T1/2 only":["Diuretic_renogram_T1_2_min"],"APD + T1/2":["APD_mm","Diuretic_renogram_T1_2_min"]}.items():
    m,s,ac=cv_eval(df[cols],y0,y0,cols,[],"Logistic Regression"); rows.append(dict(exp="baseline",setting=lab+" (LR)",model="LR",auc=m,auc_sd=s,acc=ac))
for n in ["Logistic Regression","Random Forest","SVM (RBF)","XGBoost"]:
    m,s,ac=cv_eval(df,y0,y0,num,cat,n); rows.append(dict(exp="baseline",setting="All 14 features",model=n,auc=m,auc_sd=s,acc=ac)); print(rows[-1],flush=True)
# B. label noise: flip a fraction of labels (applies to train AND test = what an observer would see; plus train-noisy/test-clean)
rng=np.random.RandomState(SEED); levels=[0,.05,.10,.20,.30]
for n in ["Logistic Regression","Random Forest","XGBoost"]:
    for f in levels:
        yn=y0.copy(); idx=rng.rand(len(yn))<f; yn[idx]=1-yn[idx]
        m,s,ac=cv_eval(df,yn,yn,num,cat,n); rows.append(dict(exp="label_noise_observed",setting=f"{int(f*100)}%",model=n,auc=m,auc_sd=s,acc=ac))
        m,s,ac=cv_eval(df,yn,y0,num,cat,n); rows.append(dict(exp="label_noise_train_only",setting=f"{int(f*100)}%",model=n,auc=m,auc_sd=s,acc=ac)); print(rows[-1],flush=True)
# C. measurement noise on APD & T1/2 (inter-observer / hydration variability), both train and test
for n in ["Logistic Regression","Random Forest","XGBoost"]:
    for k in [0,.25,.5,1.0]:
        d=df.copy(); r=np.random.RandomState(SEED)
        for c in ["APD_mm","Diuretic_renogram_T1_2_min"]: d[c]=d[c]+r.normal(0,k*d[c].std(),len(d))
        m,s,ac=cv_eval(d,y0,y0,num,cat,n); rows.append(dict(exp="measurement_noise",setting=f"{k}xSD",model=n,auc=m,auc_sd=s,acc=ac)); print(rows[-1],flush=True)
R=pd.DataFrame(rows); R.to_csv(f"{out}/noise_results.csv",index=False)
fig,ax=plt.subplots(1,2,figsize=(10,4))
for e,a,t,xl in [("label_noise_observed",ax[0],"Label noise (train+test flipped)","Label-flip rate"),("measurement_noise",ax[1],"Measurement noise on APD & T1/2","Noise SD (x feature SD)")]:
    for n,g in R[R.exp==e].groupby("model"):
        a.errorbar(range(len(g)),g.auc,g.auc_sd,marker="o",capsize=3,label=n)
    s=R[R.exp==e].setting.unique(); a.set_xticks(range(len(s))); a.set_xticklabels(s); a.set_title(t); a.set_xlabel(xl); a.set_ylabel("CV ROC-AUC"); a.set_ylim(.5,1.01)
ax[0].legend(fontsize=8); fig.tight_layout(); fig.savefig(f"{out}/noise_sensitivity.png",dpi=200)
