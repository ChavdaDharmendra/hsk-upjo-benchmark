"""Extra publication figures: renogram-curve schematic (Fig. 2) and column-width
stacked noise-sensitivity figure. Usage: python make_extra_figures.py results/"""
import sys, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
out = sys.argv[1]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.dpi": 600, "savefig.bbox": "tight", "axes.linewidth": .7})
# ---- renogram typologies (idealised schematic curves; furosemide at t = 20 min)
t = np.linspace(0, 40, 400); F = 20
def rise(t, k): return 1 - np.exp(-t / k)
g1 = np.where(t < 3, t / 3, np.exp(-(t - 3) / 6) * .95 + .05 * np.exp(-(t - 3) / 30))
g2 = rise(t, 3) * .98
g3a = np.where(t < F, rise(t, 4), rise(F, 4) * np.exp(-(t - F) / 2.5))
g3b = np.where(t < F, rise(t, 5), rise(F, 5) * (1 - .025 * (t - F)))
fig, axs = plt.subplots(2, 2, figsize=(3.5, 3.1), sharex=True, sharey=True)
for ax, y, ttl, sub in zip(axs.ravel(), [g1, g2, g3a, g3b], ["Group 1", "Group 2", "Group 3a", "Group 3b"],
                           ["Normal drainage", "Obstructed", "Stasis, no obstruction", "Partial obstruction"]):
    ax.plot(t, y, color="k", lw=1.3); ax.annotate("F", xy=(F, .02), xytext=(F, .32), ha="center", fontsize=7,
                                                    arrowprops=dict(arrowstyle="->", lw=.7))
    ax.set_title(f"{ttl}\n{sub}", fontsize=7, loc="left"); ax.set_ylim(0, 1.08); ax.set_xticks([0, 10, 20, 30, 40])
    ax.set_yticks([])
for ax in axs[1]: ax.set_xlabel("Time (min)", fontsize=7.5)
for ax in axs[:, 0]: ax.set_ylabel("Renal activity", fontsize=7.5)
fig.tight_layout(); fig.savefig(f"{out}/fig_renogram_schematic.png"); plt.close(fig)

# ---- stacked noise-sensitivity figure (single column)
NS = pd.read_csv(f"{out}/table_noise_summary.csv")
COL = {"Logistic Regression": "#1f77b4", "Random Forest": "#2ca02c", "XGBoost": "#d62728"}
MK = {"Logistic Regression": "o", "Random Forest": "^", "XGBoost": "s"}
fig, axs = plt.subplots(2, 1, figsize=(3.5, 5.0))
for ax, e, xl, title in [(axs[0], "label_observed", "Fraction of UPJO labels flipped", "(a) Label noise"),
                         (axs[1], "measurement", "Gaussian noise SD on APD_mm and T½ (× feature SD)", "(b) Measurement noise")]:
    for n in COL:
        g = NS[(NS.exp == e) & (NS.model == n)].sort_values("level")
        ax.errorbar(g.level, g["mean"], g["std"], marker=MK[n], ms=3.5, lw=1.1, capsize=2.5, color=COL[n],
                    label=f"{n} (noisy train + evaluation)" if e == "label_observed" else n)
    if e == "label_observed":
        for n in COL:
            g = NS[(NS.exp == "label_train_only") & (NS.model == n)].sort_values("level")
            ax.plot(g.level, g["mean"], ls="--", lw=.9, color=COL[n], marker=MK[n], ms=2.5, mfc="white")
        ax.plot([], [], ls="--", color="grey", label="Dashed: noisy training, clean evaluation")
        ax.set_xticks([0, .05, .1, .2, .3]); ax.set_xticklabels(["0", "5%", "10%", "20%", "30%"]); ax.set_ylim(.40, 1.0)
        ax.legend(frameon=False, fontsize=6.2, loc="lower left")
    else:
        ax.set_xticks([0, .25, .5, 1.0]); ax.set_xticklabels(["0", "0.25", "0.5", "1.0"]); ax.set_ylim(.93, 1.0)
        ax.legend(frameon=False, fontsize=6.5, loc="lower left")
    ax.set_xlabel(xl, fontsize=7.5); ax.set_ylabel("Cross-validated ROC-AUC", fontsize=7.5)
    ax.set_title(title, loc="left", fontsize=8); ax.grid(alpha=.25, lw=.5)
fig.tight_layout(); fig.savefig(f"{out}/fig_noise_stacked.png"); plt.close(fig)
print("ok")
