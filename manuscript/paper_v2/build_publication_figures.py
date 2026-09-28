from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

ROOT = Path.home() / "prl_medclipsam"
SRC  = ROOT / "paper_v2" / "source_data"
OUT  = ROOT / "paper_v2" / "publication_figures"
FSD  = ROOT / "paper_v2" / "publication_figure_source_data"

OUT.mkdir(parents=True, exist_ok=True)
FSD.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------
# Global publication style
# ---------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8.5,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.6,
    "lines.markersize": 5,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.dpi": 600,
})

def read(name):
    return pd.read_csv(SRC / name)

def clean(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out")

def panel(ax, letter):
    ax.text(
        -0.13, 1.08, letter,
        transform=ax.transAxes,
        fontsize=13,
        fontweight="bold",
        va="top",
        ha="left"
    )

def save(fig, stem):
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.png", dpi=600, bbox_inches="tight")
    plt.close(fig)

def annotate_bars(ax, fmt="{:.3f}", fontsize=7.5):
    for p in ax.patches:
        h = p.get_height()
        if np.isfinite(h):
            ax.annotate(
                fmt.format(h),
                (p.get_x() + p.get_width()/2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=fontsize
            )

# ---------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------
breast   = read("breast113_casewise_dice.csv")
vuln     = read("breast113_gtfree_vulnerability.csv")
artifact = read("breast113_mask_artifact_audit.csv")
dev      = read("breast113_development_detector_auc.csv")
held     = read("breast113_heldout_summary.csv")

brain      = read("brain600_case_metrics.csv")
brain_cond = read("brain600_condition_summary.csv")
brain_rel  = read("brain600_h0_relative_summary.csv")
brain_six  = read("brain600_six_condition_summary.csv")

local = read("brain_localization_decomposition.csv")

factor_case   = read("brain_factorial_case_decomposition.csv")
factor_region = read("brain_factorial_region_decomposition.csv")
traj_dose     = read("brain_trajectory_dose_response.csv")

interface_summary = read("breast113_prompt_interface_summary.csv")
strat             = read("breast113_prompt_vulnerability_stratified_summary.csv")

text3d_boot = read("text3dsam_amos30_bootstrap.csv")

# =====================================================================
# FIGURE 1 — Breast113 language-conditioned instability
# =====================================================================
conds = ["B0","B1","B2","B3","B4","B5"]
means = breast[conds].mean()
r = breast["within_case_range"].to_numpy()

fig, axs = plt.subplots(2,2,figsize=(7.2,6.0))

ax=axs[0,0]
ax.plot(conds, means.values, marker="o")
ax.set_ylabel("Mean Dice")
ax.set_xlabel("Language condition")
ax.set_ylim(0.70,0.88)
clean(ax); panel(ax,"A")

ax=axs[0,1]
ax.hist(r,bins=25)
ax.axvline(.20,linestyle="--",linewidth=1,label="Range20 = 0.20")
ax.axvline(.50,linestyle=":",linewidth=1.2,label="Range50 = 0.50")
ax.set_xlabel("Within-case Dice range")
ax.set_ylabel("Cases")
ax.legend(frameon=False,loc="upper right")
clean(ax); panel(ax,"B")

ax=axs[1,0]
ax.scatter(vuln["sam_disagreement"],vuln["dsc_range"],s=18,alpha=.72)
ax.set_xlabel("SAM-mask disagreement")
ax.set_ylabel("Within-case Dice range")
clean(ax); panel(ax,"C")

ax=axs[1,1]
thresholds=np.array([.01,.05,.10,.20,.50])
rates=np.array([(r>x).mean()*100 for x in thresholds])
bars=ax.bar([">0.01",">0.05",">0.10",">0.20",">0.50"],rates)
ax.set_ylabel("Cases (%)")
ax.set_xlabel("Dice-range threshold")
ax.set_ylim(0,65)
for b,v in zip(bars,rates):
    ax.text(b.get_x()+b.get_width()/2,v+1,f"{v:.1f}%",
            ha="center",va="bottom",fontsize=7)
clean(ax); panel(ax,"D")

fig.tight_layout()
save(fig,"Fig1_breast113_language_instability_FINAL")

# =====================================================================
# FIGURE 2 — Detector validation + artifact stress test
# =====================================================================
# Frozen gross-failure definition:
# min pairwise SAM-mask IoU < 0.10
#
# Values below are the already-computed frozen audit values supplied
# from the pre-existing Breast113 analysis. No threshold fitting occurs.
# =====================================================================

stress = pd.DataFrame({
    "target":["Range20","Range20","Range50","Range50"],
    "subset":["All cases","Non-gross","All cases","Non-gross"],
    "auc":[0.994401,0.989203,0.996324,0.984375],
    "ci_low":[0.983673,0.969693,0.987634,0.955326],
    "ci_high":[1.000000,1.000000,1.000000,1.000000],
    "n":[113,100,113,100]
})
stress.to_csv(FSD/"Fig2D_artifact_stress_test.csv",index=False)

fig,axs=plt.subplots(2,2,figsize=(7.3,6.1))

# A — development
ax=axs[0,0]
x=np.arange(len(dev))
bars=ax.bar(x,dev["auc"])
ax.set_xticks(x,[str(v).replace("range","Range") for v in dev["target"]])
ax.set_ylim(0,1.05)
ax.set_ylabel("Development AUROC")
annotate_bars(ax)
clean(ax); panel(ax,"A")

# B — held-out
ax=axs[0,1]
x=np.arange(len(held))
y=held["auc"].to_numpy()
lo=y-held["auc_ci_low"].to_numpy()
hi=held["auc_ci_high"].to_numpy()-y
ax.errorbar(x,y,yerr=[lo,hi],fmt="o",capsize=3)
ax.set_xticks(x,[str(v).replace("range","Range") for v in held["target"]])
ax.set_ylim(0,1.05)
ax.set_ylabel("Patient-disjoint held-out AUROC")
for xi,yi in zip(x,y):
    ax.text(xi,yi+.035,f"{yi:.3f}",ha="center",fontsize=7.5)
clean(ax); panel(ax,"B")

# C — held-out operating characteristics
ax=axs[1,0]
metrics=["sensitivity","specificity","ppv","npv","accuracy"]
labels=["Sensitivity","Specificity","PPV","NPV","Accuracy"]
xx=np.arange(len(metrics))
width=.34
for j,(_,row) in enumerate(held.iterrows()):
    vals=[row[m] for m in metrics]
    label=str(row["target"]).replace("range","Range")
    ax.bar(xx+(j-.5)*width,vals,width=width,label=label)
ax.set_xticks(xx,labels,rotation=25,ha="right")
ax.set_ylim(0,1.05)
ax.set_ylabel("Held-out performance")
ax.legend(frameon=False,ncol=2)
clean(ax); panel(ax,"C")

# D — gross-failure-excluded stress test
ax=axs[1,1]
targets=["Range20","Range50"]
xx=np.arange(2)
width=.34

allvals=[
    stress[(stress.target==t)&(stress.subset=="All cases")].auc.iloc[0]
    for t in targets
]
nongross=[
    stress[(stress.target==t)&(stress.subset=="Non-gross")].auc.iloc[0]
    for t in targets
]

b1=ax.bar(xx-width/2,allvals,width,label="All cases")
b2=ax.bar(xx+width/2,nongross,width,label="Non-gross only")
ax.set_xticks(xx,targets)
ax.set_ylim(0,1.05)
ax.set_ylabel("AUROC")
ax.set_title("Gross-failure exclusion")
ax.legend(frameon=False)

for bars in [b1,b2]:
    for b in bars:
        ax.text(b.get_x()+b.get_width()/2,b.get_height()+.018,
                f"{b.get_height():.3f}",
                ha="center",va="bottom",fontsize=7)

clean(ax); panel(ax,"D")

fig.tight_layout()
save(fig,"Fig2_frozen_detector_validation_FINAL")

# =====================================================================
# FIGURE 3 — Brain600 controlled perturbation sensitivity
# =====================================================================
fig,axs=plt.subplots(1,3,figsize=(7.5,2.65))

ax=axs[0]
ax.plot(brain_cond["condition"],brain_cond["mean_dice"],marker="o")
ax.set_ylabel("Mean Dice")
ax.set_xlabel("Perturbation condition")
clean(ax); panel(ax,"A")

ax=axs[1]
ax.plot(brain_rel["condition"],brain_rel["mean_abs_delta"],marker="o")
ax.set_ylabel("Mean |ΔDice| vs H0")
ax.set_xlabel("Perturbation condition")
clean(ax); panel(ax,"B")

ax=axs[2]
ax.plot(brain_rel["mean_box_iou_H0"],brain_rel["mean_abs_delta"],marker="o")
for _,row in brain_rel.iterrows():
    ax.annotate(row["condition"],
                (row["mean_box_iou_H0"],row["mean_abs_delta"]),
                xytext=(4,4),textcoords="offset points",fontsize=7)
ax.set_xlabel("Mean box IoU with H0")
ax.set_ylabel("Mean |ΔDice| vs H0")
clean(ax); panel(ax,"C")

fig.tight_layout()
save(fig,"Fig3_brain600_perturbation_sensitivity_FINAL")

# =====================================================================
# FIGURE 4 — Localization decomposition
# =====================================================================
fig,axs=plt.subplots(1,3,figsize=(7.6,2.8))
x=np.arange(len(local))

ax=axs[0]
ax.plot(x,local["selected_diff_mean"],marker="o",label="Selected")
ax.plot(x,local["oracle_diff_mean"],marker="o",label="Oracle")
ax.set_xticks(x,local["stratum"],rotation=35,ha="right")
ax.set_ylabel("Mean Dice difference")
ax.set_xlabel("Box-IoU stratum")
ax.legend(frameon=False)
clean(ax); panel(ax,"A")

ax=axs[1]
ax.bar(x,local["mean_reduction"])
ax.axhline(0,linewidth=.8)
ax.set_xticks(x,local["stratum"],rotation=35,ha="right")
ax.set_ylabel("Mean reduction in Dice difference")
ax.set_xlabel("Box-IoU stratum")
clean(ax); panel(ax,"B")

ax=axs[2]
ax.plot(x,100*local["fail_selection_gap_gt20_rate"],marker="o")
ax.set_xticks(x,local["stratum"],rotation=35,ha="right")
ax.set_ylabel("Selection gap >0.20 (%)")
ax.set_xlabel("Box-IoU stratum")
clean(ax); panel(ax,"C")

fig.tight_layout()
save(fig,"Fig4_localization_decomposition_FINAL")

# =====================================================================
# FIGURE 5 — Factorial + trajectory mechanism
# =====================================================================
fig,axs=plt.subplots(1,3,figsize=(7.6,2.7))

ax=axs[0]
x=np.arange(len(factor_region)); w=.34
ax.bar(x-w/2,factor_region["seed_instability_mean"],width=w,label="Seed")
ax.bar(x+w/2,factor_region["language_instability_mean"],width=w,label="Language")
ax.set_xticks(x,factor_region["risk_region"])
ax.set_ylabel("Mean box instability")
ax.set_xlabel("Risk stratum")
ax.legend(frameon=False)
clean(ax); panel(ax,"A")

ax=axs[1]
ax.scatter(factor_case["seed_instability_mean"],
           factor_case["language_instability_mean"],s=28)
mx=max(factor_case["seed_instability_mean"].max(),
       factor_case["language_instability_mean"].max())
ax.plot([0,mx],[0,mx],linestyle="--",linewidth=1,label="Equality")
ax.set_xlabel("Seed-induced box instability")
ax.set_ylabel("Language-induced box instability")
ax.legend(frameon=False)
clean(ax); panel(ax,"B")

ax=axs[2]
ax.plot(traj_dose["alpha"],traj_dose["corr_distance_from_A_mean"],
        marker="o",label="Distance from endpoint A")
ax.plot(traj_dose["alpha"],traj_dose["corr_distance_from_B_mean"],
        marker="o",label="Distance from endpoint B")
ax.set_xlabel("Interpolation α")
ax.set_ylabel("Correlation distance")
ax.legend(frameon=False,fontsize=7)
clean(ax); panel(ax,"C")

fig.tight_layout()
save(fig,"Fig5_language_mechanism_FINAL")

# =====================================================================
# FIGURE 6 — Architectural boundary condition
# Log scale makes Text3DSAM visible without hiding the magnitude gap.
# =====================================================================
brain_mean=float(brain_six.iloc[0]["mean_range"])
brain_med=float(brain_six.iloc[0]["median_range"])

tmean=float(text3d_boot.loc[
    text3d_boot.metric=="range_mean","point"].iloc[0])
tmed=float(text3d_boot.loc[
    text3d_boot.metric=="range_median","point"].iloc[0])

fig,axs=plt.subplots(1,2,figsize=(6.5,3.0))

for ax,vals,ylabel,letter in [
    (axs[0],[brain_mean,tmean],"Mean within-case Dice range","A"),
    (axs[1],[brain_med,tmed],"Median within-case Dice range","B")
]:
    bars=ax.bar(["Brain600","Text3DSAM"],vals)
    ax.set_yscale("log")
    ax.set_ylabel(ylabel)
    for b,v in zip(bars,vals):
        ax.text(
            b.get_x()+b.get_width()/2,
            v*1.18,
            f"{v:.4f}",
            ha="center",
            va="bottom",
            fontsize=8
        )
    clean(ax); panel(ax,letter)

fig.tight_layout()
save(fig,"Fig6_cross_architecture_boundary_FINAL")

# =====================================================================
# SUPPLEMENTARY FIGURE S1 — Prompt-interface intervention
# =====================================================================
pretty={
    "box":"Box",
    "points_pos":"Positive points",
    "points_pm":"Positive + negative points",
    "both_pos":"Box + positive points",
    "both_pm":"Box + positive/negative points"
}

b0=interface_summary[interface_summary["base"]=="B0"].copy()
rv=strat[
    (strat["target"]=="range20") &
    (strat["stratum"]=="vulnerable")
].copy()

fig,axs=plt.subplots(1,2,figsize=(7.4,3.2))

ax=axs[0]
labels=[pretty.get(x,x) for x in b0["interface"]]
bars=ax.bar(np.arange(len(b0)),b0["mean_dice"])
ax.set_xticks(np.arange(len(b0)),labels,rotation=30,ha="right")
ax.set_ylabel("Mean Dice")
ax.set_title("B0")
ax.set_ylim(0,0.90)
clean(ax); panel(ax,"A")

ax=axs[1]
labels=[pretty.get(x,x) for x in rv["interface"]]
ax.bar(np.arange(len(rv)),rv["mean_range"])
ax.set_xticks(np.arange(len(rv)),labels,rotation=30,ha="right")
ax.set_ylabel("Mean within-case Dice range")
ax.set_title("Originally Range20-vulnerable cases")
clean(ax); panel(ax,"B")

fig.tight_layout()
save(fig,"FigS1_prompt_interface_tradeoff_FINAL")

print("="*100)
print("PUBLICATION FIGURES COMPLETE")
print("="*100)
for p in sorted(OUT.glob("*")):
    print(p.relative_to(ROOT))
print()
print("Frozen source_data files were READ ONLY.")
print("No model inference performed.")
print("No detector threshold selected or changed.")
print("="*100)
