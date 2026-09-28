from pathlib import Path
import hashlib
import json
import shutil

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path.home() / "prl_medclipsam"
SRC = ROOT / "paper_v2" / "source_data"
FIG = ROOT / "paper_v2" / "figures"
TAB = ROOT / "paper_v2" / "tables"
FSD = ROOT / "paper_v2" / "figure_source_data"
PROV = ROOT / "paper_v2" / "provenance"

for p in [FIG, TAB, FSD, PROV]:
    p.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def read(name):
    path = SRC / name
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)

def savefig(fig, stem):
    fig.savefig(FIG / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"{stem}.png", dpi=400, bbox_inches="tight")
    plt.close(fig)

def panel(ax, letter):
    ax.text(
        -0.12, 1.08, letter,
        transform=ax.transAxes,
        fontsize=13,
        fontweight="bold",
        va="top"
    )

def clean(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

def pct(x):
    return 100.0 * np.asarray(x, dtype=float)

# ---------------------------------------------------------------------
# Load frozen sources
# ---------------------------------------------------------------------

breast = read("breast113_casewise_dice.csv")
vuln = read("breast113_gtfree_vulnerability.csv")
artifact = read("breast113_mask_artifact_audit.csv")
dev = read("breast113_development_detector_auc.csv")
held = read("breast113_heldout_summary.csv")
held_cases = read("breast113_heldout_case_predictions.csv")

interface_summary = read("breast113_prompt_interface_summary.csv")
interface_stability = read("breast113_prompt_interface_stability.csv")
strat = read("breast113_prompt_vulnerability_stratified_summary.csv")
trans = read("breast113_prompt_vulnerability_transitions.csv")

brain = read("brain600_case_metrics.csv")
brain_cond = read("brain600_condition_summary.csv")
brain_rel = read("brain600_h0_relative_summary.csv")
brain_six = read("brain600_six_condition_summary.csv")

local = read("brain_localization_decomposition.csv")
local_dist = read("brain_localization_distribution.csv")

factor_case = read("brain_factorial_case_decomposition.csv")
factor_region = read("brain_factorial_region_decomposition.csv")
mech_case = read("brain_stochastic_mechanism_case.csv")
mech_region = read("brain_stochastic_mechanism_region.csv")
traj_case = read("brain_trajectory_case_summary.csv")
traj_dose = read("brain_trajectory_dose_response.csv")

text3d = read("text3dsam_amos30_language.csv")
text3d_boot = read("text3dsam_amos30_bootstrap.csv")

# =====================================================================
# FIGURE 1
# Breast113: language-conditioned segmentation instability
# =====================================================================

conds = ["B0", "B1", "B2", "B3", "B4", "B5"]

fig, axs = plt.subplots(2, 2, figsize=(10.2, 7.8))

# A: mean Dice by condition
ax = axs[0,0]
means = breast[conds].mean()
ax.plot(conds, means.values, marker="o")
ax.set_ylabel("Mean Dice")
ax.set_xlabel("Language condition")
ax.set_ylim(max(0, means.min()-0.05), min(1, means.max()+0.05))
clean(ax)
panel(ax, "A")

# B: casewise range distribution
ax = axs[0,1]
r = breast["within_case_range"].to_numpy()
ax.hist(r, bins=25)
ax.axvline(0.20, linestyle="--", linewidth=1)
ax.axvline(0.50, linestyle=":", linewidth=1)
ax.set_xlabel("Within-case Dice range")
ax.set_ylabel("Cases")
clean(ax)
panel(ax, "B")

# C: GT-free disagreement vs Dice range
ax = axs[1,0]
ax.scatter(
    vuln["sam_disagreement"],
    vuln["dsc_range"],
    s=22,
    alpha=.75
)
ax.set_xlabel("SAM-mask disagreement")
ax.set_ylabel("Within-case Dice range")
clean(ax)
panel(ax, "C")

# D: fraction exceeding instability thresholds
ax = axs[1,1]
rates = [
    (r > .01).mean(),
    (r > .05).mean(),
    (r > .10).mean(),
    (r > .20).mean(),
    (r > .50).mean(),
]
labels = [">.01", ">.05", ">.10", ">.20", ">.50"]
ax.bar(labels, pct(rates))
ax.set_ylabel("Cases (%)")
ax.set_xlabel("Dice-range threshold")
clean(ax)
panel(ax, "D")

fig.tight_layout()
savefig(fig, "Fig1_breast113_language_instability")

pd.DataFrame({
    "condition": conds,
    "mean_dice": means.values
}).to_csv(FSD/"Fig1A.csv", index=False)

breast[["filename","within_case_range"]].to_csv(
    FSD/"Fig1B.csv", index=False
)

vuln[[
    "image","dsc_range","sam_disagreement",
    "range20","range50"
]].to_csv(FSD/"Fig1C.csv", index=False)

pd.DataFrame({
    "threshold":[.01,.05,.10,.20,.50],
    "fraction":rates
}).to_csv(FSD/"Fig1D.csv", index=False)

# =====================================================================
# FIGURE 2
# Frozen detector development -> held-out validation
# =====================================================================

fig, axs = plt.subplots(1, 3, figsize=(12, 3.8))

# A development AUROC
ax = axs[0]
x = np.arange(len(dev))
ax.bar(x, dev["auc"])
ax.set_xticks(x, dev["target"])
ax.set_ylim(0,1.05)
ax.set_ylabel("Development AUROC")
clean(ax)
panel(ax, "A")

# B held-out AUROC + CI
ax = axs[1]
x = np.arange(len(held))
y = held["auc"].to_numpy()
lo = y-held["auc_ci_low"].to_numpy()
hi = held["auc_ci_high"].to_numpy()-y
ax.errorbar(x,y,yerr=[lo,hi],fmt="o",capsize=4)
ax.set_xticks(x, held["target"])
ax.set_ylim(.85,1.01)
ax.set_ylabel("Held-out AUROC")
clean(ax)
panel(ax, "B")

# C held-out operating characteristics
ax = axs[2]
metrics = ["sensitivity","specificity","ppv","npv","accuracy"]
xx = np.arange(len(metrics))
width=.36
for j,(_,row) in enumerate(held.iterrows()):
    vals=[row[m] for m in metrics]
    ax.bar(xx+(j-.5)*width, vals, width=width, label=row["target"])
ax.set_xticks(xx, ["Sens.","Spec.","PPV","NPV","Acc."])
ax.set_ylim(0,1.05)
ax.set_ylabel("Held-out performance")
ax.legend(frameon=False)
clean(ax)
panel(ax, "C")

fig.tight_layout()
savefig(fig, "Fig2_frozen_detector_validation")

dev.to_csv(FSD/"Fig2A.csv",index=False)
held[["target","auc","auc_ci_low","auc_ci_high"]].to_csv(
    FSD/"Fig2B.csv",index=False
)
held.to_csv(FSD/"Fig2C.csv",index=False)

# =====================================================================
# FIGURE 3
# Brain600 perturbation sensitivity
# =====================================================================

fig, axs = plt.subplots(1,3,figsize=(12,3.8))

# A condition means
ax=axs[0]
ax.plot(
    brain_cond["condition"],
    brain_cond["mean_dice"],
    marker="o"
)
ax.set_ylabel("Mean Dice")
ax.set_xlabel("Condition")
clean(ax)
panel(ax,"A")

# B mean absolute delta
ax=axs[1]
ax.plot(
    brain_rel["condition"],
    brain_rel["mean_abs_delta"],
    marker="o"
)
ax.set_ylabel("Mean |Δ Dice| vs H0")
ax.set_xlabel("Condition")
clean(ax)
panel(ax,"B")

# C localization relation
ax=axs[2]
ax.plot(
    brain_rel["mean_box_iou_H0"],
    brain_rel["mean_abs_delta"],
    marker="o"
)
for _,row in brain_rel.iterrows():
    ax.annotate(
        row["condition"],
        (row["mean_box_iou_H0"],row["mean_abs_delta"]),
        xytext=(4,4),
        textcoords="offset points",
        fontsize=8
    )
ax.set_xlabel("Mean box IoU with H0")
ax.set_ylabel("Mean |Δ Dice|")
clean(ax)
panel(ax,"C")

fig.tight_layout()
savefig(fig,"Fig3_brain600_perturbation_sensitivity")

brain_cond.to_csv(FSD/"Fig3A.csv",index=False)
brain_rel[[
    "condition","mean_abs_delta","median_abs_delta",
    "mean_signed_delta"
]].to_csv(FSD/"Fig3B.csv",index=False)
brain_rel.to_csv(FSD/"Fig3C.csv",index=False)

# =====================================================================
# FIGURE 4
# Localization decomposition
# =====================================================================

fig, axs = plt.subplots(1,3,figsize=(12.5,3.9))

# A selected vs oracle difference
ax=axs[0]
x=np.arange(len(local))
ax.plot(x,local["selected_diff_mean"],marker="o",label="Selected")
ax.plot(x,local["oracle_diff_mean"],marker="o",label="Oracle")
ax.set_xticks(x,local["stratum"],rotation=35,ha="right")
ax.set_ylabel("Mean Dice difference")
ax.set_xlabel("Box-IoU stratum")
ax.legend(frameon=False)
clean(ax)
panel(ax,"A")

# B reduction
ax=axs[1]
ax.bar(x,local["mean_reduction"])
ax.axhline(0,linewidth=.8)
ax.set_xticks(x,local["stratum"],rotation=35,ha="right")
ax.set_ylabel("Mean reduction")
ax.set_xlabel("Box-IoU stratum")
clean(ax)
panel(ax,"B")

# C selection gap failures
ax=axs[2]
ax.plot(
    x,
    pct(local["fail_selection_gap_gt20_rate"]),
    marker="o"
)
ax.set_xticks(x,local["stratum"],rotation=35,ha="right")
ax.set_ylabel("Selection gap >0.20 (%)")
ax.set_xlabel("Box-IoU stratum")
clean(ax)
panel(ax,"C")

fig.tight_layout()
savefig(fig,"Fig4_localization_decomposition")
local.to_csv(FSD/"Fig4.csv",index=False)

# =====================================================================
# FIGURE 5
# Mechanism: language vs stochastic seed
# =====================================================================

fig, axs = plt.subplots(1,3,figsize=(12,3.8))

# A region-level box instability
ax=axs[0]
x=np.arange(len(factor_region))
w=.36
ax.bar(
    x-w/2,
    factor_region["seed_instability_mean"],
    width=w,
    label="Seed"
)
ax.bar(
    x+w/2,
    factor_region["language_instability_mean"],
    width=w,
    label="Language"
)
ax.set_xticks(x,factor_region["risk_region"])
ax.set_ylabel("Mean box instability")
ax.legend(frameon=False)
clean(ax)
panel(ax,"A")

# B case-level language vs seed
ax=axs[1]
ax.scatter(
    factor_case["seed_instability_mean"],
    factor_case["language_instability_mean"],
    s=38
)
mx=max(
    factor_case["seed_instability_mean"].max(),
    factor_case["language_instability_mean"].max()
)
ax.plot([0,mx],[0,mx],linestyle="--",linewidth=1)
ax.set_xlabel("Seed instability")
ax.set_ylabel("Language instability")
clean(ax)
panel(ax,"B")

# C trajectory dose response
ax=axs[2]
ax.plot(
    traj_dose["alpha"],
    traj_dose["corr_distance_from_A_mean"],
    marker="o",
    label="Distance from A"
)
ax.plot(
    traj_dose["alpha"],
    traj_dose["corr_distance_from_B_mean"],
    marker="o",
    label="Distance from B"
)
ax.set_xlabel("Interpolation α")
ax.set_ylabel("Correlation distance")
ax.legend(frameon=False)
clean(ax)
panel(ax,"C")

fig.tight_layout()
savefig(fig,"Fig5_language_mechanism")

factor_region.to_csv(FSD/"Fig5A.csv",index=False)
factor_case.to_csv(FSD/"Fig5B.csv",index=False)
traj_dose.to_csv(FSD/"Fig5C.csv",index=False)

# =====================================================================
# FIGURE 6
# Cross-system comparison: Brain600 vs Text3DSAM
# =====================================================================

brain_mean_range=float(brain_six.iloc[0]["mean_range"])
brain_med_range=float(brain_six.iloc[0]["median_range"])

tmean=float(
    text3d_boot.loc[
        text3d_boot.metric=="range_mean","point"
    ].iloc[0]
)
tmed=float(
    text3d_boot.loc[
        text3d_boot.metric=="range_median","point"
    ].iloc[0]
)

fig,axs=plt.subplots(1,2,figsize=(8.5,3.8))

ax=axs[0]
ax.bar(
    ["Brain600","Text3DSAM"],
    [brain_mean_range,tmean]
)
ax.set_ylabel("Mean within-case range")
clean(ax)
panel(ax,"A")

ax=axs[1]
ax.bar(
    ["Brain600","Text3DSAM"],
    [brain_med_range,tmed]
)
ax.set_ylabel("Median within-case range")
clean(ax)
panel(ax,"B")

fig.tight_layout()
savefig(fig,"Fig6_cross_architecture_comparison")

pd.DataFrame({
    "system":["Brain600","Text3DSAM"],
    "mean_case_range":[brain_mean_range,tmean],
    "median_case_range":[brain_med_range,tmed]
}).to_csv(FSD/"Fig6.csv",index=False)

# =====================================================================
# SUPPLEMENTARY FIGURE S1
# Prompt-interface trade-off
# =====================================================================

fig,axs=plt.subplots(1,2,figsize=(9,3.8))

# Restrict visual summary to B0 for readability.
b0=interface_summary[interface_summary["base"]=="B0"].copy()

ax=axs[0]
ax.bar(
    b0["interface"],
    b0["mean_dice"]
)
ax.tick_params(axis="x",rotation=35)
ax.set_ylabel("Mean Dice")
ax.set_title("B0")
clean(ax)
panel(ax,"A")

# Range20 vulnerable cases
rv=strat[
    (strat["target"]=="range20") &
    (strat["stratum"]=="vulnerable")
].copy()

ax=axs[1]
ax.bar(
    rv["interface"],
    rv["mean_range"]
)
ax.tick_params(axis="x",rotation=35)
ax.set_ylabel("Mean Dice range")
ax.set_title("Range20-vulnerable cases")
clean(ax)
panel(ax,"B")

fig.tight_layout()
savefig(fig,"FigS1_prompt_interface_tradeoff")

b0.to_csv(FSD/"FigS1A.csv",index=False)
rv.to_csv(FSD/"FigS1B.csv",index=False)

# =====================================================================
# TABLES
# =====================================================================

# Table 1 — dataset/system overview
table1=pd.DataFrame([
    ["Breast113",113,6,"Language-conditioned B0-B5 segmentation"],
    ["Brain600",600,6,"Prompt/localization perturbation series"],
    ["Text3DSAM AMOS liver",30,4,"Language-prompt robustness comparison"],
],columns=["cohort","n_cases","n_conditions","role"])
table1.to_csv(TAB/"Table1_study_overview.csv",index=False)

# Table 2 — Breast frozen detector
table2=held[[
    "target","heldout_n","events","non_events","threshold",
    "auc","auc_ci_low","auc_ci_high",
    "sensitivity","specificity","ppv","npv","accuracy",
    "tp","fp","tn","fn"
]].copy()
table2.to_csv(TAB/"Table2_breast_heldout_detector.csv",index=False)

# Table 3 — Brain600 perturbations
table3=brain_rel[[
    "condition","n","mean_abs_delta","median_abs_delta",
    "mean_signed_delta","abs_gt_010","abs_gt_020",
    "failure_success_switch","mean_box_iou_H0",
    "spearman_boxiou_absdelta"
]].copy()
table3.to_csv(TAB/"Table3_brain600_perturbation_summary.csv",index=False)

# Table 4 — Localization decomposition
local.to_csv(TAB/"Table4_localization_decomposition.csv",index=False)

# Supplementary interface table
strat.to_csv(
    TAB/"TableS1_prompt_interface_vulnerability_stratified.csv",
    index=False
)
trans.to_csv(
    TAB/"TableS2_prompt_interface_transitions.csv",
    index=False
)

# Supplementary Text3DSAM table
text3d_boot.to_csv(
    TAB/"TableS3_text3dsam_bootstrap.csv",
    index=False
)

# =====================================================================
# PROVENANCE / SHA256 MANIFEST
# =====================================================================

records=[]

for folder,kind in [
    (SRC,"frozen_source"),
    (FIG,"figure"),
    (TAB,"table"),
    (FSD,"figure_source_data")
]:
    for path in sorted(folder.glob("*")):
        if not path.is_file():
            continue

        h=hashlib.sha256()
        with open(path,"rb") as f:
            for block in iter(lambda:f.read(1024*1024),b""):
                h.update(block)

        records.append({
            "kind":kind,
            "path":str(path.relative_to(ROOT)),
            "sha256":h.hexdigest(),
            "bytes":path.stat().st_size
        })

manifest=pd.DataFrame(records)
manifest.to_csv(
    PROV/"paper_v2_artifact_manifest.csv",
    index=False
)

declaration={
    "source_policy":"All manuscript artifacts generated from frozen pre-existing CSV result tables.",
    "model_inference_performed":False,
    "thresholds_selected_or_changed":False,
    "frozen_source_files_modified":False,
    "figures_generated":True,
    "tables_generated":True,
    "figure_source_data_generated":True
}

with open(PROV/"BUILD_DECLARATION.json","w") as f:
    json.dump(declaration,f,indent=2)

print("="*100)
print("PAPER V2 — ARTIFACT BUILD COMPLETE")
print("="*100)
print()
print("Figures:")
for x in sorted(FIG.glob("*")):
    print(" ",x.relative_to(ROOT))

print()
print("Tables:")
for x in sorted(TAB.glob("*")):
    print(" ",x.relative_to(ROOT))

print()
print("Figure source data:")
for x in sorted(FSD.glob("*")):
    print(" ",x.relative_to(ROOT))

print()
print("Manifest:")
print(" ",(PROV/"paper_v2_artifact_manifest.csv").relative_to(ROOT))

print()
print("="*100)
print("DECLARATION")
print("="*100)
print("No model inference performed.")
print("No frozen threshold selected or changed.")
print("No frozen source result file modified.")
print("Figures/tables are derived only from paper_v2/source_data.")
print("="*100)
