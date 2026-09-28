import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score

# ============================================================
# PATHS
# ============================================================

DICE_PATH = Path(
    "parent_repro_brain600/v2_H0L5_analysis/"
    "brain600_v2_H0L5_casewise_dice.csv"
)

INTER_PATH = Path(
    "parent_repro_brain600/embedding_intervention_analysis/"
    "embedding_intervention_trajectory.csv"
)

OUTDIR = Path(
    "parent_repro_brain600/"
    "embedding_intervention_linkage"
)
OUTDIR.mkdir(parents=True, exist_ok=True)

JOIN_PATH = OUTDIR / "embedding_intervention_segmentation_join.csv"
BOOT_PATH = OUTDIR / "embedding_intervention_linkage_bootstrap.csv"
SUMMARY_PATH = OUTDIR / "embedding_intervention_linkage_summary.txt"

N_BOOT = 10000
SEED = 42

# ============================================================
# LOAD + EXACT JOIN
# ============================================================

dice = pd.read_csv(DICE_PATH)
inter = pd.read_csv(INTER_PATH)

dice = dice.rename(columns={"image": "filename"})

if dice["filename"].duplicated().any():
    raise RuntimeError("Duplicate filenames in casewise Dice table.")

if inter["filename"].duplicated().any():
    raise RuntimeError("Duplicate filenames in intervention table.")

d = dice.merge(
    inter,
    on="filename",
    how="inner",
    validate="one_to_one"
)

if len(d) != 600:
    raise RuntimeError(f"Expected 600 joined cases, found {len(d)}")

# ============================================================
# DEFINE TRUE SEGMENTATION VULNERABILITY
# ============================================================

conditions = ["H0", "H1", "H2", "L3", "L4", "L5"]

# Existing "range" is in DSC units [0,1].
d["dsc_range"] = d["range"]

# Direct H0 -> L5 segmentation response.
d["delta_H0_L5"] = d["L5"] - d["H0"]
d["abs_delta_H0_L5"] = np.abs(d["delta_H0_L5"])
d["loss_H0_L5"] = np.maximum(d["H0"] - d["L5"], 0.0)

# Severe prompt sensitivity.
d["range20"] = (d["dsc_range"] >= 0.20).astype(int)
d["range50"] = (d["dsc_range"] >= 0.50).astype(int)

# Same conceptual catastrophic-switch definition used previously:
# at least one good segmentation and at least one near-total failure.
mx = d[conditions].max(axis=1)
mn = d[conditions].min(axis=1)

d["catastrophic_switch"] = (
    (mx >= 0.80) &
    (mn < 0.10)
).astype(int)

# Direct H0 -> L5 severe deterioration.
d["H0L5_loss20"] = (
    d["delta_H0_L5"] <= -0.20
).astype(int)

d["H0L5_loss50"] = (
    d["delta_H0_L5"] <= -0.50
).astype(int)

# ============================================================
# INTERVENTION PREDICTORS
# ============================================================

predictors = [
    "endpoint_disagreement",
    "max_step_disagreement",
    "trajectory_deviation",
]

continuous_targets = [
    "dsc_range",
    "abs_delta_H0_L5",
    "loss_H0_L5",
]

binary_targets = [
    "range20",
    "range50",
    "catastrophic_switch",
    "H0L5_loss20",
    "H0L5_loss50",
]

# ============================================================
# HELPERS
# ============================================================

def pearson(x, y):
    r = pearsonr(x, y)
    return float(r[0]), float(r[1])

def spearman(x, y):
    r = spearmanr(x, y)
    return float(r[0]), float(r[1])

def auc(y, score):
    y = np.asarray(y)
    score = np.asarray(score)

    if len(np.unique(y)) != 2:
        return np.nan

    return float(roc_auc_score(y, score))

def ci(a):
    a = np.asarray(a, dtype=float)
    a = a[np.isfinite(a)]

    if len(a) == 0:
        return np.nan, np.nan

    return (
        float(np.percentile(a, 2.5)),
        float(np.percentile(a, 97.5))
    )

# ============================================================
# BASIC OUTPUT
# ============================================================

lines = []

def emit(x=""):
    print(x)
    lines.append(str(x))

emit("=" * 100)
emit("MEDCLIP-SAMV2 — CONTROLLED EMBEDDING INTERVENTION ↔ SEGMENTATION VULNERABILITY")
emit("=" * 100)

emit(f"Joined cases = {len(d)}")
emit()

emit("=" * 100)
emit("TRUE SEGMENTATION VULNERABILITY PREVALENCE")
emit("=" * 100)

for target in binary_targets:
    n = int(d[target].sum())
    emit(
        f"{target:24s}: "
        f"{n:3d}/{len(d)} = {100*n/len(d):6.2f}%"
    )

# ============================================================
# OBSERVED CORRELATIONS
# ============================================================

emit()
emit("=" * 100)
emit("CONTINUOUS ASSOCIATIONS")
emit("=" * 100)

observed_rows = []

for pred in predictors:

    emit()
    emit(pred)

    for target in continuous_targets:

        pr, pp = pearson(d[pred], d[target])
        sr, sp = spearman(d[pred], d[target])

        observed_rows.append({
            "analysis": "continuous",
            "predictor": pred,
            "target": target,
            "pearson_r": pr,
            "pearson_p": pp,
            "spearman_rho": sr,
            "spearman_p": sp,
            "auc": np.nan
        })

        emit(
            f"  {target:20s} "
            f"Pearson r={pr:+.4f} p={pp:.3e} | "
            f"Spearman rho={sr:+.4f} p={sp:.3e}"
        )

# ============================================================
# OBSERVED AUROCS
# ============================================================

emit()
emit("=" * 100)
emit("VULNERABILITY DISCRIMINATION — AUROC")
emit("=" * 100)

for pred in predictors:

    emit()
    emit(pred)

    for target in binary_targets:

        a = auc(d[target], d[pred])
        events = int(d[target].sum())

        observed_rows.append({
            "analysis": "binary",
            "predictor": pred,
            "target": target,
            "pearson_r": np.nan,
            "pearson_p": np.nan,
            "spearman_rho": np.nan,
            "spearman_p": np.nan,
            "auc": a
        })

        emit(
            f"  {target:24s} "
            f"events={events:3d}/{len(d)} "
            f"AUC={a:.4f}"
        )

# ============================================================
# BOOTSTRAP
# ============================================================

emit()
emit("=" * 100)
emit(f"IMAGE-LEVEL BOOTSTRAP 95% CI — {N_BOOT:,} RESAMPLES")
emit("=" * 100)

rng = np.random.default_rng(SEED)
n = len(d)

boot_rows = []

for pred in predictors:

    emit()
    emit(pred)

    x_all = d[pred].to_numpy(dtype=float)

    # Continuous outcomes
    for target in continuous_targets:

        y_all = d[target].to_numpy(dtype=float)

        prs = []
        srs = []

        for _ in range(N_BOOT):
            idx = rng.integers(0, n, n)

            x = x_all[idx]
            y = y_all[idx]

            # Avoid pathological constant bootstrap samples.
            if np.std(x) == 0 or np.std(y) == 0:
                continue

            prs.append(pearsonr(x, y)[0])
            srs.append(spearmanr(x, y)[0])

        pr_obs = pearsonr(x_all, y_all)[0]
        sr_obs = spearmanr(x_all, y_all)[0]

        pr_lo, pr_hi = ci(prs)
        sr_lo, sr_hi = ci(srs)

        boot_rows.append({
            "analysis": "continuous",
            "predictor": pred,
            "target": target,
            "estimate": pr_obs,
            "metric": "Pearson_r",
            "ci_low": pr_lo,
            "ci_high": pr_hi,
            "events": np.nan,
            "n": n
        })

        boot_rows.append({
            "analysis": "continuous",
            "predictor": pred,
            "target": target,
            "estimate": sr_obs,
            "metric": "Spearman_rho",
            "ci_low": sr_lo,
            "ci_high": sr_hi,
            "events": np.nan,
            "n": n
        })

        emit(
            f"  {target:20s} "
            f"Pearson={pr_obs:+.4f} "
            f"[{pr_lo:+.4f}, {pr_hi:+.4f}] | "
            f"Spearman={sr_obs:+.4f} "
            f"[{sr_lo:+.4f}, {sr_hi:+.4f}]"
        )

    # Binary outcomes
    for target in binary_targets:

        y_all = d[target].to_numpy(dtype=int)

        aucs = []

        for _ in range(N_BOOT):
            idx = rng.integers(0, n, n)

            x = x_all[idx]
            y = y_all[idx]

            if len(np.unique(y)) < 2:
                continue

            aucs.append(
                roc_auc_score(y, x)
            )

        obs = roc_auc_score(y_all, x_all)
        lo, hi = ci(aucs)

        events = int(y_all.sum())

        boot_rows.append({
            "analysis": "binary",
            "predictor": pred,
            "target": target,
            "estimate": obs,
            "metric": "AUROC",
            "ci_low": lo,
            "ci_high": hi,
            "events": events,
            "n": n
        })

        emit(
            f"  {target:24s} "
            f"AUC={obs:.4f} "
            f"95% CI [{lo:.4f}, {hi:.4f}] "
            f"events={events}"
        )

# ============================================================
# ENDPOINT DISAGREEMENT QUINTILES
# ============================================================

emit()
emit("=" * 100)
emit("CONTROLLED ENDPOINT-DISAGREEMENT QUINTILES")
emit("=" * 100)

# rank(method=first) guarantees five groups even if tied values occur.
d["intervention_quintile"] = pd.qcut(
    d["endpoint_disagreement"].rank(method="first"),
    5,
    labels=False
) + 1

for q in range(1, 6):

    z = d[d["intervention_quintile"] == q]

    emit(
        f"Q{q} "
        f"N={len(z):3d} | "
        f"intervention-dis={z['endpoint_disagreement'].mean():.4f} | "
        f"mean DSC range={100*z['dsc_range'].mean():6.2f} pp | "
        f"mean |H0-L5|={100*z['abs_delta_H0_L5'].mean():6.2f} pp | "
        f"range>=20pp={100*z['range20'].mean():5.1f}% | "
        f"range>=50pp={100*z['range50'].mean():5.1f}% | "
        f"cat.switch={100*z['catastrophic_switch'].mean():5.1f}%"
    )

# ============================================================
# TOP 10% INTERVENTION RESPONSE
# ============================================================

emit()
emit("=" * 100)
emit("TOP 10% CONTROLLED INTERVENTION-RESPONSE GROUP")
emit("=" * 100)

cut = d["endpoint_disagreement"].quantile(0.90)
top = d[d["endpoint_disagreement"] >= cut]

emit(f"Threshold endpoint disagreement = {cut:.6f}")
emit(f"N = {len(top)}")
emit(f"Mean DSC range = {100*top['dsc_range'].mean():.2f} pp")
emit(f"Mean |H0-L5| DSC = {100*top['abs_delta_H0_L5'].mean():.2f} pp")
emit(f"Range >=20pp = {100*top['range20'].mean():.2f}%")
emit(f"Range >=50pp = {100*top['range50'].mean():.2f}%")
emit(f"Catastrophic switch = {100*top['catastrophic_switch'].mean():.2f}%")
emit(f"H0->L5 loss >=20pp = {100*top['H0L5_loss20'].mean():.2f}%")
emit(f"H0->L5 loss >=50pp = {100*top['H0L5_loss50'].mean():.2f}%")

# ============================================================
# MONOTONIC VS NON-MONOTONIC CASES
# ============================================================

emit()
emit("=" * 100)
emit("FULLY MONOTONIC VS NON-MONOTONIC INTERVENTION TRAJECTORIES")
emit("=" * 100)

for flag in [1, 0]:

    z = d[d["fully_monotonic"] == flag]

    label = "Fully monotonic" if flag == 1 else "Non-monotonic"

    emit(
        f"{label:18s} "
        f"N={len(z):3d} | "
        f"DSC range={100*z['dsc_range'].mean():6.2f} pp | "
        f"|H0-L5|={100*z['abs_delta_H0_L5'].mean():6.2f} pp | "
        f"range>=20pp={100*z['range20'].mean():5.1f}% | "
        f"cat.switch={100*z['catastrophic_switch'].mean():5.1f}%"
    )

# ============================================================
# TOP CASES
# ============================================================

emit()
emit("=" * 100)
emit("TOP 20 CASES BY CONTROLLED ENDPOINT RESPONSE")
emit("=" * 100)

top20 = d.sort_values(
    "endpoint_disagreement",
    ascending=False
).head(20)

for rank, (_, r) in enumerate(top20.iterrows(), 1):

    emit(
        f"{rank:02d}. {r['filename']:<10s} | "
        f"intervention-dis={r['endpoint_disagreement']:.4f} | "
        f"max-step={r['max_step_disagreement']:.4f} | "
        f"traj-dev={r['trajectory_deviation']:.4f} | "
        f"DSC-range={100*r['dsc_range']:6.2f} pp | "
        f"H0={100*r['H0']:6.2f} | "
        f"L5={100*r['L5']:6.2f} | "
        f"Δ={100*r['delta_H0_L5']:+6.2f} pp"
    )

# ============================================================
# SAVE
# ============================================================

pd.DataFrame(observed_rows).to_csv(
    OUTDIR / "embedding_intervention_observed_statistics.csv",
    index=False
)

pd.DataFrame(boot_rows).to_csv(
    BOOT_PATH,
    index=False
)

d.to_csv(
    JOIN_PATH,
    index=False
)

with open(SUMMARY_PATH, "w") as f:
    f.write("\n".join(lines) + "\n")

emit()
emit("=" * 100)
emit("SAVED")
emit("=" * 100)
emit(f"JOIN    = {JOIN_PATH}")
emit(
    "STATS   = "
    f"{OUTDIR / 'embedding_intervention_observed_statistics.csv'}"
)
emit(f"BOOT    = {BOOT_PATH}")
emit(f"SUMMARY = {SUMMARY_PATH}")
emit()
emit("DONE")
